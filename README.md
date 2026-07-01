# Redrob Senior AI Engineer — Candidate Ranker

Ranks 100K candidates against the Senior AI Engineer JD, outputs a top-100 CSV in under 5 minutes on CPU. No LLM calls. No network during ranking.

---

## How to reproduce

**Step 1 — precompute embeddings (one-time, ~90 min on CPU)**

```bash
python redrob-ranker/precompute_embeddings.py
```

This downloads `all-MiniLM-L6-v2` (≈80 MB) on first run, embeds all 100K candidate profiles, and writes three files to `data/`:

- `candidate_embeddings.npy` — float32 [100000, 384]
- `candidate_ids.json` — row-index map
- `jd_embedding.npy` — the distilled JD vector

You only run this once. If you add new candidates later, pass `--ids CAND_XXXX,...` to append without re-embedding everything.

**Step 2 — rank (< 5 min)**

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

Streams candidates line by line (never loads the 465 MB file into memory), scores each one, keeps a fixed-size 100-slot heap, writes `submission.csv` with 100 rows, and prints wall-clock time.

---

## Scoring formula

```
total_score = base_fit_score × behavioral_multiplier × consistency_factor
```

### base_fit_score — five weighted components

| Component | Weight | What it measures |
|---|---|---|
| `title_career_fit` | 0.35 | Current title + full career history relevance to AI/ML/retrieval work |
| `semantic_fit` | 0.25 | Cosine similarity of candidate text to a distilled JD embedding |
| `experience_fit` | 0.15 | Smooth curve over years_of_experience; sweet spot 6–8 yrs |
| `production_evidence` | 0.15 | Unique production-signal keywords in career descriptions (not skills lists) |
| `location_fit` | 0.10 | Pune/Noida → 1.0; other India cities → 0.80; India + willing to relocate → 0.65 |

Weights sum to exactly 1.0. Disqualifiers (see below) can override or multiply `base_fit_score` before behavioral and consistency are applied.

---

## Component design

### title_career_fit (0.35)

The JD is explicit: "Marketing Manager is not a fit no matter how perfect their skill list looks." So the current title carries real signal. But the JD also says career history can rescue an irrelevant title if it shows a recommendation system or search work — so I blend both.

**Title gradient** (7 tiers):

| Tier | Examples | Base score |
|---|---|---|
| 1 — Core AI/ML | ML Engineer, NLP Scientist, Applied Scientist, Search Engineer | 0.88–0.95 |
| 2 — Data/ML Platform | Data Engineer, MLOps Engineer, ML Infrastructure | 0.85–0.92 |
| 3 — Backend/Cloud/DevOps | Software Engineer, SRE, Platform Engineer | 0.50–0.58 |
| 4 — Frontend/Mobile/QA | iOS Developer, QA Engineer, React Native | 0.28–0.35 |
| 5 — Other technical | Generic Developer, Architect, Analyst | 0.12–0.19 |
| 6 — Technical management | CTO, Engineering Manager | 0.10 |
| IRRELEVANT | HR, Marketing, Accountant, etc. | 0.05 |

`senior`, `staff`, `principal`, or `lead` anywhere in the title adds +0.07 within the tier (no tier-jumping).

I deliberately left no 0.5 default floor. A frontend developer who is great at their job should score lower than a mediocre ML engineer for this specific role, and the gradient reflects that.

**Career blend**: career history is scored entry-by-entry with recency weights (1.0, 0.80, 0.65, 0.52, 0.42, ...). Services companies (Infosys, Wipro, TCS, etc.) apply a 0.65× multiplier to that entry — not a zero, because the JD's own disqualifier is for an *entirely* services career, not one services role. AI/ML keyword density in the role description adds up to +0.25 boost per entry.

If the current title is irrelevant (score < 0.15), career can rescue up to 0.60 max. Otherwise: 40% current title, 60% career history weighted average.

### semantic_fit (0.25)

The keyword-overlap approach I started with had a problem: a candidate who lists "FAISS, Pinecone, embedding" in their skills will score identically to someone who actually *describes shipping an embedding retrieval system in production*. I wanted to reward the latter.

**Primary path**: embed the candidate's summary + all career description text with `all-MiniLM-L6-v2` (384-dim, fits in CPU memory, encodes ~2K candidates/sec). Compare against a pre-embedded JD text using cosine similarity (dot product of L2-normalised vectors).

The JD text I embed is intentionally production-focused and contains zero aspirational language — it describes what a hired engineer will *do*, not what they're *interested in*. This means the cosine score naturally penalises candidates who write about learning/aspiring/transitioning even before the explicit negative-signal pass runs.

**Negative-signal penalty**: if the candidate's text contains phrases like "transitioning to AI", "building competence", "aspiring", "kaggle", "currently learning", "excited to learn" — each unique match reduces the semantic score by 10%, floored at 55%. These phrases indicate someone talking *about* ML rather than someone who has *shipped* ML. The JD is explicit that it wants people for whom "retrieval was already your job before it became fashionable."

**Fallback**: if embeddings aren't precomputed for a candidate, the score falls back to a keyword-overlap proxy against `HARD_REQUIREMENTS` signal terms and `PRODUCTION_SIGNAL_KEYWORDS`. See [Known limitations](#known-limitations).

### experience_fit (0.15)

Piecewise-linear curve:

- 0–4 years: 0.20 → 0.60 (too junior, but not zero)
- 4–5 years: 0.60 → 0.85 (approaching ideal)
- 5–6 years: 0.85 → 1.00
- 6–8 years: 1.00 (sweet spot)
- 8–9 years: 1.00 → 0.95 (slight declining edge)
- 9–12 years: 0.95 → 0.75 (over-experienced risk)
- 12+ years: declining further, floor at 0.50

The JD says "5–9 years" with a note that some people hit senior judgment at 4, so I don't cliff-edge at 4 or 9. Missing YoE defaults to 0.30 — penalised but not zeroed.

### production_evidence (0.15)

Counts unique `PRODUCTION_SIGNAL_KEYWORDS` in career descriptions and the profile summary. Saturates at 20 unique hits → 1.0. Deliberately excludes the skills list, which is easy to stuff. The idea: someone who has actually shipped will naturally use words like "deployed", "latency", "A/B test", "serving", "throughput", "rollout" when describing what they did at work. Someone who learned from a tutorial usually won't.

### location_fit (0.10)

- Pune or Noida in `location` → 1.00
- Hyderabad, Mumbai, Bangalore, Delhi/NCR, Chennai → 0.80
- India, `willing_to_relocate = true` → 0.65
- India, not relocating → 0.40
- Outside India, willing to relocate → 0.35
- Outside India, not relocating → 0.15

---

## Behavioral multiplier

Applied as a multiplier to the whole score after base_fit is computed. Range: 0.5–1.0 (can never boost above base, only reduce).

Four signals from `redrob_signals`, each mapped to tiers:

**recruiter_response_rate (RRR)** — how often this candidate responds to recruiter messages. A candidate who ghosts recruiters 80% of the time is a pipeline risk regardless of how good their skills are. Tiers: ghost (<10%), low (<25%), mid (<50%), high (<75%), very_high (≥75%).

**last_active_days** — days since the candidate was last active on the platform. Active within the last month is a strong signal they're actually looking. Six tiers from very_fresh (<30d) down to very_stale (≥270d).

**open_to_work_flag** — binary. Not a knockout if false (they might be passively open), but true gives a multiplier boost.

**interview_completion_rate (ICR)** — how reliably they show up and complete scheduled interviews. Below 40% is penalised significantly (unreliable). Above 85% is rewarded (highly_reliable).

The four per-signal multipliers are weighted-averaged using weights from `BEHAVIORAL_THRESHOLDS` in `jd_requirements.py`.

---

## Consistency factor — honeypot and impossible profile detection

Applied as a multiplier after behavioral. Starts at 1.0 and deducts per violation.

**Check (a) — expert skill with zero duration** (primary honeypot pattern): A skill listed as `proficiency: expert` or `advanced` but with `duration_months: 0` is physically impossible. You cannot be an expert in something you have spent zero time on. Each such skill deducts 0.15, capped at 0.50 total deduction. These skills also get excluded from keyword scoring and reasoning — they're treated as if they don't exist.

**Check (b) — career duration sum vs YoE**: If the sum of `duration_months` across all career entries is more than 1.5× the claimed `years_of_experience * 12` (plus 24 months buffer for legitimate overlap), deducts 0.15. The buffer accounts for concurrent consulting/side roles that some platforms count separately.

**Check (c) — impossible date ranges**: `start_date > end_date` in a completed role. Each violation deducts 0.10, capped at 0.30.

**Check (d) — impossible tenure**: Any single role with `duration_months > 480` (40 years). Deducts 0.35. One flag is enough.

---

## Disqualifiers

Applied to `base_fit_score` before behavioral and consistency.

### Hard disqualifier — `pure_research`

If zero `PRODUCTION_SIGNAL_KEYWORDS` appear anywhere in the candidate's career descriptions or summary, the base score is set to 0.02. That's it. No recovery. The JD is explicit: "We will not move forward with candidates who have never shipped anything." A publications list with no deployed system is not enough.

### Soft disqualifiers — multiplicative penalties

**`services_only_career` (×0.50)**: Every single career entry is at an IT services or consulting firm (Infosys, Wipro, TCS, Cognizant, Accenture, etc.). The JD carve-out: "if you're currently at one of these but have prior product-company experience, that's fine." So this only fires when *all* entries are services, not just the current role.

**`title_chaser` (×0.75)**: Average tenure across completed roles (minimum 3 to qualify) is under 18 months. Frequent job changes at < 1.5 years average suggest either lack of depth or a pattern of overselling at interviews. Doesn't fire on early-career candidates who have fewer than 3 completed roles.

**`recent_llm_wrapper_only` (×0.60)**: LangChain/OpenAI wrapper terms present in the profile AND no pre-LLM ML production history found. The JD: "Unless you can demonstrate substantial pre-LLM-era ML production experience." Someone whose entire ML career is ChatGPT API and LangChain has not necessarily learned the retrieval fundamentals this role needs. The AND condition is important — someone who used LangChain after a decade of NLP/IR work is fine.

**`cv_speech_robotics_without_nlp_ir` (×0.65)**: Computer vision, speech, or robotics terms dominate (≥3 hits) AND NLP/IR terms are nearly absent (< 3 hits) AND CV hits outnumber NLP hits by more than 3×. The skill-set for vision tasks (CNNs, detection, segmentation) doesn't transfer strongly to text retrieval and ranking. If they have both, this doesn't fire.

Soft penalties are multiplicative and can stack. Two soft disqualifiers would give a combined multiplier of 0.50 × 0.60 = 0.30 on top of base_fit_score.

---

## Why no LLM calls at ranking time

The spec requires the ranking step to run in under 5 minutes on CPU. LLM inference on 100K candidates would take hours, and using an API introduces network latency, rate limits, and non-determinism. The scoring needs to be reproducible — the same candidate should get the same score on every run.

Everything here is deterministic given fixed data: the keyword lists, the tier thresholds, the embedding vectors, and `_TODAY = date(2026, 7, 1)`. You can re-run `rank.py` on the same JSONL and get the same CSV.

The reasoning column in the CSV is also generated from actual computed values (scores, skill names, company names, YoE), not from an LLM — so it's honest about what the scorer actually saw, and it's fast.

---

## Known limitations

**Semantic fallback for non-precomputed candidates**: if `precompute_embeddings.py` has only been run with `--limit N` (e.g. the first 50 candidates), all other candidates fall back to `_keyword_semantic_fit` — a keyword-overlap proxy that scores on `HARD_REQUIREMENTS` signal term coverage and `PRODUCTION_SIGNAL_KEYWORDS` density. The fallback is reasonably correlated with the embedding score but doesn't capture phrasing or context. Run the full precompute to get real cosine similarity across all 100K.

**Embeddings encode what candidates *write*, not what they know**: the semantic score is only as good as the text in `summary` and career `description` fields. A candidate who writes terse bullet points will score lower than one who writes descriptive paragraphs, independent of actual ability.

**Negative-signal penalty can misfire on expert candidates**: if an experienced engineer mentions "currently learning Rust" or "experimenting with multi-agent frameworks" as an aside in their summary, the phrase match might lightly penalise their semantic score even though their overall profile is strong. The floor of 0.55× and the per-phrase cap limit the damage, but it's a known rough edge.

**Behavioral multiplier is a proxy for engagement, not ability**: a candidate who is passive (low RRR, not open_to_work) scores lower here even if they're outstanding technically. The signal is about pipeline predictability, not quality — the JD's framing is "we only want to move fast on candidates who are actually available."

---

## Files

```
redrob-ranker/
  scorer.py               — all component scoring logic
  rank.py                 — streaming top-100 pipeline entry point
  precompute_embeddings.py — one-time embedding precompute
  jd_requirements.py      — all JD-derived constants (keywords, thresholds, etc.)
  README.md               — this file

data/                     — not in git (see .gitignore)
  candidates.jsonl        — full 100K candidate dataset
  candidate_embeddings.npy
  candidate_ids.json
  jd_embedding.npy

submission.csv            — ranked output (100 rows)
```
