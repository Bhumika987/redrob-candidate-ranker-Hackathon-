"""
scorer.py
Core ranking logic for the Redrob Senior AI Engineer JD.

Formula:
    total_score = base_fit_score * behavioral_multiplier * consistency_factor

    base_fit_score (0-1) — weighted sum of five components, with DISQUALIFIERS
        applied as multiplicative penalties or a near-zero override:
          title_career_fit    0.35
          semantic_fit        0.25
          experience_fit      0.15
          production_evidence 0.15
          location_fit        0.10

    behavioral_multiplier (0.5-1.0) — availability/engagement signals from
        BEHAVIORAL_THRESHOLDS: recruiter response rate, recency, open_to_work,
        interview completion.

    consistency_factor (0-1, default 1.0) — honeypot / impossible-profile
        detector: expert skills with duration_months=0, career-duration vs YoE
        mismatch, impossible date ranges.

No LLM calls. No network. Pure Python, CPU-only.
"""

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Allow: python redrob-ranker/scorer.py  OR  import scorer
sys.path.insert(0, str(Path(__file__).parent))

from jd_requirements import (
    BEHAVIORAL_THRESHOLDS,
    CV_SPEECH_ROBOTICS_TERMS,
    HARD_REQUIREMENTS,
    IRRELEVANT_TITLES,
    LANGCHAIN_WRAPPER_TERMS,
    NLP_IR_TERMS,
    PRE_LLM_ML_TERMS,
    PREFERRED_LOCATIONS,
    PRODUCTION_SIGNAL_KEYWORDS,
    SERVICES_COMPANIES,
    TARGET_EXPERIENCE_YEARS,
    TITLE_CHASER_THRESHOLD_MONTHS,
)

# =============================================================================
# Module-level constants
# =============================================================================

# Component weights — must sum to 1.0.
COMPONENT_WEIGHTS: dict = {
    "title_career_fit":    0.35,
    "semantic_fit":        0.25,
    "experience_fit":      0.15,
    "production_evidence": 0.15,
    "location_fit":        0.10,
}
assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

# Production-evidence component saturates at this many unique keyword hits.
_PROD_EVIDENCE_SAT = 20

# Base score assigned when a hard disqualifier fires.
_HARD_DISQ_BASE = 0.02

# Reference date — fixed for reproducibility across the whole scoring run.
_TODAY = date(2026, 7, 1)

# Title keyword groups for _score_single_title().
_HIGHLY_RELEVANT_TITLE_KW: tuple = (
    "ml engineer", "machine learning engineer", "machine learning",
    "ai engineer", "ai research engineer", "ai research",
    "nlp engineer", "nlp scientist", "nlp",
    "search engineer", "ranking engineer", "recommendation engineer",
    "applied scientist", "applied ml", "applied ai",
    "data scientist", "research scientist",
    "retrieval engineer", "information retrieval",
    "senior data engineer", "analytics engineer",
)

_GENERAL_TECH_TITLE_KW: tuple = (
    "software engineer", "software developer",
    "backend engineer", "backend developer",
    "full stack", "fullstack",
    "platform engineer", "cloud engineer",
    "devops engineer", "site reliability", " sre",
    "systems engineer", "infrastructure engineer",
)

# Titles where the core skill (UI rendering, native apps, manual QA) rarely
# transfers to embeddings/retrieval/ranking work.  Scored below backend.
_FRONTEND_MOBILE_QA_KW: tuple = (
    "frontend engineer", "front-end engineer",
    "frontend developer", "front-end developer",
    "ui engineer", "ux engineer", "ui developer",
    "mobile developer", "mobile engineer",
    "ios developer", "ios engineer",
    "android developer", "android engineer",
    "flutter developer", "react native",
    "qa engineer", "quality assurance",
    "test engineer", "sdet", "automation engineer",
    "java developer", ".net developer", "c# developer",
)

# Career-description keywords that signal AI/ML work within a role.
_AI_CAREER_KW: tuple = (
    "machine learning", "deep learning", "neural",
    "embedding", "embeddings", "retrieval", "ranking",
    "recommendation", "vector", "semantic search",
    "language model", "transformer", "bert", "llm", "gpt",
    "nlp", "natural language", "text classification",
    "fine-tun", "model training", "model serving",
    "inference", "feature engineering", "feature store",
    "a/b test", "personalization", "mlops",
    "recommendation engine", "search relevance",
    "predictive model", "data science",
)

# Leadership title terms that suggest code inactivity (see disqualifier).
_LEADERSHIP_KW: tuple = (
    "vp ", "vice president", "director of", "head of",
    "chief", " cto", " cio", "chief technology", "chief information",
)

# =============================================================================
# Embedding configuration (used by semantic_fit)
# =============================================================================

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Precomputed artefact paths (written by precompute_embeddings.py).
_EMBEDDINGS_NPY     = Path(__file__).parent.parent / "data" / "candidate_embeddings.npy"
_CANDIDATE_IDS_JSON = Path(__file__).parent.parent / "data" / "candidate_ids.json"
_JD_EMBEDDING_NPY   = Path(__file__).parent.parent / "data" / "jd_embedding.npy"

# Distilled JD text for embedding.
# Packs what we WANT (production, shipped, hands-on) — absent of aspiration language.
# Keep under ~200 tokens so it fits within all-MiniLM-L6-v2's 256-token window.
_JD_TEXT = (
    "Senior AI engineer who has shipped production machine learning systems. "
    "Built and deployed embeddings-based retrieval systems to real users at product companies. "
    "Operated vector search infrastructure: Pinecone, Milvus, Qdrant, Weaviate, Elasticsearch, FAISS. "
    "Designed hybrid search combining dense vector retrieval and BM25 sparse search. "
    "Created offline and online evaluation frameworks using NDCG, MRR, MAP and A/B testing. "
    "Implemented ranking and recommendation systems at scale with measurable engagement metrics. "
    "Strong Python engineer writing production code daily; five to nine years of experience. "
    "Not a researcher or academic. Not transitioning from another field. "
    "Senior individual contributor who codes, deploys, and operates ML systems in production. "
    "Experience with LLM fine-tuning using LoRA, QLoRA or PEFT. "
    "Built feature stores, data pipelines, and low-latency model serving infrastructure. "
    "Monitored embedding drift, index refresh cycles, and retrieval quality regression in production."
)

# Phrases that indicate aspiration / career-transition rather than production experience.
# Source: JD wants people who understood retrieval "before it became fashionable",
# not career-changers or recent LLM enthusiasts.
# Each unique hit applies a _NEG_PENALTY_PER_PHRASE multiplicative reduction.
NEGATIVE_SIGNAL_PHRASES: list = [
    # Career-change language
    "transitioning", "looking to transition", "transition to ai",
    "transition into", "career change", "career pivot", "pivoting to",
    # Competence-building language
    "building competence", "building my skills", "building skills in",
    "developing skills", "upskilling",
    # Aspiration
    "aspiring", "aspire to", "hope to", "hoping to", "plan to", "planning to",
    # Self-directed / learning language
    "self-directed", "self-taught in", "currently learning",
    "learning machine learning", "learning ai", "learning ml",
    "self-study", "online course", "side project", "kaggle",
    # Interested-in phrases (exploration, not production)
    "interested in moving", "interested in transitioning",
    "interested in ai", "interested in machine learning",
    "curious about ai", "excited to learn",
    # New-to language
    "new to ai", "new to machine learning", "new to ml",
    "getting into ai", "getting into machine learning",
    "just started", "recently started learning",
    # Tinkering without production context
    "experimenting with chatgpt", "experimenting with llms",
    "experimenting with ai tools", "played around with", "tinkered with",
]

_NEG_PENALTY_PER_PHRASE = 0.10  # each unique hit reduces semantic_fit by 10%
_NEG_PENALTY_FLOOR      = 0.55  # minimum multiplier (floor for heavily aspirational text)

# Module-level lazy-loaded embedding state (initialised on first semantic_fit call).
_embed_model    = None         # SentenceTransformer instance
_jd_embedding   = None         # np.ndarray [384]
_embedding_index: dict | None = None   # {candidate_id: np.ndarray[384]}

# =============================================================================
# Internal helpers
# =============================================================================

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_services_company(company: str) -> bool:
    """Case-insensitive substring match against SERVICES_COMPANIES."""
    cn = company.lower().strip()
    return any(sc in cn for sc in SERVICES_COMPANIES)


def _get_trusted_skills(candidate: dict) -> list:
    """
    Return skills with duration_months > 0.
    Skill trust rule: skills with duration_months == 0 are honeypot indicators
    and must not contribute to keyword or proficiency scoring.
    """
    return [
        s for s in candidate.get("skills", [])
        if s.get("duration_months", 0) > 0
    ]


def _get_career_text(candidate: dict) -> str:
    """All career-description + summary text. Used for production-signal counting."""
    parts = [candidate.get("profile", {}).get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
        parts.append(job.get("industry", ""))
    return " ".join(parts)


def _get_full_text(candidate: dict, trusted_only: bool = True) -> str:
    """
    All searchable text: headline + summary + career + certifications + trusted skills.
    trusted_only=True excludes skills with duration_months==0 (default; safest).
    """
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
        parts.append(cert.get("issuer", ""))
    if trusted_only:
        parts.extend(s["name"] for s in _get_trusted_skills(candidate))
    else:
        parts.extend(s["name"] for s in candidate.get("skills", []))
    return " ".join(parts)


def _score_single_title(title: str) -> float:
    """
    Score one job title for relevance to the Senior AI Engineer JD.
    Returns 0-1.  Gradient design:
        AI/ML/DS/Data-Eng    → 0.85–1.00  (Tier 1–2)
        Backend/Cloud/DevOps → 0.50–0.58  (Tier 3)
        Frontend/Mobile/QA   → 0.28–0.35  (Tier 4)
        Catch-all technical  → 0.12–0.19  (Tier 5)
        IRRELEVANT_TITLES    → 0.05       (floor; career history can still carve out)
    Seniority (senior/staff/principal/lead) adds +0.07 within each tier.
    """
    t = title.lower().strip()

    # IRRELEVANT: JD says 'Marketing Manager is not a fit, no matter how
    # perfect their skill list looks.' Tiny non-zero so career_history can
    # still provide a minimal override.
    if t in IRRELEVANT_TITLES or any(irr in t for irr in IRRELEVANT_TITLES):
        return 0.05

    senior = 0.07 if any(s in t for s in ("senior", "staff", "principal", "lead")) else 0.0

    # Tier 1 — Core AI/ML/NLP/Search/IR roles (0.88–0.95)
    if any(kw in t for kw in _HIGHLY_RELEVANT_TITLE_KW):
        return min(1.0, 0.88 + senior)

    # Tier 2 — Data Engineering / MLOps (0.85–0.92)
    # Pipelines, feature stores, and index refresh are first-class concerns in
    # the JD; a data/ML-platform engineer is a near-ideal background.
    if any(kw in t for kw in ("data engineer", "data architect",
                               "data platform", "ml platform",
                               "mlops", "ml infra")):
        return min(0.93, 0.85 + senior)

    # Tier 3 — Backend / Cloud / DevOps (0.50–0.58)
    # Solid engineering foundations, but no direct ML/retrieval signal.
    if any(kw in t for kw in _GENERAL_TECH_TITLE_KW):
        return min(0.65, 0.50 + senior)

    # Tier 4 — Frontend / Mobile / QA (0.28–0.35)
    # Core skill (UI, native apps, browser testing) rarely transfers to ranking.
    if any(kw in t for kw in _FRONTEND_MOBILE_QA_KW):
        return min(0.40, 0.28 + senior)

    # Tier 5 — Other technical (catch-all): has an engineering role but
    # unrelated to ML/data (0.12–0.19).
    if any(kw in t for kw in ("developer", "engineer", "programmer",
                               "architect", "scientist", "technical",
                               "quantitative", "analyst")):
        return min(0.22, 0.12 + senior)

    # Tier 6 — Technical management; code-inactivity risk.
    if any(kw in t for kw in ("cto", "tech lead", "engineering manager",
                               "technical manager", "product manager")):
        return 0.10

    return 0.07  # uncategorised


def _score_career_entry(job: dict) -> float:
    """
    Score one career_history entry for AI/ML relevance.
    Combines title relevance, company type, and description AI-signal keywords.
    """
    title_score = _score_single_title(job.get("title", ""))

    # Services companies get a multiplier penalty (not a zero): the JD
    # disqualifies only a career that is *entirely* at services firms, so
    # individual roles just lose some weight.
    company_mult = 0.65 if _is_services_company(job.get("company", "")) else 1.0

    # AI/ML keyword density in the role description.
    desc = job.get("description", "").lower()
    ai_hits = sum(1 for kw in _AI_CAREER_KW if kw in desc)
    desc_boost = min(0.25, ai_hits * 0.06)   # saturates at ~4+ AI keywords

    return min(1.0, title_score * company_mult + desc_boost)


# =============================================================================
# Component 1: title_career_fit
# =============================================================================

def _compute_title_career_fit(candidate: dict) -> float:
    """
    Is this person an AI/ML/data/software IC with relevant career history?

    Current title is the strong prior (JD: 'Marketing Manager is not a fit
    no matter how perfect their skill list looks'). Career history can override
    an irrelevant current title, but caps at 0.60 — the JD's carve-out only
    applies to candidates with *demonstrated* career evidence, not skills lists.

    Returns float in [0, 1].
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    current_score = _score_single_title(current_title)

    career = candidate.get("career_history", [])
    if not career:
        return current_score

    # Sort career entries newest-first.
    def _start(job: dict):
        d = _parse_date(job.get("start_date"))
        return d if d else date.min

    sorted_career = sorted(career, key=_start, reverse=True)

    # Recency weights: most recent role is most predictive of current ability.
    recency_w = [1.0, 0.80, 0.65, 0.52, 0.42] + [0.35] * 10

    wsum = 0.0
    wtot = 0.0
    for i, job in enumerate(sorted_career):
        w = recency_w[min(i, len(recency_w) - 1)]
        wsum += w * _score_career_entry(job)
        wtot += w
    career_score = wsum / wtot if wtot > 0 else 0.0

    # If current title is IRRELEVANT: career can rescue up to max 0.60.
    # JD says irrelevant title is near-disqualifying but the career history
    # carve-out applies ('if career history shows recommendation system = fit').
    if current_score < 0.15:
        return 0.20 * current_score + 0.80 * min(career_score, 0.60)

    # Otherwise blend: career history is the stronger signal (0.60 weight)
    # because current title alone doesn't prove production AI experience.
    return 0.40 * current_score + 0.60 * career_score


# =============================================================================
# Component 2: semantic_fit  — sentence-transformers cosine similarity
# =============================================================================

def _build_semantic_text(candidate: dict) -> str:
    """
    Builds the text to embed for a candidate.
    Uses summary + all career descriptions (narrative text only).
    Excludes skill names and titles — we want signal from *what they wrote*,
    not from keyword lists that are easy to stuff.
    """
    p = candidate.get("profile", {})
    parts = [p.get("summary", "").strip()]
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").strip()
        if desc:
            parts.append(desc)
    return "\n\n".join(x for x in parts if x)


def _get_embed_model():
    """Lazy-load the SentenceTransformer model (at most once per process)."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            print(f"  [semantic_fit] loading {_EMBED_MODEL_NAME} ...",
                  file=sys.stderr, flush=True)
            _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        except ImportError:
            pass  # sentence-transformers not installed; callers fall back to keyword
    return _embed_model


def _get_jd_embedding():
    """Return the JD embedding, loading or computing it lazily."""
    global _jd_embedding
    if _jd_embedding is not None:
        return _jd_embedding
    if not _HAS_NUMPY:
        return None
    # Prefer precomputed file (written by precompute_embeddings.py)
    if _JD_EMBEDDING_NPY.exists():
        _jd_embedding = np.load(str(_JD_EMBEDDING_NPY))
        return _jd_embedding
    # Fall back: compute on the fly (one short text — fast)
    model = _get_embed_model()
    if model is None:
        return None
    _jd_embedding = model.encode(_JD_TEXT, normalize_embeddings=True)
    return _jd_embedding


def _load_embedding_index() -> None:
    """
    Load precomputed candidate embeddings from disk into _embedding_index.
    Called once on first access; sets _embedding_index to {} if files are missing.
    """
    global _embedding_index
    if _embedding_index is not None:
        return
    _embedding_index = {}  # marks "load attempted" even if files are absent
    if not _HAS_NUMPY:
        return
    if not (_EMBEDDINGS_NPY.exists() and _CANDIDATE_IDS_JSON.exists()):
        return  # precompute_embeddings.py hasn't been run yet
    try:
        embeddings = np.load(str(_EMBEDDINGS_NPY))           # [N, 384] float32
        with open(_CANDIDATE_IDS_JSON, "r", encoding="utf-8") as fh:
            ids = json.load(fh)
        _embedding_index = dict(zip(ids, embeddings))
    except Exception as exc:
        print(f"  [semantic_fit] WARNING: could not load embeddings: {exc}",
              file=sys.stderr)
        _embedding_index = {}


def _get_candidate_embedding(candidate: dict):
    """
    Return the precomputed embedding for a candidate, or None.

    IMPORTANT: never loads the model or does inference here.  This function is
    on the hot path (called for every candidate during bulk ranking) so it must
    be O(dict-lookup) only.  For on-the-fly inference in ad-hoc/--ids contexts,
    call _compute_embedding_on_the_fly() explicitly instead.
    """
    _load_embedding_index()
    cid = candidate.get("candidate_id", "")
    return _embedding_index.get(cid)   # None if not precomputed


def _compute_embedding_on_the_fly(candidate: dict):
    """
    Encode one candidate with the local model and return a normalised vector.
    SLOW — only call this for a handful of candidates (e.g. --ids inspection).
    Returns np.ndarray [384] or None if sentence-transformers is unavailable.
    """
    model = _get_embed_model()
    if model is None:
        return None
    return model.encode(_build_semantic_text(candidate), normalize_embeddings=True)


def _keyword_semantic_fit(candidate: dict) -> float:
    """
    Keyword-overlap fallback used when precomputed embeddings are unavailable.
    Kept as-is from the original implementation so scoring degrades gracefully.
    """
    text = _get_full_text(candidate, trusted_only=True).lower()
    req_cov = []
    for req in HARD_REQUIREMENTS:
        hits = sum(1 for term in req.signal_terms if term.lower() in text)
        req_cov.append(min(hits / 3.0, 1.0))
    hard_req_coverage = sum(req_cov) / len(HARD_REQUIREMENTS)
    prod_hits = sum(1 for kw in PRODUCTION_SIGNAL_KEYWORDS if kw.lower() in text)
    prod_density = min(prod_hits / 20.0, 1.0)
    return 0.55 * hard_req_coverage + 0.45 * prod_density


def semantic_fit(candidate: dict) -> float:
    """
    Semantic relevance of candidate to the JD.

    PRIMARY PATH: cosine similarity between candidate text embedding and a
    distilled JD embedding using all-MiniLM-L6-v2 (384-dim, CPU-friendly).
    Precomputed embeddings are loaded from candidate_embeddings.npy at first call.

    FALLBACK: keyword-overlap proxy (_keyword_semantic_fit) when
    precompute_embeddings.py has not been run yet.

    NEGATIVE-SIGNAL PENALTY: aspirational/transitional language in the
    candidate's summary/career text reduces the score multiplicatively.
    The JD explicitly wants proven production experience, not career-changers.
    Each unique phrase from NEGATIVE_SIGNAL_PHRASES applies _NEG_PENALTY_PER_PHRASE
    reduction, floored at _NEG_PENALTY_FLOOR.

    TO SWAP THE MODEL: change _EMBED_MODEL_NAME and re-run precompute_embeddings.py.
    The rest of the pipeline (caller, output format) is unchanged.

    Returns float in [0, 1].
    """
    cand_emb = _get_candidate_embedding(candidate)
    jd_emb   = _get_jd_embedding()

    if cand_emb is None or jd_emb is None:
        return _keyword_semantic_fit(candidate)

    # Both vectors are L2-normalised by sentence-transformers, so dot product = cosine.
    cos_sim = float(np.dot(cand_emb, jd_emb))
    raw = max(0.0, min(1.0, cos_sim))  # clip: cosine can be slightly < 0

    # Negative-signal penalty.
    # High cosine similarity for aspirational text is a false positive: the
    # candidate talks about ML topics but hasn't shipped anything.
    text = _build_semantic_text(candidate).lower()
    neg_hits = sum(1 for phrase in NEGATIVE_SIGNAL_PHRASES if phrase in text)
    penalty = max(_NEG_PENALTY_FLOOR, 1.0 - neg_hits * _NEG_PENALTY_PER_PHRASE)

    return round(raw * penalty, 4)


# =============================================================================
# Component 3: experience_fit
# =============================================================================

def _compute_experience_fit(candidate: dict) -> float:
    """
    Smooth piecewise-linear curve over years_of_experience.

    Sweet spot (6-8 yrs) → 1.0. Ramps up below, gradually declines above.
    JD: '5-9 years ... some people hit senior judgment at 4 years'.
    Ideal candidate described as '6-8 years total'.

    Returns float in [0, 1].
    """
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if yoe is None:
        return 0.30  # unknown: penalise but don't zero out

    y = float(yoe)
    T = TARGET_EXPERIENCE_YEARS

    if y < 0:
        return 0.0
    if y < T["min"]:                                 # 0-4 yrs: 0.20 → 0.60
        return 0.20 + 0.40 * (y / T["min"])
    if y < T["ideal_low"]:                           # 4-5 yrs: 0.60 → 0.85
        frac = (y - T["min"]) / (T["ideal_low"] - T["min"])
        return 0.60 + 0.25 * frac
    if y < T["sweet_spot_low"]:                      # 5-6 yrs: 0.85 → 1.00
        frac = (y - T["ideal_low"]) / (T["sweet_spot_low"] - T["ideal_low"])
        return 0.85 + 0.15 * frac
    if y <= T["sweet_spot_high"]:                    # 6-8 yrs: sweet spot 1.00
        return 1.0
    if y <= T["ideal_high"]:                         # 8-9 yrs: 1.00 → 0.95
        frac = (y - T["sweet_spot_high"]) / (T["ideal_high"] - T["sweet_spot_high"])
        return 1.0 - 0.05 * frac
    if y <= T["max_considered"]:                     # 9-12 yrs: 0.95 → 0.75
        frac = (y - T["ideal_high"]) / (T["max_considered"] - T["ideal_high"])
        return 0.95 - 0.20 * frac
    return max(0.50, 0.75 - 0.05 * (y - T["max_considered"]))  # 12+ yrs: declining


# =============================================================================
# Component 4: production_evidence
# =============================================================================

def _compute_production_evidence(candidate: dict) -> float:
    """
    Count unique PRODUCTION_SIGNAL_KEYWORDS appearing in career descriptions
    and profile summary (not skills, which can be faked).

    Saturates at _PROD_EVIDENCE_SAT unique hits using a linear cap.
    JD: production deployment experience is the primary hiring signal.

    Returns float in [0, 1].
    """
    text = _get_career_text(candidate).lower()
    unique_hits = sum(1 for kw in PRODUCTION_SIGNAL_KEYWORDS if kw.lower() in text)
    return min(unique_hits / _PROD_EVIDENCE_SAT, 1.0)


# =============================================================================
# Component 5: location_fit
# =============================================================================

def _compute_location_fit(candidate: dict) -> float:
    """
    Score candidate location against PREFERRED_LOCATIONS.

    Primary (Pune/Noida)                  → 1.00
    Acceptable (Hyderabad/Mumbai/Delhi/…) → 0.80
    India + willing_to_relocate           → 0.65
    India + not relocating                → 0.40  (can still apply, bar is lower)
    Outside India + willing_to_relocate   → 0.35  (case-by-case, no visa)
    Outside India + not relocating        → 0.15

    Returns float in [0, 1].
    """
    p = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    loc = (p.get("location") or "").lower()
    country = (p.get("country") or "").lower()
    willing = bool(sig.get("willing_to_relocate", False))

    if any(city in loc for city in PREFERRED_LOCATIONS["primary"]):
        return 1.00
    if any(city in loc for city in PREFERRED_LOCATIONS["acceptable"]):
        return 0.80
    if country == "india":
        return 0.65 if willing else 0.40
    return 0.35 if willing else 0.15


# =============================================================================
# Behavioral multiplier
# =============================================================================

def _beh_rrr(v: float) -> float:
    """recruiter_response_rate → tier multiplier."""
    tiers = BEHAVIORAL_THRESHOLDS["recruiter_response_rate"]["tiers"]
    if v < 0.10: return tiers[0]["multiplier"]   # ghost
    if v < 0.25: return tiers[1]["multiplier"]   # low
    if v < 0.50: return tiers[2]["multiplier"]   # mid
    if v < 0.75: return tiers[3]["multiplier"]   # high
    return          tiers[4]["multiplier"]        # very_high


def _beh_lad(days: int) -> float:
    """days_since_active → tier multiplier."""
    tiers = BEHAVIORAL_THRESHOLDS["last_active_days"]["tiers"]
    if days <  30: return tiers[0]["multiplier"]  # very_fresh
    if days <  60: return tiers[1]["multiplier"]  # fresh
    if days < 120: return tiers[2]["multiplier"]  # recent
    if days < 180: return tiers[3]["multiplier"]  # cooling
    if days < 270: return tiers[4]["multiplier"]  # stale
    return          tiers[5]["multiplier"]         # very_stale


def _beh_icr(v: float) -> float:
    """interview_completion_rate → tier multiplier."""
    tiers = BEHAVIORAL_THRESHOLDS["interview_completion_rate"]["tiers"]
    if v < 0.40: return tiers[0]["multiplier"]   # unreliable
    if v < 0.65: return tiers[1]["multiplier"]   # average
    if v < 0.85: return tiers[2]["multiplier"]   # reliable
    return          tiers[3]["multiplier"]         # highly_reliable


def _compute_behavioral_multiplier(candidate: dict) -> tuple:
    """
    Weighted average of four per-signal tier multipliers from BEHAVIORAL_THRESHOLDS.
    Returns (multiplier: float, detail: dict).
    """
    sig = candidate.get("redrob_signals", {})

    # recruiter_response_rate
    rrr = float(sig.get("recruiter_response_rate") or 0.30)
    rrr_m = _beh_rrr(rrr)

    # last_active_days
    last_active = _parse_date(sig.get("last_active_date"))
    days = (_TODAY - last_active).days if last_active else 120  # default: cooling
    lad_m = _beh_lad(days)

    # open_to_work_flag
    otw = bool(sig.get("open_to_work_flag", False))
    otw_m = (BEHAVIORAL_THRESHOLDS["open_to_work_flag"]["tiers"][0]["multiplier"]
             if otw else
             BEHAVIORAL_THRESHOLDS["open_to_work_flag"]["tiers"][1]["multiplier"])

    # interview_completion_rate
    icr = float(sig.get("interview_completion_rate") or 0.60)
    icr_m = _beh_icr(icr)

    w_rrr = BEHAVIORAL_THRESHOLDS["recruiter_response_rate"]["weight"]
    w_lad = BEHAVIORAL_THRESHOLDS["last_active_days"]["weight"]
    w_otw = BEHAVIORAL_THRESHOLDS["open_to_work_flag"]["weight"]
    w_icr = BEHAVIORAL_THRESHOLDS["interview_completion_rate"]["weight"]

    combined = w_rrr * rrr_m + w_lad * lad_m + w_otw * otw_m + w_icr * icr_m

    detail = {
        "recruiter_response_rate": {"value": rrr, "days_active": days,
                                    "rrr_mult": rrr_m, "lad_mult": lad_m,
                                    "otw_mult": otw_m, "icr_mult": icr_m},
    }
    return round(combined, 4), detail


# =============================================================================
# Consistency factor — honeypot / impossible-profile detector
# =============================================================================

def _compute_consistency_factor(candidate: dict, flags: list) -> float:
    """
    Detects impossible or internally inconsistent profiles.
    Each violation deducts from 1.0; result is clamped to [0.0, 1.0].

    Check (a): expert/advanced skill with duration_months == 0 — the primary
               honeypot pattern found in this dataset.
    Check (b): sum of career duration_months >> years_of_experience * 12.
    Check (c): start_date > end_date in a completed role.
    Check (d): single job tenure > 480 months (40 years) — physically impossible.

    Returns float in [0, 1].
    """
    factor = 1.0
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)

    # ── (a) Expert/advanced skill with zero duration ──────────────────────────
    # From explore.py: all 5 impossible examples were exactly this pattern.
    honeypot = [
        s["name"] for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) == 0
    ]
    if honeypot:
        penalty = min(0.50, len(honeypot) * 0.15)
        factor -= penalty
        sample = ", ".join(honeypot[:3]) + ("…" if len(honeypot) > 3 else "")
        flags.append(
            f"HONEYPOT: {len(honeypot)} expert/advanced skill(s) with "
            f"duration_months=0 ({sample})"
        )

    # ── (b) Career duration sum far exceeds claimed YoE ──────────────────────
    # Allow a 1.5× buffer + 24 months for legitimate concurrent roles
    # (part-time consulting, side-projects counted by the platform, etc.).
    career_months = sum(j.get("duration_months", 0) for j in career)
    yoe_months = yoe * 12
    if yoe_months > 0 and career_months > yoe_months * 1.5 + 24:
        factor -= 0.15
        flags.append(
            f"INCONSISTENT: career sum {career_months}mo >> YoE {yoe_months:.0f}mo "
            f"(ratio {career_months / yoe_months:.1f}x)"
        )

    # ── (c) Completed role where start_date > end_date ────────────────────────
    date_violations = 0
    for job in career:
        if job.get("is_current"):
            continue
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))
        if start and end and start > end:
            date_violations += 1
            flags.append(
                f"IMPOSSIBLE DATE: '{job.get('company', '?')}' "
                f"start {start} > end {end}"
            )
    if date_violations:
        factor -= min(0.30, date_violations * 0.10)

    # ── (d) Single tenure > 40 years ─────────────────────────────────────────
    for job in career:
        dur = job.get("duration_months", 0)
        if dur > 480:
            factor -= 0.35
            flags.append(
                f"IMPOSSIBLE TENURE: '{job.get('company', '?')}' "
                f"duration_months={dur} ({dur // 12} yrs)"
            )
            break  # one flag is enough

    return max(0.0, round(factor, 4))


# =============================================================================
# DISQUALIFIERS  — applied as penalties to base_fit_score
# =============================================================================

def _apply_disqualifiers(
    candidate: dict, raw_base: float, flags: list
) -> tuple:
    """
    Evaluate DISQUALIFIERS against the candidate.

    Hard severity: override base_fit_score → _HARD_DISQ_BASE (near zero).
    Soft severity: multiply score by a penalty factor.

    Returns (adjusted_score: float, penalties: dict).
    The penalties dict maps disqualifier_id → multiplier_applied.
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    yoe = float(profile.get("years_of_experience") or 0)

    penalties: dict = {}
    score = raw_base

    # ── HARD: pure_research ───────────────────────────────────────────────────
    # 'We will not move forward.' Zero production signal anywhere in career text
    # or summary means the candidate has never shipped anything.
    career_text = _get_career_text(candidate).lower()
    prod_hits = sum(1 for kw in PRODUCTION_SIGNAL_KEYWORDS if kw.lower() in career_text)
    if prod_hits == 0:
        flags.append(
            "DISQUALIFY [hard] pure_research: no PRODUCTION_SIGNAL_KEYWORDS "
            "found in any career description or summary"
        )
        penalties["pure_research"] = _HARD_DISQ_BASE / raw_base if raw_base > 0 else 1.0
        return _HARD_DISQ_BASE, penalties

    # ── SOFT: services_only_career ────────────────────────────────────────────
    # Disqualifies only if *every* role is at a services firm.
    # JD carve-out: 'if you're currently at one of these but have prior
    # product-company experience, that's fine.'
    if career and all(_is_services_company(j.get("company", "")) for j in career):
        m = 0.50
        score *= m
        penalties["services_only_career"] = m
        flags.append(
            "PENALTY [soft] services_only_career: all career roles at "
            "IT-services / consulting firms"
        )

    # ── SOFT: title_chaser ────────────────────────────────────────────────────
    # Average tenure < 18 months across completed (non-current) roles.
    # Require at least 3 completed roles to avoid penalising early-career candidates.
    completed = [j for j in career if not j.get("is_current")]
    if len(completed) >= 3:
        avg_tenure = sum(j.get("duration_months", 0) for j in completed) / len(completed)
        if avg_tenure < TITLE_CHASER_THRESHOLD_MONTHS:
            m = 0.75
            score *= m
            penalties["title_chaser"] = m
            flags.append(
                f"PENALTY [soft] title_chaser: avg tenure {avg_tenure:.1f}mo "
                f"< {TITLE_CHASER_THRESHOLD_MONTHS}mo "
                f"across {len(completed)} completed roles"
            )

    # ── SOFT: recent_llm_wrapper_only ─────────────────────────────────────────
    # AND condition: LangChain-style keywords present AND no pre-LLM ML terms.
    # JD: 'unless you can demonstrate substantial pre-LLM-era ML production experience.'
    full_text = _get_full_text(candidate, trusted_only=True).lower()
    has_wrapper = any(kw.lower() in full_text for kw in LANGCHAIN_WRAPPER_TERMS)
    has_pre_llm = any(kw.lower() in full_text for kw in PRE_LLM_ML_TERMS)
    if has_wrapper and not has_pre_llm:
        m = 0.60
        score *= m
        penalties["recent_llm_wrapper_only"] = m
        flags.append(
            "PENALTY [soft] recent_llm_wrapper_only: LangChain/OpenAI wrapper "
            "terms present, no pre-LLM ML production history found"
        )

    # ── SOFT: cv_speech_robotics_without_nlp_ir ───────────────────────────────
    # CV/speech/robotics terms dominate AND NLP/IR terms are nearly absent.
    # Threshold: CV hits >= 3 AND CV hits > 3× NLP hits AND NLP hits < 3.
    cv_hits = sum(1 for kw in CV_SPEECH_ROBOTICS_TERMS if kw.lower() in full_text)
    nlp_hits = sum(1 for kw in NLP_IR_TERMS if kw.lower() in full_text)
    if cv_hits >= 3 and nlp_hits < 3 and cv_hits > nlp_hits * 3:
        m = 0.65
        score *= m
        penalties["cv_speech_robotics_only"] = m
        flags.append(
            f"PENALTY [soft] cv_speech_robotics_only: "
            f"CV/speech hits={cv_hits}, NLP/IR hits={nlp_hits}"
        )

    # ── SOFT: code_inactive_18m ───────────────────────────────────────────────
    # Most recent career role has a leadership title AND github_activity_score <= 0.
    # Heuristic — can't read role descriptions as precisely as a human would.
    if career:
        most_recent = max(
            career,
            key=lambda j: _parse_date(j.get("start_date")) or date.min,
        )
        recent_title = most_recent.get("title", "").lower()
        is_leadership = any(kw in recent_title for kw in _LEADERSHIP_KW)
        gh_score = sig.get("github_activity_score", 0) or 0
        if is_leadership and gh_score <= 0:
            m = 0.75
            score *= m
            penalties["code_inactive_18m"] = m
            flags.append(
                f"PENALTY [soft] code_inactive_18m: recent title "
                f"'{most_recent['title']}' suggests leadership + "
                f"github_activity_score={gh_score}"
            )

    # ── SOFT: closed_source_no_external_validation ────────────────────────────
    # 5+ years experience, no GitHub, no certifications, all large enterprises.
    # Only fires when all three signals align — avoid over-penalising Indian
    # candidates who simply don't have public GitHub profiles.
    gh_score = sig.get("github_activity_score", 0) or 0
    certs = candidate.get("certifications", [])
    all_large = career and all(
        j.get("company_size", "") in {"1001-5000", "5001-10000", "10001+"}
        for j in career
    )
    if yoe > 5 and gh_score == -1 and not certs and all_large:
        m = 0.85
        score *= m
        penalties["closed_source_no_external_validation"] = m
        flags.append(
            "PENALTY [soft] closed_source_no_external_validation: "
            "5+ yrs, no GitHub, no certs, all large-enterprise roles"
        )

    return round(score, 6), penalties


# =============================================================================
# Main public function
# =============================================================================

def score_candidate(candidate: dict) -> dict:
    """
    Score a single candidate dict against the Senior AI Engineer JD.

    Returns a dict with all scoring components and diagnostic fields so the
    caller can explain any score:
    {
        "candidate_id": str,
        "total_score": float,              # base_fit * behavioral * consistency
        "base_fit_score": float,           # weighted component sum, post-disqualifiers
        "raw_base_score": float,           # same before disqualifiers are applied
        "behavioral_multiplier": float,
        "consistency_factor": float,
        "components": {                    # per-component breakdown
            "title_career_fit": float,
            "semantic_fit": float,
            "experience_fit": float,
            "production_evidence": float,
            "location_fit": float,
        },
        "disqualifier_penalties": dict,    # {disq_id: multiplier_applied}
        "flags": [str],                    # human-readable explanation of penalties
        "behavioral_detail": dict,         # per-signal breakdown for debugging
    }
    """
    flags: list = []

    # ── 1. Score each base-fit component ─────────────────────────────────────
    components = {
        "title_career_fit":    _compute_title_career_fit(candidate),
        "semantic_fit":        semantic_fit(candidate),
        "experience_fit":      _compute_experience_fit(candidate),
        "production_evidence": _compute_production_evidence(candidate),
        "location_fit":        _compute_location_fit(candidate),
    }

    raw_base = sum(COMPONENT_WEIGHTS[k] * v for k, v in components.items())

    # ── 2. Apply disqualifiers to the base ───────────────────────────────────
    base_fit_score, disq_penalties = _apply_disqualifiers(candidate, raw_base, flags)

    # ── 3. Behavioral multiplier ─────────────────────────────────────────────
    behavioral_multiplier, behavioral_detail = _compute_behavioral_multiplier(candidate)

    # ── 4. Consistency factor (honeypot / impossible profile) ─────────────────
    consistency_factor = _compute_consistency_factor(candidate, flags)

    # ── 5. Compose final score ────────────────────────────────────────────────
    total_score = round(base_fit_score * behavioral_multiplier * consistency_factor, 6)

    return {
        "candidate_id":          candidate.get("candidate_id", "?"),
        "total_score":           total_score,
        "base_fit_score":        round(base_fit_score, 4),
        "raw_base_score":        round(raw_base, 4),
        "behavioral_multiplier": behavioral_multiplier,
        "consistency_factor":    consistency_factor,
        "components":            {k: round(v, 4) for k, v in components.items()},
        "disqualifier_penalties": disq_penalties,
        "flags":                 flags,
        "behavioral_detail":     behavioral_detail,
    }


# =============================================================================
# I/O helpers (used by __main__)
# =============================================================================

def _load_first_n(path: Path, n: int) -> list:
    """Return the first n parseable records from a JSONL file."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(out) >= n:
                break
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _load_by_ids(path: Path, target_ids: set) -> tuple:
    """
    Stream path line-by-line and collect only candidates whose
    candidate_id is in target_ids.  Stops as soon as all IDs are found.
    Returns (found: list[dict], missing: set[str]).
    """
    remaining = set(target_ids)
    found = []
    scanned = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not remaining:
                break
            scanned += 1
            if scanned % 10_000 == 0:
                print(f"  scanning ... {scanned:,} records  "
                      f"({len(found)}/{len(target_ids)} found)\r",
                      end="", file=sys.stderr, flush=True)
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = c.get("candidate_id", "")
            if cid in remaining:
                found.append(c)
                remaining.discard(cid)
    # Clear the progress line
    print(" " * 60 + "\r", end="", file=sys.stderr, flush=True)
    return found, remaining   # remaining == IDs not found


# =============================================================================
# Print helpers (shared between default and --ids modes)
# =============================================================================

_SEP  = "=" * 118
_SEP2 = "-" * 118

_TABLE_HDR = (
    f"{'CANDIDATE':<14}  "
    f"{'TOTAL':>6}  {'base':>6}  {'beh':>6}  {'cons':>6}  |  "
    f"{'title':>6}  {'sem':>6}  {'exp':>6}  {'prod':>6}  {'loc':>6}  |  "
    f"FLAGS / DISQUALIFIERS"
)


def _print_table(results: list, title: str) -> None:
    print()
    print(_SEP)
    print(f"  {title}")
    print(_SEP)
    print(_TABLE_HDR)
    print(_SEP2)
    for r in results:
        c = r["components"]
        flag_str = "; ".join(r["flags"])
        if len(flag_str) > 55:
            flag_str = flag_str[:52] + "..."
        print(
            f"{r['candidate_id']:<14}  "
            f"{r['total_score']:>6.3f}  "
            f"{r['base_fit_score']:>6.3f}  "
            f"{r['behavioral_multiplier']:>6.3f}  "
            f"{r['consistency_factor']:>6.3f}  |  "
            f"{c['title_career_fit']:>6.3f}  "
            f"{c['semantic_fit']:>6.3f}  "
            f"{c['experience_fit']:>6.3f}  "
            f"{c['production_evidence']:>6.3f}  "
            f"{c['location_fit']:>6.3f}  |  "
            f"{flag_str if flag_str else '-'}"
        )
    print(_SEP)
    print()
    print("  FORMULA : total = base_fit x behavioral x consistency")
    print("  base_fit : 0.35*title + 0.25*sem + 0.15*exp + 0.15*prod + 0.10*loc  (post-disqualifiers)")
    print("  beh      : weighted avg(recruiter_response, last_active, open_to_work, interview_completion)")
    print("  cons     : 1.0 penalised by expert-skill-with-0-duration, duration>>yoe, impossible dates")
    print()


def _print_verbose(r: dict, cand: dict) -> None:
    """Full component + flag breakdown for one candidate."""
    prof = cand.get("profile", {})
    sig  = cand.get("redrob_signals", {})
    bd   = r["behavioral_detail"].get("recruiter_response_rate", {})

    print(f"\n  {r['candidate_id']}  --  {prof.get('anonymized_name', '?')}")
    print(f"  Title   : {prof.get('current_title', '?')}")
    print(f"  Location: {prof.get('location', '?')}, {prof.get('country', '?')}")
    print(f"  YoE     : {prof.get('years_of_experience', '?')} years")
    print()
    print(f"  total_score          : {r['total_score']:.4f}")
    print(f"  raw_base_score       : {r['raw_base_score']:.4f}")
    print(f"  base_fit_score       : {r['base_fit_score']:.4f}  (after disqualifiers)")
    print(f"  behavioral_multiplier: {r['behavioral_multiplier']:.4f}")
    print(f"  consistency_factor   : {r['consistency_factor']:.4f}")
    print()
    for comp, val in r["components"].items():
        w = COMPONENT_WEIGHTS[comp]
        bar = "#" * int(val * 25) + "." * (25 - int(val * 25))
        print(f"    {comp:<22} {val:.4f}  (w={w:.2f}, contrib={w*val:.4f})  [{bar}]")
    # Semantic-fit detail: show whether embeddings were used and any negative hits.
    sem_text = _build_semantic_text(cand).lower()
    neg_found = [p for p in NEGATIVE_SIGNAL_PHRASES if p in sem_text]
    cid_v = cand.get("candidate_id", "")
    in_precomputed = _embedding_index is not None and cid_v in _embedding_index
    # For --ids inspection, try on-the-fly so the mode label is accurate.
    otf_emb = None
    if not in_precomputed:
        otf_emb = _compute_embedding_on_the_fly(cand)
    if neg_found or in_precomputed or otf_emb is not None:
        print("  Semantic-fit detail:")
        if in_precomputed:
            mode = "embedding cosine (precomputed)"
        elif otf_emb is not None:
            mode = "embedding cosine (on-the-fly)"
        else:
            mode = "keyword fallback"
        print(f"    mode         : {mode}")
        if neg_found:
            penalty = max(_NEG_PENALTY_FLOOR, 1.0 - len(neg_found) * _NEG_PENALTY_PER_PHRASE)
            print(f"    neg-signal hits : {len(neg_found)}  ->  penalty factor {penalty:.2f}")
            for phrase in neg_found:
                print(f"      - \"{phrase}\"")
        else:
            print("    neg-signal hits : 0  (no penalty)")
        print()

    print("  Behavioral signals:")
    print(f"    recruiter_response_rate  : {sig.get('recruiter_response_rate', '?')}")
    print(f"    last_active_date         : {sig.get('last_active_date', '?')}  ({bd.get('days_active', '?')}d ago)")
    print(f"    open_to_work_flag        : {sig.get('open_to_work_flag', '?')}")
    print(f"    interview_completion_rate: {sig.get('interview_completion_rate', '?')}")
    print()
    if r["disqualifier_penalties"]:
        print(f"  Disqualifier penalties: {r['disqualifier_penalties']}")
    if r["flags"]:
        print("  Flags:")
        for flag in r["flags"]:
            print(f"    - {flag}")

    # Honeypot detail: list every untrusted skill so it's easy to screenshot
    all_skills  = cand.get("skills", [])
    honeypots   = [s for s in all_skills
                   if s.get("proficiency") in ("expert", "advanced")
                   and s.get("duration_months", 0) == 0]
    trusted     = _get_trusted_skills(cand)
    untrusted   = [s for s in all_skills if s.get("duration_months", 0) == 0]
    if honeypots or untrusted:
        print()
        print(f"  Skill trust audit  (total={len(all_skills)}, "
              f"trusted={len(trusted)}, "
              f"zero-duration={len(untrusted)}):")
        print(f"    {'NAME':<30} {'PROFICIENCY':<14} {'DURATION':>8}  {'ENDORSEMENTS':>12}  TRUSTED?")
        print(f"    {'-'*30} {'-'*14} {'-'*8}  {'-'*12}  -------")
        for s in sorted(all_skills,
                        key=lambda x: (x.get("duration_months", 0) == 0,
                                       x.get("name", ""))):
            dur = s.get("duration_months", 0)
            flagged = "*" if (s.get("proficiency") in ("expert", "advanced") and dur == 0) else ""
            print(f"    {s.get('name', '?'):<30} "
                  f"{s.get('proficiency', '?'):<14} "
                  f"{dur:>8}  "
                  f"{s.get('endorsements', 0):>12}  "
                  f"{'NO ' + flagged if dur == 0 else 'yes'}")
    print(_SEP2)


# =============================================================================
# __main__
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Score candidates against the Redrob Senior AI Engineer JD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scorer.py                          # score first 20 records\n"
            "  python scorer.py --ids CAND_0003582,CAND_0016000,CAND_0033817\n"
        ),
    )
    ap.add_argument(
        "--ids",
        metavar="CAND_ID[,CAND_ID,...]",
        default=None,
        help=(
            "Comma-separated candidate IDs to score. "
            "Streams the JSONL and stops once all IDs are found. "
            "Prints the full verbose breakdown for every requested candidate."
        ),
    )
    args = ap.parse_args()

    DATA = Path(__file__).parent.parent / "data" / "candidates.jsonl"

    # ── --ids mode ────────────────────────────────────────────────────────────
    if args.ids:
        target_ids = {cid.strip() for cid in args.ids.split(",") if cid.strip()}
        print(f"\nLooking for {len(target_ids)} candidate(s) in {DATA.name} ...",
              flush=True)

        candidates, missing = _load_by_ids(DATA, target_ids)

        if missing:
            print(f"  WARNING: {len(missing)} ID(s) not found: {', '.join(sorted(missing))}",
                  file=sys.stderr)

        if not candidates:
            print("No candidates found. Check the IDs and try again.")
            sys.exit(1)

        results = [score_candidate(c) for c in candidates]
        # Preserve the requested order (by candidate_id) for easy comparison.
        id_order = [cid.strip() for cid in args.ids.split(",") if cid.strip()]
        id_rank  = {cid: i for i, cid in enumerate(id_order)}
        results.sort(key=lambda r: id_rank.get(r["candidate_id"], 999))
        # Re-sort candidates to match
        cand_map = {c["candidate_id"]: c for c in candidates}

        _print_table(
            results,
            title=f"HONEYPOT / ID CHECK  --  {len(results)} candidate(s)",
        )

        print(_SEP)
        print(f"  DETAILED BREAKDOWN -- ALL {len(results)} REQUESTED CANDIDATE(S)")
        print(_SEP)
        for r in results:
            _print_verbose(r, cand_map[r["candidate_id"]])
        print()

    # ── default mode: first 20 ────────────────────────────────────────────────
    else:
        print(f"\nLoading first 20 candidates from {DATA.name} ...", flush=True)
        candidates = _load_first_n(DATA, 20)
        results    = [score_candidate(c) for c in candidates]
        results.sort(key=lambda r: r["total_score"], reverse=True)
        cand_map   = {c["candidate_id"]: c for c in candidates}

        _print_table(results, title="SCORE BREAKDOWN  --  first 20 records, sorted by total_score desc")

        print(_SEP)
        print("  DETAILED BREAKDOWN -- TOP 3")
        print(_SEP)
        for r in results[:3]:
            _print_verbose(r, cand_map[r["candidate_id"]])
        print()
