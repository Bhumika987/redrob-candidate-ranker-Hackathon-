"""
rank.py — main entry point for the Redrob candidate ranking submission.

Reproduce command (matches spec):
    python redrob-ranker/rank.py \\
        --candidates ./data/candidates.jsonl \\
        --out ./submission.csv

Pipeline:
    1. Load precomputed embeddings (data/candidate_embeddings.npy) once.
    2. Stream candidates.jsonl, score each via scorer.py.
    3. Maintain a fixed-size heap of the top 100 throughout streaming
       (never loads all 100K results into memory).
    4. Tie-break: total_score DESC; equal scores -> candidate_id ASC.
    5. Generate per-candidate reasoning from computed values.
    6. Write UTF-8 CSV: candidate_id,rank,score,reasoning (exactly 100 rows).
    7. Print wall-clock time so the ranking step can be confirmed < 5 min.
"""

import argparse
import csv
import heapq
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scorer import (
    score_candidate,
    _build_semantic_text,
    _get_trusted_skills,
    _is_services_company,
    _load_embedding_index,
    _score_single_title,
    NEGATIVE_SIGNAL_PHRASES,
    TARGET_EXPERIENCE_YEARS,
)

TOP_N = 100

# ---------------------------------------------------------------------------
# ML / AI skill terms used to select the most relevant skills for reasoning.
# Deliberately broad so that legitimate ML skills are surfaced even under
# non-standard names (e.g. "sentence-transformers", "Pinecone", "FAISS").
# ---------------------------------------------------------------------------
_ML_KW = frozenset({
    "python", "pytorch", "tensorflow", "keras", "jax",
    "scikit-learn", "scikit", "sklearn",
    "transformers", "huggingface", "sentence-transformers",
    "bert", "gpt", "llm", "nlp", "natural language",
    "embedding", "embeddings", "vector search", "vector db",
    "faiss", "pinecone", "milvus", "qdrant", "weaviate", "chroma",
    "elasticsearch", "opensearch", "solr", "bm25",
    "retrieval", "reranking", "re-ranking", "ranking",
    "recommendation", "collaborative filtering", "matrix factorization",
    "mlflow", "kubeflow", "mlops", "feature store", "feature engineering",
    "spark", "pyspark", "kafka", "airflow", "dbt", "flink",
    "ray", "triton", "onnx", "torchserve", "model serving",
    "langchain", "llamaindex", "openai", "anthropic",
    "a/b testing", "ndcg", "mrr", "map", "precision@k",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "deep learning", "neural network", "transformer",
    "fine-tuning", "fine-tune", "lora", "qlora", "peft",
    "data pipeline", "etl", "data engineering",
    "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
    "redis", "postgresql", "sql", "pyspark", "scala",
    "r language", "golang", "rust",
})


# ---------------------------------------------------------------------------
# Heap helpers
# ---------------------------------------------------------------------------

def _cid_num(cid: str) -> int:
    """Extract the numeric suffix from CAND_XXXXXXX for tie-breaking."""
    try:
        return int(cid.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def _heap_key(r: dict) -> tuple:
    """
    Sort key: higher score = better; equal scores -> lower candidate_id = better.
    Stored as (score, -cid_num) so Python's min-heap places the worst entry at [0].
    """
    return (r["total_score"], -_cid_num(r["candidate_id"]))


# ---------------------------------------------------------------------------
# Reasoning helpers
# ---------------------------------------------------------------------------

def _top_ml_skills(cand: dict, n: int = 3) -> list:
    """
    Return up to n trusted-skill names most relevant to ML/AI.
    Ranks by: (1) is it in _ML_KW, (2) endorsements + duration_months.
    Never returns a skill with duration_months == 0.
    """
    trusted = _get_trusted_skills(cand)

    def _key(s):
        name_lc = (s.get("name") or "").lower()
        is_ml = any(kw in name_lc for kw in _ML_KW)
        strength = s.get("endorsements", 0) + s.get("duration_months", 0)
        return (is_ml, strength)

    return [s["name"] for s in sorted(trusted, key=_key, reverse=True)[:n]]


def _best_career_entry(cand: dict) -> dict:
    """
    Pick the most ML-relevant career entry for naming in the reasoning.
    Prefers product-company roles; among those, highest AI-keyword density.
    """
    career = cand.get("career_history", [])
    if not career:
        return {}

    def _key(job):
        desc = (job.get("description") or "").lower()
        hits = sum(1 for kw in _ML_KW if kw in desc)
        at_product = not _is_services_company(job.get("company", ""))
        return (at_product, hits)

    return max(career, key=_key)


def _generate_reasoning(r: dict, cand: dict) -> str:
    """
    Generate a non-templated reasoning string for one candidate.

    Properties guaranteed:
    - All numbers come from r (the scored result) or cand (the raw profile).
    - Skills named are only from _top_ml_skills(), which returns trusted skills
      (duration_months > 0). Skills with duration_months == 0 are never named.
    - Narrative shape varies across the 8 branches below, driven by what is
      most notable about this candidate (strong fit, rescued title, aspiration,
      behavioral risk, honeypot, etc.).
    - Concerns are reported honestly: aspirational language, zero-duration skills,
      services-only career, below-ideal YoE, poor reachability.
    """
    prof  = cand.get("profile", {})
    sig   = cand.get("redrob_signals", {})

    title    = prof.get("current_title") or "Unknown"
    yoe      = float(prof.get("years_of_experience") or 0)
    location = prof.get("location") or ""
    relocate = sig.get("willing_to_relocate", False)

    comps     = r["components"]
    flags     = r.get("flags", [])
    disq      = r.get("disqualifier_penalties", [])
    cons      = r["consistency_factor"]
    beh       = r["behavioral_multiplier"]
    beh_d     = (r.get("behavioral_detail") or {}).get("recruiter_response_rate", {})

    title_fit = comps["title_career_fit"]
    prod_ev   = comps["production_evidence"]
    loc_fit   = comps["location_fit"]

    # Behavioral raw values
    rrr      = sig.get("recruiter_response_rate")
    icr      = sig.get("interview_completion_rate")
    otw      = sig.get("open_to_work_flag", False)
    days_act = beh_d.get("days_active")        # days since last active

    # Current-title raw score (ignoring career history carve-out)
    title_raw = _score_single_title(title)

    # Trusted ML skills (at most 3)
    skills     = _top_ml_skills(cand, 3)
    skills_str = ", ".join(skills) if skills else None

    # Best career entry for naming
    best     = _best_career_entry(cand)
    best_co  = best.get("company", "")
    best_ttl = best.get("title", "")

    # Aspirational-language hits in narrative text
    sem_text  = _build_semantic_text(cand).lower()
    neg_hits  = [p for p in NEGATIVE_SIGNAL_PHRASES if p in sem_text]

    # YoE with context vs spec targets
    tgt = TARGET_EXPERIENCE_YEARS
    tgt_min  = tgt["min"]
    tgt_ihi  = tgt["ideal_high"]
    tgt_max  = tgt["max_considered"]
    if yoe < tgt_min:
        yoe_str = f"{yoe:.1f}yr (below {tgt_min}yr minimum)"
    elif yoe <= tgt_ihi:
        yoe_str = f"{yoe:.1f}yr"
    elif yoe <= tgt_max:
        yoe_str = f"{yoe:.1f}yr (above ideal {tgt_ihi}yr)"
    else:
        yoe_str = f"{yoe:.1f}yr (beyond {tgt_max}yr)"

    # Compact behavioral note for appending to any shape
    def _beh_clause() -> str | None:
        if rrr is None:
            return None
        recency = f"{days_act}d ago" if days_act is not None else "recency unknown"
        if beh >= 0.85:
            return f"strong availability: rrr {rrr:.2f}, active {recency}"
        elif beh >= 0.72:
            return f"rrr {rrr:.2f}, active {recency}"
        else:
            return f"reachability concern: rrr {rrr:.2f}, {recency}"

    beh_clause = _beh_clause()

    # The single most important concern for this candidate
    def _top_concern() -> str | None:
        if disq:
            msg = disq[0].replace("PENALTY [hard] ", "").replace("PENALTY [soft] ", "")
            return msg.split(":")[0].strip()
        if cons < 0.75:
            honeypot_flag = next((f for f in flags if "HONEYPOT" in f), "")
            try:
                count = int(honeypot_flag.split("HONEYPOT:")[1].split(" ")[1])
            except (IndexError, ValueError):
                count = sum(1 for f in flags if "HONEYPOT" in f)
            return f"{count} expert/advanced skills with zero practice duration"
        if beh < 0.65:
            r_str = f"{rrr:.2f}" if rrr is not None else "n/a"
            d_str = f"{days_act}d" if days_act is not None else "unknown"
            return f"low reachability (rrr {r_str}, {d_str} inactive)"
        if len(neg_hits) >= 3:
            return "summary signals ML aspiration, not established practice"
        if yoe < tgt_min:
            return f"{yoe:.1f}yr below {tgt_min}yr minimum"
        return None

    concern = _top_concern()

    # ── Shape A: Hard disqualifier dominates ─────────────────────────────────
    if any("hard" in d.lower() for d in disq):
        parts = [f"{title}, {yoe_str}"]
        parts.append(f"disqualified: {concern}")
        if beh_clause:
            parts.append(beh_clause)
        return "; ".join(parts) + "."

    # ── Shape B: Honeypot / stuffed-skills concern leads ─────────────────────
    if cons < 0.75:
        parts = [f"{title} with {yoe_str}"]
        parts.append(concern)
        if skills_str:
            parts.append(f"trusted (nonzero-duration) skills: {skills_str}")
        if beh_clause:
            parts.append(beh_clause)
        return "; ".join(parts) + "."

    # ── Shape C: Strong direct fit — title + production depth ────────────────
    if title_fit >= 0.75 and prod_ev >= 0.40:
        parts = [f"{title} with {yoe_str}"]
        if skills_str:
            parts.append(f"production skills: {skills_str}")
        if best_co and not _is_services_company(best_co):
            parts.append(f"including at {best_co}")
        if beh_clause:
            parts.append(beh_clause)
        if concern:
            parts.append(f"note: {concern}")
        return "; ".join(parts) + "."

    # ── Shape D: Career history rescues a non-ML current title ───────────────
    if title_raw < 0.50 and title_fit >= 0.50:
        lead = f"Current title ({title}) understates ML depth"
        parts = [lead]
        if best_co and best_ttl and not _is_services_company(best_co):
            parts.append(f"career shows {best_ttl} work at {best_co}")
        elif best_co:
            parts.append(f"career shows {best_ttl}")
        if skills_str:
            parts.append(f"skills: {skills_str}")
        if beh_clause:
            parts.append(beh_clause)
        if concern:
            parts.append(f"concern: {concern}")
        return "; ".join(parts) + "."

    # ── Shape E: Aspiring / transitioning profile ────────────────────────────
    if len(neg_hits) >= 2:
        parts = [f"{title} with {yoe_str}"]
        parts.append("summary indicates ML career transition, not established production practice")
        if skills_str:
            parts.append(f"trusted background: {skills_str}")
        if beh_clause:
            parts.append(beh_clause)
        if concern and concern not in parts:
            parts.append(f"note: {concern}")
        return "; ".join(parts) + "."

    # ── Shape F: Behaviorally weak — reachability is the headline ────────────
    if beh < 0.70:
        parts = [f"{title}, {yoe_str}"]
        if skills_str:
            parts.append(f"skills: {skills_str}")
        r_str = f"{rrr:.2f}" if rrr is not None else "n/a"
        d_str = f"{days_act}d inactive" if days_act is not None else "last-active unknown"
        parts.append(f"reachability concern: rrr {r_str}, {d_str}")
        if concern and concern not in parts[-1]:
            parts.append(concern)
        return "; ".join(parts) + "."

    # ── Shape G: Location is a clear positive ────────────────────────────────
    if loc_fit >= 0.80:
        city = location.split(",")[0].strip() if location else ""
        parts = []
        if city:
            extra = "; open to relocation" if relocate else ""
            parts.append(f"Based in {city}{extra}")
        parts.append(f"{title} with {yoe_str}")
        if skills_str:
            parts.append(f"skills: {skills_str}")
        if beh_clause:
            parts.append(beh_clause)
        if concern:
            parts.append(f"note: {concern}")
        return "; ".join(parts) + "."

    # ── Shape H: General moderate fit ────────────────────────────────────────
    parts = [f"{title} with {yoe_str}"]
    if skills_str:
        parts.append(f"skills: {skills_str}")
    if beh_clause:
        parts.append(beh_clause)
    if concern:
        parts.append(f"note: {concern}")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _stream_score_top_n(path: Path, n: int) -> list:
    """
    Stream the JSONL file, score each candidate, and maintain a fixed-size
    max-heap of the n best candidates using (score, -cid_num) as sort key.

    Memory: O(n) candidate dicts + O(1) per non-top-n candidate.
    """
    heap    = []   # min-heap: heap[0] is the *worst* of the current top-n
    counter = 0    # tie-breaker so Python never tries to compare dicts

    t0      = time.time()
    total   = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            r   = score_candidate(cand)
            key = _heap_key(r)

            if len(heap) < n:
                heapq.heappush(heap, (key, counter, r, cand))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, counter, r, cand))

            counter += 1

            if total % 10_000 == 0:
                elapsed = time.time() - t0
                rate    = total / elapsed
                print(
                    f"  {total:>7,} scored | {rate:>5.0f}/s",
                    end="\r", flush=True,
                )

    elapsed = time.time() - t0
    rate    = total / elapsed if elapsed > 0 else 0
    print(f"  {total:,} scored in {elapsed:.1f}s ({rate:.0f}/s)          ")
    return heap, total, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank candidates against the Redrob Senior AI Engineer JD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python redrob-ranker/rank.py \\\n"
            "      --candidates ./data/candidates.jsonl \\\n"
            "      --out ./submission.csv\n"
        ),
    )
    ap.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        metavar="PATH",
        help="Path to candidates.jsonl (default: ./data/candidates.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="./submission.csv",
        metavar="PATH",
        help="Output CSV path (default: ./submission.csv)",
    )
    args = ap.parse_args()

    candidates_path = Path(args.candidates)
    out_path        = Path(args.out)

    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found.", file=sys.stderr)
        sys.exit(1)

    t_wall_start = time.time()

    # ── Step 1: Pre-load embeddings ──────────────────────────────────────────
    print("Loading precomputed embeddings ...", flush=True)
    _load_embedding_index()

    # ── Step 2: Stream + score ───────────────────────────────────────────────
    print(f"Streaming {candidates_path} and scoring ...", flush=True)
    heap, n_scored, score_elapsed = _stream_score_top_n(candidates_path, TOP_N)

    # ── Step 3: Extract top-100, enforce tie-break sort ──────────────────────
    # Sort descending on (score, -cid_num): higher score = rank 1;
    # equal scores -> lower cid_num (ascending candidate_id) = better rank.
    top100 = sorted(heap, key=lambda x: x[0], reverse=True)

    # Sanity: scores must be non-increasing
    scores = [entry[0][0] for entry in top100]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"BUG: score at rank {i+1} ({scores[i]:.4f}) < "
            f"rank {i+2} ({scores[i+1]:.4f})"
        )

    # ── Step 4: Write CSV ────────────────────────────────────────────────────
    print(f"Generating reasoning and writing {out_path} ...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (_key, _counter, r, cand) in enumerate(top100, start=1):
            reasoning = _generate_reasoning(r, cand)
            writer.writerow([
                r["candidate_id"],
                rank,
                f"{r['total_score']:.4f}",
                reasoning,
            ])

    # ── Step 5: Summary ──────────────────────────────────────────────────────
    t_wall_end = time.time()
    wall_elapsed = t_wall_end - t_wall_start

    print()
    print("=" * 60)
    print(f"  Output         : {out_path}")
    print(f"  Rows written   : {min(len(top100), TOP_N)}")
    print(f"  Candidates scored: {n_scored:,}")
    print(f"  Scoring time   : {score_elapsed:.1f}s")
    print(f"  Wall-clock time: {wall_elapsed:.1f}s ({wall_elapsed / 60:.2f} min)")
    if wall_elapsed < 300:
        print(f"  Budget check   : PASS (ranking step under 5-minute limit)")
    else:
        print(f"  Budget check   : OVER ({wall_elapsed / 60:.1f} min > 5 min limit)")
    print("=" * 60)

    # ── Step 6: Preview top 5 ────────────────────────────────────────────────
    print()
    print("  Top 5 preview:")
    print(f"  {'RANK':<5} {'CANDIDATE':<15} {'SCORE':<7}  REASONING")
    print(f"  {'-'*5} {'-'*15} {'-'*7}  {'-'*50}")
    for rank, (_key, _counter, r, cand) in enumerate(top100[:5], start=1):
        rsn = _generate_reasoning(r, cand)
        short_rsn = rsn[:60] + "..." if len(rsn) > 60 else rsn
        print(f"  {rank:<5} {r['candidate_id']:<15} {r['total_score']:.4f}  {short_rsn}")
    print()


if __name__ == "__main__":
    main()
