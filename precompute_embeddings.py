"""
precompute_embeddings.py
Precompute sentence-transformer embeddings for candidates in the JSONL.

Run this once before scoring.  scorer.py loads the saved vectors at ranking
time so that no model inference happens during the ranking budget window.

Usage
-----
    # Full 100K run (20-40 min on CPU, ~40 MB output)
    python redrob-ranker/precompute_embeddings.py

    # Quick test: first 50 candidates only (~5 seconds)
    python redrob-ranker/precompute_embeddings.py --limit 50

    # Embed specific candidate IDs (useful for verifying honeypot handling)
    python redrob-ranker/precompute_embeddings.py --ids CAND_0000001,CAND_0003582

Output files (all written to data/)
------------------------------------
    candidate_embeddings.npy   float32 [N, 384] — one row per candidate
    candidate_ids.json         list of N candidate_id strings (row index)
    jd_embedding.npy           float32 [384] — distilled JD vector for scorer.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from scorer import _JD_TEXT, _build_semantic_text  # noqa: E402

DATA_DIR   = Path(__file__).parent.parent / "data"
JSONL_PATH = DATA_DIR / "candidates.jsonl"

OUT_EMBEDDINGS = DATA_DIR / "candidate_embeddings.npy"
OUT_IDS        = DATA_DIR / "candidate_ids.json"
OUT_JD_EMB     = DATA_DIR / "jd_embedding.npy"


def _stream_all(path: Path, limit: int | None):
    """Yield (candidate_id, text) for each record up to limit."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if limit and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield c.get("candidate_id", ""), _build_semantic_text(c)
            count += 1
            if count % 10_000 == 0:
                print(f"  collected {count:,} texts ...", flush=True)


def _stream_by_ids(path: Path, target_ids: set):
    """Yield (candidate_id, text) for specific IDs; stops when all found."""
    remaining = set(target_ids)
    scanned = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            scanned += 1
            cid = c.get("candidate_id", "")
            if cid in remaining:
                remaining.discard(cid)
                yield cid, _build_semantic_text(c)
    if remaining:
        print(f"  WARNING: {len(remaining)} IDs not found: {remaining}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Precompute sentence-transformer embeddings for ranking."
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Embed only the first N candidates (default: all 100K)"
    )
    ap.add_argument(
        "--ids", default=None, metavar="CAND_ID[,...]",
        help="Comma-separated candidate IDs to embed (instead of --limit)"
    )
    ap.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Inference batch size (default: 64)"
    )
    ap.add_argument(
        "--model", default=None,
        help="Override model name (default: uses _EMBED_MODEL_NAME from scorer.py)"
    )
    args = ap.parse_args()

    # Determine model name
    from scorer import _EMBED_MODEL_NAME
    model_name = args.model or _EMBED_MODEL_NAME

    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    dim = model.get_embedding_dimension()
    print(f"  Embedding dim : {dim}")

    # ── Embed the JD ────────────────────────────────────────────────────────
    print("\nEmbedding distilled JD text ...")
    jd_emb = model.encode(_JD_TEXT, normalize_embeddings=True)
    np.save(str(OUT_JD_EMB), jd_emb.astype("float32"))
    print(f"  Saved: {OUT_JD_EMB}  shape={jd_emb.shape}")

    # ── Collect candidate texts ──────────────────────────────────────────────
    if args.ids:
        target_ids = {cid.strip() for cid in args.ids.split(",") if cid.strip()}
        print(f"\nCollecting {len(target_ids)} specific candidates ...")
        pairs = list(_stream_by_ids(JSONL_PATH, target_ids))
    else:
        label = f"first {args.limit:,}" if args.limit else "all"
        print(f"\nCollecting texts for {label} candidates ...")
        pairs = list(_stream_all(JSONL_PATH, args.limit))

    if not pairs:
        print("ERROR: no candidates found — nothing to embed.", file=sys.stderr)
        sys.exit(1)

    all_ids, all_texts = zip(*pairs)
    all_ids   = list(all_ids)
    all_texts = list(all_texts)
    print(f"  {len(all_ids):,} candidates ready for embedding.")

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\nRunning inference (batch_size={args.batch_size}) ...")
    t0 = time.time()
    embeddings = model.encode(
        all_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    rate = len(all_ids) / elapsed if elapsed > 0 else float("inf")
    print(f"\n  Done in {elapsed:.1f}s  ({rate:.0f} candidates/sec)")

    # ── Merge with any existing index (--ids mode appends) ───────────────────
    if args.ids and OUT_EMBEDDINGS.exists() and OUT_IDS.exists():
        print("\nMerging with existing embeddings ...")
        existing_emb = np.load(str(OUT_EMBEDDINGS))
        with open(OUT_IDS, "r", encoding="utf-8") as fh:
            existing_ids = json.load(fh)
        existing_set = set(existing_ids)
        new_mask = [i for i, cid in enumerate(all_ids) if cid not in existing_set]
        if new_mask:
            merged_ids = existing_ids + [all_ids[i] for i in new_mask]
            merged_emb = np.vstack([existing_emb, embeddings[new_mask]])
        else:
            print("  All IDs already in existing index — nothing new to add.")
            merged_ids = existing_ids
            merged_emb = existing_emb
        all_ids   = merged_ids
        embeddings = merged_emb
    elif args.ids and (OUT_EMBEDDINGS.exists() or OUT_IDS.exists()):
        print("  WARNING: only one of the two output files exists — rebuilding from scratch.")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(str(OUT_EMBEDDINGS), embeddings.astype("float32"))
    with open(OUT_IDS, "w", encoding="utf-8") as fh:
        json.dump(all_ids, fh)

    size_mb = OUT_EMBEDDINGS.stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {OUT_EMBEDDINGS}  shape={embeddings.shape}  size={size_mb:.1f} MB")
    print(f"  {OUT_IDS}  ({len(all_ids):,} IDs)")
    print(f"  {OUT_JD_EMB}  shape={jd_emb.shape}")
    print("\nDone. Run scorer.py to use precomputed embeddings.")

    # Rough estimate for full 100K run
    if args.limit and args.limit < 100_000:
        secs_100k = (elapsed / len(all_ids)) * 100_000
        mins = secs_100k / 60
        print(f"\n  Full 100K estimate at this speed: ~{mins:.0f} min on CPU")
        print(f"  (run without --limit for production precompute)")


if __name__ == "__main__":
    main()
