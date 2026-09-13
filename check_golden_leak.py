"""
Golden-set leakage check for a FAISS index.

Logic:
  For every golden example, re-embed its customer_message using the SAME
  model + preprocessing your indexer.py used, then search the index for
  its nearest neighbor. If that example's own vector is sitting in the
  index, the nearest-neighbor distance will be ~0 (not just "close" --
  bit-identical, since same text + same deterministic model = same vector).

  A baseline distribution (nearest-neighbor distances for genuinely
  indexed, non-golden threads) is computed alongside for calibration --
  those SHOULD be ~0 too (they're supposed to be in the index), so the
  real signal is: are golden examples behaving like baseline (leaked) or
  like held-out examples (higher, non-zero distances)?

IMPORTANT: replace `embed()` below with however your indexer.py actually
produces vectors -- same model name, same preprocessing (e.g. whatever
normalization was applied to customer_text before embedding). Using a
different model here will produce meaningless, incomparable distances.
"""

import json
import faiss
import numpy as np
import pandas as pd

INDEX_PATH = "data/brands/uber/index.faiss"
GOLDEN_PATH = "data/golden_set_labeled.jsonl"  # the FINAL 165-example set used by 09_evaluation_harness.py -- this is what needs to be leak-free, not the pre-label draft
THREADS_PARQUET = "data/processed/threads.parquet"

# Confirmed from the README's own FAQ code snippet (SentenceTransformer
# ('all-mpnet-base-v2')) -- this is the exact model + string used to build
# the index, not a guess anymore.
def embed(texts: list[str]) -> np.ndarray:
    """Model confirmed via README -- matches the exact string used to build the index."""
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = "all-mpnet-base-v2"  # exact string from README's tested query example
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, show_progress_bar=False).astype("float32")


def main():
    index = faiss.read_index(INDEX_PATH)
    print(f"Index: {index.ntotal} vectors, dim={index.d}")

    # --- Load golden set ---
    golden = []
    with open(GOLDEN_PATH) as f:
        for line in f:
            golden.append(json.loads(line))
    golden_texts = [g["customer_message"] for g in golden]
    golden_ids = [g["thread_id"] for g in golden]
    print(f"Golden examples: {len(golden)}")

    # --- Query index with golden embeddings ---
    golden_vecs = embed(golden_texts)
    D, I = index.search(golden_vecs, k=1)  # D = L2 distances, I = nearest neighbor positions
    golden_distances = D[:, 0]

    # --- Baseline: sample genuinely-indexed threads for calibration ---
    # (adjust this filter to match whatever selection logic actually built
    # the index, so the baseline is apples-to-apples)
    threads = pd.read_parquet(THREADS_PARQUET)
    baseline_pool = threads[
        (threads.resolution_tier.isin(["resolved_explicit", "resolved_implicit"]))
        & (~threads.is_boilerplate)
        & (~threads.thread_id.isin(golden_ids))
    ]
    baseline_sample = baseline_pool.sample(n=min(len(golden), len(baseline_pool)), random_state=42)
    baseline_vecs = embed(baseline_sample["customer_text"].tolist())
    D_base, _ = index.search(baseline_vecs, k=1)
    baseline_distances = D_base[:, 0]

    # --- Report ---
    print("\n=== Golden set nearest-neighbor distances ===")
    print(f"  min:    {golden_distances.min():.6f}")
    print(f"  median: {np.median(golden_distances):.6f}")
    print(f"  max:    {golden_distances.max():.6f}")

    print("\n=== Baseline (known-indexed, non-golden) nearest-neighbor distances ===")
    print(f"  min:    {baseline_distances.min():.6f}")
    print(f"  median: {np.median(baseline_distances):.6f}")
    print(f"  max:    {baseline_distances.max():.6f}")

    # A genuinely indexed vector should match itself at ~0. If golden
    # examples show the same near-zero pattern as baseline, that's leakage.
    EPS = 1e-3
    leaked = [
        (gid, dist) for gid, dist in zip(golden_ids, golden_distances) if dist < EPS
    ]
    print(f"\nGolden examples with near-zero distance (< {EPS}): {len(leaked)} / {len(golden)}")
    if leaked:
        print("Likely-leaked thread_ids (sample):")
        for gid, dist in sorted(leaked, key=lambda x: x[1])[:20]:
            print(f"  {gid}: distance={dist:.8f}")

    if len(leaked) == 0:
        print("\nNo near-zero matches -- golden set does not appear to be leaked into the index.")
    elif len(leaked) == len(golden):
        print("\nEVERY golden example matched near-zero -- strong evidence of full leakage, "
              "or the embed() model/preprocessing here doesn't match the index and this "
              "result is not trustworthy. Sanity-check against the baseline numbers above: "
              "if baseline distances also cluster near zero as expected, but golden looks "
              "the SAME, that's leakage, not a model mismatch.")
    else:
        print(f"\n{len(leaked)} golden examples appear leaked -- partial contamination. "
              "Cross-reference these thread_ids against your indexing code's filter logic.")


if __name__ == "__main__":
    main()