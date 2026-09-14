"""
Milestone 5: Semantic Retrieval (FAISS)
Build all-mpnet-base-v2 embeddings + FAISS index for Uber conversations

Fixes applied (see decision log):
1. Golden-set thread_ids are excluded from the candidate pool BEFORE
   sampling/embedding -- previously nothing excluded them, and all 66
   eligible golden examples ended up leaked into the index.
2. Removed the redundant keyword-based re-filter on customer_text
   (threads.parquet is already Uber-only from Phase 1's ThreadFinder --
   re-filtering by keyword silently dropped valid indexable threads whose
   text didn't happen to contain one of a handful of magic words).
3. Replaced the retrieval eval's hardcoded "everything is relevant"
   placeholder with a real distance-threshold-based relevance signal, so
   NDCG@5/Recall@5/MRR are no longer mathematically guaranteed to be 1.0
   regardless of what gets retrieved.
"""

import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import faiss

print("=" * 80)
print("MILESTONE 5: SEMANTIC RETRIEVAL (FAISS)")
print("=" * 80)

# Load Phase 1 Uber threads
df = pd.read_parquet('data/processed/threads.parquet')

# threads.parquet is already Uber-only by construction (Phase 1's
# ThreadFinder only builds threads rooted at a customer tweet Uber
# actually replied to). The previous keyword-based re-filter here was
# redundant and dropped legitimate threads whose text didn't happen to
# contain 'uber'/'driver'/'ride'/etc -- removed.
uber_df = df.copy()
print(f"\nUber threads (from Phase 1): {len(uber_df):,}")

# Filter to indexable threads (resolved + not boilerplate)
indexable_df = uber_df[
    (uber_df['resolution_tier'].isin(['resolved_explicit', 'resolved_implicit'])) &
    (~uber_df['is_boilerplate'])
].reset_index(drop=True)

print(f"Indexable threads (resolved + not boilerplate): {len(indexable_df):,}")

# --- Exclude golden-set thread_ids BEFORE sampling/embedding ---
GOLDEN_PATH = 'data/golden_set_labeled.jsonl'
with open(GOLDEN_PATH) as f:
    golden_thread_ids = {json.loads(line)['thread_id'] for line in f}

pre_exclusion_count = len(indexable_df)
indexable_df = indexable_df[~indexable_df['thread_id'].isin(golden_thread_ids)].reset_index(drop=True)
excluded_count = pre_exclusion_count - len(indexable_df)

print(f"\nGolden-set thread_ids loaded: {len(golden_thread_ids)}")
print(f"Excluded from candidate pool (leakage prevention): {excluded_count}")
print(f"Indexable threads after golden exclusion: {len(indexable_df):,}")

if excluded_count == 0:
    print("WARNING: zero golden thread_ids matched the indexable pool. "
          "Either golden IDs don't overlap the indexable filter (expected "
          "for unresolved/boilerplate golden examples) or thread_id "
          "formats don't match between golden_set_labeled.jsonl and "
          "threads.parquet -- verify before trusting the 'clean' result.")

# No artificial sampling cap -- index the full eligible pool (post golden
# exclusion). Note: this will be ~39,900 threads, not 18,570 -- that
# number never matched this script's own SAMPLE_SIZE=5000 logic to begin
# with, so there's nothing meaningful to preserve by targeting it.
print(f"\nUsing full eligible pool: {len(indexable_df):,} threads (no sampling cap)")

print("\n" + "=" * 80)
print("STEP 1: LOAD EMBEDDING MODEL")
print("=" * 80)

model_name = 'all-mpnet-base-v2'
print(f"\nLoading model: {model_name}")

model = SentenceTransformer(model_name)
print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

print("\n" + "=" * 80)
print("STEP 2: GENERATE EMBEDDINGS")
print("=" * 80)

texts = indexable_df['customer_text'].tolist()

print(f"\nEmbedding {len(texts):,} customer messages...")

batch_size = 64
embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

    if (i + batch_size) % 2000 == 0 or i + batch_size >= len(texts):
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")

embeddings = np.array(embeddings)
print(f"\nGenerated {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")

print("\n" + "=" * 80)
print("STEP 3: BUILD FAISS INDEX")
print("=" * 80)

embeddings = embeddings.astype('float32')

print(f"\nBuilding IndexFlatL2 (exact search)...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"Index built with {index.ntotal} vectors")

print("\n" + "=" * 80)
print("STEP 4: SAVE INDEX AND METADATA")
print("=" * 80)

os.makedirs('data/brands/uber', exist_ok=True)

index_path = 'data/brands/uber/index.faiss'
faiss.write_index(index, index_path)
print(f"Saved FAISS index: {index_path}")

metadata = []
for idx, row in indexable_df.iterrows():
    metadata.append({
        'index_id': int(idx),
        'thread_id': row['thread_id'],
        'customer_text': row['customer_text'],
        'brand_reply': row['brand_reply'],
        'resolution_tier': row['resolution_tier'],
    })

metadata_path = 'data/brands/uber/metadata.jsonl'
with open(metadata_path, 'w') as f:
    for item in metadata:
        f.write(json.dumps(item) + '\n')

print(f"Saved metadata: {metadata_path}")

embeddings_path = 'data/brands/uber/embeddings.npy'
np.save(embeddings_path, embeddings)
print(f"Saved embeddings: {embeddings_path}")

print("\n" + "=" * 80)
print("STEP 5: EVALUATE ON GOLDEN SET")
print("=" * 80)

with open(GOLDEN_PATH, 'r') as f:
    golden_records = [json.loads(line) for line in f.readlines()]

print(f"\nEvaluating on {len(golden_records)} golden set examples...\n")


def evaluate_faiss_retrieval(golden_records, index, metadata, model, top_k=5, relevance_threshold=0.7):
    """
    Evaluate FAISS retrieval metrics using a REAL relevance signal.

    Previous version hardcoded every retrieved item as relevant, which
    made NDCG@5/Recall@5/MRR mathematically guaranteed to be 1.0
    regardless of what was actually retrieved -- not a real measurement.

    This version uses L2 distance converted to a similarity score,
    thresholded at `relevance_threshold`, as a proxy for relevance. This
    is an approximation, not human-judged ground truth -- document this
    limitation explicitly in the report (e.g. in the "what's misleading
    about my headline number" section). A stronger version would use
    human relevance judgments or intent-label agreement once the
    classifier from Milestone 6 exists for the full corpus, not just
    golden examples.
    """
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []

    for record in golden_records:
        query = record['customer_message']

        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        distances = distances[0]
        indices = indices[0]

        # Convert L2 distance to a bounded similarity score and threshold
        # it for relevance, instead of assuming everything is relevant.
        similarities = 1 / (1 + distances)
        relevances = [1 if sim >= relevance_threshold else 0 for sim in similarities]

        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(top_k))
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_scores.append(ndcg)

        recall_at_5 = sum(relevances) / top_k
        recall_scores.append(recall_at_5)

        first_relevant_rank = next((i + 1 for i, rel in enumerate(relevances) if rel == 1), None)
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        mrr_scores.append(mrr)

    return {
        'ndcg@5': np.mean(ndcg_scores),
        'recall@5': np.mean(recall_scores),
        'mrr': np.mean(mrr_scores),
        'ndcg@5_std': np.std(ndcg_scores),
    }


metrics = evaluate_faiss_retrieval(golden_records, index, metadata, model, top_k=5)

print(f"--- FAISS Retrieval Performance on Golden Set ---")
print(f"NDCG@5: {metrics['ndcg@5']:.3f} (±{metrics['ndcg@5_std']:.3f})")
print(f"Recall@5: {metrics['recall@5']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")

os.makedirs('reports', exist_ok=True)
retrieval_report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'index_type': 'IndexFlatL2',
    'model': model_name,
    'corpus_size': index.ntotal,
    'embedding_dim': embeddings.shape[1],
    'golden_thread_ids_excluded': excluded_count,
    'relevance_method': 'distance_threshold_proxy (>=0.7 similarity, not human-judged)',
    'metrics': {
        'ndcg@5': float(metrics['ndcg@5']),
        'recall@5': float(metrics['recall@5']),
        'mrr': float(metrics['mrr']),
    }
}

with open('reports/retrieval_scores.json', 'w') as f:
    json.dump(retrieval_report, f, indent=2)

print(f"\nSaved to: reports/retrieval_scores.json")

print("\n" + "=" * 80)
print("Milestone 5 COMPLETE (rebuilt, leak-excluded, real eval metric)")
print("=" * 80)

print(f"""
FAISS Index Summary:
- Model: {model_name}
- Corpus: {index.ntotal:,} indexed Uber conversations
- Golden thread_ids excluded from corpus: {excluded_count}
- Index type: IndexFlatL2 (exact search)
- Embedding dimension: {embeddings.shape[1]}

Retrieval Performance (real distance-threshold relevance, not placeholder):
- NDCG@5: {metrics['ndcg@5']:.3f}
- Recall@5: {metrics['recall@5']:.3f}
- MRR: {metrics['mrr']:.3f}

Files saved:
- data/brands/uber/index.faiss
- data/brands/uber/metadata.jsonl
- data/brands/uber/embeddings.npy
""")