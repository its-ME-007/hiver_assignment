"""
Milestone 4: Establish Baselines
Create simple reference systems before building fancy components:
- Baseline 1: Majority-class intent classifier
- Baseline 2: TF-IDF retrieval
"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import os

print("=" * 80)
print("MILESTONE 4: ESTABLISH BASELINES")
print("=" * 80)

# Load Phase 1 data (for training)
df = pd.read_parquet('data/processed/threads.parquet')

def detect_brand(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['uber', 'driver', 'ride', 'surge', 'pool', 'ubereats']):
        return 'uber'
    return None

df['inferred_brand'] = df['customer_text'].apply(detect_brand)
uber_df = df[df['inferred_brand'] == 'uber'].copy()

print(f"\nPhase 1 Uber threads: {len(uber_df):,}")

# Infer intents on full dataset
def infer_intent(customer_text, brand_reply):
    if not isinstance(customer_text, str):
        return 'other'
    
    text = customer_text.lower()
    
    if any(kw in text for kw in ['charge', 'overcharge', 'cost', 'price', 'fee', 'billing']):
        if any(kw in text for kw in ['refund', 'money', 'back', 'credit']):
            return 'payment_refund'
        return 'payment_disputed_charge'
    
    if any(kw in text for kw in ['refund', 'money back', 'reimburse', 'credit me']):
        return 'payment_refund'
    
    if any(kw in text for kw in ['cancel', 'cancelled', 'cancellation']):
        return 'ride_cancellation'
    
    if any(kw in text for kw in ['driver', 'rude', 'unprofessional', 'disrespectful', 'bad driver', 'unsafe']):
        return 'driver_quality_issue'
    
    if any(kw in text for kw in ['lost', 'left', 'forgot', 'missing item', 'left behind']):
        return 'lost_found_item'
    
    if any(kw in text for kw in ['account', 'suspend', 'ban', 'verify', 'login', 'locked']):
        return 'account_issue'
    
    if any(kw in text for kw in ['ubereats', 'delivery', 'order', 'food']):
        if any(kw in text for kw in ['late', 'delay', 'long', 'took']):
            return 'delivery_timing'
        if any(kw in text for kw in ['wrong', 'missing', 'incomplete', 'damaged', 'cold']):
            return 'delivery_quality'
        return 'delivery_quality'
    
    if any(kw in text for kw in ['pickup', 'wrong location', 'wrong place', 'stranded']):
        return 'ride_pickup_issue'
    
    if any(kw in text for kw in ['dropoff', 'wrong address', 'wrong destination', 'ended at']):
        return 'ride_dropoff_issue'
    
    if any(kw in text for kw in ['app', 'crash', 'bug', 'not working', 'error', 'glitch', 'text', 'message']):
        return 'technical_issue'
    
    if any(kw in text for kw in ['service', 'quality', 'bad', 'terrible', 'worst']):
        return 'service_quality'
    
    return 'other'

uber_df['inferred_intent'] = uber_df.apply(
    lambda row: infer_intent(row['customer_text'], row['brand_reply']),
    axis=1
)

# Load golden set for evaluation
with open('data/golden_set_labeled.jsonl', 'r') as f:
    golden_records = [json.loads(line) for line in f.readlines()]

golden_df = pd.DataFrame(golden_records)

print(f"Golden set: {len(golden_df)} examples")
print(f"Intents in golden set: {golden_df['intent_label'].nunique()}")

print("\n" + "=" * 80)
print("BASELINE 1: INTENT CLASSIFICATION (MAJORITY CLASS)")
print("=" * 80)

# Majority class = most common intent in Phase 1
intent_counts = uber_df['inferred_intent'].value_counts()
majority_intent = intent_counts.index[0]
majority_count = intent_counts.iloc[0]
majority_pct = majority_count / len(uber_df) * 100

print(f"\nMajority intent: {majority_intent} ({majority_count:,}, {majority_pct:.1f}%)")
print("\nIntent distribution (Phase 1 Uber):")
for intent, count in intent_counts.items():
    print(f"  {intent}: {count:,}")

# Evaluate majority classifier on golden set
golden_df['majority_pred'] = majority_intent
baseline1_f1 = f1_score(golden_df['intent_label'], golden_df['majority_pred'], average='macro', zero_division=0)

print(f"\n--- Baseline 1 Performance on Golden Set ---")
print(f"Macro F1: {baseline1_f1:.3f}")

# Per-intent metrics
precision, recall, f1, support = precision_recall_fscore_support(
    golden_df['intent_label'], 
    golden_df['majority_pred'], 
    average=None, 
    zero_division=0,
    labels=sorted(golden_df['intent_label'].unique())
)

print(f"\nPer-intent F1 scores:")
for i, intent in enumerate(sorted(golden_df['intent_label'].unique())):
    print(f"  {intent}: F1={f1[i]:.3f}, Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, Support={int(support[i])}")

baseline1_results = {
    'name': 'Majority Class',
    'macro_f1': float(baseline1_f1),
    'majority_intent': majority_intent,
    'description': f"Always predict {majority_intent}"
}

print("\n" + "=" * 80)
print("BASELINE 2: RETRIEVAL (TF-IDF + COSINE SIMILARITY)")
print("=" * 80)

# Build retrieval corpus from Phase 1 Uber threads
print(f"\nBuilding TF-IDF index from {len(uber_df):,} Phase 1 threads...")

# Use customer_text as retrieval corpus
corpus = uber_df['customer_text'].tolist()
corpus_intents = uber_df['inferred_intent'].tolist()

# Train TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
corpus_vectors = tfidf.fit_transform(corpus)

print(f"TF-IDF features: {corpus_vectors.shape[1]}")

# Evaluate retrieval on golden set
def evaluate_retrieval(golden_records, corpus_vectors, corpus_intents, tfidf_model, top_k=5):
    """Evaluate retrieval metrics"""
    ndcg_scores = []
    recall_scores = []
    
    for record in golden_records:
        query = record['customer_message']
        true_intent = record['intent_label']
        
        # Vectorize query
        query_vector = tfidf_model.transform([query])
        
        # Compute similarity to corpus
        similarities = cosine_similarity(query_vector, corpus_vectors)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_intents = [corpus_intents[idx] for idx in top_indices]
        top_sims = sorted(similarities, reverse=True)[:top_k]
        
        # Compute NDCG@5
        # Relevance: 1 if intent matches, 0 otherwise
        relevances = [1 if intent == true_intent else 0 for intent in top_intents]
        
        # Ideal DCG: all relevant (best case)
        num_relevant_total = sum(1 for intent in corpus_intents if intent == true_intent)
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(num_relevant_total, top_k)))
        
        # Actual DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Recall@5: % of relevant items in top-5
        num_relevant_in_top5 = sum(relevances)
        recall_at_5 = num_relevant_in_top5 / top_k  # Simple recall
        recall_scores.append(recall_at_5)
    
    return {
        'ndcg@5': np.mean(ndcg_scores),
        'recall@5': np.mean(recall_scores),
        'ndcg@5_std': np.std(ndcg_scores),
    }

print(f"\nEvaluating TF-IDF retrieval on {len(golden_df)} golden set examples...")
retrieval_metrics = evaluate_retrieval(golden_records, corpus_vectors, corpus_intents, tfidf, top_k=5)

print(f"\n--- Baseline 2 Performance on Golden Set ---")
print(f"NDCG@5: {retrieval_metrics['ndcg@5']:.3f} (±{retrieval_metrics['ndcg@5_std']:.3f})")
print(f"Recall@5: {retrieval_metrics['recall@5']:.3f}")

baseline2_results = {
    'name': 'TF-IDF Retrieval',
    'ndcg@5': retrieval_metrics['ndcg@5'],
    'recall@5': retrieval_metrics['recall@5'],
    'corpus_size': len(corpus),
    'description': 'TF-IDF vectorization + cosine similarity retrieval'
}

# Save baselines
print("\n" + "=" * 80)
print("SAVE BASELINE RESULTS")
print("=" * 80)

baseline_report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'golden_set_size': len(golden_df),
    'corpus_size': len(corpus),
    'baseline_1_intent': baseline1_results,
    'baseline_2_retrieval': baseline2_results,
}

os.makedirs('reports', exist_ok=True)

with open('reports/baseline_scores.json', 'w') as f:
    json.dump(baseline_report, f, indent=2)

print(f"\n✓ Saved to: reports/baseline_scores.json")

# Summary table
print("\n" + "=" * 80)
print("BASELINE SUMMARY")
print("=" * 80)

summary = f"""
┌─────────────────────────────────────────────────────────────────┐
│                     BASELINE PERFORMANCE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Baseline 1: Intent Majority Class                              │
│   Macro F1: {baseline1_results['macro_f1']:.3f}                           │
│   Strategy: Always predict {baseline1_results['majority_intent']}              │
│                                                                 │
│ Baseline 2: TF-IDF Retrieval                                   │
│   NDCG@5: {baseline2_results['ndcg@5']:.3f}                           │
│   Recall@5: {baseline2_results['recall@5']:.3f}                          │
│   Corpus: {baseline2_results['corpus_size']:,} Uber conversations         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ TARGETS FOR PHASE 2 SYSTEM:                                    │
│   Intent F1 > {baseline1_results['macro_f1']:.3f}                          │
│   Retrieval NDCG@5 > {baseline2_results['ndcg@5']:.3f}                 │
└─────────────────────────────────────────────────────────────────┘
"""

print(summary)

print("\n" + "=" * 80)
print("✓ Milestone 4 COMPLETE")
print("=" * 80)
print(f"""
Baselines established:

Baseline 1 (Intent): Macro F1 = {baseline1_results['macro_f1']:.3f}
- Simple majority-class classifier
- Provides floor for intent classification

Baseline 2 (Retrieval): NDCG@5 = {baseline2_results['ndcg@5']:.3f}
- TF-IDF + cosine similarity
- Provides floor for retrieval quality

Next: Milestone 5 (FAISS semantic retrieval) should beat these baselines.
""")
