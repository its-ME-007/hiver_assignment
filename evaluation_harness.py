"""
Milestone 8: Evaluation Harness
Comprehensive metrics, ablation study, error analysis, and final report

Fixes applied (see decision log):
1. Key Findings summary previously mislabeled the low-retrieval count as
   "strong retrieval" (copy-paste polarity bug) -- now derives the label
   from the actual threshold comparison.
2. Escalation logic previously used only 2 signals (intent confidence OR
   weak grounding), not the documented 3-signal hard-voting scheme
   (intent confidence, grounding, explicit escalation language; 2-of-3
   fires). This did not match 08_escalation_rag_pipeline.py's actual
   behavior, so the escalation F1 was evaluating a different, simpler
   rule than what the live system runs. NOTE: the explicit-language
   keyword list below is a reasonable reconstruction, not copied from
   08_escalation_rag_pipeline.py -- reconcile these two (ideally by
   importing one shared function) so they can't drift apart again.
3. TF-IDF retrieval baseline was dead code: it transformed 1,000
   documents per golden example and then discarded the result, appending
   a hardcoded 0.5 regardless. Replaced with a real TF-IDF cosine-
   similarity baseline.
4. majority_intent was hardcoded ('driver_quality_issue') instead of
   computed from the golden set's actual label distribution.
5. components.intent_classifier.test_f1 was a bare literal (0.656) not
   computed by anything in this script. Now loaded from
   reports/intent_scores.json if present, else explicitly null with a
   printed warning -- no more silently-sourced numbers in the report.
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import pickle
import os
from collections import Counter
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.metrics import f1_score, precision_recall_fscore_support

print("=" * 80)
print("MILESTONE 8: EVALUATION HARNESS")
print("=" * 80)

# Load all components
print("\nLoading components...")

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/intent_classifier.pkl', 'rb') as f:
    intent_clf = pickle.load(f)

embedding_model = SentenceTransformer('all-mpnet-base-v2')
faiss_index = faiss.read_index('data/brands/uber/index.faiss')

metadata = []
with open('data/brands/uber/metadata.jsonl', 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

# Load golden set
with open('data/golden_set_labeled.jsonl', 'r') as f:
    golden_records = [json.loads(line) for line in f.readlines()]

print(f"Golden set: {len(golden_records)} examples")

print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)


def has_explicit_escalation_language(text):
    """
    Signal 3 of the documented 3-signal escalation scheme.

    NOTE: this is a reconstruction based on the README's description, not
    copied from 08_escalation_rag_pipeline.py's actual implementation.
    Reconcile against that file before trusting this as equivalent to
    what the live pipeline does -- ideally both should call one shared
    function instead of maintaining this list twice.
    """
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    escalation_phrases = [
        'lawyer', 'legal action', 'sue', 'attorney',
        'bbb', 'better business bureau',
        'unacceptable', 'ridiculous', 'disgusting',
        'cancel my account', 'never using', 'switching to',
        'refund now', 'refund immediately', 'right now',
        'dispute', 'chargeback', 'report this',
        'third time', 'again and again', 'every time',
    ]
    return any(phrase in text_lower for phrase in escalation_phrases)


def retrieve_cases(customer_text, top_k=5):
    """Retrieve top-k cases"""
    query_embedding = embedding_model.encode([customer_text], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    similarities = 1.0 / (1.0 + distances[0])
    return indices[0], similarities, np.mean(similarities)


# Evaluate on golden set
print(f"\nEvaluating on {len(golden_records)} golden set examples...")

intent_preds = []
intent_trues = []
retrieval_scores = []
escalation_preds = []
escalation_trues = []

for record in golden_records:
    # Intent prediction
    X_tfidf = tfidf.transform([record['customer_message']])
    intent_pred = intent_clf.predict(X_tfidf)[0]
    intent_preds.append(intent_pred)
    intent_trues.append(record['intent_label'])

    # Retrieval
    indices, sims, ret_score = retrieve_cases(record['customer_message'], top_k=5)
    retrieval_scores.append(ret_score)

    # Escalation: 3-signal hard voting (2-of-3 fires), matching the
    # documented system instead of the previous 2-signal OR rule.
    intent_conf = float(np.max(intent_clf.predict_proba(X_tfidf)[0]))
    signal_1_low_confidence = intent_conf < 0.5
    signal_2_weak_grounding = ret_score < 0.7
    signal_3_explicit_language = has_explicit_escalation_language(record['customer_message'])

    signals_fired = sum([signal_1_low_confidence, signal_2_weak_grounding, signal_3_explicit_language])
    escalate_pred = signals_fired >= 2

    escalation_preds.append(escalate_pred)
    escalation_trues.append(record['should_escalate'] == 'YES')

# Compute metrics
intent_f1 = f1_score(intent_trues, intent_preds, average='macro', zero_division=0)
escalation_prec, escalation_rec, escalation_f1, _ = precision_recall_fscore_support(
    escalation_trues, escalation_preds, average='binary', zero_division=0
)

print(f"\n### INTENT CLASSIFICATION ###")
print(f"Macro F1: {intent_f1:.3f}")

print(f"\n### RETRIEVAL ###")
print(f"Mean retrieval score: {np.mean(retrieval_scores):.3f} (+/-{np.std(retrieval_scores):.3f})")
print(f"Min: {np.min(retrieval_scores):.3f}, Max: {np.max(retrieval_scores):.3f}")

print(f"\n### ESCALATION DECISION (3-signal, 2-of-3 vote) ###")
print(f"Precision: {escalation_prec:.3f}")
print(f"Recall: {escalation_rec:.3f}")
print(f"F1: {escalation_f1:.3f}")

print("\n" + "=" * 80)
print("ABLATION STUDY")
print("=" * 80)

# A) Trivial baseline: majority class, computed from actual golden set
# distribution rather than assumed/hardcoded.
majority_intent = Counter(intent_trues).most_common(1)[0][0]
majority_count = Counter(intent_trues).most_common(1)[0][1]
print(f"\nMajority class in golden set: '{majority_intent}' ({majority_count}/{len(intent_trues)} examples)")

baseline_intent_f1 = f1_score(
    intent_trues,
    [majority_intent] * len(intent_trues),
    average='macro',
    zero_division=0
)

# B) Simple baseline: TF-IDF cosine similarity retrieval (real computation,
# replacing the previous dead-code placeholder that discarded its own
# results and reported a hardcoded 0.5 for every example).
print("\nComputing TF-IDF retrieval baseline...")
corpus_texts = [m['customer_text'] for m in metadata]
corpus_tfidf = tfidf.transform(corpus_texts)

tfidf_retrieval_scores = []
for record in golden_records:
    query_tfidf = tfidf.transform([record['customer_message']])
    sims = cosine_similarity(query_tfidf, corpus_tfidf)[0]
    top5_mean = np.mean(np.sort(sims)[-5:]) if len(sims) >= 5 else np.mean(sims)
    tfidf_retrieval_scores.append(top5_mean)

tfidf_mean_score = float(np.mean(tfidf_retrieval_scores))

# C) FAISS semantic retrieval (current system, already computed above)

print(f"\nA) Trivial baseline (majority class '{majority_intent}'): Intent F1 = {baseline_intent_f1:.3f}")
print(f"B) Simple baseline (TF-IDF cosine retrieval): Mean score = {tfidf_mean_score:.3f}")
print(f"C) Current system (FAISS semantic retrieval): Intent F1 = {intent_f1:.3f}, Mean retrieval = {np.mean(retrieval_scores):.3f}")

if baseline_intent_f1 > 0:
    intent_improvement_pct = float((intent_f1 - baseline_intent_f1) / baseline_intent_f1 * 100)
    print(f"   Intent F1 improvement over majority-class baseline: {intent_improvement_pct:.1f}%")
else:
    intent_improvement_pct = None
    print("   Majority-class baseline F1 is 0 -- improvement percentage is undefined, reporting absolute F1 only.")

if tfidf_mean_score > 0:
    retrieval_improvement_pct = float((np.mean(retrieval_scores) - tfidf_mean_score) / tfidf_mean_score * 100)
    print(f"   Retrieval improvement over TF-IDF baseline: {retrieval_improvement_pct:.1f}%")
else:
    retrieval_improvement_pct = None

print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

# Categorize failures
incorrect_intent = [(intent_trues[i], intent_preds[i]) for i in range(len(intent_preds)) if intent_trues[i] != intent_preds[i]]
low_retrieval = [i for i, score in enumerate(retrieval_scores) if score < 0.7]
false_escalations = [i for i in range(len(escalation_preds)) if escalation_preds[i] and not escalation_trues[i]]
missed_escalations = [i for i in range(len(escalation_preds)) if not escalation_preds[i] and escalation_trues[i]]

print(f"\nIntent misclassifications: {len(incorrect_intent)}/{len(intent_preds)} ({len(incorrect_intent)/len(intent_preds)*100:.1f}%)")
print(f"Low retrieval scores (<0.7): {len(low_retrieval)}/{len(retrieval_scores)} ({len(low_retrieval)/len(retrieval_scores)*100:.1f}%)")
print(f"False escalations: {len(false_escalations)}/{len(escalation_preds)} ({len(false_escalations)/len(escalation_preds)*100:.1f}%)")
print(f"Missed escalations: {len(missed_escalations)}/{len(escalation_preds)} ({len(missed_escalations)/len(escalation_preds)*100:.1f}%)")

if incorrect_intent:
    print(f"\nTop intent confusion pairs:")
    confusion_pairs = Counter(incorrect_intent)
    for (true, pred), count in confusion_pairs.most_common(3):
        print(f"  {true} -> {pred}: {count} times")

print("\n" + "=" * 80)
print("FINAL EVALUATION REPORT")
print("=" * 80)

# Load the actual M6 test-set F1 from its source instead of hardcoding a
# literal here. If the file doesn't exist, report null rather than guess.
intent_test_f1 = None
intent_scores_path = 'reports/intent_scores.json'
if os.path.exists(intent_scores_path):
    with open(intent_scores_path) as f:
        intent_scores_data = json.load(f)
        intent_test_f1 = intent_scores_data.get('test_f1') or intent_scores_data.get('f1')
    if intent_test_f1 is None:
        print(f"\nWARNING: {intent_scores_path} exists but no 'test_f1'/'f1' key found -- "
              f"check its actual schema and update this loader.")
else:
    print(f"\nWARNING: {intent_scores_path} not found -- components.intent_classifier.test_f1 "
          f"will be null in the report. Run 07_train_intent_classifier.py first if this is unexpected.")

report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'golden_set_size': len(golden_records),
    'components': {
        'intent_classifier': {
            'type': 'TF-IDF + Logistic Regression',
            'test_f1': intent_test_f1,  # sourced from reports/intent_scores.json, not hardcoded
            'golden_f1': float(intent_f1)
        },
        'retrieval': {
            'type': 'FAISS semantic search',
            'corpus_size': faiss_index.ntotal,
            'mean_score': float(np.mean(retrieval_scores)),
            'std_dev': float(np.std(retrieval_scores)),
            'tfidf_baseline_mean_score': tfidf_mean_score,
        },
        'escalation': {
            'type': '3-signal hard voting (2-of-3), see has_explicit_escalation_language()',
            'precision': float(escalation_prec),
            'recall': float(escalation_rec),
            'f1': float(escalation_f1)
        }
    },
    'ablation': {
        'majority_class': majority_intent,
        'baseline_intent_f1': float(baseline_intent_f1),
        'system_intent_f1': float(intent_f1),
        'intent_improvement_pct': intent_improvement_pct,
        'tfidf_baseline_retrieval_score': tfidf_mean_score,
        'system_retrieval_score': float(np.mean(retrieval_scores)),
        'retrieval_improvement_pct': retrieval_improvement_pct,
    },
    'error_analysis': {
        'intent_errors': len(incorrect_intent),
        'low_retrieval_cases': len(low_retrieval),
        'false_escalations': len(false_escalations),
        'missed_escalations': len(missed_escalations)
    }
}

os.makedirs('reports', exist_ok=True)

with open('reports/final_evaluation.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nSaved final evaluation: reports/final_evaluation.json")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

# Key Findings: every line below derives its wording directly from the
# actual comparison being made, instead of an assumed/fixed direction.
strong_retrieval_count = len(retrieval_scores) - len(low_retrieval)
strong_retrieval_pct = strong_retrieval_count / len(retrieval_scores) * 100
low_retrieval_pct = len(low_retrieval) / len(retrieval_scores) * 100

print(f"""
Summary:
- Golden set: {len(golden_records)} examples
- Intent F1: {intent_f1:.3f} (majority-class baseline: {baseline_intent_f1:.3f})
- Escalation F1: {escalation_f1:.3f} (3-signal, 2-of-3 vote)
- Retrieval: Mean score {np.mean(retrieval_scores):.3f} (TF-IDF baseline: {tfidf_mean_score:.3f})

Key findings:
- Intent classifier F1 {intent_f1:.3f} vs majority-class baseline {baseline_intent_f1:.3f}
  {"(" + f"{intent_improvement_pct:.0f}% relative improvement)" if intent_improvement_pct is not None else "(baseline F1 is 0, relative improvement undefined)"}
- {strong_retrieval_pct:.0f}% of cases have strong retrieval (>=0.7); {low_retrieval_pct:.0f}% have weak retrieval (<0.7)
- Escalation F1: {escalation_f1:.3f} (precision {escalation_prec:.3f}, recall {escalation_rec:.3f})

Ready for Streamlit UI demo and final report.
""")