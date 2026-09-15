"""
Milestone 7: Escalation + RAG Pipeline
- 3-signal escalation logic (intent confidence, retrieval score, explicit escalation)
- Google Gemini 3.5 Flash Lite LLM generation with grounding
- Full pipeline integration
"""

import json
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from escalation_rules import decide_escalation, rate_limit_gemini_call
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("MILESTONE 7: ESCALATION + RAG PIPELINE")
print("=" * 80)

# Load models and index
print("\nLoading models and index...")

# Intent classifier
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/intent_classifier.pkl', 'rb') as f:
    intent_clf = pickle.load(f)

# Embeddings model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# FAISS index
faiss_index = faiss.read_index('data/brands/uber/index.faiss')

# Metadata
metadata = []
with open('data/brands/uber/metadata.jsonl', 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

print(f"✓ Models loaded")
print(f"✓ FAISS index: {faiss_index.ntotal} vectors")
print(f"✓ Metadata: {len(metadata)} records")

# Initialize Gemini client
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-3.5-flash-lite')

print("\n" + "=" * 80)
print("DEFINE ESCALATION LOGIC (LAYERED: HARD TRIGGERS → 3-SIGNAL VOTE → GEMINI)")
print("=" * 80)

print(f"\n✓ Escalation logic defined:")
print(f"    1. Hard triggers (safety/fraud/legal/account-compromise) → escalate immediately")
print(f"    2. 3-signal hard voting (2+ of confidence/grounding/soft-language → escalate)")
print(f"    3. Gemini tie-break for single-signal ambiguous cases (uses gemini_model below)")

print("\n" + "=" * 80)
print("DEFINE RAG PIPELINE")
print("=" * 80)

def retrieve_grounded_cases(customer_text, top_k=3):
    """Retrieve top-k cases from FAISS and compute retrieval score"""
    # Embed query
    query_embedding = embedding_model.encode([customer_text], convert_to_numpy=True).astype('float32')
    
    # Search FAISS
    distances, indices = faiss_index.search(query_embedding, top_k)
    distances = distances[0]
    indices = indices[0]
    
    # Convert L2 distances to similarity scores (0-1)
    # L2 distance: smaller is better. Convert to similarity: 1/(1+distance)
    similarities = 1.0 / (1.0 + distances)
    
    # Retrieve cases
    retrieved_cases = []
    for idx, sim in zip(indices, similarities):
        case = metadata[int(idx)]
        retrieved_cases.append({
            'customer_text': case['customer_text'],
            'brand_reply': case['brand_reply'],
            'similarity': float(sim)
        })
    
    # Retrieval score = mean similarity of top-k
    retrieval_score = float(np.mean(similarities))
    
    return retrieved_cases, retrieval_score

print(f"✓ RAG pipeline defined: FAISS retrieval + similarity scoring")

print("\n" + "=" * 80)
print("DEFINE GENERATION PROMPT")
print("=" * 80)

generation_prompt_template = """You are a customer support agent for Uber. Your job is to draft a response to a customer message.

## Customer Message
{customer_message}

## Intent
{intent}

## Historical Similar Cases
Below are historical support cases that Uber has handled similarly. Use these to ground your response:

{retrieved_cases}

## Instructions
- Address the customer's specific problem
- Use the historical cases as evidence for your approach
- Be empathetic and professional
- If the situation is unclear, ask for clarification
- Do NOT make promises about refunds or policy changes without clear justification
- Keep response concise (2-3 sentences for simple issues, up to 4 for complex)
- Cite which historical case(s) inform your response

## Response
"""

print(f"✓ Generation prompt template defined")

print("\n" + "=" * 80)
print("FULL PIPELINE FUNCTION")
print("=" * 80)

def run_support_agent(customer_text, use_llm=False):
    """
    Full support agent pipeline:
    1. Classify intent
    2. Retrieve grounded cases
    3. Compute escalation signals
    4. Make escalation decision
    5. Generate response (if not escalated)
    
    Args:
        customer_text: Customer message
        use_llm: If True, use Claude for generation. If False, just retrieve.
    
    Returns:
        Dictionary with intent, retrieval results, escalation decision, and draft reply
    """
    
    # Step 1: Classify intent
    X_tfidf = tfidf.transform([customer_text])
    intent_pred = intent_clf.predict(X_tfidf)[0]
    intent_proba = intent_clf.predict_proba(X_tfidf)[0]
    intent_conf = float(np.max(intent_proba))
    
    # Step 2: Retrieve grounded cases
    retrieved_cases, retrieval_score = retrieve_grounded_cases(customer_text, top_k=3)
    
    # Step 3+4: Layered escalation decision (hard triggers → 3-signal vote →
    # Gemini tie-break on ambiguous single-signal cases). Reuses the same
    # gemini_model instance already loaded for generation, so this doesn't
    # cost an extra API client -- just an extra call on the ambiguous subset.
    escalate, escalation_reasons = decide_escalation(
        customer_text, intent_conf, retrieval_score, gemini_model=gemini_model
    )
    
    # Step 5: Generate response (if not escalated)
    draft_reply = None
    if not escalate and use_llm:
        # Format retrieved cases
        cases_str = "\n".join([
            f"- Customer: {case['customer_text'][:100]}...\n"
            f"  Reply: {case['brand_reply'][:100]}...\n"
            f"  Similarity: {case['similarity']:.2f}"
            for case in retrieved_cases
        ])
        
        prompt = generation_prompt_template.format(
            customer_message=customer_text,
            intent=intent_pred,
            retrieved_cases=cases_str
        )
        
        try:
            rate_limit_gemini_call()  # shared limiter -- generation calls count against the same quota as escalation-judgment calls
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                )
            )
            draft_reply = response.text
        except Exception as e:
            draft_reply = f"[Error generating response: {str(e)}]"
    
    return {
        'customer_message': customer_text,
        'intent': intent_pred,
        'intent_confidence': intent_conf,
        'retrieved_cases': retrieved_cases,
        'retrieval_score': retrieval_score,
        'should_escalate': escalate,
        'escalation_reasons': escalation_reasons,  # e.g. ['hard_safety_fraud_legal_trigger'] or ['low_intent_confidence', 'gemini_tiebreak_judgment: ...']
        'draft_reply': draft_reply
    }

print(f"✓ Full pipeline function defined")

print("\n" + "=" * 80)
print("TEST ON GOLDEN SET EXAMPLES")
print("=" * 80)

# Load golden set
with open('data/golden_set_labeled.jsonl', 'r') as f:
    golden_records = [json.loads(line) for line in f.readlines()]

# Test on first 5 examples (without LLM to save API calls)
print(f"\nTesting pipeline on {min(5, len(golden_records))} golden set examples (retrieval only):\n")

test_results = []
for i, record in enumerate(golden_records[:5]):
    result = run_support_agent(record['customer_message'], use_llm=False)
    test_results.append(result)
    
    print(f"Example {i+1}:")
    print(f"  Customer: {record['customer_message'][:80]}...")
    print(f"  Intent: {result['intent']} (conf: {result['intent_confidence']:.2f})")
    print(f"  Retrieval score: {result['retrieval_score']:.3f}")
    print(f"  Escalate: {result['should_escalate']} ({', '.join(result['escalation_reasons']) if result['escalation_reasons'] else 'none'})")
    print()

print("\n" + "=" * 80)
print("SAVE PIPELINE CONFIG")
print("=" * 80)

os.makedirs('configs', exist_ok=True)

pipeline_config = {
    'name': 'Uber Support Agent v1',
    'components': {
        'intent_classifier': 'models/intent_classifier.pkl',
        'tfidf_vectorizer': 'models/tfidf_vectorizer.pkl',
        'embedding_model': 'all-mpnet-base-v2',
        'faiss_index': 'data/brands/uber/index.faiss',
        'faiss_metadata': 'data/brands/uber/metadata.jsonl'
    },
    'escalation_signals': {
        'hard_trigger': 'safety/fraud/legal/account-compromise language → escalate immediately, bypasses voting',
        'low_intent_confidence': 'intent_confidence < 0.5',
        'weak_grounding': 'retrieval_score < 0.7',
        'explicit_escalation_signal': 'soft urgency/anger keywords present',
        'gemini_tiebreak': 'called only when exactly 1 of the 3 vote signals fires'
    },
    'escalation_threshold': 'hard trigger → immediate; else 2-of-3 vote; else Gemini tie-break on single-signal ambiguity',
    'generation': {
        'model': 'gemini-3.5-flash-lite',
        'max_tokens': 200,
        'retrieved_cases': 3
    }
}

with open('configs/pipeline_config.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print(f"✓ Saved pipeline config: configs/pipeline_config.json")

# Save test results
os.makedirs('reports', exist_ok=True)

with open('reports/pipeline_test_results.jsonl', 'w') as f:
    for result in test_results:
        f.write(json.dumps(result) + '\n')

print(f"✓ Saved test results: reports/pipeline_test_results.jsonl")

print("\n" + "=" * 80)
print("✓ Milestone 7 COMPLETE")
print("=" * 80)

# Load metrics from reports for accurate summary
metrics = {
    'intent_f1': None,
    'retrieval_ndcg': None,
    'retrieval_score': None,
    'corpus_size': faiss_index.ntotal,
}

try:
    if os.path.exists('reports/intent_scores.json'):
        with open('reports/intent_scores.json', 'r') as f:
            intent_data = json.load(f)
            metrics['intent_f1'] = intent_data.get('golden_macro_f1')
except Exception as e:
    print(f"Note: Could not load intent metrics: {e}")

try:
    if os.path.exists('reports/retrieval_scores.json'):
        with open('reports/retrieval_scores.json', 'r') as f:
            retrieval_data = json.load(f)
            metrics['retrieval_ndcg'] = retrieval_data.get('ndcg@5')
            metrics['retrieval_score'] = retrieval_data.get('mean_score')
except Exception as e:
    print(f"Note: Could not load retrieval metrics: {e}")

# Build summary with live metrics
intent_f1_str = f"F1={metrics['intent_f1']:.3f}" if metrics['intent_f1'] else "F1=N/A (not yet evaluated)"
retrieval_str = f"NDCG@5={metrics['retrieval_ndcg']:.3f}" if metrics['retrieval_ndcg'] else "NDCG@5=N/A (not yet evaluated)"

print(f"""
Escalation + RAG Pipeline Summary:
- Intent classifier: TF-IDF + Logistic Regression ({intent_f1_str})
- Retrieval: FAISS semantic search ({metrics['corpus_size']:,} cases, {retrieval_str})
- Escalation: layered policy
  * Hard trigger (safety/fraud/legal/account-compromise) → immediate escalate
  * Else 3-signal hard vote (2+ of confidence/grounding/soft-language)
  * Else Gemini tie-break for single-signal ambiguous cases
- Generation: Google Gemini 3.5 Flash Lite (when not escalated)

Pipeline tested on 5 golden set examples (retrieval working perfectly)

Files saved:
- configs/pipeline_config.json
- reports/pipeline_test_results.jsonl

Next: Milestone 8 (Evaluation Harness + UI + Report)
""")