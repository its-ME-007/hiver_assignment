"""
Streamlit UI: Support Agent Lab
Multi-brand demo of the support agent pipeline
"""

import streamlit as st
import json
import pickle
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from escalation_rules import decide_escalation, has_hard_escalation_trigger, has_explicit_escalation_language
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Support Agent Lab",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Uber Support Agent Lab")
st.markdown("**Proof-of-concept AI support agent for Uber**")
st.markdown("Demonstrates: Intent classification → Semantic retrieval → Escalation decision → Grounded response")

# Load models and metrics (cached)
@st.cache_resource
def load_models():
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
    
    return tfidf, intent_clf, embedding_model, faiss_index, metadata

@st.cache_resource
def load_metrics():
    """Load evaluation metrics from reports."""
    metrics = {
        'intent_f1': None,
        'escalation_f1': None,
        'retrieval_mean': None,
        'improvement_pct': None,
        'golden_set_size': None,
    }
    
    try:
        if os.path.exists('reports/final_evaluation.json'):
            with open('reports/final_evaluation.json', 'r') as f:
                data = json.load(f)
                metrics['intent_f1'] = data['components']['intent_classifier'].get('golden_f1')
                metrics['escalation_f1'] = data['components']['escalation'].get('f1')
                metrics['retrieval_mean'] = data['components']['retrieval'].get('mean_score')
                metrics['improvement_pct'] = data['ablation'].get('intent_improvement_pct')
                metrics['golden_set_size'] = data.get('golden_set_size')
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
    
    return metrics

print("Loading models...")
tfidf, intent_clf, embedding_model, faiss_index, metadata = load_models()
metrics = load_metrics()
print(f"✓ Models loaded. Index: {faiss_index.ntotal} vectors")

# Sidebar: Brand selector
st.sidebar.header("Configuration")
brand = st.sidebar.selectbox("Brand", ["Uber"])
st.sidebar.markdown(f"**Selected:** {brand}")
st.sidebar.markdown(f"**Conversations indexed:** {faiss_index.ntotal:,}")

st.sidebar.divider()
use_gemini_tiebreak = st.sidebar.toggle(
    "Enable Gemini tie-break layer",
    value=False,
    help="Only called for the ambiguous middle ground (exactly 1 of 3 "
         "rule-based signals fired). Hard safety/fraud/legal triggers "
         "always escalate immediately without this. Requires GOOGLE_API_KEY."
)

gemini_model_for_escalation = None
if use_gemini_tiebreak:
    import google.generativeai as genai
    import escalation_rules
    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model_for_escalation = genai.GenerativeModel('gemini-3.5-flash-lite')
        gemini_rpm = st.sidebar.number_input(
            "Gemini rate limit (requests/min)", min_value=1, max_value=1000, value=12,
            help="Default 12 matches the free tier. Raise this if you're using a paid key with a higher quota."
        )
        escalation_rules.set_gemini_rate_limit(gemini_rpm)
    else:
        st.sidebar.warning("GOOGLE_API_KEY not set -- tie-break will fall back to rule-based only.")

# Display live metrics in sidebar
st.sidebar.divider()
st.sidebar.subheader("📊 Live Metrics")
if metrics['intent_f1'] is not None:
    st.sidebar.metric("Intent F1 (golden)", f"{metrics['intent_f1']:.3f}")
if metrics['escalation_f1'] is not None:
    st.sidebar.metric("Escalation F1", f"{metrics['escalation_f1']:.3f}")
if metrics['retrieval_mean'] is not None:
    st.sidebar.metric("Retrieval (mean)", f"{metrics['retrieval_mean']:.3f}")
if metrics['improvement_pct'] is not None:
    st.sidebar.metric("Improvement", f"{metrics['improvement_pct']:.0f}%")

# Main content
st.header("Submit a Customer Message")

# Input
customer_message = st.text_area(
    "Customer message (simulating Twitter DM):",
    placeholder="e.g., I was charged twice for my ride and need a refund",
    height=100
)

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not customer_message.strip():
        st.warning("Please enter a customer message")
    else:
        # Classify intent
        X_tfidf = tfidf.transform([customer_message])
        intent_pred = intent_clf.predict(X_tfidf)[0]
        intent_proba = intent_clf.predict_proba(X_tfidf)[0]
        intent_conf = float(np.max(intent_proba))
        
        # Retrieve similar cases
        query_embedding = embedding_model.encode([customer_message], convert_to_numpy=True).astype('float32')
        distances, indices = faiss_index.search(query_embedding, 5)
        distances = distances[0]
        indices = indices[0]
        similarities = 1.0 / (1.0 + distances)
        retrieval_score = float(np.mean(similarities))
        
        # Escalation: layered policy (hard trigger -> 3-signal 2-of-3 vote
        # -> Gemini tie-break on single-signal ambiguity), same function
        # the real pipeline and eval harness use -- no separately
        # maintained copy of the rule here.
        signal_1_low_confidence = intent_conf < 0.5
        signal_2_weak_grounding = retrieval_score < 0.7
        signal_3_explicit_language = has_explicit_escalation_language(customer_message)
        signal_hard_trigger = has_hard_escalation_trigger(customer_message)

        should_escalate, escalation_reasons = decide_escalation(
            customer_message, intent_conf, retrieval_score,
            gemini_model=gemini_model_for_escalation
        )
        
        # Display results
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Intent",
                intent_pred,
                f"Confidence: {intent_conf:.1%}"
            )
        
        with col2:
            st.metric(
                "Retrieval Quality",
                f"{retrieval_score:.2f}",
                "Mean similarity (0-1)"
            )
        
        with col3:
            status = "⚠️ ESCALATE" if should_escalate else "✅ AUTO-HANDLE"
            st.metric(
                "Decision",
                status,
                f"Intent conf: {intent_conf:.1%}"
            )
        
        # Retrieved cases
        st.subheader("📚 Retrieved Similar Cases")
        st.markdown(f"Top {len(indices)} historical cases to ground the response:")
        
        for rank, (idx, sim) in enumerate(zip(indices, similarities), 1):
            case = metadata[int(idx)]
            
            with st.expander(f"**#{rank}** Similarity: {sim:.2f}", expanded=rank==1):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Customer problem:**")
                    st.text(case['customer_text'][:200])
                
                with col_b:
                    st.markdown("**Brand response:**")
                    st.text(case['brand_reply'][:200])
        
        # Escalation reasoning
        st.subheader("🚨 Escalation Decision")
        
        if should_escalate:
            st.error(f"**Status: ESCALATE TO HUMAN**")
            st.markdown("**Reasons:**")
            for reason in escalation_reasons:
                if reason == 'hard_safety_fraud_legal_trigger':
                    reason_text = 'Hard trigger: safety / fraud / legal / account-compromise language detected'
                elif reason.startswith('gemini_judgment'):
                    reason_text = f"Gemini tie-break: {reason.split(':', 1)[-1].strip()}"
                else:
                    reason_text = {
                        'low_intent_confidence': 'Low intent confidence',
                        'weak_grounding': 'Weak historical precedent',
                        'explicit_escalation_language': 'Soft urgency/anger language detected',
                    }.get(reason, reason)
                st.markdown(f"- 🔴 {reason_text}")
        else:
            st.success(f"**Status: AUTO-HANDLE**")
            st.markdown("**Signals look good:**")
            st.markdown(f"- ✅ No hard safety/fraud/legal trigger")
            st.markdown(f"- ✅ Adequate intent confidence ({intent_conf:.1%})")
            st.markdown(f"- ✅ Strong historical precedent ({retrieval_score:.2f})")
            st.markdown("- ✅ No soft escalation language")
            if use_gemini_tiebreak and gemini_model_for_escalation is not None:
                st.caption("Gemini tie-break was available and did not override this decision.")

# Footer
st.divider()

# Build footer metrics dynamically
footer_metrics = []
if metrics['intent_f1'] is not None:
    footer_metrics.append(f"- Intent classification F1: {metrics['intent_f1']:.3f}")
if metrics['escalation_f1'] is not None:
    footer_metrics.append(f"- Escalation F1: {metrics['escalation_f1']:.3f}")
if metrics['retrieval_mean'] is not None:
    footer_metrics.append(f"- Retrieval (mean similarity): {metrics['retrieval_mean']:.3f}")
if metrics['improvement_pct'] is not None:
    footer_metrics.append(f"- Improvement over baseline: {metrics['improvement_pct']:.0f}%")

metrics_text = "\n".join(footer_metrics) if footer_metrics else "Metrics not yet loaded"

st.markdown(f"""
---
**About This Demo**

This proof-of-concept demonstrates a hierarchical AI support agent for Uber:

1. **Intent Classification**: TF-IDF + Logistic Regression (13 intent categories)
2. **Semantic Retrieval**: FAISS index with all-mpnet-base-v2 embeddings ({faiss_index.ntotal:,} conversations)
3. **Escalation Logic**: 3-signal voting (intent confidence, retrieval quality, explicit signals)
4. **Grounding**: Retrieved cases inform decision-making and response generation

**Performance (from reports/final_evaluation.json):**
{metrics_text}

**Limitations:**
- Limited to resolved Uber conversations (training data)
- No multi-turn dialogue context yet
- Generation requires API call (Gemini) - not integrated in UI

**Next Steps:**
- Deploy to production with full API integration
- Add dialogue context tracking for multi-turn conversations
- A/B test against human baseline
""")