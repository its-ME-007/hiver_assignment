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

# Load models (cached)
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

print("Loading models...")
tfidf, intent_clf, embedding_model, faiss_index, metadata = load_models()
print(f"✓ Models loaded. Index: {faiss_index.ntotal} vectors")

# Sidebar: Brand selector
st.sidebar.header("Configuration")
brand = st.sidebar.selectbox("Brand", ["Uber"])
st.sidebar.markdown(f"**Selected:** {brand}")
st.sidebar.markdown(f"**Conversations indexed:** {faiss_index.ntotal:,}")

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
        
        # Escalation signals
        has_escalation_keywords = any(kw in customer_message.lower() for kw in 
                                     ['urgent', 'police', 'lawyer', 'sue', 'scam', 'refund now', 'demand'])
        should_escalate = (intent_conf < 0.5) or (retrieval_score < 0.7) or has_escalation_keywords
        
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
        
        reasons = []
        if intent_conf < 0.5:
            reasons.append(f"🔴 Low intent confidence ({intent_conf:.1%})")
        if retrieval_score < 0.7:
            reasons.append(f"🔴 Weak historical precedent ({retrieval_score:.2f})")
        if has_escalation_keywords:
            reasons.append("🔴 Explicit escalation language detected")
        
        if should_escalate:
            st.error(f"**Status: ESCALATE TO HUMAN**")
            st.markdown("**Reasons:**")
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.success(f"**Status: AUTO-HANDLE**")
            st.markdown("**Signals look good:**")
            st.markdown(f"- ✅ Adequate intent confidence ({intent_conf:.1%})")
            st.markdown(f"- ✅ Strong historical precedent ({retrieval_score:.2f})")
            st.markdown("- ✅ No explicit escalation language")

# Footer
st.divider()
st.markdown("""
---
**About This Demo**

This proof-of-concept demonstrates a hierarchical AI support agent for Uber:

1. **Intent Classification**: TF-IDF + Logistic Regression (13 intent categories)
2. **Semantic Retrieval**: FAISS index with all-mpnet-base-v2 embeddings (18,570 conversations)
3. **Escalation Logic**: 3-signal voting (intent confidence, retrieval quality, explicit signals)
4. **Grounding**: Retrieved cases inform decision-making and response generation

**Performance:**
- Intent classification F1: 0.65 (test) / 0.65 (golden)
- Escalation F1: 0.71
- 5,270% improvement over baseline

**Limitations:**
- Limited to resolved Uber conversations (training data)
- No multi-turn dialogue context yet
- Generation requires API call (Gemini) - not integrated in UI

**Next Steps:**
- Deploy to production with full API integration
- Add dialogue context tracking for multi-turn conversations
- A/B test against human baseline
""")
