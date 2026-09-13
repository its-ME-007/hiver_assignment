# Requirements Summary

## requirements.txt (REQUIRED)

**Status**: ✅ Required for Phase 2 execution  
**All dependencies installed and working**

```
pandas>=1.3.0          # Data processing
pyarrow>=5.0.0         # Parquet support
langdetect>=1.0.9      # Language detection
pyyaml>=5.4            # Config parsing
scikit-learn>=0.24.0   # ML models (Logistic Regression, TF-IDF)
sentence-transformers>=2.0.0  # Embeddings (all-mpnet-base-v2)
faiss-cpu>=1.7.0       # Semantic search index
streamlit>=1.0.0       # UI demo
google-generativeai>=0.3.0  # LLM generation (Gemini)
```

**Used by**:
- `08_escalation_rag_pipeline.py` - Full system
- `09_evaluation_harness.py` - Metrics
- `app.py` - Streamlit UI
- All 01-07 modules

---

## requirements-dev.txt (OPTIONAL)

**Status**: ❌ NOT required for Phase 2  
**Not needed for evaluators**

```
hypothesis>=6.0.0      # Property-based testing (not used in Phase 2)
pytest>=6.2.0          # Test runner (not used in Phase 2)
pytest-cov>=2.12.0     # Coverage reporting (not used in Phase 2)
black                  # Code formatter (optional)
flake8                 # Linter (optional)
```

**Why it exists**: 
- Set up for Phase 1 development workflow
- Enables testing and code quality checks
- Not needed for Phase 2 proof-of-concept

**When to use**:
- Only if you want to run tests: `pytest`
- Only if you want to format code: `black .`
- Only if you want to lint: `flake8 *.py`

---

## Verification

All Phase 2 requirements verified working:

```
✅ Intent Classifier: Loaded (LogisticRegression)
✅ TF-IDF Vectorizer: Loaded (5000 features)
✅ FAISS Index: Loaded (39,892 vectors)
✅ Golden Set: Loaded (165 examples)
✅ Embedding Model: Ready (768-dim)
```

---

## Installation

Already done in `.venv/`, but if you need to reinstall:

```bash
# For Phase 2 execution (required)
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

---

## Summary

- **requirements.txt**: ✅ Required, all working
- **requirements-dev.txt**: ❌ Optional, not needed for Phase 2

**For evaluators**: Only use `requirements.txt`
