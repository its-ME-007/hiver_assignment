# Uber Support Agent

This repository contains a Python prototype for customer-support routing on Uber-style conversations. The system combines intent classification, retrieval over historical support examples, escalation logic, and a lightweight Streamlit demo for interactive inspection.

## Problem framing

The project addresses a practical support-routing problem: given an inbound customer message, determine the likely support intent, retrieve similar historical examples, and decide whether the case should be escalated to a human agent or handled automatically.

This is a research and prototype system, not a production support operation. The goal is to demonstrate an end-to-end pipeline with transparent intermediate signals rather than to claim a deployment-ready production service.

## What is in this repository

The project includes:
- `app.py` — Streamlit demo for manual inspection
- `evaluation_harness.py` — end-to-end evaluation and metric export
- `brand_analysis.py` — dataset selection and brand filtering logic
- `intent_taxonomy.py` — intent definitions and categories
- `golden_set_app.py` — exploratory labeling and review workflow
- `establish_baselines.py` — majority-class and TF-IDF retrieval baseline setup
- `build_faiss_index.py` — FAISS index construction
- `train_intent_classifier.py` — TF-IDF + logistic regression training
- `escalation_rules.py` — rule-based escalation logic and Gemini-assisted tie-break logic
- `escalation_rag_pipeline.py` — end-to-end support-routing flow
- `configs/labeling_rubric.md` — labeling rubric and methodology
- `src/phase1/` — processing utilities for thread reconstruction and data preparation

## Setup

The project expects a Python environment with the dependencies listed in `requirements.txt`.

```powershell
cd D:\hiver
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To enable the optional Gemini-assisted escalation path, copy the example environment file and provide a local API key:

```powershell
copy .env.example .env
```

Then set the required local variable in `.env` without committing real credentials:

```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MAX_RPM=12
```

This project should be treated as a local evaluator setup. The actual key must be supplied externally and not committed to version control.

## Local retrieval artifacts

The FAISS index, its embedding matrix, and matching metadata are generated local runtime artifacts and are intentionally excluded from Git. This keeps the repository below GitHub's file-size limit while preserving reproducibility.

Before running the Streamlit demo, retrieval pipeline, or evaluation harness in a fresh clone, generate the artifacts from the included processed data:

```powershell
python build_faiss_index.py
```

This creates the following local files under `data/brands/uber/`:

- `index.faiss`
- `embeddings.npy`
- `metadata.jsonl`

The build can take time and may download the configured sentence-transformer model on its first run. Existing local artifacts may be retained for normal use, but they should not be committed.

## Quick-start evaluator guide

The repository includes a short evaluator guide in `EVALUATOR_START_HERE.txt`. That file should be treated as the fast-start path for reviewers and remains intentionally lightweight.

## Run the Streamlit demo

```powershell
streamlit run app.py
```

The app allows a reviewer to inspect:
- the predicted intent,
- retrieved similar support cases,
- the retrieval score,
- the escalation decision and reasons.

## Run the evaluation harness

```powershell
python evaluation_harness.py
```

The harness reads the labeled golden set and writes the verified summary to `reports/final_evaluation.json`.

## Architecture and workflow

```text
Customer message
    ↓
Intent classification (TF-IDF + Logistic Regression)
    ↓
Dense semantic retrieval (FAISS)
    ↓
Escalation policy (rule-based, Gemini-assisted modes)
    ↓
Manual review / evaluation output
```

The pipeline is intentionally modular:
- Intent classification answers the question: what is the user primarily reporting?
- Retrieval provides the most relevant historical support examples for grounding.
- Escalation checks whether automation is safe or whether the case should be routed to a human.

## Golden set and labeling process

The repository includes a hand-labeled golden set of 165 examples in `data/golden_set_labeled.jsonl`.

The labeling process is described in `configs/labeling_rubric.md` and includes:
- a single primary-intent label per message,
- escalation labeling (`YES` / `NO`),
- a support-quality rubric for response quality,
- a process for handling ambiguous or multi-issue cases by choosing the primary problem.

This is a carefully curated evaluation reference set, not a full production dataset. The project reports only the verified number of examples from the current repository state.

## Baselines and evaluation modes

The current harness evaluates multiple modes and compares them against baselines:

- majority-class intent baseline
- TF-IDF retrieval baseline
- rule-only escalation
- Gemini tiebreak escalation
- Gemini always-final-call escalation

The latest checked-in evaluation output in `reports/final_evaluation.json` reports the following verified values:

| Component | Verified value |
|-----------|---------------|
| Golden set size | 165 |
| Intent classifier F1 | 0.5199 |
| Majority-class baseline F1 | 0.0166 |
| Intent improvement vs majority baseline | 3025.6% |
| TF-IDF retrieval mean score | 0.4227 |
| FAISS semantic retrieval mean score | 0.6747 |
| Retrieval improvement vs TF-IDF baseline | 59.6% |
| Rule-only escalation F1 | 0.4912 |
| Gemini tiebreak escalation F1 | 0.6203 |
| Gemini always-final-call escalation F1 | 0.6667 |

The project must be interpreted as a prototype benchmark. These values are evidence-backed and should not be overstated beyond the actual implementation and evaluation configuration.

## Limitations and what was not built

The current repo should be understood as a constrained proof of concept. It does not claim the following:
- production-ready support automation,
- perfect retrieval,
- fully validated production deployment,
- universal generalization beyond the selected dataset and task framing.

Known limitations include:
- class imbalance for underrepresented intents,
- moderate retrieval quality on ambiguous requests,
- escalation sensitivity to confidence and grounding signals,
- dependence on external Gemini API access and rate limits,
- a focused support-routing prototype rather than a broad customer-support platform.

## Attribution and external dependencies

The project relies on external libraries and tooling, including:
- Python scientific and ML stack (pandas, scikit-learn, sentence-transformers, faiss-cpu)
- Streamlit for the demo interface
- Google Generative AI for the optional Gemini-assisted escalation path
- PyTorch-related components used indirectly through the chosen embedding and ML stack

The project also uses a support dataset and processed support-thread data for the Uber-focused prototype. Where external assets are present in the repository, they should be treated as repository-local artifacts rather than a claim of broad production sponsorship or endorsement.

## Final notes

This repository demonstrates a working end-to-end support-routing prototype with transparent intermediate outputs and a reproducible harness. The checked-in evaluation output is the source of truth for the reported benchmark values, and the documentation should reflect those values conservatively.

For the primary final report, see `report.md`.
