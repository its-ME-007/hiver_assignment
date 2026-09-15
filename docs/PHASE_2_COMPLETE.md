# Phase summary

This repository is a working prototype for an Uber support-routing assistant.

## Included

- intent classification using a TF-IDF + logistic regression pipeline
- FAISS-based retrieval over historical support examples
- escalation policy checks with optional Gemini tie-break logic
- a Streamlit demo for live testing
- an evaluation harness that exports metrics to `reports/final_evaluation.json`

## Current results

The current evaluated metrics are:
- intent F1: 0.5199
- majority baseline F1: 0.0166
- improvement: 3,025.6%
- retrieval mean score: 0.6747
- escalation F1: 0.6667

## Caveat

These results represent the project as it exists in the checked-in repo and should be reported accurately in any submission.
