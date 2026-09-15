# Project report

This directory mirrors the root-level submission documentation and reflects the current repo state.

## Summary

The repository contains a proof-of-concept support-routing system for Uber-style conversations. It uses a TF-IDF + logistic regression intent classifier, FAISS-based semantic retrieval, and a layered escalation policy with optional Gemini tie-break logic.

## Current measured results

The checked-in evaluation summary is:
- golden set size: 165
- intent classifier F1: 0.5199
- majority-class baseline F1: 0.0166
- improvement: 3,025.6%
- retrieval mean score: 0.6747
- escalation F1: 0.6667 (Gemini-enabled compare_all mode)

These values come from `reports/final_evaluation.json` and should be treated as the current canonical evaluation results.

## Implementation notes

- `app.py` offers the interactive demo
- `evaluation_harness.py` computes the end-to-end metrics
- `escalation_rules.py` encapsulates the escalation logic
- `build_faiss_index.py` and `train_intent_classifier.py` create the retrieval and classification artifacts

## Risks and next steps

- sparse label coverage for rare intents
- moderate retrieval quality on ambiguous requests
- need for stronger human review around escalations before production use
