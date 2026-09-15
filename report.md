# Uber Support Agent: Final Report

## 1. Problem framing

This project addresses a customer-support routing task for Uber-style conversations. The system is designed to determine the likely support intent behind a customer message, retrieve relevant historical examples, and decide whether an issue should be escalated to a human agent or handled automatically.

The core task is not to replace human support end-to-end, but to provide a transparent prototype for triage and routing assistance. The repository implements a pragmatic support-agent workflow using intent classification, dense semantic retrieval, and layered escalation logic.

## 2. Definition of “good”

For this project, a “good” system should do the following reliably on a fixed evaluation set:
- classify the primary support intent correctly,
- retrieve support examples that are relevant to the issue at hand,
- avoid over-escalating low-risk messages,
- identify cases where routing to a human is warranted,
- provide interpretable intermediate evidence rather than opaque model-only output.

The evaluation is therefore judged on measured classification quality, retrieval relevance, and escalation effectiveness, rather than on broad claims about deployment readiness or product maturity.

## 3. What was not built

This repository is intentionally not a full customer-support platform. The project does not include:
- production-grade authentication or authorization,
- a multi-tenant support workspace,
- a deployment stack or service orchestration layer,
- high-volume monitoring and alerting,
- CRM or ticketing integration,
- broad human-agent tooling,
- a full live production inference system.

The scope is a proof-of-concept evaluation pipeline and a local demo interface.

## 4. Data and methodology

The project uses a hand-labeled evaluation set of 165 examples and a local retrieval index built from Uber-style support threads. The evaluation harness compares a baseline and a full prototype pipeline against the same fixed examples.

The evaluated pipeline includes:
- TF-IDF + logistic regression for intent classification,
- sentence-transformers embeddings for dense retrieval,
- FAISS over local support-thread metadata and embeddings,
- rule-based escalation logic with optional Gemini-assisted reasoning,
- a deterministic set of evaluation metrics produced by the checked-in harness.

The repository’s canonical evaluation output is `reports/final_evaluation.json`.

## 5. Evaluation results against baselines

The latest checked-in evaluation output reports the following verified values:

| Metric | Value |
|--------|-------|
| Golden set size | 165 |
| Intent classifier F1 | 0.5199 |
| Majority-class baseline F1 | 0.0166 |
| Intent improvement vs majority baseline | 3025.6% |
| TF-IDF retrieval mean score | 0.4227 |
| FAISS retrieval mean score | 0.6747 |
| Retrieval improvement vs TF-IDF baseline | 59.6% |
| Rule-only escalation F1 | 0.4912 |
| Gemini tiebreak escalation F1 | 0.6203 |
| Gemini always-final-call escalation F1 | 0.6667 |

These values are the repository’s current source of truth. They should be reported as measured values from the checked-in evaluation output and not inflated through extrapolation.

## 6. Baseline comparisons

The project compares the main system against two relevant baselines:
- Majority-class intent baseline: establishes the minimum baseline for intent classification on the labeled set.
- TF-IDF retrieval baseline: provides a lexical retrieval baseline against the dense semantic retrieval system.

The observed pattern is consistent with the design intent: the semantic retrieval path and the escalation logic provide a meaningful improvement over the simpler baseline configuration.

## 7. Top five failure modes with real examples and hypotheses

### 1. Rare intents are underrepresented
Hypothesis: when an intent occurs infrequently in the dataset, the intent model has less supervised signal and tends to misclassify those messages as more common categories.

Example pattern: a rare but concrete support issue gets assigned to a dominant intent class because the model learns the majority pattern more strongly than the minority one.

### 2. Ambiguous customer wording leads to weak retrieval
Hypothesis: the retrieval system can struggle when the user’s phrasing is indirect, emotionally charged, or uses slang rather than the exact terminology in historical tickets.

Example pattern: a complaint that refers to a delayed payment or cancellation without explicit keyword overlap may retrieve only partially relevant examples.

### 3. Borderline escalation cases are sensitive to confidence and grounding
Hypothesis: the escalation decision depends on the interaction of low intent confidence, weak retrieval grounding, and explicit escalation language. Small changes in any of these signals can shift the decision boundary.

Example pattern: a message with emotional phrasing but low technical grounding may trigger a human handoff or be handled automatically depending on the selected escalation mode.

### 4. Rule-based escalation is conservative and misses some cases
Hypothesis: a deterministic rule-based path does not fully capture nuanced support situations, especially when the user is upset but the explicit signals are not cleanly present.

Evidence: the checked-in evaluation output shows lower F1 for the rule-only mode than for the Gemini-assisted modes.

### 5. LLM-assisted escalation is more sensitive to API access and rate limits
Hypothesis: when the Gemini-assisted path is enabled, the result depends on external API availability, quota, latency, and the selected mode configuration.

Evidence: the evaluation harness explicitly notes rate limits and mode-specific costs. This is a real operational dependency and a limitation of the prototype evaluation setup.

## 8. What is misleading about my headline number?

A headline number can be misleading when it is taken out of context. In this project, the strongest single headline is the Gemini-assisted escalation F1 of 0.667, but that number is conditional on:
- the exact dataset and golden set used,
- the selected evaluation mode,
- the presence of API access and the configured rate limit,
- the specific scoring function used in the harness,
- the fact that the system is a prototype and not a production deployment.

The more cautious interpretation is that the project demonstrates meaningful improvement over a naive majority baseline and a lexical retrieval baseline, without claiming general-purpose superiority or production readiness. A single aggregate metric should therefore be read alongside the baseline comparisons and the mode-specific results.

## 9. One-week next steps

The next most valuable work in one week would be:
1. document the labeling procedure and ensure it is fully consistent with the final golden set,
2. add a concise requirement compliance review and final submission checklist,
3. verify all documentation claims against the checked-in evaluation JSON,
4. simplify the root documentation to avoid duplicate or conflicting guidance,
5. remove or isolate generated artifacts and secrets from the working tree,
6. clarify the final report structure so it is aligned with the assignment outline.

These steps are intentionally modest and do not imply a production deployment path.

## 10. Limitations and honest interpretation

This project should be evaluated as a focused prototype:
- it is not a full customer-support production system,
- it is not validated for all production traffic or edge cases,
- it depends on the selected support dataset and final evaluation harness,
- the secondary Gemini-assisted decision path depends on API access and rate limits,
- its strongest claim is that it provides a transparent, measurable support-routing prototype.

## 11. Attribution and external work

The repository uses several external components and artifacts, including:
- `sentence-transformers` for embedding generation,
- `scikit-learn` for TF-IDF and logistic regression,
- `faiss-cpu` for local vector search,
- `Streamlit` for interactive inspection,
- Google Generative AI for optional Gemini-assisted escalation logic,
- the repository-local dataset and support-thread artifacts used to build the Uber support evaluation slice.

Where the repository does not contain explicit provenance for an asset, the documentation should be conservative and avoid claiming more than the repo actually supports.

## 12. Conclusion

This project demonstrates a modest but meaningful support-routing prototype for Uber-style conversations. It is strongest when described as a transparent and measured research artifact, not as a production-ready solution. The repository’s checked-in evaluation output provides the authoritative values for the final metrics, and the documentation should remain consistent with those values.
