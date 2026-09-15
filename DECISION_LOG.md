# Decision Log

This decision log records the major design choices reflected in the current repository documentation and evaluation setup. The entries below are intentionally analytical and conservative: they describe the reasoning behind the prototype, the trade-offs that were accepted, and the known limitations of the current implementation.

## 1. Why Uber was selected as the dataset/brand

Decision: Use Uber support conversations as the core evaluation and retrieval domain.

Rationale: The selected dataset provided a sufficiently diverse set of support issues, including ride, cancellation, payment, delivery, account, pickup/dropoff, and technical problems. This diversity was useful for a prototype that needed both classification and retrieval to work across multiple support intents.

Alternatives considered: narrower single-intent datasets, a generic support corpus, or a smaller brand-specific slice with less diversity.

Trade-offs/consequences: The broader Uber dataset increased coverage and realism but also introduced more label imbalance and more varied phrasing than a narrow toy corpus.

## 2. Why an intent classifier was introduced

Decision: Add a dedicated intent classifier before retrieval and escalation.

Rationale: The classifier creates a structured first-stage understanding of the user’s main request. This supports routing, helps narrow retrieval to relevant examples, and makes the pipeline auditable through intent-level evaluation metrics.

Alternatives considered: skip intent classification and rely only on retrieval or LLM reasoning.

Trade-offs/consequences: This adds a clear evaluation objective and more interpretable model behavior, but also requires a fixed taxonomy and creates failure modes in underrepresented classes.

## 3. Why a Gemini-assisted decision system was used

Decision: Combine deterministic signals with Gemini-assisted evaluation for escalation decisions.

Rationale: The project wanted a rule-based floor plus a more context-sensitive prompt-based review when needed. The checked-in evaluation output shows the tiered approach materially improves escalation performance compared with rule-only logic.

Alternatives considered: pure rules, pure retrieval-plus-LLM routing, or no Gemini involvement.

Trade-offs/consequences: The system becomes more sensitive to API latency, rate limits, and external availability, but the evaluation suggests better context-aware decisions than the purely rule-based baseline.

## 4. Why Gemini was selected

Decision: Use Gemini as the practical LLM integration for the prototype.

Rationale: Gemini was selected within the project’s environment because it was available, practical, and easy to integrate with the evaluation workflow. The decision was pragmatic rather than a claim that Gemini is universally the best model for this task.

Alternatives considered: other model providers or a completely rule-driven escalation layer.

Trade-offs/consequences: This choice reduces operational complexity relative to a full model evaluation sweep, but it does not justify an absolute claim of model superiority.

## 5. Why FAISS was run locally on CPU

Decision: Use a local FAISS index rather than a hosted vector database.

Rationale: A local CPU-based index keeps the prototype simple, portable, inexpensive, and straightforward to reproduce. It does not require a cloud deployment, external database credentials, or extra operational infrastructure.

Alternatives considered: cloud vector stores, remote hosted embeddings, or a simpler keyword-only retrieval baseline.

Trade-offs/consequences: Local FAISS is efficient for a prototype and relatively small project scale, but it is not as well suited to distributed serving or highly concurrent production workloads.

## 6. Why retrieval-augmented grounding was used

Decision: Add retrieval over historical support examples before final decision making.

Rationale: The support data is domain-specific and the model should use project-specific examples for grounding. Retrieval provides evidence that is anchored in the actual dataset rather than only in broader pretrained knowledge.

Alternatives considered: direct prediction without retrieval, or a fully prompt-only pipeline.

Trade-offs/consequences: Retrieval improves grounding but introduces additional complexity and can propagate retrieval errors if the nearest examples are weak or irrelevant.

## 7. Why a simpler RAG design was preferred over more advanced retrieval setups

Decision: Keep retrieval relatively simple rather than adding re-ranking, BM25, hybrid retrieval, or more advanced architectures.

Rationale: The scope was a proof-of-concept, not a full retrieval-engine research project. The project already combines intent classification, dense retrieval, escalation heuristics, and LLM reasoning; the expected marginal benefit of more advanced retrieval mechanisms was not enough to justify the added complexity and tuning burden.

Alternatives considered: hybrid retrieval, re-ranking, BM25, and multilingual or multi-stage retrieval strategies.

Trade-offs/consequences: The design prioritizes reproducibility and clarity over optimizing every retrieval nuance. It remains a deliberate prototype choice, not a claim that alternative retrieval methods are ineffective.

## 8. How escalation thresholds were selected

Decision: Use empirically shaped threshold-style escalation logic supported by rule-based signals.

Rationale: The escalation path combines low intent confidence, weak retrieval grounding, and explicit escalation language. The rules are transparent and interpretable; they also provide a baseline against which Gemini-assisted choices can be compared.

Alternatives considered: pure hard-coded rules, fully learned escalation decisions, or a single threshold without evidence review.

Trade-offs/consequences: The thresholds are prototype-level and empirical rather than theoretically optimal or production-calibrated. This is a practical design choice for an evaluation-focused project.

## 9. Why intent and escalation were treated as separate decisions

Decision: Keep intent classification and escalation as separate tasks with distinct evaluation metrics.

Rationale: Intent answers “what is the issue?” while escalation answers “should the system route this to a human?” These are related but not identical decisions. Keeping them separate allows the project to measure each stage independently and to understand where failure occurs.

Alternatives considered: one combined decision or a single routing label.

Trade-offs/consequences: Separate tasks create clearer evaluation and debugging, but they are more demanding to interpret and can obscure the connection between classification errors and escalation errors.

## 10. Why a single primary intent was assigned

Decision: Use one primary intent per message.

Rationale: Standard classification metrics rely on a single target label per example. This also makes the routing workflow easier to evaluate and explain. Real messages can have multiple issues, but the project chose a consistent primary-label policy to maintain auditable evaluation.

Alternatives considered: multi-label classification or free-form issue tagging.

Trade-offs/consequences: This simplifies evaluation but can understate the complexity of messages with multiple relevant complaints or concerns.

## 11. Why a golden set was created

Decision: Build a fixed hand-labeled golden set for evaluation.

Rationale: A manually reviewed golden set creates a stable reference target that is independent of model predictions. It allows the project to compare intent quality, retrieval quality, and escalation behavior under a consistent evaluation frame.

Alternatives considered: trust only the training data, or evaluate on the full dataset without a fixed reference set.

Trade-offs/consequences: A fixed golden set gives clear comparability and makes benchmarking reproducible, but it is smaller and more curated than a full production dataset.

## 12. Why TF-IDF + Logistic Regression was used as the classifier baseline

Decision: Use a lightweight TF-IDF + Logistic Regression model as the standard baseline for intent classification.

Rationale: This baseline is interpretable, fast, and easy to reproduce. It provides a conventional text-classification benchmark against which the more complex semantic and escalation pipeline can be compared.

Alternatives considered: a deep neural baseline, a zero-shot model, or no baseline.

Trade-offs/consequences: This yields a simple, strong benchmark but does not reflect every possible modern text-classification approach.

## 13. Why dense semantic retrieval was compared with TF-IDF retrieval

Decision: Measure dense retrieval against a lexical baseline.

Rationale: TF-IDF retrieval depends on word overlap, while dense semantic retrieval can better capture paraphrases and semantically similar customer descriptions. The checked-in evaluation output reports the verified comparison between the two approaches.

Alternatives considered: evaluate only dense retrieval or only lexicon-based retrieval.

Trade-offs/consequences: The comparison is useful for understanding the contribution of semantic matching, but it is still limited to the selected dataset and scoring method.

## 14. Why the project was kept as a proof of concept rather than a production system

Decision: Treat the project as a prototype rather than a production-grade support platform.

Rationale: The work is intentionally scoped to demonstrate the full workflow: classification, retrieval, escalation, and manual inspection. It does not aim to provide a production service with monitoring, deployment, or full operational risk controls.

Alternatives considered: framing the project as a deployment-ready production system or broad customer support suite.

Trade-offs/consequences: The conservative framing preserves honesty about the project’s maturity and avoids unsupported claims about production readiness.

## 15. Why evaluation modes were compared instead of trusting one configuration

Decision: Compare rule-only, Gemini tiebreak, and Gemini always-final-call modes.

Rationale: The project wanted to show how much the LLM-assisted path changes escalation behaviour and whether the added complexity is worth it. The checked-in evaluation output shows meaningful differences among the modes, which is more informative than assuming a single mode is always appropriate.

Alternatives considered: choose one escalation mode and never compare it to the simpler alternatives.

Trade-offs/consequences: The comparison increases transparency about precision/recall trade-offs, rate-limit dependence, and API cost, but it also makes the system more complex to explain and reproduce.
