# Phase 2 Requirements: Uber Support Agent — Indexing, Classification & Evaluation

## Introduction

Phase 2 builds the machine learning infrastructure for the Uber Support Agent system on top of Phase 1's cleaned thread dataset. This phase has two major checkpoints: (1) **Golden Set Creation** (Week 1)—hand-labeling 150-250 stratified examples as a validation checkpoint and ground truth for evaluation; and (2) **FAISS Indexing & Evaluation** (Week 2)—building a high-recall vector index and measuring pipeline quality. Weeks 3-4 focus on intent classification, escalation logic tuning, LLM generation grounding, and end-to-end evaluation.

## Glossary

- **Golden Set**: A hand-labeled subset of 150-250 threads (stratified by intent, resolution tier, and difficulty) used as a validation checkpoint and ground truth for evaluation metrics.
- **Stratified Sampling**: Dividing Phase 1 threads by categorical attributes (intent, resolution_tier, difficulty) and sampling proportionally from each stratum.
- **FAISS Index**: A dense vector index using IndexFlatL2 (exact search) or IndexHNSWFlat (approximate search, ~95% recall).
- **Metadata Store**: JSON mapping thread_id to {customer_text, brand_reply, resolution_tier, intent, turn_count}.
- **Indexable Filtering**: Inclusion: resolution_tier in {resolved_explicit, resolved_implicit} AND is_boilerplate == False (approx. 40-50k threads).
- **Embedding Model**: all-mpnet-base-v2 (768-dim, sentence-transformers) for semantic similarity.

## Requirements

### Requirement 1: Generate Hand-Labeled Golden Set (Checkpoint 1)

**User Story:** As a data scientist, I want to create a stratified sample of 150-250 hand-labeled examples from Phase 1 threads early in Phase 2, so that I have a validation checkpoint for measuring retrieval quality and a ground truth set for intent classification and escalation tuning.

#### Acceptance Criteria

1. THE GoldenSetGenerator SHALL load threads.parquet and apply stratified random sampling by: (a) intent (target 12-15 per intent), (b) resolution_tier (50% explicit, 40% implicit, 10% unresolved), (c) difficulty (50% easy, 50% hard).
2. OUTPUT: data/golden_set_raw.jsonl with columns: customer_text, intent_predicted, resolution_tier, is_boilerplate, turn_count, brand_reply.
3. THE GoldenSetLabeler (human) SHALL produce data/golden_set_labeled.jsonl with: customer_text, intent_true, should_escalate_true, escalation_reason (if escalate), reply_quality_rubric (1-4 scale).
4. THE DataValidator SHALL verify all required fields are present and intent_true is in predefined set; report intent_true distribution, should_escalate_true rate, and reply_quality_rubric distribution.
5. IF any field is missing or invalid, Phase 2 SHALL halt and require re-labeling (no graceful degradation).

---

### Requirement 2: Build FAISS Dense Vector Index (Checkpoint 2)

**User Story:** As a data engineer, I want to build a FAISS index from 40-50k customer_text embeddings, so that I can perform fast semantic similarity retrieval for grounding LLM responses and validating intent classification.

#### Acceptance Criteria

1. THE FAISSBuilder SHALL load threads.parquet and filter to indexable threads: resolution_tier in {resolved_explicit, resolved_implicit} AND is_boilerplate == False.
2. THE FAISSBuilder SHALL log the indexable thread count (expected: 40-50k ±20%).
3. EMBEDDINGS: Use all-mpnet-base-v2 sentence transformer to embed customer_text strings → 768-dim vectors.
4. INDEX TYPE: Use IndexFlatL2 (exact search) for Week 2 validation; may switch to IndexHNSWFlat (approximate, ~95% recall) for production.
5. OUTPUT: data/faiss_index/index.faiss (FAISS index) + data/faiss_index/metadata.json (thread_id → {customer_text, brand_reply, resolution_tier, intent, turn_count}).
6. CONFIG: Write data/faiss_index/index_config.json with: index_type, embedding_model, vector_dimension (768), thread_count, creation_timestamp, indexable_filter_criteria.
7. VALIDATION: Load index, query one golden set example, retrieve top-5 results with distances; IF query fails, HALT Phase 2.

---

### Requirement 3: Validate FAISS Retrieval Quality Against Golden Set

**User Story:** As a data scientist, I want to measure FAISS retrieval quality on the golden set to ensure top-5 results are semantically relevant, establishing a baseline for retrieval improvements.

#### Acceptance Criteria

1. FOR each golden set query (customer_text), THE RetrieverValidator SHALL retrieve top-5 threads from FAISS and compute NDCG@5.
2. COMPUTE: (a) mean NDCG@5 across all golden set queries, (b) retrieval rate (% with NDCG@5 >= 0.70), (c) top-1 accuracy.
3. IF mean NDCG@5 >= 0.70, Phase 2 MAY proceed to component builds (Weeks 3-4).
4. IF mean NDCG@5 < 0.70, generate detailed error report (low-NDCG queries, example retrieved results, recommendations); continue but flag for investigation.
5. OUTPUT: data/faiss_validation_report.json with: ndcg_mean, ndcg_list, top1_accuracy, queries_with_low_ndcg, recommendations.

---

### Requirement 4: Train Intent Classification Model

**User Story:** As an ML engineer, I want to train a Logistic Regression + TF-IDF intent classifier on all Phase 1 threads, so that I can infer intent for new queries and evaluate accuracy on the golden set.

#### Acceptance Criteria

1. LOAD: All threads from threads.parquet (both indexable and non-indexable).
2. GROUND TRUTH INFERENCE: Use Phase 1 thread metadata or heuristic keyword classifier (e.g., "billing" → billing_issue).
3. VECTORIZATION: TF-IDF (sklearn, stop_words='english', max_features=5000, ngram_range=(1,2)).
4. MODEL: Logistic Regression (max_iter=1000, multi_class='multinomial', random_state=42), trained on 90% of Phase 1 threads (stratified split).
5. EVAL on held-out 10%: weighted F1, per-class precision/recall, confusion matrix.
6. EVAL on golden set (intent_true labels): weighted F1; compare to Phase 1 held-out F1 (should be ±5%).
7. PERSIST: data/intent_classifier.pkl (model + vectorizer) + data/intent_classifier_config.json (params, F1 scores, training_timestamp).

---

### Requirement 5: Implement Escalation Voting Framework

**User Story:** As a product manager, I want to establish a 5-signal voting framework for determining when to escalate customer issues, so that complex issues reach human agents and simple issues are automated.

#### Acceptance Criteria

1. DEFINE FIVE SIGNALS:
   - (a) low_intent_confidence (intent classifier prob < 0.60)
   - (b) unresolved_tier (resolution_tier == 'unresolved_or_ongoing')
   - (c) high_boilerplate_risk (is_boilerplate == True)
   - (d) long_conversation (turn_count > 5)
   - (e) explicit_escalation_signal (customer text contains: "urgent", "escalate", "manager", "supervisor")
2. VOTING RULE: Hard voting (sum of votes >= threshold); configurable threshold (test 2/5, 3/5, 4/5).
3. OUTPUT per query: (a) should_escalate (boolean), (b) escalation_score (0.0-1.0 or 0-5), (c) active_signals (list of signals voting for escalation).
4. EVAL on golden set: precision, recall, F1-score compared to should_escalate_true.
5. THRESHOLD TUNING: Grid search over thresholds; choose threshold maximizing F1.
6. PERSIST: data/escalation_framework.json (config, weights, threshold) + data/escalation_report.json (metrics, recommendations).

---

### Requirement 6: Implement LLM Generation with Grounding

**User Story:** As an LLM engineer, I want to build a prompt template and generation pipeline using FAISS retrieval for grounding, so that generated replies are consistent with past resolutions.

#### Acceptance Criteria

1. RETRIEVAL: For each customer query, retrieve top-5 threads from FAISS; extract brand_reply and resolution_tier from metadata.
2. GROUNDING CONTEXT: Format as "Similar past issue: [customer_text] was resolved by: [brand_reply]" for top-3 or top-5 results.
3. PROMPT: System message (Uber support role) + grounding context + current query + instruction to prioritize consistency with past resolutions.
4. LLM CALL: Claude 3 Sonnet API (anthropic library); return: (a) generated_reply, (b) confidence (0.0-1.0), (c) used_grounding (boolean).
5. EVAL on golden set: BERTScore (F1 >= 0.50 compared to brand_reply), length similarity, retrieval_relevance (NDCG@5).
6. COST TRACKING: Log all API calls (input/output tokens); generate data/generation_cost_report.json for budget planning.
7. PERSIST: data/generation_config.json (prompt template, grounding instruction, API config) + data/generation_sample_outputs.json (10 example queries with grounding and replies).

---

### Requirement 7: Build Evaluation & Metrics Harness

**User Story:** As a data scientist, I want to build a comprehensive evaluation framework measuring intent classification, retrieval quality, generation quality, escalation accuracy, and end-to-end performance, enabling systematic improvement.

#### Acceptance Criteria

1. LOAD golden set and run full pipeline: retrieve intent, top-5 threads, check escalation, generate reply.
2. METRICS by category:
   - **Intent**: weighted F1, per-class F1, confusion matrix, accuracy
   - **Retrieval**: NDCG@5, top-1 accuracy, MRR, retrieval_rate (% with >= 1 relevant result)
   - **Escalation**: precision, recall, F1, confusion matrix, per-signal analysis
   - **Generation**: BERTScore (F1), length similarity, grounding_coverage (% results used)
   - **End-to-End**: success_rate (% resolved without escalation), escalation_rate, average latency (ms)
3. ABLATION: Disable components and measure impact (e.g., "no grounding" vs "with grounding").
4. OUTPUT: data/phase2_evaluation_report.json (all metrics, ablation results, per-query results) + data/phase2_evaluation_report.md (human-readable with visualizations, error analysis, recommendations).
5. CONFIDENCE TIERS: Allow thresholds (e.g., only evaluate predictions >= 0.70) and produce separate reports per tier.
6. ALERTS: IF any metric below threshold (intent F1 < 0.70, NDCG@5 < 0.70, escalation F1 < 0.60), flag "ALERT" and recommend investigation.

---

### Requirement 8: Implement Evaluation Visualization & Error Analysis

**User Story:** As a product manager, I want visualizations and error analysis to understand failure modes and prioritize improvements.

#### Acceptance Criteria

1. CHARTS (matplotlib/plotly): confusion matrix (heatmap), per-class F1 (bar), NDCG distribution (histogram), PR curve (escalation), BERTScore distribution (histogram).
2. ERROR CATEGORIZATION:
   - Intent Misclassification: query, predicted, true, examples
   - Retrieval Failure: query, top-5 results, why irrelevant
   - Generation Failure: query, generated vs expected, BERTScore, reason
   - Escalation Error: query, predicted, true, mismatch reason
3. OUTPUT: data/phase2_error_analysis.md with error distribution, top-10 errors, patterns, "easy wins" (high-impact low-effort fixes).
4. CHARTS: Write PNG files to data/phase2_evaluation_charts/ and embed in markdown report.

---

### Requirement 9: Validate Data Consistency & Schema Compliance

**User Story:** As a data engineer, I want to validate all Phase 2 outputs conform to expected schemas, preventing downstream errors and enabling confident handoff.

#### Acceptance Criteria

1. **Golden Set**: All required fields present, correct types, no nulls in required fields, intent_true in predefined set.
2. **FAISS Index**: metadata.json valid JSON, correct keys for all thread_ids, embedding dimension = 768, index file size plausible, index loads and queries successfully.
3. **Models/Configs**: Config JSONs valid and contain required fields, model files loadable (pickle), class labels/thresholds reasonable (not NaN, within ranges).
4. **Evaluation Reports**: JSON contains required metric sections, all metrics numeric (or valid categorical strings), no NaN/Inf, per-query results match golden set size, confusion matrices square and sum correctly.
5. IF validation fails, output detailed error messages (file path, line/record id, field name, expected type, actual value) and HALT Phase 2.

---

### Requirement 10: Document Phase 2 Decisions & Limitations

**User Story:** As a data engineer, I want to document all Phase 2 decisions and limitations, so that Phase 3 and downstream teams understand system reliability and failure modes.

#### Acceptance Criteria

1. GENERATE: data/decision_log_phase2.md with:
   - Golden set sampling strategy (rationale for 150-250 size)
   - FAISS configuration (IndexFlatL2 vs IndexHNSWFlat trade-off)
   - Intent classification approach (TF-IDF + LR rationale, class heuristic)
   - Escalation framework design (5-signal choice, voting rule, threshold tuning results)
   - LLM generation prompt template (grounding strategy, API cost implications)
   - Evaluation metrics (why NDCG@5, why BERTScore, success thresholds)
2. **KNOWN LIMITATIONS**:
   - Intent inference is heuristic-based; may not generalize to new query types
   - Retrieval uses customer_text only (brand_reply omitted to avoid boilerplate bias)
   - FAISS approximation may miss ~5% relevant threads if IndexHNSWFlat used
   - Golden set is small (may not cover all intent/resolution combinations)
   - LLM generation is Claude 3 Sonnet only (not compared to other models)
   - Escalation voting is unweighted hard voting (may not reflect true cost of false positives vs negatives)
3. **METRICS SUMMARY**: All Phase 2 results (intent F1, NDCG@5, escalation F1, BERTScore, success_rate, thread_count, golden set size, timestamps).
4. **PHASE 3 RECOMMENDATIONS**: Next improvements (expand golden set, fine-tune embedding model, weighted voting), retraining frequency, production monitoring strategy.

---

### Requirement 11: Set Up Reproducible Pipeline & Experiment Tracking

**User Story:** As an ML engineer, I want Phase 2 fully reproducible with versioned experiments and checksums, so that improvements are auditable and models can be deployed confidently.

#### Acceptance Criteria

1. VERSIONING: Each run gets unique run_id (timestamp + git commit hash); outputs stored in data/phase2_runs/{run_id}/.
2. METADATA: Record inputs (threads.parquet path), hyperparameters (TF-IDF, LR, embedding model, index type, escalation thresholds), random seeds, software versions, execution metadata (start/end time, user, machine).
3. CHECKSUMS: Compute SHA-256 for all major outputs; store in metadata.json.
4. COMPARISON: Load metadata from two runs, produce diff report (data/phase2_comparison_{run_id1}_vs_{run_id2}.json) showing what changed and how results changed.
5. VERSIONING FOR DEPLOYMENT: Tag best run (e.g., v1.0.0) by copying data/phase2_runs/{run_id}/* to data/phase2_models_v1.0.0/.

---

### Requirement 12: Establish Phase 2 Completion Criteria & Sign-Off

**User Story:** As a project manager, I want clear Phase 2 completion criteria and documented sign-off, so that Phase 3 can proceed confidently with a validated foundation.

#### Acceptance Criteria

**PHASE 2 IS COMPLETE** only when ALL criteria met:
- ✅ **Checkpoint 1**: Golden set (150-250 labeled examples) created, validated, all required fields present
- ✅ **Checkpoint 2**: FAISS index built, NDCG@5 >= 0.70 (Requirement 3)
- ✅ **Intent Classifier**: Test F1 >= 0.70, golden set F1 >= 0.65 (Requirement 4)
- ✅ **Escalation**: F1 >= 0.60 on golden set (Requirement 5)
- ✅ **LLM Generation**: Grounding pipeline working, cost report generated, sample outputs reviewed (Requirement 6)
- ✅ **Evaluation**: Comprehensive report with all metrics, no "ALERT"-level failures (Requirement 7-8)
- ✅ **Data Validation**: All schema checks pass (Requirement 9)
- ✅ **Documentation**: decision_log_phase2.md complete (Requirement 10)
- ✅ **Reproducibility**: All outputs versioned with checksums (Requirement 11)

**SIGN-OFF**: ProjectManager produces data/phase2_sign_off.md with completion checklist, key metrics, known limitations, and team approvals.

**PHASE 3 SHALL NOT BEGIN** until sign-off is finalized and approved.

