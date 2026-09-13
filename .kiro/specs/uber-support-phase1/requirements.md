# Phase 1 Requirements: Uber Support Twitter Data Pipeline

## Introduction

Phase 1 implements the data cleaning and thread reconstruction pipeline for the Uber Support Agent system. The system processes a complete Twitter customer support dataset (3M rows) to extract meaningful Uber Support brand conversations, reconstruct customer-support exchange threads, infer resolution outcomes, detect repetitive patterns, and normalize content for downstream embedding and classification tasks. The output is a cleaned, structured parquet file containing 1000-5000 conversation threads (estimated from Phase 0 profiling) with resolution status, boilerplate flags, and rich metadata.

## Glossary

- **Thread**: A complete conversation between a customer and Uber Support, starting from a customer's initial tweet and including all replies from both parties. Threads are reconstructed chronologically forward from their root customer tweet.
- **Root Customer Tweet**: The initial customer tweet in a thread (marked as `inbound=True`) that initiates the conversation, not a reply to a prior tweet.
- **Brand Reply**: A tweet authored by Uber Support (author_id in UBER_HANDLES) that is a direct response to a customer tweet.
- **Inbound Tweet**: A tweet from a customer (inbound=True in the dataset), indicating the customer is the originator.
- **Outbound Tweet**: A tweet from Uber Support (inbound=False in the dataset), indicating the brand is responding.
- **Thread Alternation**: The property that conversation participants alternate between customer and brand. Valid threads alternate: customer → brand → customer → brand, etc.
- **Resolution Tier**: A three-level classification of how a thread concluded: `resolved_explicit` (customer explicitly confirmed resolution), `resolved_implicit` (thread ended on a brand reply with no follow-up), or `unresolved_or_ongoing` (ambiguous or ongoing).
- **Boilerplate Reply**: A brand reply that is repeated frequently or shares identical normalized text with other brand replies (appears ≥5 times or represents ≥0.5% of all brand replies), indicating a templated/automated response.
- **Normalized Text**: Tweet text after removing URLs, mentions, IDs, and excess whitespace, while preserving case and emojis. Used for duplicate detection and embedding.
- **Dropped Branch**: A valid alternative reply in a thread that was not selected for the primary thread reconstruction due to the branch-selection rule (earliest chronological continuation is preferred). Dropped branches are logged but excluded from the final thread.
- **Phase 0 Profiling Numbers**: Statistical baselines established in preliminary data exploration (e.g., expected thread count, boilerplate prevalence, resolution tier distribution).

## Requirements

### Requirement 1: Identify and Filter Uber Support Brand Threads

**User Story:** As a data engineer, I want to identify all customer-support conversations involving the Uber Support brand, so that I can isolate the brand's interactions from the broader 3M-row dataset without missing customer context.

#### Acceptance Criteria

1. THE DataPipeline SHALL confirm the exact Uber Support handle(s) by performing a value_counts() on author_id and identifying all brand-owned handles (expected: "Uber_Support" or similar set).
2. WHEN the DataPipeline processes the dataset, THE ThreadFinder SHALL construct a complete thread index by iterating through all Brand Replies, looking up their parent tweets, and including only root tweets where parent.inbound is True; thread index construction SHALL complete successfully before returning results.
3. THE ThreadFinder SHALL return a list of root customer tweet IDs (the seeds for thread reconstruction) and a tweet_lookup dictionary for O(1) reference during thread building.
4. WHEN thread discovery is complete, THE DataPipeline SHALL log the count of unique root customer tweets identified and store this in the Phase 1 output metadata for validation against Phase 0 projections.

---

### Requirement 2: Reconstruct Conversation Threads Forward from Root

**User Story:** As a data engineer, I want to reconstruct complete customer-support conversations from root customer tweet through all responses, so that I can capture the full context and resolution pathway.

#### Acceptance Criteria

1. WHEN a root customer tweet ID is provided, THE ThreadReconstructor SHALL start from that tweet and traverse forward through response chains, adding tweets to the thread in chronological order (by tweet_id / timestamp).
2. WHILE reconstructing a thread, THE ThreadReconstructor SHALL enforce Thread Alternation: each new tweet must have an inbound value opposite to the previous tweet (customer ↔ brand); WHERE the alternation rule is configurable, THE ThreadReconstructor MAY be configured to handle consecutive same-party tweets as an alternative mode.
3. WHEN a tweet has multiple valid replies (all meeting alternation rules), THE ThreadReconstructor SHALL select the chronologically earliest reply (by tweet_id) as the next thread continuation.
4. WHEN a tweet has multiple valid replies and the chronologically earliest is not selected, THE ThreadReconstructor SHALL record the unselected tweets in a dropped_branches list and log them with a "branch_selection_rule" decision log entry stating: "When a tweet has multiple valid replies, the chronologically earliest is kept; others are discarded to avoid parallel resolution paths."
5. WHEN a thread reaches its natural conclusion (no further valid replies), THE ThreadReconstructor SHALL stop expanding and return the completed thread; WHERE valid replies exist, THE ThreadReconstructor MAY continue beyond 6 turns (the 6-turn limit is a soft guideline for typical threads, not a hard stop).
6. THE ThreadReconstructor SHALL output each reconstructed thread as an ordered list of tweet objects, preserving the conversation sequence for downstream analysis.

---

### Requirement 3: Infer Resolution Status Using Three-Tier Heuristic

**User Story:** As a data engineer, I want to classify each thread's resolution outcome into one of three explicit tiers, so that I can understand the reliability of each thread as grounding material and document uncertainty in the dataset.

#### Acceptance Criteria

1. WHEN a thread is analyzed, THE ResolutionClassifier SHALL inspect the last tweet in the thread and determine its author (customer or brand).
2. IF the last tweet is a customer tweet AND its normalized text contains any of the positive closer phrases ("thanks", "thank you", "appreciate", "resolved", "fixed", "got it", "sorted", "perfect", "great, thanks", "issue fixed", "problem solved"), THEN THE ResolutionClassifier SHALL assign the thread resolution_tier as "resolved_explicit".
3. IF the last tweet is a brand reply (author is not customer), THEN THE ResolutionClassifier SHALL assign the thread resolution_tier as "resolved_implicit", AND add a metadata flag noting: "implicit resolution tier has medium confidence; silence does not guarantee satisfaction."
4. IF the last tweet is a customer tweet AND its normalized text does not contain any positive closer phrases, THEN THE ResolutionClassifier SHALL assign the thread resolution_tier as "unresolved_or_ongoing".
5. WHEN Phase 1 processing completes, THE DataPipeline SHALL print the distribution of resolution_tier values (counts and percentages for each tier) to the Phase 1 report; WHERE report printing encounters an error, THE DataPipeline SHALL continue processing (graceful degradation).

---

### Requirement 4: Detect Boilerplate and Repetitive Brand Replies

**User Story:** As a data engineer, I want to identify templated and frequently repeated brand replies, so that I can flag them as lower-confidence grounding material and understand the extent of automation in the support process.

#### Acceptance Criteria

1. WHEN processing brand replies, THE BoilerplateDetector SHALL normalize each brand reply text by removing mentions (@\w+), URLs (http\S+), numeric sequences (\d+), and excess whitespace, converting to lowercase.
2. THE BoilerplateDetector SHALL count the frequency of each normalized reply text across all brand replies and calculate the percentage of total replies for each unique normalized text.
3. WHEN a normalized text appears ≥5 times AND represents ≥0.5% of all brand replies, THE BoilerplateDetector SHALL flag it as boilerplate (both thresholds must be met) and mark all tweets with that text with is_boilerplate = True.
4. WHEN Phase 1 processing completes, THE DataPipeline SHALL print the boilerplate detection statistics: total unique boilerplate signatures, count of boilerplate replies, and percentage of all brand replies marked as boilerplate.
5. THE BoilerplateDetector SHALL allow the min_occurrences (default: 5) and min_share (default: 0.005) parameters to be tuned based on Phase 0 profiling findings; parameter values used SHALL be logged in the Phase 1 report.

---

### Requirement 5: Normalize Tweet Text for Downstream Use

**User Story:** As a data engineer, I want to normalize tweet text consistently across all tweets, so that text is ready for embedding, classification, and deduplication without requiring additional preprocessing.

#### Acceptance Criteria

1. WHEN a tweet's text is normalized, THE TextNormalizer SHALL apply the following transformations in sequence:
   - Unescape HTML entities (e.g., "&amp;" → "&")
   - Replace all URLs with the placeholder "<URL>"
   - Replace all mentions (@username) with the placeholder "<MENTION>"
   - Replace sequences of 6+ consecutive digits with the placeholder "<ID>" (order numbers, trip IDs, etc.)
   - Collapse multiple whitespace characters to a single space
   - Strip leading/trailing whitespace
2. THE TextNormalizer SHALL preserve emojis, capitalization, and punctuation (except for whitespace collapse).
3. WHEN a tweet's language is determined, THE LanguageDetector SHALL use langdetect to identify the language; IF the detected language is English, THE tweet SHALL be included in the primary index; IF the detected language is not English, THE DataPipeline SHALL flag the tweet with a language marker.
4. WHERE the processing configuration specifies language filtering, THE DataPipeline SHALL exclude non-English threads (any thread containing a non-English tweet) from the primary index but keep them in threads.parquet for future analysis (atomic operation: ensure threads.parquet storage whenever language exclusion is attempted).
5. THE TextNormalizer SHALL store both original_text and normalized_text for each tweet, preserving the ability to audit transformations.

---

### Requirement 6: Deduplicate Exact Duplicate Tweets

**User Story:** As a data engineer, I want to remove exact duplicate tweets, so that the dataset does not artificially inflate thread counts or include redundant records.

#### Acceptance Criteria

1. WHEN Phase 1 processing completes data loading and normalization, THE Deduplicator SHALL identify exact duplicate records using author_id and normalized_text as the deduplication key.
2. WHEN duplicates are found, THE Deduplicator SHALL keep only the first occurrence (by original dataset order) and remove all subsequent duplicates.
3. THE Deduplicator SHALL also remove any tweets with empty normalized_text (after normalization produces blank or only whitespace); IF all tweets in a batch have empty normalized_text after normalization, THEN the entire batch SHALL be removed.
4. WHEN deduplication is complete and duplicates are actually found and removed, THE DataPipeline SHALL log the count of duplicate records removed in the Phase 1 report.

---

### Requirement 7: Generate Phase 1 Output Parquet File

**User Story:** As a data engineer, I want to export cleaned and reconstructed threads to a structured parquet file, so that Phase 2 can reliably load the data for indexing and embedding.

#### Acceptance Criteria

1. WHEN Phase 1 processing completes, THE DataExporter SHALL write a parquet file to `data/processed/threads.parquet` containing one row per thread with the following schema:
   - `thread_id` (string): Unique identifier for the thread
   - `customer_text` (string): The normalized text of the root customer tweet
   - `brand_reply` (string): The text of the most recent brand reply in the thread (normalized or original—to be decided by the team; currently planning normalized)
   - `resolution_tier` (string): One of "resolved_explicit", "resolved_implicit", or "unresolved_or_ongoing"
   - `is_boilerplate` (boolean): True if the final brand reply is flagged as boilerplate
   - `turn_count` (integer): Number of tweets in the thread (1-6+)
   - `timestamp` (datetime): Timestamp of the root customer tweet
   - `dropped_branch_count` (integer): Count of alternative replies that were not selected during thread reconstruction
2. THE DataExporter SHALL ensure all data types match the schema exactly (no unexpected nulls or type mismatches).
3. WHERE output directory `data/processed/` does not exist, THE DataExporter MAY allow directory creation to fail, BUT SHALL only fail the Phase 1 process at the file writing step.
4. WHEN the parquet file is written successfully, THE DataPipeline SHALL log the path, file size, row count, and schema to the Phase 1 report.

---

### Requirement 8: Validate Phase 1 Output Against Acceptance Criteria

**User Story:** As a data engineer, I want to validate Phase 1 output before proceeding to Phase 2, so that I can detect data quality issues early and document any deviations from expectations.

#### Acceptance Criteria

1. WHEN threads.parquet is loaded, THE Validator SHALL compare the row count to Phase 0 profiling numbers and flag any significant deviation (>50% difference) with a warning.
2. THE Validator SHALL randomly sample exactly 20 threads from threads.parquet, print their full details (customer_text, brand_reply, turn_count, resolution_tier, dropped_branch_count), and require manual review to confirm that reconstructed threads are coherent and correctly ordered (manual review SHALL proceed only after exactly 20 threads are sampled).
3. WHEN manual review is complete, THE Validator SHALL record the result (pass/fail) and any comments in a validation_log.txt file.
4. WHEN Phase 1 processing completes, THE DataPipeline SHALL print the distribution of dropped_branch_count values (0 branches dropped, 1 branch, 2+ branches) and the percentage of threads with branching decisions.
5. WHEN Phase 1 processing completes, THE DataPipeline SHALL print resolution_tier distribution (counts and percentages for "resolved_explicit", "resolved_implicit", "unresolved_or_ongoing") and the boilerplate_rate (percentage of threads where is_boilerplate=True).
6. IF the row count is within ±50% of Phase 0 projections, manual spot-check passes, and all statistics are plausible, THEN Phase 1 is considered PASSED; IF any validation criterion fails, THEN Phase 1 SHALL halt, and Phase 2 SHALL NOT proceed until all criteria are met.
7. IF any validation check fails, THEN Phase 1 SHALL log detailed diagnostics (sample problematic threads, schema mismatches, etc.) and halt, requiring manual investigation before Phase 2 starts.

---

### Requirement 9: Document Data Cleaning Decisions and Limitations

**User Story:** As a data engineer, I want to document all data cleaning decisions, assumptions, and limitations, so that downstream teams understand the reliability and applicability of the cleaned dataset.

#### Acceptance Criteria

1. WHEN Phase 1 processing completes, THE DataPipeline SHALL generate a decision_log.md file in the output directory documenting:
   - Uber Support handle(s) confirmed and count of threads found per handle
   - Thread reconstruction branch-selection rule (earliest chronological reply preferred) and frequency of branching decisions
   - Resolution tier heuristic and known limitations (resolved_implicit is medium-confidence)
   - Boilerplate detection parameters used (min_occurrences, min_share) and justification based on Phase 0 data
   - Language filtering rules applied (if any)
   - Deduplication counts and any edge cases encountered
2. THE decision_log.md SHALL include a "Known Limitations" section stating:
   - "Resolved_implicit threads may include unresolved issues where the customer simply stopped replying"
   - "Branch-selection rule discards parallel resolution paths; may lose valid alternative resolutions"
   - "Boilerplate detection relies on text normalization; semantically similar but syntactically different replies may not be flagged"
   - "Language detection may misclassify code snippets, URLs, or mixed-language content"
3. THE decision_log.md SHALL include a "Metrics Summary" section with counts for all major pipeline steps (threads found, threads kept after dedup, boilerplate rate, resolution tier distribution, dropped branch statistics); WHERE decision_log.md generation fails, THE DataPipeline SHALL complete Phase 1 with warnings (graceful degradation).

---

## Acceptance Criteria Mapping to Testing

| Requirement | Key Acceptance Criteria | Test Type | Rationale |
|-------------|------------------------|-----------|-----------|
| 1 | Thread discovery count matches Phase 0 projection | Integration (1-2 examples) | External data; deterministic count; not input-dependent |
| 2 | Thread reconstruction preserves order and alternation | Property-based (100+ samples) | Behavior varies with thread structure; pure code logic; catches edge cases |
| 3 | Resolution tier assignment matches heuristic | Property-based (100+ samples) | Input-dependent; different tweet endings trigger different logic |
| 4 | Boilerplate detection produces consistent flags | Property-based (100+ samples) | Normalization and counting logic; benefits from randomized text inputs |
| 5 | Text normalization produces valid output | Property-based (100+ samples) | Regex transformations; various input text patterns |
| 6 | Deduplication removes exact duplicates | Property-based (100+ samples) | Duplicate detection logic is input-dependent |
| 7 | Parquet schema matches specification | Integration (1-2 examples) | External I/O; deterministic schema validation |
| 8 | Validation metrics are accurate | Integration (spot-check) | Comparison with external Phase 0 data; manual review required |
| 9 | Decision log is complete and accurate | Integration (1 example) | Log generation is a one-time, deterministic output |

