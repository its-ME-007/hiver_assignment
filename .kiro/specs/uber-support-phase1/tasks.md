# Implementation Plan: Uber Support Phase 1 Data Pipeline

## Overview

This document outlines the implementation roadmap for Phase 1 of the Uber Support Agent system. Phase 1 is a data cleaning and thread reconstruction pipeline that processes 3M raw tweets into 1000-5000 reconstructed customer-support threads with resolution classification, boilerplate detection, and normalized text for downstream embedding and classification.

The implementation follows a staged approach:
1. **Infrastructure & Setup** - Project structure, configuration, logging
2. **Data Models** - Tweet and Thread classes, schema definitions
3. **Core Pipeline** - Thread discovery, reconstruction, resolution classification
4. **Data Processing** - Text normalization, language detection, deduplication
5. **Export & Validation** - Output generation, quality assurance
6. **Comprehensive Testing** - Property-based tests (10 properties), integration tests, end-to-end validation
7. **Documentation & Runbooks** - Final polish and knowledge transfer

Each task builds incrementally on previous work. Critical path tasks are marked with ⭐. Optional tasks that can be deferred to Phase 2 are marked with ✨.

---

## Group 1: Infrastructure & Setup

### 1.1 Create Project Structure and Initialize Repository

**Description:** Set up the Phase 1 project directory structure, version control, and core configuration files.

**Tasks:**
- Create project root directory with subdirectories:
  - `src/phase1/` - Main package
  - `tests/` - Test files
  - `data/raw/` - Input data directory
  - `data/processed/` - Output directory
  - `logs/` - Application logs
  - `docs/` - Documentation
- Initialize Git repository
- Create `.gitignore` (exclude data/, logs/, __pycache__, .pytest_cache/)
- Create `setup.py` or `pyproject.toml` with package metadata

**Acceptance Criteria:**
- Directory structure matches layout above
- Git is initialized with first commit
- Project can be imported as `from phase1 import ...`

**Dependencies:** None
**Critical Path:** ⭐
**Complexity:** Small

---

### 1.2 Install and Configure Dependencies

**Description:** Install all core and development dependencies specified in requirements.

**Tasks:**
- Create `requirements.txt` with core dependencies:
  - pandas >= 1.3.0
  - pyarrow >= 5.0.0
  - langdetect >= 1.0.9
  - pyyaml >= 5.4
  - python >= 3.8
- Create `requirements-dev.txt` with dev dependencies:
  - hypothesis >= 6.0.0
  - pytest >= 6.2.0
  - pytest-cov >= 2.12.0
  - black (code formatting)
  - flake8 (linting)
- Run `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`
- Verify imports work: `python -c "import pandas, pyarrow, langdetect, yaml"`

**Acceptance Criteria:**
- All dependencies installed successfully
- No version conflicts
- All imports work in Python interactive session

**Dependencies:** 1.1
**Critical Path:** ⭐
**Complexity:** Small

---

### 1.3 Create Configuration System (config.yaml and loader)

**Description:** Implement configuration management for Phase 1 parameters and paths.

**Tasks:**
- Create `phase1/config.yaml` with all configurable parameters:
  - UBER_HANDLES: ["Uber_Support", "Uber_SFO", "Uber_NYC"]
  - MAX_TURNS: 6
  - ENFORCE_ALTERNATION: true
  - POSITIVE_CLOSERS: list of phrases
  - BOILERPLATE_MIN_OCCURRENCES: 5
  - BOILERPLATE_MIN_SHARE: 0.005
  - LANGUAGE_FILTERING: true
  - ACCEPTED_LANGUAGES: ["en"]
  - Input/output paths
  - PHASE0_PROJECTIONS: {thread_count: 2500, ...}
  - LOG_LEVEL: "INFO"
- Implement `phase1/config_loader.py` with `load_config()` function:
  - Load YAML file and return dict
  - Validate required keys present
  - Support environment variable overrides
  - Raise ValueError if config invalid
- Write unit tests for config loading

**Acceptance Criteria:**
- `config.yaml` exists with all required keys
- `load_config()` successfully loads and returns dict
- Environment variable overrides work correctly
- Config validation catches missing keys and raises ValueError
- At least 3 unit tests cover error cases

**Dependencies:** 1.1
**Critical Path:** ⭐
**Complexity:** Small

---

### 1.4 Set Up Logging and Monitoring Infrastructure

**Description:** Implement structured logging throughout the pipeline for audit trail and debugging.

**Tasks:**
- Create `phase1/logger.py` with custom logger setup:
  - Log to file (`logs/phase1.log`) and console
  - Use structured logging format: `[TIMESTAMP] [LEVEL] [MODULE] message`
  - Support LOG_LEVEL configuration (DEBUG, INFO, WARNING, ERROR)
  - Include context (row counts, decision reasons) in log messages
- Implement decision logger for tracking pipeline decisions:
  - Thread reconstruction branch selections
  - Boilerplate flagging reasons
  - Deduplication events
  - Resolution classification edge cases
- Create `phase1/metrics.py` for tracking pipeline statistics:
  - Row counts at each stage
  - Decision counts (branching, boilerplate, etc.)
  - Performance metrics (execution time)
  - Expose as dict for final reporting

**Acceptance Criteria:**
- Logger correctly formats messages with timestamp and level
- Log level filtering works (DEBUG, INFO, WARNING, ERROR)
- Decision events are logged with full context
- Metrics dict includes: rows_in, rows_after_load, rows_after_dedup, etc.
- At least 2 unit tests verify logging behavior

**Dependencies:** 1.1, 1.3
**Critical Path:** ⭐
**Complexity:** Small

---

## Group 2: Data Models

### 2.1 Define Tweet and Thread Data Classes

**Description:** Implement Tweet and Thread classes with validation and schema enforcement.

**Tasks:**
- Create `phase1/models.py` with Tweet dataclass:
  ```python
  @dataclass
  class Tweet:
      tweet_id: int
      author_id: str
      inbound: bool
      text: str
      original_text: str
      normalized_text: str
      language: str
      created_at: str
      timestamp: int
      in_response_to_tweet_id: Optional[int]
      response_tweet_id: Optional[int]
      is_boilerplate: bool = False
  ```
- Create Thread dataclass:
  ```python
  @dataclass
  class Thread:
      thread_id: str
      root_tweet_id: int
      tweets: List[Tweet]
      turn_count: int
      dropped_branches: List[int]
      dropped_branch_count: int
      resolution_tier: str
      is_boilerplate: bool
      metadata: Dict[str, Any]
  ```
- Implement Tweet validation methods:
  - `is_valid()` - verify required fields non-null
  - `to_dict()` - convert to dict for serialization
  - `__repr__()` - human-readable representation
- Implement Thread validation methods:
  - `validate_schema()` - ensure all fields match spec
  - `to_parquet_row()` - convert to output schema
  - `__len__()` - return turn_count
- Write at least 5 unit tests verifying model structure and validation

**Acceptance Criteria:**
- Tweet and Thread classes defined with all required fields
- Validation methods work correctly
- to_dict() and to_parquet_row() produce correct output
- All tests pass
- Models can be serialized and deserialized

**Dependencies:** 1.1
**Critical Path:** ⭐
**Complexity:** Small

---

### 2.2 Define Output Parquet Schema and Validation

**Description:** Implement schema definitions and validation for output parquet file.

**Tasks:**
- Create `phase1/schema.py` defining output schema:
  ```python
  THREADS_PARQUET_SCHEMA = {
      "thread_id": str,
      "customer_text": str,
      "brand_reply": str,
      "resolution_tier": str,  # enum: resolved_explicit, resolved_implicit, unresolved_or_ongoing
      "is_boilerplate": bool,
      "turn_count": int,
      "timestamp": "datetime64[ns]",
      "dropped_branch_count": int,
  }
  ```
- Implement `validate_parquet_schema(df: pd.DataFrame) -> bool`:
  - Check all required columns present
  - Verify data types match
  - Verify resolution_tier values in allowed set
  - Verify turn_count >= 1
  - Verify dropped_branch_count >= 0
  - Raise TypeError if validation fails
- Implement `get_parquet_dtypes() -> dict`:
  - Return dtype specification for parquet write
  - Enforce datetime conversion for timestamp
- Write at least 3 unit tests verifying schema validation

**Acceptance Criteria:**
- Schema definition matches requirements exactly
- validate_parquet_schema() correctly validates valid data
- validate_parquet_schema() correctly rejects invalid data
- All tests pass
- Schema can be enforced during parquet write

**Dependencies:** 2.1
**Critical Path:** ⭐
**Complexity:** Small

---

## Group 3: Core Pipeline - Thread Discovery & Reconstruction

### 3.1 Implement ThreadFinder - Discover Uber Support Threads

**Description:** Discover all Uber Support customer-support threads from the raw dataset.

**Tasks:**
- Create `phase1/thread_finder.py` with ThreadFinder class:
  - Constructor: `__init__(config: dict, logger: Logger)`
  - Main function: `find_uber_threads(df: pd.DataFrame) -> Tuple[List[int], Dict[int, Tweet]]`
- Implement thread discovery algorithm:
  - Identify Uber Support handle(s) via value_counts() on author_id
  - Confirm handles match config UBER_HANDLES
  - Filter brand_replies (inbound == False, author in UBER_HANDLES)
  - For each brand reply, look up parent tweet
  - Verify parent.inbound == True (is customer)
  - Build root_ids list and tweet_lookup dict
  - Log confirmation message: "Found {N} brand handles, {M} root customer tweets"
- Build Tweet objects for all tweets in dataset:
  - Load from CSV into DataFrame
  - Create Tweet instance for each row
  - Store in tweet_lookup dict keyed by tweet_id
  - Validate all referenced tweets exist
- Handle edge cases:
  - Log warning if parent tweet not found
  - Skip brand replies with missing parents
  - Raise ValueError if no Uber handles found
- Write at least 3 integration tests with sample data

**Acceptance Criteria:**
- Correctly identifies Uber Support handles from data
- Discovers all root customer tweets
- tweet_lookup contains all required tweets
- Missing parent tweets are logged as warnings
- Returns (root_ids, tweet_lookup) tuple
- All integration tests pass
- **Property Validation:** N/A (deterministic data discovery)

**Dependencies:** 1.1, 1.3, 1.4, 2.1
**Critical Path:** ⭐
**Complexity:** Medium

---

### 3.2 Implement ThreadReconstructor - Build Threads Forward

**Description:** Reconstruct complete conversation threads forward from root customer tweets.

**Tasks:**
- Create `phase1/thread_reconstructor.py` with ThreadReconstructor class:
  - Constructor: `__init__(config: dict, logger: Logger, metrics: Metrics)`
  - Main function: `build_thread(root_id: int, tweet_lookup: Dict[int, Tweet]) -> Tuple[Thread, List[int]]`
- Implement forward traversal algorithm:
  - Start with root_tweet from tweet_lookup
  - Assert root_tweet.inbound == True
  - Initialize thread with [root_tweet]
  - Loop: Find all replies to current tweet
  - If ENFORCE_ALTERNATION: Filter to opposite inbound value
  - If multiple replies: Sort by tweet_id, select first (chronologically earliest)
  - If >1 reply: Log dropped_branches
  - Append selected reply to thread
  - Continue until no valid replies or max_turns reached
  - Return (thread, dropped_branches)
- Build reverse index for O(1) reply lookup:
  - Create dict: {in_response_to_tweet_id → [response_ids]}
  - Use for efficient reply discovery
- Implement alternation enforcement:
  - Verify consecutive tweets have opposite inbound values
  - Reject branches that violate alternation
  - Log violations
- Write at least 5 property-based tests (hypothesis)

**Acceptance Criteria:**
- Forward traversal correctly reconstructs threads
- Alternation is enforced (consecutive tweets have opposite inbound)
- Chronologically earliest reply is selected when branching
- Dropped branches correctly logged
- Thread termination is correct (no valid replies or max_turns reached)
- Soft limit on max_turns (continues beyond 6 if valid replies exist)
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 1 (Alternation), Property 2 (Chronological Ordering), Property 3 (Termination & Completeness)

**Dependencies:** 3.1, 1.4, 2.1
**Critical Path:** ⭐
**Complexity:** Large

---

### 3.3 Implement ResolutionClassifier - Infer Resolution Status

**Description:** Classify thread resolution outcomes into three tiers based on final tweet.

**Tasks:**
- Create `phase1/resolution_classifier.py` with ResolutionClassifier class:
  - Constructor: `__init__(config: dict, logger: Logger)`
  - Main function: `classify_resolution(thread: Thread) -> str`
- Implement resolution heuristic:
  - Get last_tweet = thread.tweets[-1]
  - IF last_tweet.inbound == True (customer):
    - Normalize text: normalize_text(last_tweet.text)
    - Check if any positive closer phrase present
    - Return "resolved_explicit" if yes, "unresolved_or_ongoing" if no
  - ELSE (last_tweet from brand):
    - Return "resolved_implicit"
    - Log: "implicit resolution has medium confidence"
- Load POSITIVE_CLOSERS from config (configurable list)
- Implement phrase matching:
  - Case-insensitive matching after normalization
  - Use regex with word boundaries
  - Handle emoji and special characters
- Write at least 4 property-based tests (hypothesis)

**Acceptance Criteria:**
- Correctly identifies "resolved_explicit" when customer ends with closer phrase
- Correctly identifies "resolved_implicit" when brand ends thread
- Correctly identifies "unresolved_or_ongoing" for ambiguous endings
- Resolution tier is deterministic (same thread = same tier)
- Positive closers list is configurable and reloadable
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 4 (Resolution Tier Determinism)

**Dependencies:** 3.2, 1.3, 1.4
**Critical Path:** ⭐
**Complexity:** Medium

---

### 3.4 Implement BoilerplateDetector - Flag Templated Replies

**Description:** Detect templated and frequently repeated brand replies using dual thresholds.

**Tasks:**
- Create `phase1/boilerplate_detector.py` with BoilerplateDetector class:
  - Constructor: `__init__(config: dict, logger: Logger, metrics: Metrics)`
  - Main function: `detect_boilerplate(brand_replies: List[Tweet]) -> pd.Series[bool]`
- Implement boilerplate detection algorithm:
  - Normalize all brand reply texts using normalize_for_dedup()
  - Count frequency of each unique normalized text
  - Calculate total count N
  - For each unique text:
    - freq = count[text]
    - share = freq / N
    - is_boilerplate = (freq >= min_occurrences) AND (share >= min_share)
  - Return Series indexed by tweet_id with boolean values
- Implement normalize_for_dedup():
  - Remove mentions: @\w+ → ""
  - Remove URLs: http\S+ → ""
  - Remove 6+ digit sequences: \d{6,} → ""
  - Lowercase text
  - Collapse whitespace
  - Strip edges
- Log boilerplate statistics:
  - Unique boilerplate signatures
  - Total boilerplate replies
  - Boilerplate rate (percentage)
- Write at least 4 property-based tests (hypothesis)

**Acceptance Criteria:**
- Correctly normalizes brand reply text
- Applies dual thresholds (freq AND share, not OR)
- Correctly flags replies meeting both thresholds
- Returns boolean Series indexed by tweet_id
- Logs all boilerplate signatures for audit
- Dual threshold logic is verified (not single threshold)
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 5 (Boilerplate Detection Dual Thresholds)

**Dependencies:** 1.3, 1.4, 2.1
**Critical Path:** ⭐
**Complexity:** Medium

---

## Group 4: Data Processing - Text Normalization & Deduplication

### 4.1 Implement TextNormalizer - Normalize Tweet Text

**Description:** Normalize tweet text for embedding, deduplication, and downstream analysis.

**Tasks:**
- Create `phase1/text_normalizer.py` with TextNormalizer class:
  - Constructor: `__init__(logger: Logger)`
  - Function: `normalize_text(text: str) -> str`
- Implement normalization pipeline (in order):
  - Unescape HTML entities: html.unescape()
  - Replace URLs: http\S+ or https\S+ → "<URL>"
  - Replace mentions: @\w+ → "<MENTION>"
  - Replace 6+ digit sequences: \d{6,} → "<ID>"
  - Collapse whitespace: \s{2,} → " "
  - Strip leading/trailing whitespace
- Preserve during normalization:
  - Emojis (not removed)
  - Capitalization (case preserved)
  - Punctuation (except whitespace)
- Implement normalize_for_dedup():
  - Similar to normalize_text but also lowercase
  - Used for boilerplate and deduplication
- Write at least 5 property-based tests (hypothesis)

**Acceptance Criteria:**
- normalize_text() removes URLs, mentions, IDs as specified
- Preserves emojis, case, punctuation
- Idempotent: normalize(normalize(x)) == normalize(x)
- Deterministic: same input = same output
- normalize_for_dedup() additionally lowercases
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 6 (Text Normalization Determinism & Preservation)

**Dependencies:** 1.4, 2.1
**Critical Path:** ⭐
**Complexity:** Medium

---

### 4.2 Implement LanguageDetector - Identify Non-English Text

**Description:** Detect language of tweets and filter non-English content if configured.

**Tasks:**
- Create `phase1/language_detector.py` with LanguageDetector class:
  - Constructor: `__init__(config: dict, logger: Logger, metrics: Metrics)`
  - Function: `is_english(text: str) -> bool`
  - Function: `detect_language(text: str) -> str`
- Use langdetect library:
  - Detect language code (e.g., "en", "es", "fr")
  - Handle langdetect failures gracefully (default to True/include)
  - Log exceptions and failed detections
- Implement language filtering:
  - Read LANGUAGE_FILTERING from config
  - If enabled: Flag non-English tweets for filtering
  - If disabled: Include all tweets
- Write at least 3 property-based tests (hypothesis)

**Acceptance Criteria:**
- Correctly identifies English text
- Correctly identifies non-English text
- Gracefully handles langdetect exceptions
- is_english() is consistent (same text = same result)
- Language filtering can be toggled via config
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 7 (Language Detection Consistency)

**Dependencies:** 1.2, 1.3, 1.4
**Critical Path:** ⭐
**Complexity:** Small

---

### 4.3 Implement Deduplicator - Remove Exact Duplicates

**Description:** Remove exact duplicate tweets and empty normalized text records.

**Tasks:**
- Create `phase1/deduplicator.py` with Deduplicator class:
  - Constructor: `__init__(logger: Logger, metrics: Metrics)`
  - Function: `deduplicate(threads_df: pd.DataFrame) -> pd.DataFrame`
- Implement deduplication algorithm:
  - Step 1: Remove empty normalized_text
    - Filter: normalized_text.str.strip() == ""
    - Log count of empty tweets removed
  - Step 2: Identify exact duplicates
    - Key: (author_id, normalized_text)
    - Use pd.duplicated(subset=[...], keep='first')
    - Log count of duplicates found
  - Step 3: Remove duplicates
    - Keep first occurrence, remove rest
    - Return deduplicated DataFrame
  - Step 4: Verify deduplication
    - Confirm removed rows = empty_count + duplicate_count
    - Log verification message
- Handle edge cases:
  - Single occurrence per author: not flagged as duplicate
  - Same text, different authors: not duplicates
  - Empty text: always removed
- Write at least 4 property-based tests (hypothesis)

**Acceptance Criteria:**
- Removes tweets with empty normalized_text
- Removes exact duplicates by (author_id, normalized_text)
- Keeps first occurrence, removes subsequent duplicates
- Deduplication is idempotent: dedup(dedup(x)) == dedup(x)
- All property tests pass (100+ iterations each)
- **Property Validation:** Property 8 (Deduplication), Property 9 (No Data Loss)

**Dependencies:** 1.4, 2.1, 4.1
**Critical Path:** ⭐
**Complexity:** Small

---

## Group 5: Export & Validation

### 5.1 Implement DataExporter - Export to Parquet and Decision Log

**Description:** Export reconstructed threads to parquet and generate decision log documentation.

**Tasks:**
- Create `phase1/data_exporter.py` with DataExporter class:
  - Constructor: `__init__(config: dict, logger: Logger, metrics: Metrics)`
  - Function: `export_parquet(threads_list: List[Thread], output_path: str) -> str`
  - Function: `generate_decision_log(metadata: Dict, output_dir: str) -> str`
- Implement parquet export:
  - Build DataFrame from threads_list
  - Map Thread objects to output schema:
    - thread_id, customer_text, brand_reply, resolution_tier, is_boilerplate, turn_count, timestamp, dropped_branch_count
  - Validate types match specification
  - Write to parquet with snappy compression
  - Log: path, file size, row count, schema
- Implement decision log generation:
  - Generate markdown report with sections:
    - Dataset Overview (handles, threads found)
    - Thread Reconstruction (branch selection rule, stats)
    - Resolution Classification (tier distribution)
    - Boilerplate Detection (parameters, stats)
    - Text Normalization (language filtering, dedup stats)
    - Known Limitations (4 documented limitations)
    - Metrics Summary (all major counts)
  - Write to decision_log.md
  - Handle file write failures gracefully
- Write at least 2 integration tests

**Acceptance Criteria:**
- Parquet file matches schema exactly
- All data types correct (no unexpected nulls)
- Parquet is readable by pandas.read_parquet()
- decision_log.md contains all required sections
- Boilerplate statistics match detection results
- Resolution tier distribution matches classification
- All integration tests pass
- **Property Validation:** Property 10 (Output Schema Conformance)

**Dependencies:** 2.2, 3.2, 3.3, 3.4, 4.3, 1.4
**Critical Path:** ⭐
**Complexity:** Medium

---

### 5.2 Implement Validator - Quality Assurance and Acceptance Criteria

**Description:** Validate Phase 1 output against acceptance criteria before proceeding to Phase 2.

**Tasks:**
- Create `phase1/validator.py` with Validator class:
  - Constructor: `__init__(config: dict, logger: Logger)`
  - Function: `validate_output(threads_df: pd.DataFrame) -> ValidationReport`
- Implement validation checks:
  - Check 1: Row count comparison
    - Compare to phase0_projections['thread_count']
    - Calculate deviation percentage
    - PASS if <= 50%, WARNING if <= 100%, FAIL if > 100%
  - Check 2: Schema validation
    - Verify all required columns present
    - Verify all types match specification
    - Raise TypeError if mismatch
  - Check 3: Sample review (reproducible)
    - Sample exactly 20 rows: df.sample(n=20, random_state=42)
    - Print full thread details for manual review
    - Return list of sampled threads
  - Check 4: Data quality
    - No NULLs in critical columns
    - turn_count >= 1 for all rows
    - resolution_tier in valid set
    - dropped_branch_count >= 0 for all rows
  - Check 5: Statistics plausibility
    - Distribution of dropped_branch_count: 0, 1, 2+
    - Distribution of resolution_tier by percentage
    - is_boilerplate rate (percentage of True)
- Implement ValidationReport:
  - status: "PASS" | "FAIL" | "WARNING"
  - row_count_check: {expected, actual, deviation_pct, result}
  - schema_check: {result, mismatches}
  - sample_review: {sampled_threads, manual_review_result, comments}
  - statistics: {dropped_branch_distribution, resolution_tier_distribution, boilerplate_rate}
  - data_quality_issues: [list of issues]
- Write at least 2 integration tests

**Acceptance Criteria:**
- Row count validation produces correct deviation percentage
- Schema validation correctly identifies mismatches
- Sample review returns exactly 20 threads
- Data quality checks identify all null/invalid values
- Statistics calculations are accurate
- ValidationReport matches specification
- All integration tests pass
- **Property Validation:** Property 10 (Output Schema Conformance)

**Dependencies:** 2.2, 5.1, 1.4
**Critical Path:** ⭐
**Complexity:** Medium

---

## Group 6: Comprehensive Testing

### 6.1 Write Property Tests for All Correctness Properties

**Description:** Implement property-based tests for all 10 correctness properties defined in design.

**Tasks:**

#### Property 1: Thread Alternation Invariant
- [~] 6.1.1 Write property test for alternation (hypothesis)
  - **Property 1: Thread Alternation Invariant**
  - **Validates: Requirements 2.2**
  - Generate random tweet sequences with controlled inbound patterns
  - Verify alternation enforcement: consecutive tweets have opposite inbound
  - Run 100+ iterations, report any failures
  - _Requirements: 2.2_

#### Property 2: Chronological Ordering Preservation
- [~] 6.1.2 Write property test for chronological ordering (hypothesis)
  - **Property 2: Chronological Ordering Preservation**
  - **Validates: Requirements 2.3, 2.4**
  - Generate tweets with multiple chronologically ordered replies
  - Verify earliest reply is selected
  - Verify unselected replies logged as dropped_branches
  - Run 100+ iterations
  - _Requirements: 2.3, 2.4_

#### Property 3: Thread Termination and Completeness
- [~] 6.1.3 Write property test for termination (hypothesis)
  - **Property 3: Thread Termination and Completeness**
  - **Validates: Requirements 2.5, 2.6**
  - Generate threads of varying lengths and complexity
  - Verify termination when no valid replies exist
  - Verify soft limit (max_turns) doesn't hard-stop reconstruction
  - Run 100+ iterations
  - _Requirements: 2.5, 2.6_

#### Property 4: Resolution Tier Determinism
- [~] 6.1.4 Write property test for resolution determinism (hypothesis)
  - **Property 4: Resolution Tier Determinism**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
  - Generate threads with various ending tweets (customer/brand, with/without closers)
  - Verify resolution tier matches heuristic
  - Verify idempotence: same thread = same tier on repeated calls
  - Run 100+ iterations
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

#### Property 5: Boilerplate Dual Thresholds
- [~] 6.1.5 Write property test for boilerplate detection (hypothesis)
  - **Property 5: Boilerplate Detection Dual Thresholds**
  - **Validates: Requirements 4.1, 4.2, 4.3**
  - Generate brand replies with controlled frequency distributions
  - Verify dual thresholds (freq >= 5 AND share >= 0.5%)
  - Verify both thresholds required (not single threshold)
  - Run 100+ iterations
  - _Requirements: 4.1, 4.2, 4.3_

#### Property 6: Text Normalization Determinism
- [~] 6.1.6 Write property test for text normalization (hypothesis)
  - **Property 6: Text Normalization Determinism and Preservation**
  - **Validates: Requirements 5.1, 5.2**
  - Generate random tweet text with emojis, mentions, URLs, IDs, capitalization
  - Verify normalization is idempotent: normalize(normalize(x)) == normalize(x)
  - Verify preservation: emojis present, case preserved, punctuation preserved
  - Run 100+ iterations
  - _Requirements: 5.1, 5.2_

#### Property 7: Language Detection Consistency
- [~] 6.1.7 Write property test for language detection (hypothesis)
  - **Property 7: Language Detection Consistency**
  - **Validates: Requirements 5.3, 5.4**
  - Generate text in English and various other languages
  - Verify detection consistency: same text = same result
  - Verify correctness: English text detected as English
  - Run 100+ iterations
  - _Requirements: 5.3, 5.4_

#### Property 8: Deduplication Preservation
- [~] 6.1.8 Write property test for deduplication (hypothesis)
  - **Property 8: Deduplication Preservation and Completeness**
  - **Validates: Requirements 6.1, 6.2, 6.3**
  - Generate datasets with known duplicate patterns
  - Verify exact duplicates removed while keeping first occurrence
  - Verify idempotence: dedup(dedup(x)) == dedup(x)
  - Run 100+ iterations
  - _Requirements: 6.1, 6.2, 6.3_

#### Property 9: No Data Loss During Processing
- [~] 6.1.9 Write property test for data loss detection (hypothesis)
  - **Property 9: No Data Loss During Processing**
  - **Validates: Requirements 1.4, 6.4**
  - Track row counts at each pipeline stage
  - Verify: count_before_dedup - count_after_dedup == count_duplicates + count_empty
  - Run 100+ iterations with varying dataset sizes
  - _Requirements: 1.4, 6.4_

#### Property 10: Output Schema Conformance
- [~] 6.1.10 Write property test for schema conformance (hypothesis)
  - **Property 10: Output Schema Conformance**
  - **Validates: Requirements 7.1, 7.2**
  - Export sample threads to parquet
  - Verify schema matches specification exactly
  - Verify data types correct (str, bool, int, datetime64)
  - Verify no null values in critical columns
  - Run 100+ iterations
  - _Requirements: 7.1, 7.2_

**Acceptance Criteria:**
- All 10 property tests implemented and passing
- Each test runs 100+ iterations (configured in hypothesis settings)
- Tests use deterministic seeds for reproducibility
- All tests tagged with property number and validated requirements
- Test file: `tests/test_properties.py`

**Dependencies:** 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 5.1, 5.2
**Critical Path:** ⭐
**Complexity:** Large

---

### 6.2✨ Write Integration Tests - End-to-End Data Pipeline

**Description:** Implement integration tests validating component interactions and I/O operations.

**Tasks:**
- Create `tests/test_integration.py` with integration test suite
- Test thread discovery:
  - Load sample CSV with known Uber Support tweets
  - Verify thread discovery finds all roots
  - Verify tweet_lookup completeness
- Test thread reconstruction:
  - Load sample threads with branching
  - Verify threads are correctly reconstructed
  - Verify alternation is enforced
  - Verify dropped branches are tracked
- Test parquet export:
  - Generate sample threads
  - Export to parquet
  - Load back and verify schema matches
- Test validation:
  - Load threads.parquet with known row counts
  - Verify validation checks work correctly
  - Verify sample review functionality
- Test end-to-end pipeline:
  - Load sample CSV
  - Run full pipeline (discovery → reconstruction → normalization → export)
  - Verify output parquet is valid and readable
- Write at least 5 integration tests

**Acceptance Criteria:**
- All integration tests pass
- Test file: `tests/test_integration.py`
- Coverage includes: discovery, reconstruction, export, validation
- End-to-end test verifies complete pipeline

**Dependencies:** 3.1, 3.2, 3.3, 3.4, 4.1, 4.3, 5.1, 5.2
**Critical Path:** ⭐
**Complexity:** Large

---

### 6.3✨ Write Unit Tests - Component-Level Error Handling

**Description:** Write unit tests for error handling, edge cases, and component interactions.

**Tasks:**
- Test config loading:
  - Valid config loads correctly
  - Missing required keys raises ValueError
  - Environment variable overrides work
- Test data models:
  - Tweet validation catches invalid data
  - Thread schema validation works
  - Serialization/deserialization works
- Test text normalization edge cases:
  - Empty text handling
  - Unicode and emoji preservation
  - Multiple consecutive whitespace
  - Mixed case and punctuation
- Test deduplication edge cases:
  - Empty dataframe
  - No duplicates
  - All duplicates
  - Empty normalized_text removal
- Test logging:
  - Log levels filtering works
  - Messages are formatted correctly
  - Decision events are logged
- Write at least 10 unit tests across all components

**Acceptance Criteria:**
- All unit tests pass
- Test file: `tests/test_units.py`
- Coverage includes: error handling, edge cases, validation
- Each component has at least 1 unit test

**Dependencies:** 1.3, 1.4, 2.1, 2.2, 4.1, 4.3
**Critical Path:** ✨
**Complexity:** Medium

---

### 6.4✨ Test Coverage Report and Optimization

**Description:** Generate test coverage report and optimize tests for maximum coverage.

**Tasks:**
- Run pytest with coverage: `pytest --cov=phase1 --cov-report=html`
- Generate coverage report
- Identify gaps (functions/branches with <80% coverage)
- Add tests to reach minimum 80% coverage across all modules
- Document any uncovered code with rationale (e.g., graceful degradation paths)
- Generate coverage badge/summary

**Acceptance Criteria:**
- Minimum 80% code coverage across all modules
- Coverage report generated and documented
- All critical paths have >90% coverage
- Optional paths documented with rationale

**Dependencies:** 6.1, 6.2, 6.3
**Critical Path:** ✨
**Complexity:** Small

---

## Group 7: Validation & Documentation

### 7.1⭐ Create Phase 1 Runbook and Usage Guide

**Description:** Document how to run Phase 1 pipeline and interpret results.

**Tasks:**
- Create `docs/PHASE1_RUNBOOK.md` with:
  - Prerequisites and setup instructions
  - Configuration guide (config.yaml parameters)
  - Data format requirements (input CSV schema)
  - Running the pipeline: `python -m phase1.main`
  - Expected output files (threads.parquet, decision_log.md, validation_log.txt)
  - Troubleshooting guide (common errors and solutions)
  - Performance expectations (runtime estimates)
- Create `docs/DATA_FORMAT.md` with:
  - Input CSV schema and requirements
  - Tweet object structure
  - Thread object structure
  - Output parquet schema
- Create `docs/DECISION_RULES.md` with:
  - Thread reconstruction decision rules
  - Resolution classification heuristic
  - Boilerplate detection logic
  - Text normalization pipeline
  - Deduplication algorithm

**Acceptance Criteria:**
- Runbook is complete and accurate
- All configuration options documented
- Example commands provided
- Troubleshooting section covers common issues
- Data formats are clearly described
- Decision rules are documented with examples

**Dependencies:** 3.1, 3.2, 3.3, 3.4, 5.1, 5.2
**Critical Path:** ⭐
**Complexity:** Medium

---

### 7.2⭐ Add Comprehensive Code Documentation and Docstrings

**Description:** Document all functions, classes, and modules with docstrings and type hints.

**Tasks:**
- Add module-level docstrings to all .py files in phase1/
- Add docstrings to all public classes:
  - Description of purpose and responsibility
  - Constructor parameters and defaults
  - Public methods with input/output types
  - Example usage
- Add docstrings to all public functions:
  - Description of algorithm and behavior
  - Args with types
  - Returns with types
  - Raises with exception types
  - Example code if complex
- Add type hints to all functions:
  - Parameter types
  - Return types
  - Optional/Union types as needed
- Add inline comments for complex logic:
  - Algorithm explanations
  - Decision rationale
  - Edge case handling

**Acceptance Criteria:**
- All public functions have docstrings and type hints
- All classes have docstrings
- All modules have module-level docstrings
- Type hints are correct and complete
- Examples are provided for complex functions

**Dependencies:** 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 5.1, 5.2
**Critical Path:** ⭐
**Complexity:** Medium

---

### 7.3⭐ Create Phase 1 Architecture Document

**Description:** Document the architecture, component interactions, and design decisions.

**Tasks:**
- Create `docs/ARCHITECTURE.md` with:
  - System overview diagram (ASCII or reference to design.md)
  - Component diagram showing dependencies
  - Data flow through pipeline
  - Module responsibilities and interfaces
  - Key algorithms (thread reconstruction, classification, boilerplate detection)
  - Error handling and graceful degradation strategy
  - Performance considerations and optimization opportunities
- Include:
  - Why each component exists
  - How components interact
  - Data flow between components
  - Validation points in pipeline
  - Testing strategy overview

**Acceptance Criteria:**
- Architecture document is complete
- Diagrams accurately represent system
- Component responsibilities are clear
- Data flow is documented
- Design decisions are explained

**Dependencies:** All Group 3, 4, 5 tasks
**Critical Path:** ⭐
**Complexity:** Medium

---

### 7.4⭐ Create Test Documentation and Property Mapping

**Description:** Document testing strategy, property definitions, and test organization.

**Tasks:**
- Create `docs/TESTING_GUIDE.md` with:
  - Testing philosophy (property-based + integration + unit)
  - How to run tests: `pytest`, `pytest --cov`, `pytest -v`
  - Property-based testing explanation (hypothesis)
  - Property definitions with examples
  - Mapping of properties to requirements
  - Known test limitations and edge cases
  - How to add new tests
- Create `docs/PROPERTY_MAPPING.md` with:
  - Table of all 10 properties
  - For each property:
    - Property statement (formal specification)
    - Test strategy and examples
    - Validates which requirements
    - Test location and function name
  - Requirements to properties cross-reference

**Acceptance Criteria:**
- Testing guide is comprehensive
- All 10 properties documented
- Property-requirement mapping is clear
- Test organization is documented
- Examples are provided

**Dependencies:** 6.1, 6.2, 6.3
**Critical Path:** ⭐
**Complexity:** Small

---

### 7.5⭐ Validate Phase 1 Acceptance Criteria

**Description:** Verify all Phase 1 acceptance criteria are met before completion.

**Tasks:**
- Requirement 1: Verify thread discovery works correctly
  - Load sample data, confirm Uber handles identified
  - Verify root thread count matches expectations
- Requirement 2: Verify thread reconstruction enforces alternation
  - Generate test threads, verify alternation enforcement
  - Verify branching decisions are logged
- Requirement 3: Verify resolution classification works
  - Test all three resolution tiers
  - Verify determinism and correctness
- Requirement 4: Verify boilerplate detection applies dual thresholds
  - Generate test data, verify dual threshold logic
  - Verify not single threshold
- Requirement 5: Verify text normalization preserves required elements
  - Test emoji, case, punctuation preservation
  - Verify URLs, mentions, IDs are replaced
- Requirement 6: Verify deduplication removes exact duplicates
  - Test deduplication with known duplicates
  - Verify empty text removal
- Requirement 7: Verify parquet export matches schema
  - Export threads, verify schema exactly matches
  - Verify data types correct
- Requirement 8: Verify validation checks work correctly
  - Test row count comparison
  - Test schema validation
  - Test data quality checks
- Requirement 9: Verify decision log is complete
  - Check all required sections present
  - Verify known limitations documented
- Create acceptance_criteria_checklist.md documenting verification

**Acceptance Criteria:**
- All requirements verified met
- Checklist document created
- All tests passing
- No outstanding issues

**Dependencies:** All previous tasks
**Critical Path:** ⭐
**Complexity:** Medium

---

### 7.6✨ Performance Optimization and Profiling

**Description:** Profile Phase 1 pipeline and optimize bottlenecks (optional, can defer).

**Tasks:**
- Profile pipeline with real or representative data
- Identify bottlenecks using `cProfile` or similar
- Optimize slow components:
  - Consider vectorization opportunities
  - Optimize DataFrame operations
  - Consider caching or memoization
- Benchmark before/after optimizations
- Document performance improvements
- Create performance baseline for regression testing

**Acceptance Criteria:**
- Pipeline completes in <60 seconds for 3M row dataset (or documented estimate)
- Bottlenecks identified and optimized
- Performance report generated

**Dependencies:** 3.1-3.4, 4.1-4.3, 5.1-5.2, 6.1-6.2
**Critical Path:** ✨
**Complexity:** Medium

---

### 7.7✨ Create Phase 2 Readiness Assessment

**Description:** Document readiness for Phase 2 embedding and indexing (optional).

**Tasks:**
- Create `docs/PHASE2_READINESS.md` with:
  - Output data characteristics (row count, schema, quality metrics)
  - Known limitations and caveats for Phase 2
  - Recommendations for Phase 2 design
  - Data quality issues requiring attention
  - Performance expectations and scaling considerations

**Acceptance Criteria:**
- Readiness assessment is complete
- Recommendations are actionable for Phase 2 team

**Dependencies:** 7.5
**Critical Path:** ✨
**Complexity:** Small

---

## Checkpoint Tasks

- [~] **Checkpoint 1: Infrastructure Ready**
  - Ensure all tests pass, ask the user if questions arise.
  - _After: 1.1-1.4, 2.1-2.2_

- [~] **Checkpoint 2: Core Pipeline Implemented**
  - Ensure all tests pass, ask the user if questions arise.
  - _After: 3.1-3.4_

- [~] **Checkpoint 3: Data Processing Complete**
  - Ensure all tests pass, ask the user if questions arise.
  - _After: 4.1-4.3_

- [~] **Checkpoint 4: Export & Validation Complete**
  - Ensure all tests pass, ask the user if questions arise.
  - _After: 5.1-5.2_

- [~] **Checkpoint 5: All Tests Passing**
  - Ensure all property tests, integration tests, and unit tests pass.
  - Verify test coverage meets requirements.
  - _After: 6.1-6.4_

- [~] **Checkpoint 6: Phase 1 Acceptance Criteria Met**
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all requirements met, all acceptance criteria validated.
  - _After: 7.1-7.5_

---

## Notes

- **Task Parallelization:** Many tasks can run in parallel:
  - Group 1 tasks are sequential (infrastructure dependency)
  - Group 2 tasks are parallel (data models independent)
  - Group 3 core tasks are sequential (each depends on previous)
  - Group 4 tasks are parallel (independent data processing)
  - Group 5 depends on Groups 3, 4
  - Group 6 depends on Groups 3, 4, 5
  - Group 7 depends on all previous

- **Critical Path:** Tasks marked ⭐ are on critical path and must be completed. Tasks marked ✨ are optional and can be deferred to Phase 2 if needed for faster MVP.

- **Property-Based Testing:** Each property test marked with property number and validates specific requirements. Tests run with 100+ iterations minimum using hypothesis framework with deterministic seeds.

- **Testing Requirements:**
  - All property tests MUST pass before Phase 1 acceptance
  - All integration tests MUST pass before Phase 1 acceptance
  - Unit tests highly recommended but not blocking acceptance
  - Minimum 80% code coverage for critical paths

- **Documentation:** All code must have docstrings and type hints. Public APIs must have examples.

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2"]
    },
    {
      "id": 1,
      "tasks": ["3.1", "4.1", "4.2"]
    },
    {
      "id": 2,
      "tasks": ["3.2", "3.3", "3.4", "4.3"]
    },
    {
      "id": 3,
      "tasks": ["5.1", "5.2"]
    },
    {
      "id": 4,
      "tasks": ["6.1.1", "6.1.2", "6.1.3", "6.1.4", "6.1.5", "6.1.6", "6.1.7", "6.1.8", "6.1.9", "6.1.10", "6.2", "6.3"]
    },
    {
      "id": 5,
      "tasks": ["6.4"]
    },
    {
      "id": 6,
      "tasks": ["7.1", "7.2", "7.3", "7.4", "7.5"]
    },
    {
      "id": 7,
      "tasks": ["7.6", "7.7"]
    }
  ]
}
```

