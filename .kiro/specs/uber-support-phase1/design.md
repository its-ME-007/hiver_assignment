# Phase 1 Design: Uber Support Twitter Data Pipeline

## Overview

Phase 1 implements a data cleaning and thread reconstruction pipeline that transforms a raw 3M-row Twitter customer support dataset into a cleaned, structured parquet file containing 1000-5000 reconstructed conversation threads with metadata for downstream embedding and classification tasks.

**Key Goals:**
- Identify Uber Support brand conversations from the dataset
- Reconstruct complete customer-support threads with alternation enforcement
- Classify resolution outcomes into three tiers
- Detect and flag templated/boilerplate brand replies
- Normalize text for downstream processing
- Deduplicate records and validate output quality

**Key Constraints:**
- Thread reconstruction must enforce strict customer↔brand alternation
- Branch selection is deterministic (earliest chronological reply preferred)
- Resolution classification uses heuristics with documented confidence levels
- Boilerplate detection requires both frequency thresholds (≥5 occurrences AND ≥0.5% of total)
- All data cleaning decisions must be logged and validated

---

## Architecture

### High-Level System Design

```
Raw CSV Input (twcs.csv: 3M rows)
         ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 DATA PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Data Loading & Preprocessing]                            │
│  - Load CSV into DataFrame                                 │
│  - Validate required columns present                       │
│            ↓                                                │
│  [ThreadFinder]                                            │
│  - Identify Uber Support handles                           │
│  - Discover brand replies                                  │
│  - Find root customer tweets                               │
│  - Build tweet_lookup index                                │
│            ↓                                                │
│  [ThreadReconstructor]                                     │
│  - Start from each root customer tweet                     │
│  - Traverse forward through response chains                │
│  - Enforce customer↔brand alternation                      │
│  - Select earliest reply for branching decisions           │
│  - Log dropped branches                                    │
│            ↓                                                │
│  [Text Normalization & Classification] (Parallel)          │
│  - TextNormalizer: URLs, mentions, IDs → placeholders      │
│  - LanguageDetector: Identify language, filter if needed   │
│  - ResolutionClassifier: Assign resolution tier            │
│  - BoilerplateDetector: Flag templated replies             │
│            ↓                                                │
│  [Deduplicator]                                            │
│  - Remove exact duplicates (author_id + normalized_text)   │
│  - Remove empty normalized text                            │
│            ↓                                                │
│  [DataExporter]                                            │
│  - Write threads.parquet with schema validation            │
│  - Generate decision_log.md                                │
│            ↓                                                │
│  [Validator]                                               │
│  - Validate row count vs Phase 0 projections               │
│  - Sample and review 20 threads manually                   │
│  - Generate validation_log.txt                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
   threads.parquet    decision_log.md      validation_log.txt
   (cleaned threads)  (decisions/limits)   (validation report)
```

### Component Responsibilities

| Component | Input | Output | Key Logic |
|-----------|-------|--------|-----------|
| **ThreadFinder** | Raw DataFrame | root_ids, tweet_lookup | Identify brand handles; walk brand replies backward to roots |
| **ThreadReconstructor** | root_id, tweet_lookup | thread, dropped_branches | Forward traversal with alternation enforcement; branch selection |
| **ResolutionClassifier** | thread | resolution_tier | Heuristic based on last tweet (author + text content) |
| **BoilerplateDetector** | brand_replies | is_boilerplate mask | Normalize text; count frequencies; apply dual thresholds |
| **TextNormalizer** | text | normalized_text | Regex-based: URLs, mentions, IDs, whitespace; preserve case/emoji |
| **LanguageDetector** | text | language | langdetect library; flag non-English |
| **Deduplicator** | threads_df | cleaned_df | Exact match on author_id + normalized_text; remove empty text |
| **DataExporter** | threads_df, metadata | .parquet, .md | Write schema-validated parquet; generate decision log |
| **Validator** | threads_df, phase0_metrics | validation_report | Compare counts; sample review; statistics |

---

## Data Models

### Input Schema (twcs.csv)

```python
{
    "tweet_id": int,                      # Unique tweet identifier
    "author_id": str,                     # Tweet author identifier
    "inbound": bool,                      # True if customer; False if brand
    "created_at": str,                    # ISO timestamp of tweet creation
    "text": str,                          # Raw tweet text
    "in_response_to_tweet_id": int,      # ID of parent tweet (nullable)
    "response_tweet_id": int,            # ID of reply (nullable, for lookup)
    "timestamp": int,                     # Unix timestamp
}
```

### Intermediate Tweet Object

```python
{
    "tweet_id": int,
    "author_id": str,
    "inbound": bool,
    "text": str,
    "original_text": str,                 # Preserved for audit
    "normalized_text": str,               # Cleaned for dedup/embedding
    "language": str,                      # "en", "es", etc.
    "created_at": str,
    "timestamp": int,
    "in_response_to_tweet_id": int | None,
    "response_tweet_id": int | None,
}
```

### Thread Object

```python
{
    "thread_id": str,                     # UUID or f"{root_id}_{timestamp}"
    "root_tweet_id": int,                 # Initial customer tweet
    "tweets": List[Tweet],                # Ordered list of tweets in thread
    "turn_count": int,                    # Length of thread (≥1)
    "dropped_branches": List[int],        # Tweet IDs of unselected replies
    "dropped_branch_count": int,          # Count of dropped branches
    "resolution_tier": str,               # "resolved_explicit" | "resolved_implicit" | "unresolved_or_ongoing"
    "is_boilerplate": bool,               # True if final brand reply is boilerplate
    "metadata": {
        "root_customer_text": str,        # Normalized text of root tweet
        "final_brand_reply": str,         # Normalized text of last brand reply (if exists)
        "final_author_inbound": bool,     # Is final tweet from customer?
    }
}
```

### Output Schema (threads.parquet)

```python
{
    "thread_id": str,                     # Unique thread identifier
    "customer_text": str,                 # Normalized root customer tweet text
    "brand_reply": str,                   # Text of most recent brand reply (normalized)
    "resolution_tier": str,               # "resolved_explicit" | "resolved_implicit" | "unresolved_or_ongoing"
    "is_boilerplate": bool,               # True if final brand reply is flagged as boilerplate
    "turn_count": int,                    # Number of tweets in thread
    "timestamp": datetime,                # Timestamp of root customer tweet
    "dropped_branch_count": int,          # Number of alternative replies discarded
}
```

**Data Types:** All types must match exactly (no unexpected nulls). ParquetSchema validation enforced at export.

### Decision Log Schema (decision_log.md)

```markdown
# Phase 1 Data Cleaning Decision Log

## Dataset Overview
- Uber Support Handle(s): [list of confirmed handles]
- Threads Found: N
- Threads After Deduplication: M

## Major Processing Steps

### Thread Reconstruction
- Branch Selection Rule: [description]
- Threads with Branching Decisions: N (X%)
- Average Dropped Branches per Thread: Y

### Resolution Classification
- Resolved Explicit: N (X%)
- Resolved Implicit: N (X%) [Note: medium confidence]
- Unresolved/Ongoing: N (X%)

### Boilerplate Detection
- Min Occurrences Threshold: 5
- Min Share Threshold: 0.5%
- Unique Boilerplate Signatures: N
- Total Boilerplate Replies: N
- Boilerplate Rate: X%

### Text Normalization
- Language Filtering: [enabled/disabled]
- Non-English Threads: N
- Deduplication Records Removed: N

## Known Limitations
- Resolved_implicit threads may include unresolved issues where customer stopped replying
- Branch-selection rule discards parallel resolution paths; may lose valid alternative resolutions
- Boilerplate detection relies on text normalization; semantically similar but syntactically different replies may not be flagged
- Language detection may misclassify code snippets, URLs, or mixed-language content
```

---

## Module Design

### 1. thread_finder.py

**Purpose:** Identify Uber Support handles and discover all customer-support conversation roots.

**Primary Function:**
```python
def find_uber_threads(df: pd.DataFrame) -> Tuple[List[int], Dict[int, Tweet]]:
    """
    Discover all Uber Support threads from the dataset.
    
    Args:
        df: Raw tweet DataFrame
    
    Returns:
        root_ids: List of root customer tweet IDs that start threads
        tweet_lookup: Dict mapping tweet_id to Tweet object for O(1) access
    
    Algorithm:
    1. Identify Uber Support handle(s) via value_counts() on author_id
    2. Filter brand_replies = df[df.author_id in UBER_HANDLES and df.inbound == False]
    3. For each brand reply:
       - Look up parent tweet: parent_id = brand_reply.in_response_to_tweet_id
       - If parent_id not found or parent.inbound != True, skip
       - parent is a root customer tweet
    4. De-duplicate root_ids (in case multiple brand replies from same root)
    5. Build tweet_lookup: {tweet_id -> Tweet} for all tweets
    
    Raises:
        ValueError: If no Uber Support handles found or dataset malformed
    """
```

**Implementation Details:**
- Confirm UBER_HANDLES via value_counts() and log
- Handle missing parent tweets gracefully (log and continue)
- Return early if no brand replies found
- Validate tweet_lookup contains all referenced tweets

---

### 2. thread_reconstructor.py

**Purpose:** Reconstruct complete conversation threads forward from root customer tweets.

**Primary Function:**
```python
def build_thread(
    root_id: int,
    tweet_lookup: Dict[int, Tweet],
    max_turns: int = 6,
    enforce_alternation: bool = True
) -> Tuple[Thread, List[int]]:
    """
    Reconstruct a conversation thread forward from a root customer tweet.
    
    Args:
        root_id: Tweet ID of root customer tweet
        tweet_lookup: Dict of all tweets for lookup
        max_turns: Soft limit on thread length (not hard stop)
        enforce_alternation: Enforce customer↔brand alternation
    
    Returns:
        thread: Reconstructed Thread object with all tweets
        dropped_branches: List of tweet IDs that were unselected
    
    Algorithm (Forward Traversal with Alternation):
    1. Start with root_tweet = tweet_lookup[root_id]
    2. Assert root_tweet.inbound == True (is customer)
    3. Initialize thread.tweets = [root_tweet], current_author_inbound = True
    4. WHILE true:
       a. Find all replies to last tweet in thread
       b. If enforce_alternation: filter to replies with opposite inbound value
       c. If no valid replies: BREAK (thread complete)
       d. If multiple valid replies: sort by tweet_id (chronological), select first
       e. If >1 reply available: add unused to dropped_branches list, log "branch_selection_rule"
       f. Append selected reply to thread.tweets
       g. Update current_author_inbound = selected_reply.inbound
       h. If thread length >= max_turns: continue (soft limit, not hard stop)
    5. Return thread, dropped_branches
    
    Alternation Enforcement:
    - Customer tweet → Brand reply → Customer tweet → Brand reply → ...
    - Reject consecutive tweets from same party
    
    Chronological Selection:
    - When multiple replies exist, select earliest by tweet_id
    - Log all unselected replies (dropped_branches)
    
    Raises:
        KeyError: If root_id or referenced tweets not in tweet_lookup
        AssertionError: If root_tweet is not from customer (inbound != True)
    """
```

**Implementation Details:**
- Use tweet.response_tweet_id to find replies (O(1) lookup)
- Build reverse index: {in_response_to_tweet_id → [response_ids]} for efficient reply lookup
- Log every branch decision with reason
- Preserve tweet order (traversal order, not timestamp order)
- Handle no-replies case (single-tweet thread)

**Algorithm Correctness Invariants:**
- Thread alternation always preserved: tweets[i].inbound != tweets[i+1].inbound ∀i
- Chronological ordering: tweets are in traversal order (earliest reply selected)
- All tweets exist: tweet_id references in tweet_lookup
- Root is customer: tweets[0].inbound == True

---

### 3. resolution_classifier.py

**Purpose:** Classify thread resolution outcome into three tiers.

**Primary Function:**
```python
def classify_resolution(thread: Thread) -> str:
    """
    Classify a thread's resolution status.
    
    Args:
        thread: Reconstructed Thread object
    
    Returns:
        resolution_tier: One of "resolved_explicit", "resolved_implicit", "unresolved_or_ongoing"
    
    Heuristic:
    1. Get last_tweet = thread.tweets[-1]
    2. IF last_tweet.inbound == True (customer's last message):
        a. normalized = normalize_text(last_tweet.text)
        b. positive_closers = ["thanks", "thank you", "appreciate", "resolved", 
                               "fixed", "got it", "sorted", "perfect", "great, thanks",
                               "issue fixed", "problem solved"]
        c. IF any closer in normalized: RETURN "resolved_explicit"
        d. ELSE: RETURN "unresolved_or_ongoing"
    3. ELSE (last_tweet.inbound == False, brand's last message):
        a. RETURN "resolved_implicit"
        b. Log metadata: "implicit resolution has medium confidence; silence ≠ satisfaction"
    
    Confidence Levels:
    - resolved_explicit: HIGH (customer explicitly stated resolution)
    - resolved_implicit: MEDIUM (brand replied but customer didn't respond)
    - unresolved_or_ongoing: LOW (ambiguous, customer stopped responding without closure)
    """
```

**Implementation Details:**
- Use regex to find closers (case-insensitive after normalization)
- Handle empty threads edge case (return "unresolved_or_ongoing")
- Log confidence metadata for resolved_implicit

**Positive Closer Phrases (Configurable):**
```python
POSITIVE_CLOSERS = {
    "thanks", "thank you", "appreciate", "appreciated",
    "resolved", "fixed", "fixed it", "works", "working",
    "got it", "sorted", "sorted out", "perfect", "great thanks",
    "issue fixed", "problem solved", "sorted now"
}
```

---

### 4. boilerplate_detector.py

**Purpose:** Identify templated/frequently repeated brand replies.

**Primary Function:**
```python
def detect_boilerplate(
    brand_replies: List[Tweet],
    min_occurrences: int = 5,
    min_share: float = 0.005
) -> pd.Series[bool]:
    """
    Detect boilerplate (templated) brand replies.
    
    Args:
        brand_replies: List of tweets from Uber Support
        min_occurrences: Minimum frequency threshold (default: 5)
        min_share: Minimum percentage of total (default: 0.5%)
    
    Returns:
        is_boilerplate: Boolean Series indexed by tweet_id, True if boilerplate
    
    Algorithm:
    1. Normalize all brand reply texts using normalize_for_dedup()
    2. Count frequency of each unique normalized text: Counter(normalized_texts)
    3. Calculate total replies: N = len(brand_replies)
    4. FOR each unique normalized text:
       a. freq = count[text]
       b. share = freq / N
       c. IS_BOILERPLATE = (freq >= min_occurrences) AND (share >= min_share)
       d. Mark all tweets with this text as boilerplate = IS_BOILERPLATE
    5. Log: unique_signatures, total_boilerplate_count, boilerplate_rate
    
    Dual Threshold Logic:
    - BOTH thresholds must be met (AND operation, not OR)
    - min_occurrences: Absolute frequency filter (reject rare patterns)
    - min_share: Relative percentage filter (reject isolated patterns)
    
    Returns:
        Series with tweet_id as index, boolean value
    """
```

**Normalization for Dedup** (`normalize_for_dedup(text)`):
```python
def normalize_for_dedup(text: str) -> str:
    """
    Normalize text for boilerplate frequency counting.
    Removes: mentions, URLs, numbers (6+ digits)
    Converts to lowercase.
    
    Pipeline:
    1. Remove mentions: @\w+ → ""
    2. Remove URLs: http\S+ → ""
    3. Remove 6+ digit sequences: \d{6,} → ""
    4. Lowercase: text.lower()
    5. Collapse whitespace: \s+ → " "
    6. Strip edges: text.strip()
    
    Example:
    "@user Check http://example.com or call 1-800-123-4567 for help"
    → "check or call 1-800- for help"
    """
```

**Implementation Details:**
- Use collections.Counter for frequency counting
- Both thresholds must be met (enforce AND logic)
- Log all boilerplate signatures for audit trail
- Return None-safe Series (tweets not in brand_replies get False)

---

### 5. text_normalizer.py

**Purpose:** Normalize text for downstream embedding and deduplication.

**Primary Functions:**

```python
def normalize_text(text: str) -> str:
    """
    Full normalization pipeline for primary analysis.
    Preserves case and emoji.
    
    Transformations (in order):
    1. Unescape HTML entities: "&amp;" → "&", "&lt;" → "<", etc.
    2. Replace URLs: http\S+ or https\S+ → "<URL>"
    3. Replace mentions: @\w+ → "<MENTION>"
    4. Replace 6+ digit sequences: \d{6,} → "<ID>"
    5. Collapse whitespace: (\s{2,}) → " "
    6. Strip edges: leading/trailing whitespace
    
    Preserves:
    - Emojis (not removed)
    - Capitalization
    - Punctuation
    - Single spaces
    
    Example:
    "Hey @support! Check http://t.co/xyz or call 1-800-123-4567 for trip #ABC123"
    → "Hey <MENTION>! Check <URL> or call 1-800- for trip #ABC123"
    """

def is_english(text: str) -> bool:
    """
    Detect if text is English using langdetect.
    
    Args:
        text: Tweet text
    
    Returns:
        True if detected language is "en"
    
    Error Handling:
    - If langdetect fails: return True (conservative, include by default)
    - Log failures for audit
    """
```

**Implementation Details:**
- Use `html.unescape()` for entity unescaping
- Use `re.sub()` for all regex replacements
- langdetect exceptions should not crash pipeline (graceful degradation)
- Store both original_text and normalized_text

---

### 6. deduplicator.py

**Purpose:** Remove exact duplicate tweets and invalid records.

**Primary Function:**
```python
def deduplicate(threads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate tweets and empty records.
    
    Args:
        threads_df: DataFrame of tweets with normalized_text
    
    Returns:
        cleaned_df: Deduplicated DataFrame
    
    Algorithm:
    1. Remove tweets with empty normalized_text:
       empty = threads_df[threads_df['normalized_text'].str.strip() == '']
       Log count of empty tweets removed
    2. Identify duplicates:
       duplicates = threads_df.duplicated(subset=['author_id', 'normalized_text'], keep='first')
    3. Remove duplicates:
       cleaned_df = threads_df[~duplicates]
    4. Log count of duplicates removed
    5. Verify: if (original_count - cleaned_count) > 0, confirm removal happened
    
    Deduplication Key:
    - (author_id, normalized_text): Two tweets from same author with same normalized text
    - Keep first occurrence (by DataFrame order)
    - Remove all subsequent occurrences
    
    Edge Cases:
    - Empty normalized_text: Always removed
    - Single tweet per author: Never flagged as duplicate
    - Tweets with same text, different authors: Not duplicates
    """
```

**Implementation Details:**
- Use `pd.duplicated(subset=..., keep='first')`
- Remove empty text before dedup (separate pass)
- Log both empty-text removals and dedup removals
- Return DataFrame with same structure, reduced rows

---

### 7. data_exporter.py

**Purpose:** Export cleaned threads to parquet and generate decision log.

**Primary Functions:**

```python
def export_parquet(threads_list: List[Thread], output_path: str) -> str:
    """
    Export reconstructed threads to parquet file.
    
    Args:
        threads_list: List of reconstructed Thread objects
        output_path: Path to output .parquet file (e.g., "data/processed/threads.parquet")
    
    Returns:
        output_path: Confirmed path (for logging)
    
    Algorithm:
    1. Build DataFrame from threads_list:
       df = pd.DataFrame([
           {
               "thread_id": t.thread_id,
               "customer_text": t.metadata["root_customer_text"],
               "brand_reply": t.metadata["final_brand_reply"] or "",
               "resolution_tier": t.resolution_tier,
               "is_boilerplate": t.is_boilerplate,
               "turn_count": t.turn_count,
               "timestamp": t.tweets[0].timestamp,
               "dropped_branch_count": t.dropped_branch_count,
           }
           for t in threads_list
       ])
    2. Ensure schema matches specification:
       - thread_id: str
       - customer_text: str
       - brand_reply: str
       - resolution_tier: str (one of 3 allowed values)
       - is_boilerplate: bool
       - turn_count: int
       - timestamp: datetime64
       - dropped_branch_count: int
    3. Validate types: df.astype({schema}) or raise TypeError
    4. Write to parquet: df.to_parquet(output_path, index=False, compression='snappy')
    5. Log: path, file_size, row_count, schema
    
    Error Handling:
    - If output directory doesn't exist: attempt creation, fail at write if needed
    - Type validation: raise TypeError if schema doesn't match
    - Write failure: exception propagates (fatal)
    """

def generate_decision_log(
    metadata: Dict,
    output_dir: str = "data/processed"
) -> str:
    """
    Generate decision_log.md documenting all data cleaning decisions.
    
    Args:
        metadata: Dict containing pipeline statistics
        output_dir: Directory for output file
    
    Returns:
        output_path: Path to generated decision_log.md
    
    Sections:
    1. Dataset Overview (handles, thread counts)
    2. Thread Reconstruction (branch selection rule, stats)
    3. Resolution Classification (tier distribution)
    4. Boilerplate Detection (parameters, stats)
    5. Text Normalization (language filtering, dedup stats)
    6. Known Limitations (4 documented limitations)
    7. Metrics Summary (all major counts)
    
    Error Handling:
    - If file write fails: log warning, continue (graceful degradation)
    - Missing metadata keys: use "N/A" placeholder
    """
```

**Implementation Details:**
- Use `pd.DataFrame.to_parquet()` with snappy compression
- Schema validation before write
- Log file metadata (size, row count, schema)
- Markdown formatting for decision log

---

### 8. validator.py

**Purpose:** Validate Phase 1 output quality before proceeding to Phase 2.

**Primary Function:**
```python
def validate_output(
    threads_df: pd.DataFrame,
    phase0_projections: Dict[str, int],
    sample_size: int = 20
) -> ValidationReport:
    """
    Validate Phase 1 output against acceptance criteria.
    
    Args:
        threads_df: Output DataFrame from threads.parquet
        phase0_projections: Expected row count from Phase 0
        sample_size: Threads to sample for manual review (default: 20)
    
    Returns:
        ValidationReport: Pass/fail verdict with diagnostics
    
    Validation Checks:
    1. Row Count Validation:
       - expected = phase0_projections['thread_count']
       - actual = len(threads_df)
       - deviation = |actual - expected| / expected
       - PASS if deviation <= 50%
       - WARNING if 50% < deviation <= 100%
       - FAIL if deviation > 100%
    
    2. Schema Validation:
       - Verify all required columns present
       - Verify all types match specification
       - PASS if schema correct, FAIL if mismatch
    
    3. Sample Review:
       - Sample exactly 20 rows: threads_df.sample(n=20, random_state=42)
       - Print full details: thread_id, customer_text, brand_reply, turn_count, resolution_tier, dropped_branch_count
       - Manual review: PASS/FAIL (user input)
       - Store in validation_log.txt
    
    4. Statistics:
       - dropped_branch_count distribution: 0, 1, 2+
       - resolution_tier distribution: counts and percentages
       - is_boilerplate rate: percentage of True values
       - PASS if all are reasonable (no NaNs, valid ranges)
    
    5. Data Quality Checks:
       - No NULLs in critical columns
       - turn_count >= 1
       - resolution_tier in valid set
       - dropped_branch_count >= 0
    
    Verdict:
    - PASS: All checks pass and manual review confirms thread coherence
    - FAIL: Any check fails; halt and require investigation
    """
```

**ValidationReport Structure:**
```python
{
    "status": "PASS" | "FAIL" | "WARNING",
    "row_count_check": {
        "expected": int,
        "actual": int,
        "deviation_pct": float,
        "result": "PASS" | "FAIL"
    },
    "schema_check": {
        "result": "PASS" | "FAIL",
        "mismatches": List[str]
    },
    "sample_review": {
        "sampled_threads": List[Dict],
        "manual_review_result": "PASS" | "FAIL" | "PENDING",
        "reviewer_comments": str
    },
    "statistics": {
        "dropped_branch_distribution": Dict,
        "resolution_tier_distribution": Dict,
        "boilerplate_rate": float
    },
    "data_quality_issues": List[str]
}
```

**Implementation Details:**
- Use `threads_df.sample(n=20, random_state=42)` for reproducible sampling
- Print thread details in human-readable format
- Store validation log with timestamp and reviewer notes
- Include diagnostics if any check fails

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Thread Alternation Invariant

*For any* reconstructed thread, consecutive tweets must alternate between customer and brand authors. No two consecutive tweets should have the same inbound value.

**Validates: Requirements 2.2**

**Property Statement:**
```
For all threads T:
  For all tweet pairs (T[i], T[i+1]):
    T[i].inbound != T[i+1].inbound
```

**Test Strategy:** Generate random tweet sequences with controlled alternation patterns; verify reconstruction enforces strict alternation or rejects invalid sequences.

---

### Property 2: Chronological Ordering Preservation

*For any* thread with multiple possible reply branches, the selected thread must follow chronological order (earliest reply selected), and all unselected replies are logged as dropped branches.

**Validates: Requirements 2.3, 2.4**

**Property Statement:**
```
For all threads T with branching:
  selected_tweet.tweet_id == min(valid_replies.tweet_id)
  AND unselected_replies ⊆ dropped_branches
```

**Test Strategy:** Generate tweets with multiple chronologically ordered replies; verify earliest is selected and dropped_branches contains all alternatives.

---

### Property 3: Thread Termination and Completeness

*For any* root customer tweet, thread reconstruction must terminate when no valid replies exist and return a complete, properly ordered thread.

**Validates: Requirements 2.5, 2.6**

**Property Statement:**
```
For all threads T:
  T.tweets is a list of valid Tweet objects in traversal order
  AND no further valid replies exist for T.tweets[-1]
  OR T.turn_count >= max_turns (soft limit reached)
```

**Test Strategy:** Generate threads of varying lengths and complexity; verify termination is correct and thread structure is valid.

---

### Property 4: Resolution Tier Determinism

*For any* thread, the resolution tier assignment is deterministic and depends only on the final tweet (author and text). Same thread always produces same resolution tier.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

**Property Statement:**
```
For all threads T:
  last_tweet = T.tweets[-1]
  IF last_tweet.inbound AND contains_positive_closer(last_tweet.text):
    tier == "resolved_explicit"
  ELSE IF NOT last_tweet.inbound:
    tier == "resolved_implicit"
  ELSE:
    tier == "unresolved_or_ongoing"
```

**Test Strategy:** Generate threads with various ending tweets (customer with/without closers, brand reply); verify resolution tier matches heuristic consistently.

---

### Property 5: Boilerplate Detection Dual Thresholds

*For any* collection of brand replies, boilerplate flagging is consistent: a reply is flagged as boilerplate if and only if its normalized text appears ≥5 times AND ≥0.5% of all brand replies (both thresholds must be met).

**Validates: Requirements 4.1, 4.2, 4.3**

**Property Statement:**
```
For all brand_replies B:
  For all unique_text T in B:
    is_boilerplate(T) ↔ (count(T) >= 5 AND count(T)/len(B) >= 0.005)
  AND is_boilerplate is consistent across multiple runs
```

**Test Strategy:** Generate brand replies with controlled frequency distributions; verify boilerplate detection applies dual thresholds correctly (not single threshold).

---

### Property 6: Text Normalization Determinism and Preservation

*For any* tweet text, normalization is deterministic, idempotent, and preserves emojis, capitalization, and punctuation while removing URLs, mentions, and IDs.

**Validates: Requirements 5.1, 5.2**

**Property Statement:**
```
For all tweet_text T:
  normalize(T) == normalize(normalize(T))  [idempotent]
  AND normalize(T1) == normalize(T2) implies T1 and T2 are equivalent
  AND contains_emoji(T) ↔ contains_emoji(normalize(T))
  AND has_case(T) is preserved in normalize(T)
```

**Test Strategy:** Generate random tweet text with emojis, mentions, URLs, IDs, capitalization; verify normalization is idempotent, deterministic, and preserves required elements.

---

### Property 7: Language Detection Consistency

*For any* tweet text, language detection returns the same result on repeated calls and correctly identifies English vs. non-English text.

**Validates: Requirements 5.3, 5.4**

**Property Statement:**
```
For all tweet_text T:
  is_english(T) == is_english(T)  [deterministic]
  AND filter(is_english=True) removes all non-English threads
  AND filter(is_english=False) contains only non-English threads
```

**Test Strategy:** Generate text in English and various other languages; verify language detection is consistent and filtering logic works correctly.

---

### Property 8: Deduplication Preservation and Completeness

*For any* dataset with exact duplicates (same author_id + normalized_text), deduplication removes all duplicates while keeping the first occurrence, and no data is lost except intended duplicates.

**Validates: Requirements 6.1, 6.2, 6.3**

**Property Statement:**
```
For all datasets D:
  deduplicate(D) has no exact duplicates by (author_id, normalized_text)
  AND first_occurrence(D) ⊆ deduplicate(D)
  AND deduplicate(D) ⊆ D
  AND deduplicate(deduplicate(D)) == deduplicate(D)  [idempotent]
```

**Test Strategy:** Generate datasets with known duplicate patterns; verify deduplication removes exactly the right records and is idempotent.

---

### Property 9: No Data Loss During Processing

*For any* thread during the entire pipeline (reconstruction, classification, normalization, deduplication), the count of tweets and thread structures must be trackable, and no data should be lost except intentional duplicates and empty normalized text.

**Validates: Requirements 1.4, 6.4**

**Property Statement:**
```
For all threads T during pipeline:
  count_before_dedup - count_after_dedup 
    == count_duplicates_removed + count_empty_text_removed
  AND all_valid_threads ⊆ output_threads
```

**Test Strategy:** Track row counts at each pipeline stage; verify no unexpected losses occur.

---

### Property 10: Output Schema Conformance

*For any* thread data exported to parquet, the output schema must match the specification exactly with correct data types and no unexpected nulls in critical columns.

**Validates: Requirements 7.1, 7.2**

**Property Statement:**
```
For all rows in threads.parquet:
  typeof(thread_id) == str
  AND typeof(customer_text) == str
  AND typeof(brand_reply) == str
  AND typeof(resolution_tier) ∈ {"resolved_explicit", "resolved_implicit", "unresolved_or_ongoing"}
  AND typeof(is_boilerplate) == bool
  AND typeof(turn_count) == int AND turn_count >= 1
  AND typeof(timestamp) == datetime64
  AND typeof(dropped_branch_count) == int AND dropped_branch_count >= 0
  AND NOT NULL(critical_columns)
```

**Test Strategy:** Export sample threads to parquet; verify schema and types match exactly.

---

## Testing Strategy

### Dual Testing Approach

Phase 1 uses **both property-based testing and integration testing** for comprehensive coverage:

1. **Property-Based Tests (100+ iterations each):**
   - Test universal properties across randomized inputs
   - Verify algorithmic correctness (alternation, ordering, determinism)
   - Catch edge cases through generated data
   - Tags: `# Feature: uber-support-phase1, Property N: [property_text]`

2. **Integration Tests (1-3 examples each):**
   - Test external I/O (CSV loading, parquet export)
   - Verify data discovery (thread finding) against real data patterns
   - Validate logging and reporting
   - Test component interactions end-to-end

3. **Unit Tests (Examples):**
   - Configuration loading and parameter validation
   - Error handling paths (missing directories, malformed data)
   - Specific edge cases and corner cases

### Property-Based Test Mapping

| Property | Suitable for PBT | Implementation Library | Iterations |
|----------|------------------|------------------------|------------|
| Property 1: Alternation Invariant | ✓ YES | hypothesis (Python) | 100+ |
| Property 2: Chronological Ordering | ✓ YES | hypothesis | 100+ |
| Property 3: Termination & Completeness | ✓ YES | hypothesis | 100+ |
| Property 4: Resolution Tier Determinism | ✓ YES | hypothesis | 100+ |
| Property 5: Boilerplate Dual Thresholds | ✓ YES | hypothesis | 100+ |
| Property 6: Text Normalization | ✓ YES | hypothesis | 100+ |
| Property 7: Language Detection | ✓ YES | hypothesis | 100+ |
| Property 8: Deduplication | ✓ YES | hypothesis | 100+ |
| Property 9: No Data Loss | ✓ YES | hypothesis | 100+ |
| Property 10: Schema Conformance | ✓ YES | hypothesis | 100+ |

### Integration Test Mapping

| Requirement | Test Focus | Strategy |
|-------------|-----------|----------|
| 1: Thread Discovery | Find all root customer tweets | Load sample data, verify all expected roots found |
| 3: Resolution Stats | Print distribution | Verify statistics output format |
| 4: Boilerplate Stats | Print detection metrics | Verify metrics logging |
| 7: Parquet Export | Write schema-validated file | Export sample threads, verify parquet is readable |
| 8: Validation | Compare counts, sample review | Load threads.parquet, verify validation checks |
| 9: Decision Log | Generate complete log | Verify log contains all required sections |

### Unit Test Examples

```python
# Configuration Loading
def test_load_config():
    config = load_config("config.yaml")
    assert config["UBER_HANDLES"] is not None
    assert config["min_occurrences"] == 5

# Error Handling: Missing Directory
def test_export_parquet_missing_dir():
    with pytest.raises(FileNotFoundError):
        export_parquet(threads, "/nonexistent/path/threads.parquet")

# Empty Text Removal
def test_dedup_removes_empty_text():
    df = pd.DataFrame({
        "author_id": ["A", "B", "C"],
        "normalized_text": ["hello", "  ", ""]
    })
    result = deduplicate(df)
    assert len(result) == 1

# Positive Closers Detection
def test_positive_closers():
    thread = create_thread_with_last_tweet("Thanks for your help!")
    assert classify_resolution(thread) == "resolved_explicit"
```

### Running Property-Based Tests

**Configuration:**
```python
@given(
    tweets=st.lists(
        st.fixed_dictionaries({
            "tweet_id": st.integers(min_value=1),
            "inbound": st.booleans(),
            "text": st.text(),
        }),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100)  # Minimum 100 iterations
def test_thread_alternation_property(tweets):
    # Feature: uber-support-phase1, Property 1: Thread Alternation Invariant
    # Implementation
    pass
```

**Minimum Iterations:** 100 per property test
**Random Seed:** Fixed seed for reproducibility
**Shrinking:** Enabled (hypothesis/fast-check automatically shrinks failing examples)

---

## Configuration & Parameters

### Main Configuration File (config.yaml)

```yaml
# Uber Support Handles
UBER_HANDLES:
  - "Uber_Support"
  - "Uber_SFO"
  - "Uber_NYC"

# Thread Reconstruction
MAX_TURNS: 6  # Soft limit on thread length
ENFORCE_ALTERNATION: true

# Resolution Classification
POSITIVE_CLOSERS:
  - "thanks"
  - "thank you"
  - "appreciate"
  - "appreciated"
  - "resolved"
  - "fixed"
  - "got it"
  - "sorted"
  - "perfect"
  - "great thanks"
  - "issue fixed"
  - "problem solved"

# Boilerplate Detection
BOILERPLATE_MIN_OCCURRENCES: 5
BOILERPLATE_MIN_SHARE: 0.005  # 0.5%

# Text Normalization
LANGUAGE_FILTERING: true
ACCEPTED_LANGUAGES:
  - "en"

# Input/Output
INPUT_CSV: "data/raw/twcs.csv"
OUTPUT_DIR: "data/processed"
OUTPUT_PARQUET: "data/processed/threads.parquet"
DECISION_LOG: "data/processed/decision_log.md"
VALIDATION_LOG: "data/processed/validation_log.txt"

# Phase 0 Projections (for validation)
PHASE0_PROJECTIONS:
  thread_count: 2500  # Expected ±50%
  boilerplate_rate: 0.15
  branching_rate: 0.22

# Logging
LOG_LEVEL: "INFO"
LOG_FILE: "logs/phase1.log"
```

### Environment Variables

```bash
# Dataset paths
export UBER_SUPPORT_CSV="data/raw/twcs.csv"
export UBER_OUTPUT_DIR="data/processed"

# Optional overrides
export UBER_LOG_LEVEL="DEBUG"
export UBER_MAX_TURNS="8"
```

### Runtime Parameter Override (CLI)

```bash
python run_phase1.py \
  --input data/raw/twcs.csv \
  --output-dir data/processed \
  --max-turns 6 \
  --min-occurrences 5 \
  --min-share 0.005 \
  --language-filtering true \
  --log-level DEBUG
```

---

## Error Handling & Graceful Degradation

### Error Categories

| Error Type | Handling Strategy | Impact |
|-----------|------------------|--------|
| **Missing CSV file** | Log error, raise FileNotFoundError, halt | Fatal |
| **Malformed CSV columns** | Log missing columns, raise ValueError, halt | Fatal |
| **Missing parent tweet in lookup** | Log warning, skip brand reply, continue | Non-fatal (thread discovery affected) |
| **Empty normalized text** | Remove silently, log count in report | Non-fatal (dedup) |
| **Language detection failure** | Log warning, default to True (include), continue | Non-fatal (conservative) |
| **Directory creation failure** | Attempt creation, let write operation fail at parquet export | Fatal at export stage |
| **Report generation failure** | Log warning, complete Phase 1 with graceful degradation | Non-fatal (decision log incomplete) |
| **Duplicate detected in dedup** | Remove second occurrence, log count, continue | Non-fatal (expected) |

### Graceful Degradation Points

1. **Language Filtering Failure:** Default to including text (conservative approach)
2. **Decision Log Generation:** If markdown generation fails, complete Phase 1 with warnings in main log
3. **Validation Report:** If manual review unavailable, flag as "PENDING_REVIEW" instead of failing

### Exception Hierarchy

```python
class Phase1Error(Exception):
    """Base exception for Phase 1 pipeline"""
    pass

class DataLoadError(Phase1Error):
    """Fatal: Cannot load input data"""
    pass

class ThreadReconstructionError(Phase1Error):
    """Fatal: Cannot reconstruct threads (missing data)"""
    pass

class ExportError(Phase1Error):
    """Fatal: Cannot write output files"""
    pass

class ValidationError(Phase1Error):
    """Fatal: Validation checks failed"""
    pass
```

---

## Dependencies & Libraries

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.3.0 | DataFrame manipulation, CSV/parquet I/O |
| pyarrow | >= 5.0.0 | Parquet file format support |
| langdetect | >= 1.0.9 | Language detection |
| pyyaml | >= 5.4 | Configuration file parsing |
| python | >= 3.8 | Core language |

### Development Dependencies (Testing)

| Library | Version | Purpose |
|---------|---------|---------|
| hypothesis | >= 6.0.0 | Property-based testing framework |
| pytest | >= 6.2.0 | Unit test framework |
| pytest-cov | >= 2.12.0 | Code coverage reporting |

### Optional Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| tqdm | >= 4.60.0 | Progress bars (nice to have) |

### Installation

```bash
pip install -r requirements.txt

# For development/testing
pip install -r requirements-dev.txt
```

---

## Validation Strategy

### Pre-Launch Validation

1. **Schema Validation:** Verify input CSV has all required columns
2. **Configuration Validation:** Load config.yaml and verify all required keys present
3. **Directory Validation:** Confirm output directory writeable (test write on startup)

### During Processing

1. **Row Count Tracking:** Log row count at each pipeline stage
2. **Null Check:** Warn if any critical column becomes null
3. **Type Checking:** Verify data types remain consistent

### Post-Processing Validation (Requirement 8)

1. **Row Count Comparison:** Compare output row count to Phase 0 projections (±50% tolerance)
2. **Schema Conformance:** Verify output parquet matches exact schema
3. **Sample Manual Review:** Randomly sample 20 threads, print for human review
4. **Statistics Plausibility:** Check distributions are reasonable (no NaNs, valid ranges)
5. **Validation Report:** Generate validation_log.txt with pass/fail verdict

### Phase 1 Pass Criteria (All must pass)

- ✓ Row count within ±50% of Phase 0 projections
- ✓ Output schema matches specification exactly
- ✓ Manual spot-check confirms thread coherence
- ✓ All statistics are plausible (no invalid values)
- ✓ Decision log generated with all required sections

---

## Implementation Roadmap

### Phase 1a: Core Infrastructure
- [ ] Project structure and module setup
- [ ] Config loading and parameter management
- [ ] Logging configuration
- [ ] Data models (Tweet, Thread classes)

### Phase 1b: Core Pipeline
- [ ] ThreadFinder implementation and tests
- [ ] ThreadReconstructor implementation and tests
- [ ] ResolutionClassifier implementation and tests
- [ ] BoilerplateDetector implementation and tests

### Phase 1c: Data Processing
- [ ] TextNormalizer implementation and tests
- [ ] LanguageDetector implementation and tests
- [ ] Deduplicator implementation and tests

### Phase 1d: Export & Validation
- [ ] DataExporter implementation and tests
- [ ] Validator implementation and tests
- [ ] Integration tests (end-to-end)

### Phase 1e: Polish & Documentation
- [ ] Error handling and edge cases
- [ ] Logging and diagnostics
- [ ] Decision log generation
- [ ] Documentation and runbooks

---

## Summary

Phase 1 architecture implements a robust, validated data pipeline with:

**Strengths:**
- Clear separation of concerns (9 focused modules)
- Deterministic, testable algorithms
- Comprehensive property-based testing (10 properties)
- Graceful error handling and degradation
- Full audit trail (decision log, dropped branches logging)
- Rigorous validation against Phase 0 projections

**Key Design Decisions:**
- Chronological branch selection (earliest reply) provides deterministic reconstruction
- Dual boilerplate thresholds (frequency AND percentage) prevent false positives
- Three-tier resolution classification balances signal with documented uncertainty
- Forward thread traversal ensures chronological coherence
- Dropped branch logging enables future analysis of alternative resolutions

**Quality Assurance:**
- 10 correctness properties validated with 100+ iterations each
- Integration tests for all I/O operations
- Manual validation checkpoint (20-thread sample review)
- Metrics tracking at every pipeline stage
- Graceful degradation for non-fatal errors

---

## Next Steps

1. **Requirements Review:** Confirm design addresses all Phase 1 requirements ✓
2. **Architecture Review:** Validate high-level system design and component interactions
3. **Implementation:** Begin Phase 1a (Core Infrastructure) following roadmap
4. **Testing Setup:** Configure hypothesis-based testing framework
5. **Phase 2 Readiness:** Design Phase 2 embedding/indexing pipeline based on Phase 1 output schema
