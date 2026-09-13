# Phase 1 Data Cleaning Decision Log

Generated: 2026-09-11 15:48:39

## Dataset Overview

- Uber Support Handle(s): Uber_Support, Uber_SFO, Uber_NYC

- Threads Found: 55182

- Threads After Deduplication: 55143


## Major Processing Steps

### Thread Reconstruction

- Branch Selection Rule: When a tweet has multiple valid replies, the chronologically earliest (lowest tweet_id) is kept; others are discarded to avoid parallel resolution paths

- Threads with Branching Decisions: 4252

- Total Dropped Branches: 6107


### Resolution Classification

- Resolved Explicit (customer confirmed): N/A

- Resolved Implicit (brand last, no customer response): N/A

- Unresolved/Ongoing (ambiguous): N/A

- Note: Implicit resolution has medium confidence; silence does not guarantee satisfaction


### Boilerplate Detection

- Min Occurrences Threshold: 5

- Min Share Threshold: 0.5%

- Unique Boilerplate Signatures: 21

- Total Boilerplate Replies: 11475

- Boilerplate Rate: 16.98%


### Text Normalization & Deduplication

- Language Filtering: False

- Accepted Languages: en

- Threads Removed by Language Filter: 0

- Empty Text Removed: 0

- Duplicates Removed: 39


## Known Limitations

- **Resolved_implicit threads may include unresolved issues**: Where customer simply stopped replying without indicating satisfaction

- **Branch-selection rule discards parallel resolution paths**: May lose valid alternative resolutions that could have been more helpful

- **Boilerplate detection relies on text normalization**: Semantically similar but syntactically different replies may not be flagged as boilerplate

- **Language detection may misclassify code snippets, URLs, or mixed-language content**: Affecting filtering accuracy


## Metrics Summary

- boilerplate_rate: 0.17

- boilerplate_replies: 11475

- boilerplate_signatures: 21

- dedup_duplicates_removed: 39

- dedup_empty_removed: 0

- dropped_branches: 6107

- initial_row_count: 2811774

- language_filtered_threads: 0

- parquet_file_size_mb: 5.48

- parquet_rows: 55143

- root_threads_found: 55182

- threads_after_dedup: 55143

- threads_capped_at_max_turns: 2230

- threads_reconstructed: 55182

- threads_with_branches: 4252

- total_turns: 144290
