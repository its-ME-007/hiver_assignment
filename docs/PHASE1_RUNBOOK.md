# Phase 1 Runbook: Uber Support Data Pipeline

## Quick Start

```bash
# Run the complete Phase 1 pipeline
python -m src.phase1.main

# With custom CSV path
python -m src.phase1.main src/phase1/config.yaml data/raw/twcs.csv

# Run tests
python -m pytest tests/ -v --cov=src/phase1

# Generate HTML coverage report
python -m pytest tests/ --cov=src/phase1 --cov-report=html
```

## Prerequisites

- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt requirements-dev.txt`

## Configuration

Configure Phase 1 via `src/phase1/config.yaml`:

```yaml
UBER_HANDLES:                          # Brand handles to identify
  - "Uber_Support"
  - "Uber_SFO"
  - "Uber_NYC"

MAX_TURNS: 6                           # Soft limit on thread length
ENFORCE_ALTERNATION: true             # Customer ↔ Brand alternation

BOILERPLATE_MIN_OCCURRENCES: 5        # Min frequency threshold
BOILERPLATE_MIN_SHARE: 0.005          # Min 0.5% of total

LANGUAGE_FILTERING: true               # Filter non-English
ACCEPTED_LANGUAGES:
  - "en"

INPUT_CSV: "data/raw/twcs.csv"        # Input dataset
OUTPUT_DIR: "data/processed"           # Output directory
OUTPUT_PARQUET: "threads.parquet"      # Output file
LOG_LEVEL: "INFO"                      # Logging verbosity
```

## Data Format

### Input CSV Schema

Required columns:
- `tweet_id` (int): Unique tweet identifier
- `author_id` (str): Tweet author ID (e.g., "@user", "Uber_Support")
- `inbound` (bool): True if customer, False if brand
- `text` (str): Raw tweet text
- `created_at` (str): ISO timestamp
- `timestamp` (int): Unix timestamp
- `in_response_to_tweet_id` (int, nullable): Parent tweet ID

### Output Parquet Schema

- `thread_id` (str): Unique thread identifier
- `customer_text` (str): Normalized root customer tweet
- `brand_reply` (str): Normalized final brand reply
- `resolution_tier` (str): One of:
  - `resolved_explicit`: Customer explicitly confirmed resolution
  - `resolved_implicit`: Brand replied but customer didn't respond (medium confidence)
  - `unresolved_or_ongoing`: Ambiguous or ongoing
- `is_boilerplate` (bool): True if final brand reply is templated
- `turn_count` (int): Number of tweets in thread
- `timestamp` (int): Timestamp of root customer tweet
- `dropped_branch_count` (int): Alternative replies discarded

## Expected Output

### Files Generated

1. **threads.parquet** - Cleaned and reconstructed threads (1000-5000 rows expected)
2. **decision_log.md** - Complete documentation of data cleaning decisions
3. **validation_log.txt** - Validation results and spot-check findings

### Metrics Captured

- Thread discovery: Uber handles found, root threads identified
- Thread reconstruction: Branching decisions, dropped branches
- Resolution classification: Tier distribution
- Boilerplate detection: Signatures found, boilerplate rate
- Deduplication: Empty records and duplicates removed
- Output validation: Schema conformance, row count deviation

## Performance Expectations

- **Input**: 3M raw tweets
- **Output**: 1000-5000 reconstructed threads (±50% from Phase 0 projection)
- **Runtime**: ~5-15 minutes (depending on hardware)
- **Output Size**: ~10-50MB parquet (depending on thread count)

## Troubleshooting

### No threads found

- Verify input CSV has correct column names
- Check that brand handles match UBER_HANDLES in config
- Ensure inbound boolean values are correct (True/False, not 1/0)

### Low thread count vs Phase 0 projection

- Check data quality: nulls, malformed timestamps, missing parents
- Verify thread reconstruction termination logic (alternation enforcement)
- Review decision_log.md for skipped threads

### Boilerplate detection issues

- Verify both thresholds are met (frequency AND share, not OR)
- Check text normalization is working (URLs, mentions removed)
- Review boilerplate signatures in decision_log.md

### Validation failures

- Check schema: all required columns, correct data types
- Verify resolution tier values in valid set
- Ensure turn_count >= 1, dropped_branch_count >= 0
- Manual spot-check sample of 20 threads for coherence

## Next Steps

After Phase 1 completion:

1. Review decision_log.md for known limitations
2. Spot-check sample threads from validation report
3. Validate metrics against Phase 0 projections
4. Proceed to Phase 2: Embedding and indexing

## Support

For issues or questions:
1. Check logs in `logs/phase1.log`
2. Review decision_log.md Known Limitations section
3. Run with LOG_LEVEL=DEBUG for detailed logging
