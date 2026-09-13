"""
Exact golden-set leakage check using metadata.jsonl.

Run from project root. This is the preferred check over
check_golden_leakage.py's embedding-recompute approach IF metadata.jsonl
actually stores a thread_id per indexed record -- it's deterministic and
doesn't depend on guessing/matching an embedding model at all.

Adjust the field name below if your metadata.jsonl uses a different key
than "thread_id". Note: golden_set_labeled.csv's columns (id, intent_label,
should_escalate, ...) don't show a thread_id column -- if golden_set_labeled
.jsonl also dropped it during labeling, this script will fail loudly on
the missing-key check below rather than silently mismatching. If that
happens, fall back to matching on the pre-label data/golden_set.jsonl
(which does carry thread_id) via its "id" field correspondence to the
labeled set, or re-derive thread_id by joining on customer_message text.
"""

import json

METADATA_PATH = "data/brands/uber/metadata.jsonl"
GOLDEN_PATH = "data/golden_set_labeled.jsonl"  # the FINAL 165-example set used by 09_evaluation_harness.py
THREAD_ID_FIELD = "thread_id"  # change if your metadata.jsonl uses a different key


def load_ids(path: str, field: str) -> set:
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if field not in rec:
                raise KeyError(
                    f"'{field}' not found in {path} record: {list(rec.keys())}. "
                    f"Update THREAD_ID_FIELD to match your actual schema."
                )
            ids.add(rec[field])
    return ids


def main():
    indexed_ids = load_ids(METADATA_PATH, THREAD_ID_FIELD)
    golden_ids = load_ids(GOLDEN_PATH, THREAD_ID_FIELD)

    overlap = golden_ids & indexed_ids

    print(f"Indexed records (metadata.jsonl): {len(indexed_ids)}")
    print(f"Golden examples:                  {len(golden_ids)}")
    print(f"Overlap (leaked thread_ids):       {len(overlap)}")

    if overlap:
        print("\nLEAKED thread_ids:")
        for tid in sorted(overlap):
            print(f"  {tid}")
    else:
        print("\nNo overlap -- golden set is not present in the index.")


if __name__ == "__main__":
    main()