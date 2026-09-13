import json, pandas as pd

golden = [json.loads(l) for l in open('data/golden_set_labeled.jsonl')]
golden_ids = {g['thread_id'] for g in golden}  # adjust field name if needed

threads = pd.read_parquet('data/processed/threads.parquet')
eligible = threads[
    (threads.resolution_tier.isin(['resolved_explicit','resolved_implicit']))
    & (~threads.is_boilerplate)
    & (threads.thread_id.isin(golden_ids))
]
print(f"Golden examples eligible for indexing: {len(eligible)}")