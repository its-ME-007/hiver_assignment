import json

# Show first 5 records from golden set
with open('data/golden_set.jsonl', 'r') as f:
    records = [json.loads(line) for line in f.readlines()[:5]]

for rec in records:
    print('='*80)
    print(f"ID: {rec['id']}")
    print(f"Intent (inferred): {rec['intent_inferred']}")
    print(f"Difficulty: {rec['difficulty']}")
    print(f"Turn count: {rec['turn_count']}")
    print(f"\nCustomer: {rec['customer_message'][:150]}")
    print(f"\nBrand reply: {rec['brand_reply'][:150] if rec['brand_reply'] else 'None'}")
    print()
