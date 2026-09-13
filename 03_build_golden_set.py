"""
Milestone 3: Build Golden Set
Generate 150-250 stratified, hand-labeled golden set examples
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

# Load Phase 1 threads, filter to Uber only
df = pd.read_parquet('data/processed/threads.parquet')

def detect_brand(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['uber', 'driver', 'ride', 'surge', 'pool', 'ubereats']):
        return 'uber'
    return None

df['inferred_brand'] = df['customer_text'].apply(detect_brand)
uber_df = df[df['inferred_brand'] == 'uber'].copy()

print("=" * 80)
print("MILESTONE 3: BUILD GOLDEN SET")
print("=" * 80)

# --- Define intent inference (same as before) ---
def infer_intent(customer_text, brand_reply):
    if not isinstance(customer_text, str):
        return 'other'
    
    text = customer_text.lower()
    
    if any(kw in text for kw in ['charge', 'overcharge', 'cost', 'price', 'fee', 'billing']):
        if any(kw in text for kw in ['refund', 'money', 'back', 'credit']):
            return 'payment_refund'
        return 'payment_disputed_charge'
    
    if any(kw in text for kw in ['refund', 'money back', 'reimburse', 'credit me']):
        return 'payment_refund'
    
    if any(kw in text for kw in ['cancel', 'cancelled', 'cancellation']):
        return 'ride_cancellation'
    
    if any(kw in text for kw in ['driver', 'rude', 'unprofessional', 'disrespectful', 'bad driver', 'unsafe']):
        return 'driver_quality_issue'
    
    if any(kw in text for kw in ['lost', 'left', 'forgot', 'missing item', 'left behind']):
        return 'lost_found_item'
    
    if any(kw in text for kw in ['account', 'suspend', 'ban', 'verify', 'login', 'locked']):
        return 'account_issue'
    
    if any(kw in text for kw in ['ubereats', 'delivery', 'order', 'food']):
        if any(kw in text for kw in ['late', 'delay', 'long', 'took']):
            return 'delivery_timing'
        if any(kw in text for kw in ['wrong', 'missing', 'incomplete', 'damaged', 'cold']):
            return 'delivery_quality'
        return 'delivery_quality'
    
    if any(kw in text for kw in ['pickup', 'wrong location', 'wrong place', 'stranded']):
        return 'ride_pickup_issue'
    
    if any(kw in text for kw in ['dropoff', 'wrong address', 'wrong destination', 'ended at']):
        return 'ride_dropoff_issue'
    
    if any(kw in text for kw in ['app', 'crash', 'bug', 'not working', 'error', 'glitch', 'text', 'message']):
        return 'technical_issue'
    
    if any(kw in text for kw in ['service', 'quality', 'bad', 'terrible', 'worst']):
        return 'service_quality'
    
    return 'other'

uber_df['inferred_intent'] = uber_df.apply(
    lambda row: infer_intent(row['customer_text'], row['brand_reply']),
    axis=1
)

print(f"\nUber threads: {len(uber_df):,}")
print(f"Intent distribution:")
for intent, count in uber_df['inferred_intent'].value_counts().items():
    print(f"  {intent}: {count:,}")

# --- Stratified sampling ---
print("\n" + "=" * 80)
print("STRATIFIED SAMPLING")
print("=" * 80)

TARGET_SIZE = 200  # Can be 150-250, using 200 as middle ground

# Stratification factors
# 1. By intent (balanced across intents)
# 2. By resolution_tier
# 3. By difficulty (proxy: turn_count)
# 4. By boilerplate

def estimate_difficulty(row):
    """Estimate difficulty: complex = longer convo + unclear resolution"""
    if row['turn_count'] >= 4 and row['resolution_tier'] == 'unresolved_or_ongoing':
        return 'hard'
    elif row['turn_count'] <= 2 and row['resolution_tier'] == 'resolved_explicit':
        return 'easy'
    else:
        return 'medium'

uber_df['difficulty'] = uber_df.apply(estimate_difficulty, axis=1)

print(f"\nDifficulty distribution:")
print(uber_df['difficulty'].value_counts())

print(f"\nResolution distribution:")
print(uber_df['resolution_tier'].value_counts())

# Sample stratified by: intent, resolution_tier, difficulty, boilerplate
np.random.seed(42)

stratified_samples = []
intent_counts = defaultdict(int)
difficulty_counts = defaultdict(int)

intents = uber_df['inferred_intent'].unique()
n_intents = len(intents)

# Target: ~15-20 examples per intent
samples_per_intent = TARGET_SIZE // n_intents

print(f"\nTarget: {TARGET_SIZE} total samples, ~{samples_per_intent} per intent")
print("\nSampling...")

for intent in sorted(intents):
    intent_df = uber_df[uber_df['inferred_intent'] == intent]
    
    # Within each intent, stratify by resolution + difficulty
    # Aim for 50% easy, 50% hard
    easy_samples = intent_df[intent_df['difficulty'].isin(['easy', 'medium'])].sample(
        n=min(samples_per_intent // 2, len(intent_df[intent_df['difficulty'].isin(['easy', 'medium'])])),
        random_state=42
    )
    hard_samples = intent_df[intent_df['difficulty'] == 'hard'].sample(
        n=min(samples_per_intent // 2, len(intent_df[intent_df['difficulty'] == 'hard'])),
        random_state=42
    )
    
    samples = pd.concat([easy_samples, hard_samples])
    stratified_samples.append(samples)
    
    print(f"  {intent}: {len(samples)} samples")

golden_df = pd.concat(stratified_samples, ignore_index=True)

print(f"\nTotal sampled: {len(golden_df)} examples")
print(f"Coverage by intent:")
for intent, count in golden_df['inferred_intent'].value_counts().items():
    print(f"  {intent}: {count}")

# --- Create labeling template ---
print("\n" + "=" * 80)
print("CREATE LABELING TEMPLATE")
print("=" * 80)

# Reset indices for clean IDs
golden_df = golden_df.reset_index(drop=True)

# Create template for manual labeling
golden_records = []

for idx, row in golden_df.iterrows():
    record = {
        'id': f'gold_{idx+1:04d}',
        'thread_id': row['thread_id'],
        'customer_message': row['customer_text'],
        'brand_reply': row['brand_reply'],
        'intent_inferred': row['inferred_intent'],
        'intent_label': None,  # TO BE FILLED BY HUMAN
        'should_escalate': None,  # TO BE FILLED BY HUMAN
        'escalation_reason': None,  # TO BE FILLED BY HUMAN
        'reply_quality': None,  # 1-4, TO BE FILLED BY HUMAN
        'difficulty': row['difficulty'],
        'resolution_tier': row['resolution_tier'],
        'turn_count': int(row['turn_count']),
        'notes': None,  # TO BE FILLED BY HUMAN
    }
    golden_records.append(record)

# Save as JSONL
print(f"\nExporting {len(golden_records)} records to data/golden_set.jsonl...")

os.makedirs('data', exist_ok=True)

with open('data/golden_set.jsonl', 'w') as f:
    for record in golden_records:
        f.write(json.dumps(record) + '\n')

# Also save as CSV for easier spreadsheet editing
golden_df_export = pd.DataFrame([
    {
        'id': r['id'],
        'customer_message': r['customer_message'][:100],
        'intent_inferred': r['intent_inferred'],
        'intent_label': '',
        'should_escalate': '',
        'escalation_reason': '',
        'reply_quality': '',
        'difficulty': r['difficulty'],
        'notes': ''
    }
    for r in golden_records
])

golden_df_export.to_csv('data/golden_set_labeling_template.csv', index=False)

print(f"✓ Exported to:")
print(f"  - data/golden_set.jsonl (full records)")
print(f"  - data/golden_set_labeling_template.csv (for spreadsheet labeling)")

# --- Show instructions ---
print("\n" + "=" * 80)
print("LABELING INSTRUCTIONS")
print("=" * 80)

instructions = """
## Golden Set Labeling Instructions

You have 200 Uber support conversations to label. Each should take 1-2 minutes.

### For each record, fill in:

1. **intent_label** (required)
   - Must be one of the 13 intents from configs/labeling_rubric.md
   - Use the intent definitions to guide your choice
   - If unsure, pick the PRIMARY problem

2. **should_escalate** (required)
   - YES if: customer angry, needs refund, account issue, or brand reply unclear
   - NO if: brand provided clear next steps, issue resolved, or helpful guidance

3. **escalation_reason** (if escalate=YES)
   - Brief reason why (e.g., "Needs refund", "Account suspended", "Angry customer")

4. **reply_quality** (required)
   - 1 = Unhelpful/dismissive
   - 2 = Acknowledges but doesn't solve
   - 3 = Offers reasonable path to resolution
   - 4 = Excellent response, likely resolves issue

5. **notes** (optional)
   - Any observations or ambiguities

### Time estimate
- 200 records × 1.5 min = ~5 hours for a single person
- Best: 2 people labeling in parallel, then spot-check 10% agreement

### Files
- Input: data/golden_set_labeling_template.csv (easy to edit)
- Full data: data/golden_set.jsonl (use if programmatic access needed)
- Reference: configs/labeling_rubric.md

### After labeling
1. Export labeled CSV back to data/golden_set_labeled.csv
2. Run 03_validate_golden_set.py to check for missing fields
3. Convert to JSONL for downstream pipelines
"""

print(instructions)

print("\n" + "=" * 80)
print("✓ Milestone 3 CHECKPOINT READY")
print("=" * 80)
print(f"""
Golden set created: {len(golden_records)} examples
Stratified by: intent, difficulty, resolution_tier

Next steps:
1. HUMAN LABELING (required, ~5 hours)
   - Open data/golden_set_labeling_template.csv in Excel/Google Sheets
   - Fill in intent_label, should_escalate, escalation_reason, reply_quality, notes
   - Save as data/golden_set_labeled.csv

2. VALIDATION (automatic, ~5 min)
   - Run: python 03_validate_golden_set.py
   - Checks for missing fields, consistency, coverage

3. CONVERSION (automatic, ~1 min)
   - Exports to JSONL for evaluation pipeline

Roadmap:
- After labeling → Milestone 4: Establish Baselines
- Baselines → Milestone 5: Build FAISS index
- FAISS → Milestone 6: Intent classifier
- etc.
""")
