"""
Milestone 2: Define Intent Taxonomy
Analyze Uber conversations to extract human-defined intents
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import json

# Load Phase 1 threads, filter to Uber only
df = pd.read_parquet('data/processed/threads.parquet')

def detect_brand(text):
    """Detect brand from conversation text"""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['uber', 'driver', 'ride', 'surge', 'pool', 'ubereats']):
        return 'uber'
    return None

df['inferred_brand'] = df['customer_text'].apply(detect_brand)
uber_df = df[df['inferred_brand'] == 'uber'].copy()

print("=" * 80)
print("MILESTONE 2: DEFINE INTENT TAXONOMY")
print("=" * 80)

print(f"\nUber conversations: {len(uber_df):,}")

# --- Sample and manually categorize conversations ---
np.random.seed(42)
sample_size = 200
sample_indices = np.random.choice(len(uber_df), min(sample_size, len(uber_df)), replace=False)
sample = uber_df.iloc[sample_indices].copy()

print(f"Analyzing {len(sample)} sample conversations...")

# Manual intent categorization based on problem types
def infer_intent(customer_text, brand_reply):
    """Infer intent from conversation"""
    if not isinstance(customer_text, str):
        return 'other'
    
    text = customer_text.lower()
    reply = (brand_reply or "").lower()
    
    # Payment/Charging issues
    if any(kw in text for kw in ['charge', 'overcharge', 'overcharged', 'cost', 'price', 'expensive', 'fee', 'billing']):
        if any(kw in text for kw in ['refund', 'money', 'back', 'credit']):
            return 'payment_refund'
        return 'payment_disputed_charge'
    
    # Refunds specifically
    if any(kw in text for kw in ['refund', 'money back', 'reimburse', 'credit me', 'get my money']):
        return 'payment_refund'
    
    # Cancellations
    if any(kw in text for kw in ['cancel', 'cancelled', 'cancellation']):
        return 'ride_cancellation'
    
    # Driver quality/behavior
    if any(kw in text for kw in ['driver', 'rude', 'unprofessional', 'disrespectful', 'bad driver', 'unsafe', 'speeding']):
        if any(kw in text for kw in ['cancel', 'remove', 'ban', 'report']):
            return 'driver_quality_issue'
        return 'driver_quality_issue'
    
    # Lost items
    if any(kw in text for kw in ['lost', 'left', 'forgot', 'missing item', 'left behind']):
        return 'lost_found_item'
    
    # Account/access issues
    if any(kw in text for kw in ['account', 'suspend', 'suspend', 'ban', 'verify', 'login', 'password', 'locked']):
        return 'account_issue'
    
    # Delivery/order issues (UberEats)
    if any(kw in text for kw in ['ubereats', 'delivery', 'order', 'food', 'restaurant']):
        if any(kw in text for kw in ['late', 'delay', 'delayed', 'long', 'took']):
            return 'delivery_timing'
        if any(kw in text for kw in ['wrong', 'missing', 'incomplete', 'damaged', 'cold']):
            return 'delivery_quality'
        return 'delivery_order'
    
    # Ride issues
    if any(kw in text for kw in ['pickup', 'wrong location', 'wrong place', 'didn\'t pick', 'stranded', 'couldn\'t find']):
        return 'ride_pickup_issue'
    
    if any(kw in text for kw in ['dropoff', 'wrong address', 'wrong destination', 'ended at', 'dropped at']):
        return 'ride_dropoff_issue'
    
    # Technical issues
    if any(kw in text for kw in ['app', 'crash', 'bug', 'not working', 'error', 'glitch', 'broken', 'text', 'message']):
        return 'technical_issue'
    
    # Service quality / general complaint
    if any(kw in text for kw in ['service', 'quality', 'bad', 'terrible', 'worst', 'complaint']):
        return 'service_quality'
    
    return 'other'

sample['inferred_intent'] = sample.apply(
    lambda row: infer_intent(row['customer_text'], row['brand_reply']),
    axis=1
)

print("\n" + "=" * 80)
print("INTENT DISTRIBUTION (sample)")
print("=" * 80)

intent_counts = sample['inferred_intent'].value_counts()
for intent, count in intent_counts.items():
    pct = count / len(sample) * 100
    print(f"  {intent:.<35} {count:>3} ({pct:>5.1f}%)")

# --- Show examples per intent ---
print("\n" + "=" * 80)
print("EXAMPLES PER INTENT")
print("=" * 80)

for intent in intent_counts.index:
    intent_sample = sample[sample['inferred_intent'] == intent].head(2)
    print(f"\n--- {intent} ---")
    for idx, row in intent_sample.iterrows():
        print(f"Customer: {row['customer_text'][:120]}")
        print()

# --- Define final taxonomy ---
taxonomy = {
    "payment_disputed_charge": {
        "description": "Customer disputes a charge or claims to be overcharged",
        "examples": [
            "you need to correct false charges",
            "why was I charged twice",
            "this is way too expensive"
        ]
    },
    "payment_refund": {
        "description": "Customer explicitly requests a refund or money back",
        "examples": [
            "I want my money back",
            "refund me please",
            "can I get a credit"
        ]
    },
    "ride_cancellation": {
        "description": "Customer cancelled a ride or has issues with cancellation",
        "examples": [
            "I cancelled but was still charged",
            "why did it charge after cancellation",
            "cancellation fee is unfair"
        ]
    },
    "driver_quality_issue": {
        "description": "Customer complains about driver behavior or quality",
        "examples": [
            "rude driver",
            "driver was unsafe",
            "worst driver ever",
            "unprofessional"
        ]
    },
    "lost_found_item": {
        "description": "Customer lost item in Uber or reporting a found item",
        "examples": [
            "I left my phone in the car",
            "lost my wallet",
            "forgot my keys"
        ]
    },
    "account_issue": {
        "description": "Account access, verification, suspension, or login problems",
        "examples": [
            "account suspended",
            "can't login",
            "need to verify account",
            "account banned"
        ]
    },
    "delivery_quality": {
        "description": "UberEats delivery had quality issues (wrong, missing, cold, damaged)",
        "examples": [
            "order was wrong",
            "missing items",
            "food was cold",
            "damaged delivery"
        ]
    },
    "delivery_timing": {
        "description": "UberEats delivery was late or took too long",
        "examples": [
            "delivery took forever",
            "extremely late",
            "delayed order"
        ]
    },
    "ride_pickup_issue": {
        "description": "Issues with ride pickup (wrong location, couldn't find driver, etc)",
        "examples": [
            "driver couldn't find me",
            "picked up at wrong location",
            "stranded at pickup"
        ]
    },
    "ride_dropoff_issue": {
        "description": "Issues with ride dropoff (wrong destination, wrong address)",
        "examples": [
            "dropped me at wrong address",
            "ended trip at wrong place",
            "wrong destination"
        ]
    },
    "technical_issue": {
        "description": "App crashes, bugs, errors, or technical problems",
        "examples": [
            "app keeps crashing",
            "can't send messages",
            "not receiving texts",
            "bug in app"
        ]
    },
    "service_quality": {
        "description": "General service quality complaint",
        "examples": [
            "terrible service",
            "customer service is useless",
            "worst experience ever"
        ]
    },
    "other": {
        "description": "Other issues not fitting above categories",
        "examples": []
    }
}

print("\n" + "=" * 80)
print("FINAL INTENT TAXONOMY")
print("=" * 80)

for intent_name, intent_info in taxonomy.items():
    print(f"\n{intent_name}")
    print(f"  Description: {intent_info['description']}")
    if intent_info['examples']:
        print(f"  Examples: {', '.join(intent_info['examples'][:2])}")

# Save taxonomy
with open('configs/intents.yaml', 'w') as f:
    import yaml
    yaml.dump(taxonomy, f, default_flow_style=False)

# Save as JSON for reference
with open('data/intent_taxonomy.json', 'w') as f:
    json.dump(taxonomy, f, indent=2)

print("\n" + "=" * 80)
print("LABELING RUBRIC")
print("=" * 80)

rubric = """
# Intent Labeling Rubric for Uber Support Golden Set

## Instructions
For each customer message, assign ONE primary intent from the taxonomy.
If multiple intents are present, choose the PRIMARY problem the customer is reporting.

## Intent Definitions

1. **payment_disputed_charge** - Customer claims the charge was incorrect/too high
2. **payment_refund** - Customer explicitly requests refund or money back
3. **ride_cancellation** - Customer has issues with cancellation (charged after cancel, etc)
4. **driver_quality_issue** - Driver was rude, unsafe, unprofessional, or low quality
5. **lost_found_item** - Customer lost something in an Uber or reporting found item
6. **account_issue** - Account suspended, banned, can't login, needs verification
7. **delivery_quality** - UberEats food was wrong, missing, cold, or damaged
8. **delivery_timing** - UberEats delivery was late or took too long
9. **ride_pickup_issue** - Driver couldn't find pickup location or customer stranded at pickup
10. **ride_dropoff_issue** - Dropped at wrong location or wrong destination
11. **technical_issue** - App crash, bug, can't send messages, not receiving texts
12. **service_quality** - General complaint about customer service quality
13. **other** - Doesn't fit above categories

## Escalation Decision
Mark TRUE if:
- Customer is angry or highly escalated
- Issue involves potential refund (payment_*, lost_*, delivery_quality)
- Issue is account-related (account_issue)
- Issue appears unresolved in the brand reply

Mark FALSE if brand appears to have resolved or offered clear next steps.

## Quality Scoring (1-4)
1 = Brand response is unhelpful or dismissive
2 = Brand response acknowledges but doesn't solve
3 = Brand response offers reasonable path to resolution
4 = Brand response is excellent and likely resolves the issue
"""

with open('configs/labeling_rubric.md', 'w') as f:
    f.write(rubric)

print(rubric)

print("\n" + "=" * 80)
print("✓ Milestone 2 COMPLETE")
print("=" * 80)
print(f"\nTaxonomy: 13 intents (12 specific + 1 other)")
print(f"Saved to: configs/intents.yaml, data/intent_taxonomy.json")
print(f"Rubric: configs/labeling_rubric.md")
print(f"\nReady for Milestone 3: Build Golden Set")
