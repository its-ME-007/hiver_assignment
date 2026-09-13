"""
Semi-automatic labeling of golden set
Uses heuristics to pre-fill labels, then human spot-checks/corrects
This dramatically speeds up the labeling process
"""

import json
import pandas as pd
import re

# Load golden set
with open('data/golden_set.jsonl', 'r') as f:
    records = [json.loads(line) for line in f.readlines()]

print("=" * 80)
print("SEMI-AUTOMATIC GOLDEN SET LABELING")
print("=" * 80)
print(f"\nPre-labeling {len(records)} records using heuristics...\n")

# Define heuristics
def auto_label_intent(record):
    """Pre-fill intent based on inferred + customer text"""
    # Start with inferred intent
    return record['intent_inferred']

def auto_label_escalation(record):
    """Determine if should escalate"""
    text = (record['customer_message'] or "").lower()
    brand_reply = (record['brand_reply'] or "").lower()
    
    # Escalation signals
    escalate = False
    reasons = []
    
    # Signal 1: Customer is angry/frustrated
    angry_words = ['worst', 'terrible', 'useless', 'ridiculous', 'scam', 'disaster', 
                   'angry', 'furious', 'outraged', 'ugh', 'wtf', 'fuck', 'hate', 'stupid']
    if any(word in text for word in angry_words):
        escalate = True
        reasons.append("Angry/frustrated customer")
    
    # Signal 2: Financial issue (payment, refund, lost money)
    if any(kw in text for kw in ['refund', 'money back', 'charge', 'billing', 'overcharge', 
                                    'lost', 'missed payment', 'dispute']):
        escalate = True
        reasons.append("Financial issue")
    
    # Signal 3: Account suspended/banned
    if any(kw in text for kw in ['suspend', 'ban', 'disabled', 'locked']):
        escalate = True
        reasons.append("Account issue")
    
    # Signal 4: Brand reply is generic/unhelpful (just "contact us")
    if brand_reply and len(brand_reply.split()) < 20:
        if any(kw in brand_reply for kw in ['send us', 'dm us', 'contact us', 'here to help']):
            # Generic response without actual help
            if not any(kw in brand_reply for kw in ['found', 'located', 'refund', 'credit']):
                escalate = True
                reasons.append("Brand reply is generic/unhelpful")
    
    # Signal 5: Unresolved in metadata
    if record['resolution_tier'] == 'unresolved_or_ongoing':
        escalate = True
        reasons.append("Unresolved in metadata")
    
    return escalate, "; ".join(reasons) if reasons else None

def auto_label_quality(record):
    """Estimate reply quality based on brand_reply"""
    brand_reply = (record['brand_reply'] or "").lower()
    
    if not brand_reply:
        return 1  # No reply = bad
    
    # Quality indicators
    if any(kw in brand_reply for kw in ['found', 'located', 'refund', 'credit', 'resolved', 'fixed']):
        return 4  # Solved
    
    elif any(kw in brand_reply for kw in ['apolog', 'sorry', 'assist', 'look into', 'investigate']):
        return 3  # Offers to help
    
    elif any(kw in brand_reply for kw in ['contact us', 'send us', 'dm', 'here to help']):
        return 2  # Generic acknowledgment + redirect
    
    else:
        return 1  # Unclear or dismissive

# Apply auto-labeling
for record in records:
    record['intent_label'] = auto_label_intent(record)
    escalate, reason = auto_label_escalation(record)
    record['should_escalate'] = "YES" if escalate else "NO"
    record['escalation_reason'] = reason
    record['reply_quality'] = auto_label_quality(record)

# Statistics
print("Auto-labeled results:")
print(f"  Escalate=YES: {sum(1 for r in records if r['should_escalate'] == 'YES')} ({sum(1 for r in records if r['should_escalate'] == 'YES')/len(records)*100:.1f}%)")
print(f"  Escalate=NO: {sum(1 for r in records if r['should_escalate'] == 'NO')} ({sum(1 for r in records if r['should_escalate'] == 'NO')/len(records)*100:.1f}%)")

print(f"\nQuality distribution:")
for q in range(1, 5):
    count = sum(1 for r in records if r['reply_quality'] == q)
    print(f"  Quality={q}: {count} ({count/len(records)*100:.1f}%)")

print(f"\nIntent distribution:")
for intent in sorted(set(r['intent_label'] for r in records)):
    count = sum(1 for r in records if r['intent_label'] == intent)
    print(f"  {intent}: {count}")

# Save labeled set
with open('data/golden_set_labeled.jsonl', 'w') as f:
    for record in records:
        f.write(json.dumps(record) + '\n')

print(f"\n✓ Saved to: data/golden_set_labeled.jsonl")

# Export as CSV for easy review/correction
df_export = pd.DataFrame([{
    'id': r['id'],
    'intent_label': r['intent_label'],
    'should_escalate': r['should_escalate'],
    'escalation_reason': r['escalation_reason'],
    'reply_quality': r['reply_quality'],
    'difficulty': r['difficulty'],
    'customer_message': r['customer_message'][:80],
    'brand_reply': (r['brand_reply'] or '')[:80],
} for r in records])

df_export.to_csv('data/golden_set_labeled.csv', index=False)
print(f"✓ Saved to: data/golden_set_labeled.csv (for review in spreadsheet)")

# Show some examples
print("\n" + "=" * 80)
print("SAMPLE PRE-LABELED RECORDS")
print("=" * 80)

for i in range(min(5, len(records))):
    r = records[i]
    print(f"\n{r['id']} | Intent: {r['intent_label']} | Escalate: {r['should_escalate']} | Quality: {r['reply_quality']}")
    if r['should_escalate'] == 'YES':
        print(f"  Reason: {r['escalation_reason']}")
    print(f"  Customer: {r['customer_message'][:100]}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
The golden set has been PRE-LABELED using heuristics.

Now you have two options:

OPTION A: Fast validation (5-10 min)
1. Open data/golden_set_labeled.csv
2. Spot-check 20-30 random records
3. If heuristics are good (>90% agreement), approve and move on
4. If heuristics are off, adjust the heuristics and re-run

OPTION B: Full manual review (1-2 hours)
1. Open data/golden_set_labeled.csv
2. Review ALL records
3. Correct any misclassifications
4. Add notes for ambiguous cases

Recommendation: Option A if heuristics look good, then proceed to Milestone 4
""")

print("\n✓ Milestone 3 COMPLETE (with auto-labeling)")
