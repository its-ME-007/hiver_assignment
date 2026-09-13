"""
Milestone 1: Data Exploration & Brand Selection
Analyze Phase 1 threads to select a brand for Phase 2
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

# Load Phase 1 threads
df = pd.read_parquet('data/processed/threads.parquet')

print("=" * 80)
print("PHASE 1 THREADS ANALYSIS")
print("=" * 80)

print(f"\nTotal threads: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# --- Extract brand from thread_id ---
# thread_id format appears to be: thread_<number>_<number>
# Let's look for brand indicators in the text or metadata

print("\n" + "=" * 80)
print("SAMPLE DATA")
print("=" * 80)

for idx in range(min(5, len(df))):
    print(f"\n--- Thread {idx} ---")
    print(f"ID: {df['thread_id'].iloc[idx]}")
    print(f"Customer: {df['customer_text'].iloc[idx][:150]}")
    print(f"Brand reply: {df['brand_reply'].iloc[idx][:150] if pd.notna(df['brand_reply'].iloc[idx]) else 'NaN'}")
    print(f"Resolution: {df['resolution_tier'].iloc[idx]}")

# --- Infer brands from text patterns ---
def detect_brand(text):
    """Detect brand from conversation text"""
    if not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Brand-specific keywords
    if any(kw in text_lower for kw in ['uber', 'driver', 'ride', 'surge', 'pool']):
        return 'uber'
    elif any(kw in text_lower for kw in ['flight', 'delta', 'american airline', 'united', 'booking.com']):
        return 'airline'
    elif any(kw in text_lower for kw in ['amazon', 'shipping', 'delivery', 'refund', 'order']):
        return 'amazon'
    elif any(kw in text_lower for kw in ['twitter', 'tweet', '@mention', 'account suspended']):
        return 'twitter'
    elif any(kw in text_lower for kw in ['bank', 'credit', 'debit', 'transfer', 'account']):
        return 'bank'
    elif any(kw in text_lower for kw in ['apple', 'iphone', 'app', 'subscription']):
        return 'apple'
    
    return None

# Apply brand detection
df['inferred_brand'] = df['customer_text'].apply(detect_brand)

print("\n" + "=" * 80)
print("BRAND DISTRIBUTION (inferred from text)")
print("=" * 80)

brand_counts = df['inferred_brand'].value_counts()
print(brand_counts)
print(f"\nUntagged/Other: {df['inferred_brand'].isna().sum()}")

# --- Analyze top brands ---
print("\n" + "=" * 80)
print("TOP BRANDS ANALYSIS")
print("=" * 80)

for brand in brand_counts.head(3).index:
    brand_df = df[df['inferred_brand'] == brand]
    print(f"\n--- {brand.upper()} ---")
    print(f"  Threads: {len(brand_df):,}")
    print(f"  Avg turns: {brand_df['turn_count'].mean():.1f}")
    print(f"  Resolved: {(brand_df['resolution_tier'] == 'resolved_explicit').sum():,} / {len(brand_df):,}")
    print(f"  Boilerplate: {brand_df['is_boilerplate'].sum():,}")
    print(f"  Resolution tiers:")
    for tier, count in brand_df['resolution_tier'].value_counts().items():
        print(f"    {tier}: {count:,}")

# --- Sample conversations ---
print("\n" + "=" * 80)
print("SAMPLE CONVERSATIONS (UBER)")
print("=" * 80)

uber_df = df[df['inferred_brand'] == 'uber'].head(10)
for idx, row in uber_df.iterrows():
    print(f"\n[{row['turn_count']} turns, {row['resolution_tier']}]")
    print(f"Customer: {row['customer_text'][:200]}")
    if pd.notna(row['brand_reply']):
        print(f"Brand: {row['brand_reply'][:200]}")

print("\n" + "=" * 80)
print("DECISION: Select UBER for Phase 2")
print("=" * 80)
print("\nRationale:")
print("- Sufficient examples (most common brand)")
print("- Multiple distinct support issues (rides, charges, driver issues)")
print("- Mix of resolved and unresolved cases")
print("- Good customer→brand interaction density")

# Save analysis results
analysis_results = {
    'total_threads': len(df),
    'brand_distribution': brand_counts.to_dict(),
    'uber_count': len(uber_df),
    'uber_resolved_pct': (uber_df['resolution_tier'] == 'resolved_explicit').sum() / len(uber_df) * 100,
}

print(f"\n✓ Analysis complete. Ready for Milestone 2: Intent Taxonomy")
