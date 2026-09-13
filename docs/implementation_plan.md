# Phase 1 & 2 — Detailed Implementation Plan
### Uber Support Agent (Hiver take-home) — data cleaning, thread reconstruction, chunking/indexing

Build and validate Phase 1 fully before starting Phase 2. Each phase has a
concrete acceptance check at the end — don't proceed past it until it passes.

---

## PHASE 1 — Cleaning + thread reconstruction

### 1.1 Filter to the brand (don't do this with a naive row filter)

A naive `df[df.author_id == 'Uber_Support']` only gets you Uber's own tweets,
not the customer tweets they're replying to. Build threads from the brand
outward instead:

```python
UBER_HANDLES = ["Uber_Support"]  # confirm exact handle(s) via value_counts() first

def find_uber_threads(df):
    tweet_lookup = df.set_index('tweet_id').to_dict('index')
    uber_replies = df[df.author_id.isin(UBER_HANDLES)]

    roots = []
    for _, reply in uber_replies.iterrows():
        parent_id = reply['in_response_to_tweet_id']
        if pd.notna(parent_id) and int(parent_id) in tweet_lookup:
            parent = tweet_lookup[int(parent_id)]
            if parent.get('inbound'):   # confirm it's a genuine customer tweet
                roots.append(int(parent_id))
    return list(set(roots)), tweet_lookup
```

This finds every customer tweet that Uber actually replied to, without
walking the full 3M-row graph.

### 1.2 Reconstruct each thread forward from its root

```python
import re

def get_response_ids(row):
    val = row.get('response_tweet_id')
    if pd.isna(val):
        return []
    return [int(x) for x in str(val).split(',')]

def build_thread(root_id, tweet_lookup, max_turns=6):
    thread = [tweet_lookup[root_id]]
    current = tweet_lookup[root_id]
    dropped_branches = []

    for _ in range(max_turns - 1):
        next_ids = get_response_ids(current)
        if not next_ids:
            break
        candidates = [tweet_lookup[i] for i in next_ids if i in tweet_lookup]
        if not candidates:
            break

        expected_inbound = not current['inbound']  # must alternate
        valid = [c for c in candidates if c['inbound'] == expected_inbound]
        if not valid:
            break

        # branch selection rule: take the chronologically earliest valid
        # continuation; log the rest as dropped for the decision log
        valid.sort(key=lambda c: c['tweet_id'])
        next_tweet = valid[0]
        dropped_branches.extend(valid[1:])

        thread.append(next_tweet)
        current = next_tweet

    return thread, dropped_branches
```

**State this rule explicitly in the decision log**: "when a tweet has
multiple valid replies, we keep the chronologically earliest and discard the
rest — this may drop legitimate parallel resolution paths, accepted for
simplicity." Log a count of how often branching actually occurred so you can
size the impact.

### 1.3 Infer resolution status (three-tier heuristic, not binary)

There's no ground-truth label, so use a tiered heuristic and keep the tier
visible in the output — don't collapse it to a single boolean silently.

```python
POSITIVE_CLOSERS = ['thanks', 'thank you', 'appreciate', 'resolved', 'fixed',
                     'got it', 'sorted', 'perfect', 'great, thanks']

def classify_resolution(thread):
    last = thread[-1]
    last_is_customer = last['inbound']
    last_text = last['text'].lower()

    if last_is_customer and any(p in last_text for p in POSITIVE_CLOSERS):
        return "resolved_explicit"
    if not last_is_customer:
        # thread ends on a brand reply with no customer follow-up
        return "resolved_implicit"   # confidence: medium — silence != satisfaction
    return "unresolved_or_ongoing"
```

`resolved_implicit` is the noisiest tier — flag it as such, it's exactly the
kind of thing to name in your "what's misleading about my headline number"
report section later.

### 1.4 Detect boilerplate replies

Don't reach for embeddings yet (Phase 2 infra doesn't exist yet at this
point in the build) — use cheap string-normalization + frequency counting:

```python
from collections import Counter

def normalize_for_dedup(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def flag_boilerplate(brand_replies, min_occurrences=5, min_share=0.005):
    normalized = brand_replies.apply(normalize_for_dedup)
    counts = Counter(normalized)
    total = len(normalized)
    boilerplate_set = {
        t for t, c in counts.items()
        if c >= min_occurrences or (c / total) >= min_share
    }
    return normalized.isin(boilerplate_set)
```

Tune `min_occurrences`/`min_share` against what Phase 0 profiling actually
showed for reply repetition — don't guess blind.

### 1.5 Normalize text (for embedding/classification use, applied per-tweet)

```python
import html

def normalize_text(text):
    text = html.unescape(text)
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<MENTION>', text)
    text = re.sub(r'\b\d{6,}\b', '<ID>', text)   # order/trip numbers, phone-like sequences
    text = re.sub(r'\s+', ' ', text).strip()
    return text  # keep emojis — real sentiment/urgency signal
```

Apply langdetect and drop (or flag, your call — flagging is more auditable)
non-English threads:

```python
from langdetect import detect, LangDetectException

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
```

### 1.6 Dedup exact duplicates

```python
threads_df = threads_df.drop_duplicates(subset=['author_id', 'normalized_text'], keep='first')
threads_df = threads_df[threads_df['normalized_text'].str.len() > 0]
```

### Phase 1 output schema (`data/processed/threads.parquet`)

```
thread_id, customer_text, brand_reply, resolution_tier
    (resolved_explicit | resolved_implicit | unresolved_or_ongoing),
is_boilerplate (bool), turn_count (int), timestamp,
dropped_branch_count (int)
```

### Phase 1 acceptance check
- Row/thread count is sane relative to Phase 0 profiling numbers.
- Manually spot-check 20 random reconstructed threads for correctness.
- Print `dropped_branch_count` distribution — if branching is rare (<2% of
  threads), the branch-selection rule barely matters; if common, revisit it.
- Print resolution_tier distribution and boilerplate rate — both go straight
  into the report's problem-framing section.

---

## PHASE 2 — Chunking / indexing strategy

Only start this once Phase 1's output passes its acceptance check.

### 2.1 What gets embedded (the actual "chunking" decision)

Not token-window chunking — tweets are short. The decision is: **embed only
`customer_text`, never `brand_reply`.** Embedding the reply mixes issue
similarity with reply-style similarity and biases retrieval toward whichever
phrasing repeats most (i.e., boilerplate).

### 2.2 What goes into the index

Include only:
- `resolution_tier in {resolved_explicit, resolved_implicit}`
- `is_boilerplate == False`

Exclude everything else from the primary retrieval index (keep it in
`threads.parquet` for classifier training — it's just not valid grounding
material).

### 2.3 Build the index

```python
# embedding model: all-MiniLM-L6-v2 (local, free) or text-embedding-3-small
# vector store: Chroma (lightweight, fine at this scale)

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/processed/vector_index")
collection = client.get_or_create_collection("uber_support_issues")

def build_index(threads_df):
    indexable = threads_df[
        (threads_df.resolution_tier.isin(['resolved_explicit', 'resolved_implicit']))
        & (~threads_df.is_boilerplate)
    ]
    embeddings = model.encode(indexable['customer_text'].tolist()).tolist()

    collection.add(
        ids=indexable['thread_id'].astype(str).tolist(),
        embeddings=embeddings,
        documents=indexable['customer_text'].tolist(),
        metadatas=indexable[[
            'brand_reply', 'resolution_tier', 'turn_count'
        ]].to_dict('records')
    )
```

### 2.4 Query the index

```python
def retrieve(query_text, k=5):
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results  # ids, distances, metadatas (brand_reply is the grounding payload)
```

### Phase 2 acceptance check
- Index size matches expected count from Phase 1 output (indexable subset).
- Spot-check 10 manual queries against the index — does the retrieved
  `brand_reply` metadata actually look like a sane resolution for that query?
  (This is a cheap sanity pass before you build the formal `retrieval_eval.py`
  harness later.)

---

## PHASE 3 — brief flag only, not detailed here

Next up after this is building the golden evaluation set (150-250
hand-labeled examples, sampled to cover your intent categories and both
resolution tiers, not just randomly) — we'll spec that out in detail
separately once Phase 1 and 2 are actually running against real data.