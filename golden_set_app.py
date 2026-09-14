"""
Golden Set Labeling Dashboard
-----------------------------
Streamlit app to review/relabel data/golden_set_labeled.jsonl by hand.

Run with:
    streamlit run label_golden_set_app.py

Place this file anywhere, then either:
  - run it from a folder where 'data/golden_set_labeled.jsonl' exists, or
  - type the correct path in the sidebar "File path" box.

Each save writes the FULL jsonl back to disk immediately, so you can close
the tab any time without losing progress. A 'reviewed' flag is added to
each record so you can filter to "only show what's left".
"""

import json
import os
import streamlit as st

st.set_page_config(page_title="Golden Set Labeler", layout="centered")

DEFAULT_PATH = "data/golden_set_labeled.jsonl"
INTENTS = [
    "payment_refund",
    "payment_disputed_charge",
    "ride_cancellation",
    "driver_quality_issue",
    "lost_found_item",
    "account_issue",
    "delivery_timing",
    "delivery_quality",
    "ride_pickup_issue",
    "ride_dropoff_issue",
    "technical_issue",
    "service_quality",
    "other",
]

# ---------- Load / persist ----------

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    for r in records:
        r.setdefault("reviewed", False)
    return records


def save_records(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------- Sidebar: file + progress ----------

st.sidebar.title("Golden Set Labeler")
path = st.sidebar.text_input("File path", value=DEFAULT_PATH)

if not os.path.exists(path):
    st.sidebar.error(f"File not found: {path}")
    st.stop()

if "records" not in st.session_state or st.session_state.get("loaded_path") != path:
    st.session_state.records = load_records(path)
    st.session_state.loaded_path = path
    st.session_state.idx = 0

records = st.session_state.records
n = len(records)
reviewed_count = sum(1 for r in records if r.get("reviewed"))

st.sidebar.progress(reviewed_count / n if n else 0)
st.sidebar.write(f"{reviewed_count} / {n} reviewed")

if st.sidebar.button("Jump to next unreviewed"):
    for i, r in enumerate(records):
        if not r.get("reviewed"):
            st.session_state.idx = i
            break
    else:
        st.sidebar.success("Everything is reviewed!")

st.sidebar.markdown("---")
jump_id = st.sidebar.number_input(
    "Jump to record #", min_value=1, max_value=n, value=st.session_state.idx + 1
)
if st.sidebar.button("Go"):
    st.session_state.idx = jump_id - 1

# ---------- Main: current record ----------

idx = st.session_state.idx
record = records[idx]

st.title(f"Record {idx + 1} / {n}")
status = "✅ reviewed" if record.get("reviewed") else "◻️ not yet reviewed"
st.caption(f"id: {record.get('id')} · thread_id: {record.get('thread_id')} · {status}")

st.markdown("**Customer message**")
st.info(record.get("customer_message", ""))

st.markdown("**Brand reply**")
st.write(record.get("brand_reply") or "*(no reply recorded)*")

col1, col2 = st.columns(2)
col1.metric("Heuristic-inferred intent", record.get("intent_inferred", "—"))
col2.metric("Difficulty", record.get("difficulty", "—"))

st.markdown("---")
st.subheader("Your label")

current_intent = record.get("intent_label") or record.get("intent_inferred") or "other"
if current_intent not in INTENTS:
    current_intent = "other"

intent_label = st.selectbox(
    "Intent", INTENTS, index=INTENTS.index(current_intent), key=f"intent_{idx}"
)

current_escalate = record.get("should_escalate") == "YES"
should_escalate = st.radio(
    "Should escalate?", ["NO", "YES"],
    index=1 if current_escalate else 0,
    horizontal=True, key=f"escalate_{idx}",
)

escalation_reason = st.text_input(
    "Escalation reason (if YES)",
    value=record.get("escalation_reason") or "",
    key=f"reason_{idx}",
)

current_quality = record.get("reply_quality") or 2
reply_quality = st.slider(
    "Reply quality (1=unhelpful, 4=excellent)", 1, 4,
    value=int(current_quality), key=f"quality_{idx}",
)

notes = st.text_area(
    "Notes (optional)", value=record.get("notes") or "", key=f"notes_{idx}"
)

st.markdown("---")
nav1, nav2, nav3 = st.columns(3)

def apply_and_save(mark_reviewed=True):
    record["intent_label"] = intent_label
    record["should_escalate"] = should_escalate
    record["escalation_reason"] = escalation_reason if should_escalate == "YES" else None
    record["reply_quality"] = reply_quality
    record["notes"] = notes or None
    record["reviewed"] = mark_reviewed
    save_records(path, records)

if nav1.button("⬅ Back", disabled=(idx == 0)):
    apply_and_save(mark_reviewed=record.get("reviewed", False))
    st.session_state.idx = max(0, idx - 1)
    st.rerun()

if nav2.button("💾 Save (stay here)"):
    apply_and_save(mark_reviewed=True)
    st.success("Saved.")

if nav3.button("Save & Next ➡", type="primary", disabled=(idx == n - 1)):
    apply_and_save(mark_reviewed=True)
    st.session_state.idx = min(n - 1, idx + 1)
    st.rerun()

if idx == n - 1 and record.get("reviewed"):
    st.balloons()
    st.success("You've reached the last record and it's marked reviewed. 🎉")