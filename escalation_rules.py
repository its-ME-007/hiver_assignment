"""
Shared escalation logic for support agent pipeline.
This module centralizes all escalation-related constants and functions
to prevent drift across app.py, escalation_rag_pipeline.py, and evaluation_harness.py

Layered escalation policy
--------------------------
1. HARD TRIGGERS (deterministic, never subject to voting or LLM override):
   explicit safety / fraud / legal / account-compromise language ->
   escalate immediately, full stop. These used to live in the same
   keyword list as soft urgency words and were only counted as ONE of
   three voting signals -- meaning a safety report with high intent
   confidence and strong retrieval similarity could fail to escalate
   under 2-of-3 voting. Splitting them out and short-circuiting on them
   fixes that gap.
2. RULE-BASED VOTE (existing 3-signal, 2-of-3 hard voting) for
   everything else: low intent confidence, weak grounding, and softer
   urgency/anger/repeat-complaint language.
3. LLM JUDGMENT (Gemini), called ONLY for the ambiguous middle -- cases
   where exactly one rule-based signal fired. This is where situational
   context (is the customer describing something happening right now
   vs. a hypothetical, has a standard fix already been ruled out, is
   there a concrete near-term consequence) matters more than keyword
   presence. 'stranded' was removed from the keyword list entirely for
   this reason: the same word meant escalate, ambiguous, and
   not-escalate across three different real messages depending on
   context a keyword match can't see -- that's exactly the LLM layer's
   job, not a keyword's.

Rate limiting
-------------
All Gemini calls in this module go through a shared sliding-60-second-
window limiter so a batch run (e.g. evaluation_harness.py over 165
golden examples) can't blow through the free tier's request cap and
start erroring out mid-run. Configurable two ways:
  - Environment variable GEMINI_MAX_RPM (read once at import time).
  - Setting escalation_rules.GEMINI_MAX_REQUESTS_PER_MINUTE directly
    after import, e.g. for an evaluator with a paid key and a higher
    quota: `escalation_rules.GEMINI_MAX_REQUESTS_PER_MINUTE = 60`.
  - Set to 0 or None to disable rate limiting entirely.
Default is 12 (the free-tier cap at time of writing) -- confirm your
actual quota if it differs, since providers change these periodically.
"""

import os
import time
from collections import deque

# Configurable Gemini rate limit (requests per rolling 60-second window).
# 0 or None disables rate limiting. See module docstring for how to
# override this for a different quota tier.
GEMINI_MAX_REQUESTS_PER_MINUTE = int(os.environ.get('GEMINI_MAX_RPM', '12'))

_gemini_call_timestamps = deque()


def set_gemini_rate_limit(requests_per_minute):
    """
    Explicit setter, for callers who'd rather do
    `escalation_rules.set_gemini_rate_limit(60)` than poke the module
    global directly. Pass 0 or None to disable rate limiting.
    """
    global GEMINI_MAX_REQUESTS_PER_MINUTE
    GEMINI_MAX_REQUESTS_PER_MINUTE = requests_per_minute


def rate_limit_gemini_call():
    """
    Block (sleep) until it's safe to make another Gemini API call,
    given GEMINI_MAX_REQUESTS_PER_MINUTE. Uses a sliding 60-second
    window (not a fixed per-minute bucket that resets on the clock),
    so a burst of calls can't sneak two windows' worth of requests
    through right at a minute boundary.
    """
    limit = GEMINI_MAX_REQUESTS_PER_MINUTE
    if not limit or limit <= 0:
        return  # rate limiting disabled

    now = time.monotonic()
    while _gemini_call_timestamps and now - _gemini_call_timestamps[0] >= 60:
        _gemini_call_timestamps.popleft()

    if len(_gemini_call_timestamps) >= limit:
        sleep_time = 60 - (now - _gemini_call_timestamps[0]) + 0.05
        if sleep_time > 0:
            time.sleep(sleep_time)
        now = time.monotonic()
        while _gemini_call_timestamps and now - _gemini_call_timestamps[0] >= 60:
            _gemini_call_timestamps.popleft()

    _gemini_call_timestamps.append(time.monotonic())

# Signal 3a: HARD TRIGGERS -- deterministic, bypasses voting entirely.
# Safety, fraud, legal, and account-compromise language. If present,
# escalate immediately regardless of intent confidence or retrieval score.
HARD_ESCALATION_KEYWORDS = [
    # Safety
    'unsafe', 'inappropriate', 'harassed', 'harassment', 'assault', 'assaulted',
    'scared', 'followed me', 'stalking', 'threatened', 'threatening me',

    # Fraud / unauthorized access
    'fraud', 'scam', 'unauthorized', 'hacked', 'trying to get into my account',
    'without my permission', 'not me', "wasn't me", 'someone accessed',

    # Legal
    'lawyer', 'attorney', 'legal action', 'sue', 'criminal', 'police',
]

# Signal 3b: SOFT urgency / anger / repeated-complaint language.
# Contributes as ONE vote among three (see compute_escalation_signals) --
# intentionally does not force escalation on its own, since these words
# show up constantly in routine, low-stakes complaints too.
SOFT_ESCALATION_KEYWORDS = [
    'urgent', 'immediately', 'right now', 'asap', 'demand',
    'bbb', 'better business bureau', 'complaint',
    'refund now', 'refund immediately', 'money back now',
    'unacceptable', 'ridiculous', 'disgusting',
    'never using', 'cancel my account', 'switching to', 'deleting account',
    'dispute', 'chargeback', 'report this', 'third time', 'again and again', 'every time',
]


def has_hard_escalation_trigger(text):
    """
    Check for deterministic, always-escalate language (Signal 3a):
    safety, fraud, legal, or account-compromise. If True, the message
    should escalate immediately -- no voting, no LLM judgment needed.
    """
    if not isinstance(text, str):
        return False
    return any(kw in text.lower() for kw in HARD_ESCALATION_KEYWORDS)


def has_explicit_escalation_language(text):
    """
    Check if text contains SOFT explicit-escalation language (Signal 3b
    of the 3-signal voting scheme):
    1. Low intent confidence (< 0.5)
    2. Weak grounding (retrieval_score < 0.7)
    3. Soft urgency/anger/repeat-complaint language (this function)

    Note: hard safety/fraud/legal triggers are handled separately by
    has_hard_escalation_trigger() and are NOT included here -- see the
    module docstring for why they can't just be "one vote among three."

    Args:
        text: Customer message text

    Returns:
        bool: True if soft escalation language detected, False otherwise
    """
    if not isinstance(text, str):
        return False

    text_lower = text.lower()
    return any(kw in text_lower for kw in SOFT_ESCALATION_KEYWORDS)


def compute_escalation_signals(customer_text, intent_conf, retrieval_score):
    """
    Compute 3 escalation signals (canonical implementation).
    
    Args:
        customer_text: Customer message
        intent_conf: Intent classifier confidence (0-1)
        retrieval_score: FAISS retrieval score (0-1)
        
    Returns:
        list: List of (signal_name, triggered) tuples
    """
    signals = []
    
    # Signal 1: Low intent confidence
    signal_1_low_confidence = intent_conf < 0.5
    signals.append(('low_intent_confidence', signal_1_low_confidence))
    
    # Signal 2: Weak grounding (low retrieval score)
    signal_2_weak_grounding = retrieval_score < 0.7
    signals.append(('weak_grounding', signal_2_weak_grounding))
    
    # Signal 3: Explicit escalation language
    signal_3_explicit_language = has_explicit_escalation_language(customer_text)
    signals.append(('explicit_escalation_language', signal_3_explicit_language))
    
    return signals


def make_escalation_decision(signals):
    """
    Hard voting: escalate if 2+ signals are triggered.
    
    Args:
        signals: List of (signal_name, triggered) tuples from compute_escalation_signals()
        
    Returns:
        tuple: (should_escalate, reasons)
            - should_escalate (bool): True if 2+ signals triggered
            - reasons (list): List of triggered signal names
    """
    triggered = [name for name, triggered in signals if triggered]
    should_escalate = len(triggered) >= 2
    
    return should_escalate, triggered


# ---------------------------------------------------------------------------
# LLM judgment layer (Gemini) -- used only for the ambiguous middle ground
# left over after hard triggers and rule-based voting have been applied.
# ---------------------------------------------------------------------------

GEMINI_ESCALATION_PROMPT = """You are deciding whether a customer support message needs a human agent (ESCALATE) or can be safely auto-handled by a bot (AUTO_HANDLE).

Judge based on the CURRENT, CONCRETE situation described -- not just emotional tone or isolated keywords. A message can sound urgent or use dramatic language and still be routine; a calmly-worded message can still need a human if the underlying situation is serious.

Escalate when:
- The customer is describing something happening RIGHT NOW that a bot's generic reply can't meaningfully help with (e.g. actively stranded with a stated blocking cause, an account-access attack in progress).
- The customer has explicitly stated that the standard/expected fix has already failed or doesn't apply, so repeating it would be useless or dismissive.
- There is a concrete, near-term consequence tied to not resolving this now (e.g. losing the customer's business tonight, missing a flight).

Do NOT escalate when:
- The concern is about a hypothetical or future situation, not a current one.
- The message is emotionally intense (angry, sarcastic, dramatic) but the underlying issue is low-stakes and has a standard, factual answer.
- The message is simply vague -- in that case the right bot action is to ask a clarifying question, which counts as AUTO_HANDLE, not escalation.

Examples:
1. "My ride keeps getting denied even though I have money in my account, and I'm stranded right now." -> ESCALATE (current situation, concrete blocking cause)
2. "I am stranded in my college town without a ride." -> AUTO_HANDLE (no stated cause; ask what's happening before deciding)
3. "I couldn't buy the ride pass before it sold out. I'm terrified of being stranded now!!" -> AUTO_HANDLE (hypothetical future concern, real issue is a missed promo)
4. "Someone has been trying to get into my Uber account all morning." -> ESCALATE (account-security event in progress)
5. "Why can't I schedule a ride in advance? Don't show me the tips, it doesn't work in my app." -> AUTO_HANDLE (routine feature issue; standard tip already ruled out, but a different clarifying question still works, no serious stakes)
6. "Your help section has no suggestions and there's no easy way to contact you, if you want my business tonight, help me fix this." -> ESCALATE (self-serve path exhausted + concrete near-term business-loss consequence)

Customer message:
"{customer_text}"

Respond with exactly one word on the first line: ESCALATE or AUTO_HANDLE
Then on the second line, a one-sentence reason.
"""


def gemini_escalation_judgment(customer_text, gemini_model):
    """
    Ask Gemini to break the tie on an AMBIGUOUS escalation case -- one
    where rule-based signals didn't clearly resolve it (see
    decide_escalation below for when this gets called).

    Fails closed to AUTO_HANDLE (with the error captured in the reason
    string) if the API call errors -- this is safe because hard
    safety/fraud/legal triggers are already caught deterministically
    before this layer ever runs; this layer only adjudicates genuinely
    ambiguous, lower-stakes cases.

    Returns:
        tuple: (escalate: bool, reason: str)
    """
    try:
        prompt = GEMINI_ESCALATION_PROMPT.format(customer_text=customer_text)
        rate_limit_gemini_call()
        response = gemini_model.generate_content(
            prompt,
            generation_config={'max_output_tokens': 60},
        )
        lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
        verdict = lines[0].upper() if lines else ''
        reason = lines[1] if len(lines) > 1 else ''
        return verdict.startswith('ESCALATE'), reason
    except Exception as e:
        return False, f"[gemini_judgment_error: {e}]"


# ---------------------------------------------------------------------------
# "gemini_always" mode: instead of only breaking ties on ambiguous cases,
# Gemini sees the heuristic's computed signals + verdict as INPUT context
# and makes the actual final call on every message (still after the
# deterministic hard-trigger check, which is never up for negotiation).
# This is a genuinely different design from the tie-break mode above --
# run both and compare, since the tie-break mode is far cheaper (1 API
# call per ~1-signal case) and may already capture most of the benefit.
# ---------------------------------------------------------------------------

GEMINI_FULL_JUDGMENT_PROMPT = """You are the final decision-maker on whether a customer support message needs a human agent (ESCALATE) or can be auto-handled by a bot (AUTO_HANDLE).

A simple rule-based heuristic has already looked at this message and computed some signals. Treat its signals as useful evidence, not as the final answer -- you have the full context of the message and can override it if the heuristic looks wrong.

Heuristic signals for this message:
{signal_summary}
Heuristic's own recommendation: {heuristic_verdict}

Judge based on the CURRENT, CONCRETE situation described -- not just emotional tone or isolated keywords. A message can sound urgent or use dramatic language and still be routine; a calmly-worded message can still need a human if the underlying situation is serious.

Escalate when:
- The customer is describing something happening RIGHT NOW that a bot's generic reply can't meaningfully help with (e.g. actively stranded with a stated blocking cause, an account-access attack in progress).
- The customer has explicitly stated that the standard/expected fix has already failed or doesn't apply, so repeating it would be useless or dismissive.
- There is a concrete, near-term consequence tied to not resolving this now (e.g. losing the customer's business tonight, missing a flight).

Do NOT escalate when:
- The concern is about a hypothetical or future situation, not a current one.
- The message is emotionally intense (angry, sarcastic, dramatic) but the underlying issue is low-stakes and has a standard, factual answer.
- The message is simply vague -- in that case the right bot action is to ask a clarifying question, which counts as AUTO_HANDLE, not escalation.

Examples:
1. "My ride keeps getting denied even though I have money in my account, and I'm stranded right now." -> ESCALATE (current situation, concrete blocking cause)
2. "I am stranded in my college town without a ride." -> AUTO_HANDLE (no stated cause; ask what's happening before deciding)
3. "I couldn't buy the ride pass before it sold out. I'm terrified of being stranded now!!" -> AUTO_HANDLE (hypothetical future concern, real issue is a missed promo)
4. "Someone has been trying to get into my Uber account all morning." -> ESCALATE (account-security event in progress)
5. "Why can't I schedule a ride in advance? Don't show me the tips, it doesn't work in my app." -> AUTO_HANDLE (routine feature issue; standard tip already ruled out, but a different clarifying question still works, no serious stakes)
6. "Your help section has no suggestions and there's no easy way to contact you, if you want my business tonight, help me fix this." -> ESCALATE (self-serve path exhausted + concrete near-term business-loss consequence)

Customer message:
"{customer_text}"

Respond with exactly one word on the first line: ESCALATE or AUTO_HANDLE
Then on the second line, a one-sentence reason (mention if you're agreeing with or overriding the heuristic).
"""


def gemini_full_judgment(customer_text, signals, heuristic_escalate, gemini_model):
    """
    Ask Gemini to make the FINAL call for this message, given the
    heuristic's signals as context -- not just to break a tie. Called
    for every non-hard-trigger message when mode='gemini_always'.

    Fails closed to the heuristic's own verdict (not blindly to
    AUTO_HANDLE) if the API call errors, so an outage degrades this
    mode back to plain rule-based behavior rather than silently
    suppressing all escalations.

    Returns:
        tuple: (escalate: bool, reason: str)
    """
    signal_summary = "\n".join(f"- {name}: {fired}" for name, fired in signals)
    heuristic_verdict = 'ESCALATE' if heuristic_escalate else 'AUTO_HANDLE'
    try:
        prompt = GEMINI_FULL_JUDGMENT_PROMPT.format(
            customer_text=customer_text,
            signal_summary=signal_summary,
            heuristic_verdict=heuristic_verdict,
        )
        rate_limit_gemini_call()
        response = gemini_model.generate_content(
            prompt,
            generation_config={'max_output_tokens': 60},
        )
        lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
        verdict = lines[0].upper() if lines else ''
        reason = lines[1] if len(lines) > 1 else ''
        if not verdict:
            return heuristic_escalate, "[gemini_returned_empty_response, fell back to heuristic]"
        return verdict.startswith('ESCALATE'), reason
    except Exception as e:
        return heuristic_escalate, f"[gemini_judgment_error, fell back to heuristic: {e}]"


def decide_escalation(customer_text, intent_conf, retrieval_score, gemini_model=None, mode='tiebreak'):
    """
    Full layered escalation policy -- the single entry point every
    caller (app.py, escalation_rag_pipeline.py, evaluation_harness.py)
    should use instead of calling compute_escalation_signals /
    make_escalation_decision directly, so the three layers can't drift
    apart the way the plain keyword list already did once.

    Layer 1 (always, both modes): hard triggers (safety/fraud/legal/
    account-compromise) -> escalate immediately. No voting, no LLM call,
    never overridden by Gemini in either mode.

    After that, two selectable modes for how Gemini participates:

      mode='tiebreak' (default):
        2. Rule-based 3-signal hard vote (2-of-3) on intent confidence,
           retrieval grounding, and soft urgency/anger language.
        3. Gemini is called ONLY if exactly one signal fired (the
           ambiguous zone) to break the tie.
        4. Otherwise, auto-handle.
        Cheapest: one API call per ~1-signal case, not per message.

      mode='gemini_always':
        2. Rule-based signals are still computed, but only as CONTEXT
           handed to Gemini, not as a gate.
        3. Gemini is called for every non-hard-trigger message and
           makes the actual final call, able to agree with or override
           the heuristic's own verdict.
        Most expensive (one API call per message), but tests whether
        giving Gemini full authority catches more than tie-breaking
        alone -- compare its precision/recall against 'tiebreak' mode
        before deciding which one to ship.

    Args:
        customer_text: Customer message
        intent_conf: Intent classifier confidence (0-1)
        retrieval_score: FAISS retrieval score (0-1)
        gemini_model: Optional configured genai.GenerativeModel. If None,
            both modes fall back to rule-based-only behavior (tiebreak
            skips step 3; gemini_always effectively behaves like
            tiebreak-with-zero-calls, i.e. just the heuristic verdict) --
            callers should log when this happens so it's visible in
            reports rather than silently changing outcomes.
        mode: 'tiebreak' (default) or 'gemini_always'.

    Returns:
        tuple: (should_escalate: bool, reasons: list[str])
    """
    if has_hard_escalation_trigger(customer_text):
        return True, ['hard_safety_fraud_legal_trigger']

    signals = compute_escalation_signals(customer_text, intent_conf, retrieval_score)
    heuristic_should_escalate, reasons = make_escalation_decision(signals)

    if mode == 'gemini_always':
        if gemini_model is None:
            return heuristic_should_escalate, reasons + (['gemini_unavailable_used_heuristic'] if heuristic_should_escalate else [])
        gemini_escalate, gemini_reason = gemini_full_judgment(
            customer_text, signals, heuristic_should_escalate, gemini_model
        )
        final_reasons = reasons + [f'gemini_final_judgment: {gemini_reason}']
        return gemini_escalate, final_reasons

    # mode == 'tiebreak' (default)
    if heuristic_should_escalate:
        return True, reasons

    num_fired = len([1 for _, fired in signals if fired])
    if num_fired == 1 and gemini_model is not None:
        gemini_escalate, gemini_reason = gemini_escalation_judgment(customer_text, gemini_model)
        if gemini_escalate:
            return True, reasons + [f'gemini_tiebreak_judgment: {gemini_reason}']

    return False, reasons