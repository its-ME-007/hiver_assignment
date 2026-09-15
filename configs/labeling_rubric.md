# Intent Labeling Rubric for the Uber Support Golden Set

## Purpose

This rubric defines how messages in the golden set are labeled for intent classification and escalation analysis. The goal is to create a consistent, hand-labeled reference set that can be used for evaluation and comparison across modeling choices.

This document describes the rubric used by the project and should be read together with the current evaluation harness and the checked-in golden-set artifact in `data/golden_set_labeled.jsonl`.

## Primary rule

For each customer message, assign one primary intent only. If multiple issues are present, choose the primary problem the customer is reporting at the time of the message.

This matters because the evaluation pipeline expects a single target label per example for intent classification metrics and baseline comparisons.

## Intent taxonomy

1. **payment_disputed_charge** — customer says a charge is incorrect, unexpected, or too high
2. **payment_refund** — customer explicitly requests a refund, reimbursement, or money back
3. **ride_cancellation** — issue related to a cancellation, cancellation fee, or post-cancel charge
4. **driver_quality_issue** — driver was rude, unsafe, unprofessional, or otherwise poor quality
5. **lost_found_item** — customer reports a lost item or found item related to a ride or trip
6. **account_issue** — account access, suspension, verification, login, profile, or similar account problems
7. **delivery_quality** — UberEats order was wrong, missing, damaged, cold, or otherwise poor quality
8. **delivery_timing** — UberEats order was late or took too long
9. **ride_pickup_issue** — driver could not find pickup, customer was stranded, or pickup was problematic
10. **ride_dropoff_issue** — customer was dropped at the wrong place or destination
11. **technical_issue** — app crash, bug, inability to send messages, failed notifications, or other technical problems
12. **service_quality** — general complaint about customer service quality or support experience
13. **other** — does not fit the above categories or is too ambiguous to confidently assign

## Sampling and review approach

The final evaluated golden set contains 165 examples. These examples were reviewed manually and used as the fixed evaluation set for the repository’s final metric snapshots.

The rubric is intended to support a consistent review process in which each message is judged independently against the taxonomy. A label should be chosen based on the most likely intent, not on the brand response or the presence of a later resolution in the thread.

## Handling ambiguous cases

Ambiguous examples should be handled conservatively:
- choose the most direct primary issue described by the customer,
- do not switch labels based on the brand’s eventual response,
- avoid assigning a broad “other” label when a more specific support category is clear,
- when the issue is genuinely unclear, use the most defensible narrow category rather than inventing a new label.

## Escalation labeling

The escalation label reflects whether the case is likely to require a human handoff rather than the brand’s final resolution status.

A case should be marked as escalated when:
- the customer appears highly escalated or angry,
- the issue involves a refund or monetary dispute,
- the issue concerns account access or account restrictions,
- the matter appears unresolved or risky in the context of the message.

A case should generally not be marked as escalated when:
- the response appears to provide clear next steps,
- the issue appears already resolved by the brand,
- the message is a routine request without signs of escalation risk.

## Distinguishing intent from escalation

Intent and escalation are intentionally separate decisions.
- Intent asks: what is the user mainly reporting?
- Escalation asks: should this case be routed to a human or handled automatically?

This distinction matters because a correctly classified issue can still require human intervention if confidence is low, the issue is sensitive, or the case indicates dissatisfaction or unresolved risk.

## Quality scoring for support-answer quality

The project also includes a 1–4 response-quality rubric for evaluating a brand response:
- 1 = unhelpful or dismissive
- 2 = acknowledges the issue but does not resolve it
- 3 = provides a reasonable path to resolution
- 4 = strong and likely to resolve the issue

This quality rubric is support-oriented and should be interpreted as an evidence-based review of the response, not as a general quality metric for the whole support system.

## Consistency and auditability

To maintain consistency across examples:
- use one primary label per message,
- rely on the taxonomy rather than a narrative impression,
- keep the escalation label distinct from the intent label,
- record ambiguous cases for review rather than silently changing labels.

This approach supports repeatable evaluation and clear comparison between model outputs and the fixed golden set.
