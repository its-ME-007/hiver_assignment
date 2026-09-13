
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
