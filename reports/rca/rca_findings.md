# Root Cause Analysis - what drives the fraud the model sees (and misses)

_Operating threshold: 0.504. All figures from the out-of-time test set._

## 1. Top fraud drivers (permutation importance)

The features whose removal most degrades PR-AUC:

- **txn_count_1h** (importance 0.2110)
- **device_change** (importance 0.1206)
- **V12** (importance 0.1025)
- **V3** (importance 0.0901)
- **V4** (importance 0.0889)
- **txn_gap_s** (importance 0.0817)
- **V14** (importance 0.0520)
- **corridor_new** (importance 0.0506)

> Plain English: the model leans hardest on the anonymized network signals plus the behavioral flags we engineered - amount-anomaly-vs-card, velocity in the last hour, new-device/new-corridor, and account age. That is exactly the signature of stolen instruments, account takeover, and mule cash-out.

## 2. Where fraud slips through (recall by fraud mode)

| Fraud mode (remittance analog) | Fraud txns | Missed | Recall |
|---|---:|---:|---:|
| mule | 20 | 12 | 40.0% |
| stolen_instrument | 28 | 12 | 57.1% |
| structuring | 16 | 4 | 75.0% |
| account_takeover | 25 | 3 | 88.0% |

**Hardest segment: `mule`.** This is where investigators should focus. Below is the average profile of the *missed* cases in this segment:

- Amount: 50.36
- amount_zscore_vs_card: 2.41
- txn_count_1h: 1.17
- device_change: 0.00
- corridor_new: 0.08
- near_threshold: 0.00
- account_age_days: 393.75

> Plain English: the misses in this segment look the most like normal behavior on the engineered signals (smaller amount anomaly / lower velocity), so they hide in the noise. Recommended fixes: a targeted rule for this segment, additional data (e.g., richer device/clickstream signals), and a lower review threshold for transactions matching this profile.

## 3. Recommended actions for Product & Compliance

- Prioritize the hardest fraud mode above with a dedicated detection rule or step-up check, since the ML model alone under-catches it.
- Feed the top drivers into analyst review UIs so investigators see *why* a case was flagged.
- Collect more signal where the model is blind (the missed-case profile points to which fields are thin).
