# Executive Summary — Remittance Fraud Detection Model
**Audience: Product & Compliance · One page · Plain English**

> *Scope note: built on public card-fraud data mapped to a remittance context to
> demonstrate the approach. Not real Remitly data. All figures below are measured
> from the model on a held-out, out-of-time test set (nothing is estimated).*

## What we built
A fraud-scoring model for cross-border money transfers, plus the analytics around
it: what fraud looks like, where the model is strong or weak, and a tested idea
for reducing fraud. Fraud is rare — about **1 in 580 transactions (0.17%)** — so
the whole system is tuned to find needles in a haystack, not to be "accurate."

## What the model catches
- On transactions it has never seen (a **future time period**, the fair test), the
  model ranks fraud extremely well: **ROC-AUC 0.97**. In practice that means if you
  review the highest-risk transactions, you find far more fraud per review than
  random checking.
- The strongest warning signs it learned are intuitive and map to real playbooks:
  **a burst of transactions in the last hour** (velocity), a **new device**, a
  **first-time destination corridor**, and **amounts that are unusual for that
  customer**. These are the fingerprints of account takeover, stolen cards, and
  layering/structuring.

## The tradeoff you have to choose (precision vs. recall)
There is no free lunch. Catch more fraud → more good customers get flagged.
We measured three settings so the choice is explicit:

| Setting | Fraud caught (recall) | Of the flags, how many are real fraud (precision) | Best when |
|---|---:|---:|---|
| **Aggressive** (chosen default) | **65%** | 11% | a missed fraud costs more than the friction of a false flag |
| **Balanced** | 33% | **57%** | review-team capacity is the bottleneck |
| **Wide net** | 87% | 2% | only as a backstop behind other checks |

**Recommendation:** start at the **Balanced** point if analyst review capacity is
limited (most flags are real, less customer friction), and move toward
**Aggressive** for high-risk corridors or segments where missed fraud is costliest.
This is a business dial, not a technical one — Product and Compliance should own it.

## Where fraud still slips through (root-cause finding)
The model is strong on account-takeover (**catches 88%**) but weak on **mule
accounts (catches only 40%)**. Reason: mule activity looks a lot like normal
brand-new-customer behavior, so it hides in the noise. **Action:** add a targeted
rule and/or extra verification for new accounts moving money to cash-pickup and
bank-deposit destinations, and collect richer onboarding signals there — the model
alone will keep under-catching this segment.

## The intervention we tested (A/B)
We simulated adding a **step-up verification** (one-time passcode / ID re-check) on
medium-high-risk transfers. It reduced fraud that gets through from **3.7% to 2.0%
— a 47% drop** — with only ~5% legitimate-customer friction. **But the test as run
was too small to be statistically conclusive (p = 0.06).** Honest read: promising
effect, not yet proven. **Recommendation: run it longer to reach the required
sample size before rolling out.** We deliberately do **not** ship on a large-but-
unproven number.

## Recommended next steps
1. **Pick the operating point** (Balanced vs. Aggressive) with Compliance and Ops.
2. **Extend the step-up A/B** to full power (~800 per group), then decide on rollout.
3. **Close the mule gap** with a new-account rule + better onboarding data.
4. **Stand up monitoring** (drift, weekly precision/recall, friction guardrails)
   before and after launch, with clear alert thresholds.
5. Feed confirmed fraud back into monthly retraining so the model keeps up with
   changing tactics.

*Supporting detail: `reports/evaluation/metrics.json`, `reports/rca/rca_findings.md`,
`reports/ab_test/ab_results.md`, and the dashboard at `dashboard/index.html`.*
