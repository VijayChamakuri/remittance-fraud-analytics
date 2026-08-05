# Production Monitoring Note (Pipeline step 8)

How I would monitor this fraud model once it is scoring live remittance traffic.
Companion SQL: [`sql/04_model_monitoring.sql`](../sql/04_model_monitoring.sql).

## 1. Data & feature drift
- **Score distribution (PSI):** compute Population Stability Index of the daily
  model-score distribution vs. the training baseline. Alert when PSI > 0.2
  (material shift), warn at > 0.1.
- **Feature drift:** track the mean/percentiles of the top drivers (amount
  z-score, velocity, new-device/new-corridor rates, account age). A sudden jump
  in `txn_count_1h` or `corridor_new` usually means either a real attack wave or
  an upstream logging change — both need eyes.
- **Missingness:** watch null rates per feature; a feature going silently null
  (broken join) quietly degrades the model.

## 2. Performance over time (the metrics the model is judged on)
- **Precision / recall / PR-AUC by day and by segment,** computed once labels
  mature (chargeback / confirmed-fraud lag). Because labels arrive late, report
  on a trailing matured window and annotate the label-maturity cutoff.
- **Alert bands:** page if daily recall drops below its 4-week rolling mean minus
  2 sigma, or if precision falls below the level that keeps review-queue volume
  within ops capacity.
- **Volume guardrails:** flagged-rate and false-positive rate — a spike means
  customer friction and review-team overload even if fraud catch looks fine.

## 3. Business / customer guardrails
- Step-up challenge rate and legitimate-customer abandonment (from the A/B
  framework) — fraud prevention must not quietly tax good customers.
- Dollars of fraud prevented vs. dollars of friction, reported weekly to Product
  and Compliance.

## 4. Operational hygiene
- **Retraining trigger:** scheduled monthly retrain, plus event-driven retrain if
  PSI > 0.2 or recall breaches its alert band for 3 consecutive days.
- **Challenger models & shadow scoring:** run the next candidate in shadow and
  compare PR-AUC before promotion.
- **Reproducibility:** every scoring run logs model version, feature version, and
  the code commit, so any decision can be reconstructed for audit/Compliance.
- **Feedback loop:** confirmed-fraud and analyst dispositions flow back into the
  training set, and RCA on new false negatives feeds the rule/feature backlog.
