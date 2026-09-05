# Data & Honesty

This project demonstrates fraud-detection **methodology** on **public card-fraud
data**, mapped onto a **cross-border remittance** context. This page is the full,
unabridged provenance note - kept out of the README so results lead, but never
hidden. The honesty here is a feature, not a disclaimer.

## What the data is (and is not)

* **It is card-fraud data, not remittance data, and it is not real Remitly data.**
  Nothing in this repo uses or implies access to any Remitly system.
* The pipeline is built for two real public datasets:
  * **ULB Credit Card Fraud** (`mlg-ulb/creditcardfraud`) - 284,807 transactions,
    492 fraud (0.172%), columns `Time, V1..V28, Amount, Class`.
  * **IEEE-CIS Fraud Detection** - richer entity/identity fields.
* **This repo was built in an offline sandbox** (no Kaggle/GitHub network), so the
  committed results were produced on a **deterministic synthetic dataset** that
  reproduces:
  * the ULB schema (`Time, V1..V28, Amount, Class`),
  * the real class imbalance (**0.172%** fraud, 1:580),
  * plus **IEEE-CIS-style entity/behavioral fields** the anonymized ULB set lacks
    (`card_id, device_type, channel, corridor, merchant_category,
    account_age_days, device_change, corridor_new, n_clicks_session,
    session_duration_s`) so velocity, entity-aggregation, and behavioral features
    can be demonstrated.

## Why the metrics are honest

* **No number in this repo is hand-typed.** Every metric is computed by the model
  on the data and written to JSON by the pipeline (`reports/**/**.json`).
* The synthetic signal strengths are fixed for reproducibility, but they were
  iteratively adjusted after an early version produced a near-perfect ROC-AUC.
  Effects were weakened and overlap and label noise were added to create a more
  useful demonstration benchmark. The resulting **ROC-AUC ≈ 0.97 / PR-AUC
  ≈ 0.41** describes this synthetic design only. It is not an estimate of
  performance on remittance transactions or the real ULB dataset.
* Evaluation uses a **strict out-of-time split** (train on the earliest 60% of the
  timeline, tune on the next 10%, test on the final 30%), never random shuffling,
  so there is no temporal leakage. Entity aggregates use **only prior
  transactions** per card.
* The A/B "true effect" is a **stated simulation assumption**; the statistical
  machinery around it (power analysis, two-proportion z-test, effect-size gating)
  is real, and the reported result honestly comes back **underpowered and not
  significant** rather than being massaged into a win.

## Remittance mapping

The four synthetic fraud modes map to real cross-border remittance risks:

| Fraud mode | Remittance risk it represents |
|---|---|
| `stolen_instrument` | stolen card/instrument used in a new corridor |
| `account_takeover` | ATO: device change + velocity spike on an aged account |
| `mule` | mule account: brand-new account cashing out |
| `structuring` | many just-under-threshold sends to evade limits |

On the real **IEEE-CIS** data, the equivalent real fields (`card1..6`,
`DeviceType`, `id_*`, email domains, address/geo) would carry these signals; the
synthetic fields are named to make that mapping obvious.

## Reproduce on the REAL dataset

```bash
pip install kagglehub kaggle
# put your kaggle.json in ~/.kaggle/  (https://www.kaggle.com/docs/api)
python src/download_data.py          # pulls mlg-ulb/creditcardfraud -> data/creditcard.csv
python run_pipeline.py --real        # identical pipeline, real data
```

The pipeline auto-detects `data/creditcard.csv` and uses the real
`Time + V1..V28 + Amount` schema. Entity/behavioral features are simply skipped
when their source columns are absent (verified: the real-schema path runs
end-to-end and produces 34 features instead of 69).

## Known limitations

* Absolute metric values will differ on the real ULB/IEEE data - the **code is
  identical**, only the data changes.
* SHAP is optional; if it is not installed the RCA falls back to permutation
  importance (what the committed run used). Install `shap` for per-transaction
  explanations.
* The synthetic entity fields make the behavioral-feature demonstration possible
  but are, by construction, cleaner than messy production logs. Treat the feature
  *methodology* as the transferable artifact, not the exact importances.
