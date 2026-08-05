# Cross-Border Remittance Fraud Analytics

An end-to-end fraud-detection analytics project: ingest → clean → EDA → feature
engineering → modeling → evaluation → root-cause analysis → A/B testing →
monitoring → dashboard. Built to mirror the workflow of a **Fraud Data Analyst**
(SQL + Python + experimentation + stakeholder translation).

---

## ⚠️ Read this first — honest framing (data provenance)

This project demonstrates fraud-detection **methodology** on **public card-fraud
data**, mapped onto a **cross-border remittance** context. A few things stated
plainly:

* **This is card-fraud data, not remittance data, and it is not real Remitly
  data.** Nothing here uses or implies access to any Remitly system.
* The pipeline is designed for the real **ULB Credit Card Fraud** dataset
  (`mlg-ulb/creditcardfraud`, 284,807 transactions, 0.172% fraud) and the
  **IEEE-CIS Fraud Detection** dataset.
* **This repo was built in an offline sandbox** with no Kaggle/GitHub access, so
  the committed results were produced on a **deterministic synthetic dataset**
  that reproduces the real ULB schema (`Time, V1..V28, Amount, Class`), the real
  0.172% class imbalance, and adds IEEE-CIS-style **entity/behavioral fields**
  (card, device, corridor, account age, clickstream) that the anonymized ULB set
  lacks — so velocity, entity-aggregation, and behavioral features can be shown.
* **No metric in this repo is hand-typed.** Every number comes from the model
  running on the data. The synthetic signal was set from plausible priors and
  **not tuned to hit a target score** — the model lands at an honest
  **ROC-AUC ≈ 0.97 / PR-AUC ≈ 0.41**, not a fake 0.99.
* **To reproduce on the REAL dataset:** `python src/download_data.py`
  (needs a Kaggle token) then `python run_pipeline.py --real`. The entire
  downstream pipeline is column-compatible and runs unchanged.

The four synthetic fraud "modes" map to real remittance risks:

| Fraud mode | Remittance risk |
|---|---|
| `stolen_instrument` | stolen card/instrument used in a new corridor |
| `account_takeover` | ATO: device change + velocity spike on an aged account |
| `mule` | mule account: brand-new account cashing out |
| `structuring` | many just-under-threshold sends to evade limits |

---

## Quick start (clone-and-run)

```bash
pip install -r requirements.txt      # or: make install
python run_pipeline.py               # or: make run   (generates data + runs everything)
open dashboard/index.html            # static dashboard, no server needed
```

One command (`python run_pipeline.py`) regenerates the deterministic dataset
(seed=42) and runs every stage. Total runtime ≈ 25s on a laptop.

Optional interactive dashboard: `streamlit run dashboard/app.py`.

### Run on the real ULB dataset instead
```bash
pip install kagglehub kaggle
# put your kaggle.json in ~/.kaggle/  (see https://www.kaggle.com/docs/api)
python src/download_data.py
python run_pipeline.py --real
```

---

## Repository structure

```
remittance-fraud-analytics/
├── README.md                     ← you are here
├── requirements.txt  Makefile  run_pipeline.py
├── src/
│   ├── generate_synthetic_data.py  ← deterministic ULB/IEEE-style data
│   ├── download_data.py            ← real Kaggle download path
│   ├── ingest_clean.py             ← step 1: ingest + clean + schema/imbalance
│   ├── eda.py                      ← step 2: EDA (rate, segments, imbalance)
│   ├── features.py                 ← step 3: velocity/entity/behavioral features
│   ├── train.py                    ← step 4: LogReg + GBT, imbalance handling
│   ├── evaluate.py                 ← step 5: PR/ROC/F1, confusion, OOT holdout
│   ├── rca.py                      ← step 6: RCA + permutation/SHAP importance
│   ├── ab_test.py                  ← step 7: step-up A/B (power + z-test)
│   └── build_dashboard.py          ← step 9: static HTML dashboard
├── sql/                          ← BigQuery-style versions of key steps
│   ├── 01_fraud_rate_and_imbalance.sql
│   ├── 02_fraud_by_segment.sql
│   ├── 03_velocity_and_entity_features.sql
│   └── 04_model_monitoring.sql
├── dashboard/  index.html (static)  app.py (Streamlit)
├── reports/
│   ├── eda/           figures + data_quality.json + feature_dictionary.md
│   ├── evaluation/    metrics.json + ROC/PR/confusion/threshold PNGs
│   ├── rca/           rca_findings.md + importance/FN figures
│   ├── ab_test/       ab_results.md + figure
│   ├── monitoring_note.md
│   └── executive_summary.md        ← one-pager for Product/Compliance
├── data/sample_1000.csv          ← small preview (full data is regenerated)
└── models/                       ← trained models + predictions (regenerated)
```

---

## Results summary (out-of-time holdout)

Trained on the first 60% of the time axis, threshold tuned on the next 10%,
evaluated on the **untouched final 30% (85,443 txns, 89 fraud)**.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression (baseline) | 0.965 | 0.322 |
| **Gradient-Boosted Trees (main)** | **0.970** | **0.408** |

Because fraud is 0.17% of volume, **PR-AUC and recall matter far more than
accuracy** (a "never fraud" model is 99.83% accurate and useless). The
precision/recall tradeoff at three operating points on the GBT model:

| Operating point | Threshold | Precision | Recall | F1 | What it's for |
|---|---:|---:|---:|---:|---|
| Cost-minimizing (chosen) | 0.50 | 11% | **65%** | 0.19 | catch the most fraud when a miss costs more than friction |
| F1-optimal | 0.88 | **57%** | 33% | 0.41 | balance catches vs. review-queue load |
| High-recall | 0.13 | 2% | 87% | 0.04 | last-line net; too many false positives to run alone |

**Imbalance handling:** class weighting (`class_weight='balanced'`) on both models
— chosen over SMOTE/oversampling because it adds no synthetic rows, is leakage-free
across the time split, and keeps scores closer to the true base rate.

**Root cause (RCA):** the top fraud drivers are **1-hour velocity
(`txn_count_1h`)**, **device change**, several anonymized signals (V12/V3/V4),
inter-transaction gap, and **new-corridor** — the exact signature of ATO, stolen
instruments, and structuring. Detection recall is highest for account-takeover
(88%) and lowest for **mule accounts (40%)**, because mule behavior overlaps with
legitimate new-account activity. → RCA writeup: [`reports/rca/rca_findings.md`](reports/rca/rca_findings.md).

**A/B test (simulated):** a **step-up verification** on medium-high-risk
transactions cut the completed-fraud rate from **3.7% → 2.0% (−47% relative)**,
but the test as sized was **underpowered** (658 vs. 781 needed per arm) so the
result was **not significant (p=0.056)**. Decision: **do not ship yet — extend
the test to reach power.** This is deliberate: we gate on *both* effect size and
significance, not a single near-miss p-value.
→ A/B writeup: [`reports/ab_test/ab_results.md`](reports/ab_test/ab_results.md).

**Monitoring:** drift (PSI), precision/recall over time on matured labels, and
customer-friction guardrails — see [`reports/monitoring_note.md`](reports/monitoring_note.md)
and [`sql/04_model_monitoring.sql`](sql/04_model_monitoring.sql).

**Executive summary for Product/Compliance:**
[`reports/executive_summary.md`](reports/executive_summary.md).

---

## Tools used (mapped to the role)

Python (pandas, scikit-learn, scipy, matplotlib), **SQL / BigQuery-style** models
of the key steps in `sql/`, statistical experimentation (power analysis +
two-proportion z-test), and a lightweight dashboard (static HTML + Streamlit).

## Limitations & honesty notes
- Results are on synthetic-but-realistic data; absolute numbers will differ on
  the real ULB/IEEE data (the code is identical — just run `--real`).
- SHAP is optional; if not installed the RCA falls back to permutation importance
  (used here). Install `shap` for per-transaction explanations.
- The A/B "true effect" is a stated simulation assumption; a live test would
  measure it. The statistical machinery (power, z-test, gating) is real.

MIT licensed.
