"""
features.py  (Pipeline step 3: FEATURE ENGINEERING)
===================================================
Builds fraud-signal features with documented rationale. Works on both the
synthetic set (rich entity fields) and real ULB (Time + V1..V28 + Amount only) --
entity/behavioral features are added only when their source columns exist.

LEAKAGE SAFETY: all entity aggregates (per-card history) use ONLY prior
transactions for that card, ordered by Time. We never use a transaction's own
row (or future rows) to build its historical stats. Categorical encodings are
frequency-based (computed on full data but label-agnostic, so no target leakage).

Each feature's rationale is written to reports/eda/feature_dictionary.json/.md.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, PROCESSED_CSV, REPORTS_DIR, STRUCTURING_THRESHOLD, TARGET

FEATURE_DOC = {}  # name -> rationale


def _add(df, name, series, rationale):
    df[name] = series
    FEATURE_DOC[name] = rationale


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    has_entity = "card_id" in df.columns
    out = pd.DataFrame(index=df.index)

    # ---- Always-available: anonymized signals + amount + time ---------------
    vcols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    for c in vcols:
        _add(out, c, df[c].values,
             "Anonymized PCA-style signal from the card network; carries most of "
             "the separating power in ULB-style data.")
    _add(out, "Amount", df["Amount"].values, "Transaction amount ($). Fraud skews to higher amounts.")
    _add(out, "log_amount", np.log1p(df["Amount"].values),
         "Log-scaled amount; stabilizes the heavy right tail for linear models.")
    if "Time" in df.columns:
        hour = (df["Time"].values % 86400) // 3600
        _add(out, "hour_of_day", hour, "Local hour; fraud over-indexes in off-peak/night hours.")
        _add(out, "is_night", (hour < 6).astype(int), "Night flag (0-6h): elevated fraud window.")

    # near-threshold + round-amount are computable from Amount alone (structuring)
    _add(out, "near_threshold",
         ((df["Amount"] >= 0.9 * STRUCTURING_THRESHOLD) & (df["Amount"] < STRUCTURING_THRESHOLD)).astype(int).values,
         "Amount within 90-100% of the reporting/limit threshold -> classic structuring signal.")
    _add(out, "is_round_amount", (df["Amount"] % 100 == 0).astype(int).values,
         "Round-number amount; weakly associated with scripted/automated fraud.")

    if not has_entity:
        feats = list(out.columns)
        return out.assign(**{TARGET: df[TARGET].values,
                             "Time": df.get("Time", pd.Series(0, index=df.index)).values,
                             "fraud_mode": df.get("fraud_mode", "unknown")}), feats

    # ======================================================================
    # Entity / behavioral / velocity features (synthetic or IEEE-CIS-style)
    # ======================================================================
    # Direct behavioral signals already present in the raw data
    passthrough = {
        "account_age_days": "Days since the sending account was created; mule accounts are brand new.",
        "device_change": "Session device differs from the card's usual device -> account-takeover signal.",
        "corridor_new": "First time this card sends on this corridor -> stolen-instrument / new-geo risk.",
        "n_clicks_session": "Clickstream depth; rushed ATO sessions have fewer, faster clicks.",
        "session_duration_s": "Session length; very short sessions correlate with scripted fraud.",
        "txn_gap_s": "Seconds since this card's previous txn; small gaps = velocity/burst behavior.",
        "card_txn_rank": "Nth transaction for this card in-window; captures account tenure/activity.",
        "txn_count_1h": "Txns by this card in the trailing hour; velocity is a core fraud signal.",
    }
    for c, r in passthrough.items():
        if c in df.columns:
            _add(out, c, df[c].values, r)

    _add(out, "is_new_account", (df["account_age_days"] < 30).astype(int).values,
         "Account younger than 30 days -> mule / bust-out risk.")
    _add(out, "log_txn_gap", np.log1p(df["txn_gap_s"].clip(lower=0)).values,
         "Log of inter-transaction gap; compresses the huge range of idle times.")
    _add(out, "amount_per_click", (df["Amount"] / df["n_clicks_session"].clip(lower=1)).values,
         "Dollars moved per click; high value = low deliberation, typical of ATO cash-out.")

    # ---- Leak-safe per-card historical aggregates (ordered by Time) --------
    d = df[["card_id", "Time", "Amount"]].copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values(["card_id", "Time"]).reset_index()
    g = d.groupby("card_id")
    prior_count = g.cumcount().values                    # excludes current row
    cum_sum = g["Amount"].cumsum().values
    prior_sum = cum_sum - d["Amount"].values
    d["_sq"] = d["Amount"].values ** 2
    cum_sumsq = d.groupby("card_id")["_sq"].cumsum().values
    prior_sumsq = cum_sumsq - d["Amount"].values ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, prior_sum / np.maximum(prior_count, 1), df["Amount"].median())
        prior_var = np.where(prior_count > 1,
                             prior_sumsq / np.maximum(prior_count, 1) - prior_mean ** 2, 0.0)
        prior_std = np.sqrt(np.clip(prior_var, 1e-6, None))
        amt_z = (d["Amount"].values - prior_mean) / prior_std
        amt_ratio = d["Amount"].values / np.maximum(prior_mean, 1.0)

    # trailing-24h count per card via searchsorted
    def _count_24h(grp):
        t = grp["Time"].values
        lo = np.searchsorted(t, t - 86400, side="left")
        return pd.Series(np.arange(len(t)) - lo + 1, index=grp.index)
    d["_c24"] = d.groupby("card_id", group_keys=False).apply(_count_24h, include_groups=False)

    # map back to original row order
    order = d["_ord"].values
    inv = np.argsort(order)
    amt_z = amt_z[inv]; amt_ratio = amt_ratio[inv]
    c24 = d["_c24"].values[inv]
    prior_count_o = prior_count[inv]

    _add(out, "amount_zscore_vs_card", np.clip(amt_z, -20, 20),
         "How anomalous this amount is vs the card's own history (z-score); "
         "amount anomalies flag stolen instruments & ATO cash-outs.")
    _add(out, "amount_to_card_mean", np.clip(amt_ratio, 0, 100),
         "Ratio of amount to the card's historical average spend.")
    _add(out, "txn_count_24h", c24,
         "Txns by this card in the trailing 24h -> sustained velocity / structuring bursts.")
    _add(out, "card_hist_txn_count", prior_count_o,
         "Number of prior transactions seen for this card (entity tenure).")
    _add(out, "device_change_x_new_corridor",
         (df["device_change"].values * df["corridor_new"].values),
         "Interaction: new device AND new corridor together is a strong takeover/stolen signal.")

    # ---- Categorical one-hot (low cardinality) -----------------------------
    for col in ["device_type", "channel", "merchant_category", "corridor"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            for c in dummies.columns:
                _add(out, c, dummies[c].values, f"One-hot indicator for {col} == '{c.split('=')[-1]}'.")

    feats = list(out.columns)
    out[TARGET] = df[TARGET].values
    out["Time"] = df["Time"].values if "Time" in df.columns else 0
    out["fraud_mode"] = df["fraud_mode"].values if "fraud_mode" in df.columns else "unknown"
    return out, feats


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "clean.parquet"))
    feat_df, feats = build(df)
    feat_df.to_parquet(PROCESSED_CSV, index=False)

    with open(os.path.join(REPORTS_DIR, "eda", "feature_dictionary.json"), "w") as f:
        json.dump({"n_features": len(feats), "features": FEATURE_DOC}, f, indent=2)
    # markdown version
    lines = ["# Feature dictionary\n",
             f"{len(feats)} model features. Target = `Class` (1 = fraud).\n",
             "| Feature | Rationale |", "|---|---|"]
    for k in feats:
        lines.append(f"| `{k}` | {FEATURE_DOC.get(k, '')} |")
    with open(os.path.join(REPORTS_DIR, "eda", "feature_dictionary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[features] built {len(feats)} features -> {PROCESSED_CSV}")


if __name__ == "__main__":
    main()
