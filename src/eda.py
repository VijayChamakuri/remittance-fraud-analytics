"""
eda.py  (Pipeline step 2: EDA)
==============================
Fraud rate, fraud by segment (amount bands, time-of-day, device, corridor,
merchant category), and a clear picture of the imbalance problem. Saves figures
to reports/eda/ and a machine-readable summary reports/eda/eda_summary.json used
by the dashboard.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, REPORTS_DIR, TARGET

EDA = os.path.join(REPORTS_DIR, "eda")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def _seg_rate(df, col):
    g = df.groupby(col)[TARGET].agg(["mean", "sum", "count"])
    g = g.rename(columns={"mean": "fraud_rate", "sum": "n_fraud", "count": "n"})
    return g.sort_values("fraud_rate", ascending=False)


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "clean.parquet"))
    summary = {}

    fr = float(df[TARGET].mean())
    summary["overall_fraud_rate"] = fr
    summary["n_rows"] = int(len(df))
    summary["n_fraud"] = int(df[TARGET].sum())

    # ---- Imbalance figure ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax[0].bar(["Legit", "Fraud"], counts.values, color=["#4C78A8", "#E45756"])
    ax[0].set_title("Class counts (log scale)")
    ax[0].set_yscale("log")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    ax[1].pie([counts.get(0, 0), counts.get(1, 0)], labels=["Legit", "Fraud"],
              autopct=lambda p: f"{p:.2f}%", colors=["#4C78A8", "#E45756"],
              startangle=90, explode=(0, 0.3))
    ax[1].set_title(f"Fraud is {fr*100:.3f}% of volume")
    fig.suptitle("Class imbalance", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "class_imbalance.png")); plt.close(fig)

    # ---- Amount bands -------------------------------------------------------
    bands = [0, 10, 50, 100, 250, 500, 1000, 5000, np.inf]
    labels = ["0-10", "10-50", "50-100", "100-250", "250-500", "500-1k", "1k-5k", "5k+"]
    df["_amt_band"] = pd.cut(df["Amount"], bins=bands, labels=labels)
    amt = df.groupby("_amt_band", observed=True)[TARGET].mean()
    summary["fraud_rate_by_amount_band"] = {str(k): float(v) for k, v in amt.items()}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(amt.index.astype(str), amt.values * 100, color="#E45756")
    ax.axhline(fr * 100, color="#333", ls="--", lw=1, label=f"overall {fr*100:.2f}%")
    ax.set_ylabel("Fraud rate (%)"); ax.set_xlabel("Amount band ($)")
    ax.set_title("Fraud rate by amount band"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "fraud_by_amount.png")); plt.close(fig)

    # ---- Time-of-day --------------------------------------------------------
    if "Time" in df.columns:
        df["_hour"] = (df["Time"] % 86400) // 3600
        hod = df.groupby("_hour")[TARGET].mean()
        summary["fraud_rate_by_hour"] = {int(k): float(v) for k, v in hod.items()}
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(hod.index, hod.values * 100, color="#F58518")
        ax.axhline(fr * 100, color="#333", ls="--", lw=1)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Fraud rate (%)")
        ax.set_title("Fraud rate by hour of day")
        fig.tight_layout(); fig.savefig(os.path.join(EDA, "fraud_by_hour.png")); plt.close(fig)

    # ---- Segment tables (entity fields, if present) -------------------------
    seg_figs = []
    for col, pretty in [("device_type", "Device"), ("corridor", "Corridor"),
                        ("merchant_category", "Merchant category"), ("channel", "Channel")]:
        if col in df.columns:
            t = _seg_rate(df, col)
            summary[f"fraud_rate_by_{col}"] = {
                str(k): {"fraud_rate": float(r["fraud_rate"]), "n": int(r["n"]),
                         "n_fraud": int(r["n_fraud"])}
                for k, r in t.iterrows()
            }
            seg_figs.append((col, pretty, t))

    if seg_figs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (col, pretty, t) in zip(axes.ravel(), seg_figs):
            ax.barh(t.index.astype(str), t["fraud_rate"].values * 100, color="#72B7B2")
            ax.axvline(fr * 100, color="#333", ls="--", lw=1)
            ax.set_title(f"Fraud rate by {pretty}"); ax.set_xlabel("Fraud rate (%)")
            ax.invert_yaxis()
        for ax in axes.ravel()[len(seg_figs):]:
            ax.axis("off")
        fig.suptitle("Fraud concentration by segment", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(EDA, "fraud_by_segment.png")); plt.close(fig)

    # ---- Amount distribution by class --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, color, lab in [(0, "#4C78A8", "Legit"), (1, "#E45756", "Fraud")]:
        vals = np.log1p(df.loc[df[TARGET] == cls, "Amount"])
        ax.hist(vals, bins=60, density=True, alpha=0.55, color=color, label=lab)
    ax.set_xlabel("log(1 + Amount)"); ax.set_ylabel("density")
    ax.set_title("Amount distribution: fraud skews higher"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(EDA, "amount_distribution.png")); plt.close(fig)

    with open(os.path.join(EDA, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eda] overall fraud rate {fr*100:.3f}% | figures + eda_summary.json written")


if __name__ == "__main__":
    main()
