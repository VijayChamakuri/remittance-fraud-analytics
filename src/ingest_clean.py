"""
ingest_clean.py  (Pipeline step 1: INGEST + CLEAN)
==================================================
Loads the dataset (real ULB creditcard.csv if present, else the synthetic
stand-in), documents the schema and class imbalance, handles missing values,
and writes a data-quality report.

Design choices (documented):
  * Real IEEE-CIS data has heavy missingness; ULB has none. We implement median
    imputation for numeric columns + a "<col>_was_missing" flag so the same code
    is correct on either dataset. On the synthetic set missingness is ~0, so the
    flags are all-zero and harmless.
  * We never drop the minority (fraud) class rows.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, RAW_CSV, REPORTS_DIR, TARGET


def load_raw() -> tuple[pd.DataFrame, str]:
    real = os.path.join(DATA_DIR, "creditcard.csv")
    if os.path.exists(real):
        print(f"[ingest] using REAL dataset: {real}")
        return pd.read_csv(real), "real_ulb"
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"No dataset found. Run `python src/generate_synthetic_data.py` "
            f"or `python src/download_data.py` first."
        )
    print(f"[ingest] using SYNTHETIC dataset: {RAW_CSV}")
    return pd.read_csv(RAW_CSV), "synthetic"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    report["n_rows"] = int(len(df))
    report["n_cols"] = int(df.shape[1])
    report["columns"] = list(df.columns)

    # class imbalance
    pos = int(df[TARGET].sum())
    report["n_fraud"] = pos
    report["n_legit"] = int(len(df) - pos)
    report["fraud_rate"] = float(pos / len(df))
    report["imbalance_ratio"] = float((len(df) - pos) / max(pos, 1))

    # missingness (real, per column)
    miss = df.isna().sum()
    report["missing_by_column"] = {c: int(v) for c, v in miss.items() if v > 0}
    report["total_missing_cells"] = int(miss.sum())

    # median-impute numeric + missing flags
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c + "_was_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(df[c].median())
    # categorical: fill with "unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("unknown")

    # duplicate rows
    dups = int(df.duplicated().sum())
    report["duplicate_rows"] = dups

    # basic amount / time sanity
    if "Amount" in df.columns:
        report["amount_stats"] = {
            "min": float(df["Amount"].min()), "median": float(df["Amount"].median()),
            "mean": float(df["Amount"].mean()), "max": float(df["Amount"].max()),
        }
    return df, report


def main():
    df, source = load_raw()
    df, report = clean(df)
    report["source"] = source

    out = os.path.join(DATA_DIR, "clean.parquet")
    df.to_parquet(out, index=False)

    with open(os.path.join(REPORTS_DIR, "eda", "data_quality.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ingest] source={source} rows={report['n_rows']:,} "
          f"fraud={report['n_fraud']:,} ({report['fraud_rate']*100:.3f}%) "
          f"imbalance=1:{report['imbalance_ratio']:.0f} "
          f"missing_cells={report['total_missing_cells']:,} dups={report['duplicate_rows']}")
    print(f"[ingest] wrote {out}")


if __name__ == "__main__":
    main()
