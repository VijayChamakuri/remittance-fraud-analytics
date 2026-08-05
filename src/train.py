"""
train.py  (Pipeline step 4: MODEL)
==================================
Trains two models on an OUT-OF-TIME split:
  * Logistic Regression (scaled, class_weight='balanced')  -- interpretable baseline
  * HistGradientBoosting  (gradient-boosted trees, balanced sample weights) -- main model

IMBALANCE HANDLING (explained): fraud is ~0.17% of rows, so plain training would
let a model reach 99.8% accuracy by predicting "never fraud". We (a) train on a
proper time split, (b) reweight the minority class ('balanced' -> weight inversely
proportional to class frequency) so both models optimize for catching fraud, and
(c) evaluate with PR-AUC / recall (not accuracy). We prefer class-weighting over
SMOTE/oversampling because it adds no synthetic rows, is leakage-free across the
time split, and keeps calibration closer to the true base rate.

Split: sort by Time; train = first 60%, valid = next 10% (threshold tuning),
test = last 30% (untouched out-of-time holdout).
"""
from __future__ import annotations
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import (MODELS_DIR, PROCESSED_CSV, REPORTS_DIR, SEED, TARGET,
                    TRAIN_FRAC, VALID_FRAC)


def time_split(df):
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * TRAIN_FRAC), int(n * VALID_FRAC)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def main():
    df = pd.read_parquet(PROCESSED_CSV)
    feats = [c for c in df.columns if c not in (TARGET, "Time", "fraud_mode")]
    train, valid, test = time_split(df)
    print(f"[train] split sizes  train={len(train):,}  valid={len(valid):,}  test(OOT)={len(test):,}")
    print(f"[train] fraud in train={int(train[TARGET].sum())}  valid={int(valid[TARGET].sum())}  test={int(test[TARGET].sum())}")

    Xtr, ytr = train[feats], train[TARGET].values
    Xva, yva = valid[feats], valid[TARGET].values
    Xte, yte = test[feats], test[TARGET].values

    sw = compute_sample_weight(class_weight="balanced", y=ytr)

    # ---- Logistic Regression baseline --------------------------------------
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, n_jobs=None)),
    ])
    logreg.fit(Xtr, ytr)

    # ---- Gradient-boosted trees (main) -------------------------------------
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=350, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=SEED,
    )
    hgb.fit(Xtr, ytr, sample_weight=sw)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": logreg, "features": feats}, os.path.join(MODELS_DIR, "logreg.joblib"))
    joblib.dump({"model": hgb, "features": feats}, os.path.join(MODELS_DIR, "hgb.joblib"))

    # ---- Save predictions for downstream steps -----------------------------
    def preds(m):
        return {
            "valid": m.predict_proba(Xva)[:, 1],
            "test": m.predict_proba(Xte)[:, 1],
        }
    P = {"logreg": preds(logreg), "hgb": preds(hgb)}

    pred_out = {
        "features": feats,
        "valid": {"y": yva.tolist(), "Time": valid["Time"].tolist(),
                  "fraud_mode": valid["fraud_mode"].tolist(),
                  "Amount": valid["Amount"].tolist() if "Amount" in valid.columns else None,
                  "logreg": P["logreg"]["valid"].tolist(), "hgb": P["hgb"]["valid"].tolist()},
        "test": {"y": yte.tolist(), "Time": test["Time"].tolist(),
                 "fraud_mode": test["fraud_mode"].tolist(),
                 "Amount": test["Amount"].tolist() if "Amount" in test.columns else None,
                 "logreg": P["logreg"]["test"].tolist(), "hgb": P["hgb"]["test"].tolist()},
    }
    # attach the raw feature matrices for RCA/SHAP on the test set
    test[feats + [TARGET, "fraud_mode"]].to_parquet(os.path.join(MODELS_DIR, "test_matrix.parquet"), index=False)

    with open(os.path.join(MODELS_DIR, "predictions.json"), "w") as f:
        json.dump(pred_out, f)

    meta = {"n_train": len(train), "n_valid": len(valid), "n_test": len(test),
            "n_features": len(feats), "seed": SEED,
            "fraud_train": int(train[TARGET].sum()), "fraud_test": int(test[TARGET].sum())}
    with open(os.path.join(REPORTS_DIR, "evaluation", "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] models + predictions saved ({len(feats)} features)")


if __name__ == "__main__":
    main()
