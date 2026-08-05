"""
evaluate.py  (Pipeline step 5: EVALUATION)
==========================================
Reports precision, recall, F1, ROC-AUC, PR-AUC, a confusion matrix, and proper
OUT-OF-TIME holdout results. Sweeps the decision threshold, discusses the
precision/recall tradeoff, and picks an operating point by minimizing an
explicit business cost (false-positive friction vs. dollars of fraud missed).

All numbers here are computed from real model predictions on the untouched
out-of-time test set -- nothing is hand-entered.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from config import (COST_FALSE_NEGATIVE_MULT, COST_FALSE_POSITIVE, MODELS_DIR,
                    REPORTS_DIR)

EV = os.path.join(REPORTS_DIR, "evaluation")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(threshold=float(thr), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(prec), recall=float(rec), f1=float(f1))


def choose_threshold_by_cost(y, p, amount):
    """Pick threshold minimizing expected $ cost on the validation set."""
    amount = np.asarray(amount, float) if amount is not None else np.ones_like(p)
    grid = np.unique(np.quantile(p, np.linspace(0.90, 0.99999, 400)))
    best, best_cost = 0.5, np.inf
    for thr in grid:
        yhat = (p >= thr).astype(int)
        fp = int(((yhat == 1) & (y == 0)).sum())
        miss = (yhat == 0) & (y == 1)
        fn_cost = float((amount[miss] * COST_FALSE_NEGATIVE_MULT).sum())
        cost = fp * COST_FALSE_POSITIVE + fn_cost
        if cost < best_cost:
            best_cost, best = cost, float(thr)
    return best, best_cost


def main():
    P = json.load(open(os.path.join(MODELS_DIR, "predictions.json")))
    yva = np.array(P["valid"]["y"]); yte = np.array(P["test"]["y"])
    va_amt = P["valid"].get("Amount"); te_amt = np.array(P["test"].get("Amount") or [1] * len(yte), float)

    avg_fraud_amt = float(te_amt[yte == 1].mean()) if (yte == 1).any() else 0.0
    implied_ratio = avg_fraud_amt * COST_FALSE_NEGATIVE_MULT / COST_FALSE_POSITIVE
    report = {"cost_assumptions": {
        "false_positive_$": COST_FALSE_POSITIVE,
        "false_negative_mult_x_amount": COST_FALSE_NEGATIVE_MULT,
        "avg_fraud_amount_$": round(avg_fraud_amt, 2),
        "implied_cost_ratio_missed_fraud_to_false_positive": round(implied_ratio, 1),
    }}
    models = {"logreg": "Logistic Regression", "hgb": "Gradient-Boosted Trees"}

    # ---- Headline AUCs (out-of-time test) ----------------------------------
    for m in models:
        pte = np.array(P["test"][m])
        report[m] = {
            "roc_auc": float(roc_auc_score(yte, pte)),
            "pr_auc": float(average_precision_score(yte, pte)),
        }

    # ---- Operating point: tune on VALID by cost, report on TEST ------------
    primary = "hgb"
    pva = np.array(P["valid"][primary]); pte = np.array(P["test"][primary])
    thr_cost, _ = choose_threshold_by_cost(yva, pva, va_amt)

    # also compute F1-optimal (on valid) and a high-recall point for the tradeoff table
    prec_v, rec_v, thr_v = precision_recall_curve(yva, pva)
    f1_v = 2 * prec_v * rec_v / np.clip(prec_v + rec_v, 1e-9, None)
    thr_f1 = float(thr_v[max(0, np.argmax(f1_v) - 1)]) if len(thr_v) else 0.5
    # high-recall: lowest threshold achieving >=0.90 recall on valid
    hi_idx = np.where(rec_v >= 0.90)[0]
    thr_hr = float(thr_v[hi_idx[-1] - 1]) if len(hi_idx) and hi_idx[-1] - 1 < len(thr_v) else thr_f1

    report["operating_points"] = {
        "cost_minimizing": metrics_at(yte, pte, thr_cost),
        "f1_optimal": metrics_at(yte, pte, thr_f1),
        "high_recall_0.90": metrics_at(yte, pte, thr_hr),
    }
    report["chosen_operating_point"] = "cost_minimizing"
    report["primary_model"] = primary

    # ---- Threshold sweep table (test) --------------------------------------
    sweep = []
    for thr in np.round(np.linspace(0.05, 0.95, 19), 3):
        sweep.append(metrics_at(yte, pte, float(thr)))
    report["threshold_sweep"] = sweep

    # =====================  FIGURES  ========================================
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, name in models.items():
        fpr, tpr, _ = roc_curve(yte, np.array(P["test"][m]))
        ax.plot(fpr, tpr, label=f"{name} (AUC={report[m]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (out-of-time test)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(EV, "roc_curve.png")); plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, name in models.items():
        pr, rc, _ = precision_recall_curve(yte, np.array(P["test"][m]))
        ax.plot(rc, pr, label=f"{name} (PR-AUC={report[m]['pr_auc']:.3f})")
    ax.axhline(yte.mean(), color="k", ls="--", lw=1, label=f"base rate={yte.mean():.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (out-of-time test)"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(EV, "pr_curve.png")); plt.close(fig)

    # Confusion matrix at chosen op point
    op = report["operating_points"]["cost_minimizing"]
    cm = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontweight="bold")
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(f"Confusion matrix @ thr={op['threshold']:.3f}\n"
                 f"precision={op['precision']:.2f}  recall={op['recall']:.2f}  F1={op['f1']:.2f}")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(EV, "confusion_matrix.png")); plt.close(fig)

    # Precision/recall vs threshold
    fig, ax = plt.subplots(figsize=(7, 4.5))
    thr_axis = [s["threshold"] for s in sweep]
    ax.plot(thr_axis, [s["precision"] for s in sweep], "-o", label="Precision", color="#4C78A8")
    ax.plot(thr_axis, [s["recall"] for s in sweep], "-o", label="Recall", color="#E45756")
    ax.plot(thr_axis, [s["f1"] for s in sweep], "-o", label="F1", color="#54A24B")
    ax.axvline(op["threshold"], color="#333", ls="--", lw=1, label=f"chosen thr={op['threshold']:.3f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs threshold (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(EV, "threshold_tradeoff.png")); plt.close(fig)

    with open(os.path.join(EV, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("[eval] OUT-OF-TIME TEST RESULTS")
    for m, name in models.items():
        print(f"   {name:26s} ROC-AUC={report[m]['roc_auc']:.4f}  PR-AUC={report[m]['pr_auc']:.4f}")
    print(f"   Chosen op point (cost-min, thr={op['threshold']:.3f}): "
          f"precision={op['precision']:.3f} recall={op['recall']:.3f} F1={op['f1']:.3f} "
          f"[TP={op['tp']} FP={op['fp']} FN={op['fn']}]")


if __name__ == "__main__":
    main()
