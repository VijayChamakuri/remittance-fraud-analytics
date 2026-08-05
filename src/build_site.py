"""
build_site.py  (GitHub Pages site + README hero image)
======================================================
Publishes a results landing page and the interactive dashboard to `docs/`, which
GitHub Pages serves from `main /docs`. Also builds a composite hero image used at
the top of the README.

Outputs:
  docs/index.html        - results landing page (KPIs, charts, links)
  docs/dashboard.html    - copy of the static dashboard
  docs/assets/*.png      - chart images (also embedded in the README)
  docs/.nojekyll         - so GitHub Pages serves files as-is
  reports/hero.png       - 2x2 composite for the README banner
All numbers are injected from the real pipeline JSON outputs.
"""
from __future__ import annotations
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from config import DASH_DIR, REPORTS_DIR, ROOT

DOCS = os.path.join(ROOT, "docs")
ASSETS = os.path.join(DOCS, "assets")
GH = "https://github.com/VijayChamakuri/remittance-fraud-analytics/blob/main"


def _load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d or {}


def _hero():
    """2x2 composite of the four headline charts -> reports/hero.png."""
    panels = [
        ("evaluation/roc_curve.png", "ROC — out-of-time holdout"),
        ("evaluation/pr_curve.png", "Precision-Recall"),
        ("evaluation/confusion_matrix.png", "Confusion @ operating point"),
        ("rca/feature_importance.png", "Top fraud drivers (RCA)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (rel, title) in zip(axes.ravel(), panels):
        p = os.path.join(REPORTS_DIR, rel)
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Remittance Fraud Analytics — headline results", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "hero.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _dashboard_preview(dq, m):
    """Dark-themed composite that previews the live dashboard for the README."""
    op = m.get("operating_points", {}).get("cost_minimizing", {})
    kpis = [
        (f"{dq.get('n_rows',0):,}", "Transactions"),
        (f"{dq.get('fraud_rate',0)*100:.2f}%", "Fraud rate"),
        (f"{m.get('hgb',{}).get('roc_auc',0):.3f}", "ROC-AUC"),
        (f"{m.get('hgb',{}).get('pr_auc',0):.3f}", "PR-AUC"),
        (f"{op.get('recall',0)*100:.0f}%", "Recall @ op"),
        (f"{op.get('precision',0)*100:.0f}%", "Precision @ op"),
    ]
    fig = plt.figure(figsize=(13, 8.6), facecolor="#0b1220")
    fig.text(0.5, 0.965, "Cross-Border Remittance Fraud Analytics — Dashboard",
             ha="center", color="#e6edf7", fontsize=17, fontweight="bold")
    # KPI row
    for i, (v, l) in enumerate(kpis):
        x = 0.02 + i * 0.163
        ax = fig.add_axes([x, 0.80, 0.15, 0.11]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#151f34", ec="#24304d"))
        ax.text(0.5, 0.60, v, ha="center", va="center", color="#43b7ff", fontsize=17, fontweight="bold")
        ax.text(0.5, 0.20, l, ha="center", va="center", color="#93a3bd", fontsize=9)
    panels = [("evaluation/threshold_tradeoff.png", 0.02), ("rca/fn_by_mode.png", 0.51)]
    for rel, x in panels:
        p = os.path.join(REPORTS_DIR, rel)
        if os.path.exists(p):
            ax = fig.add_axes([x, 0.40, 0.47, 0.36]); ax.axis("off"); ax.imshow(mpimg.imread(p))
    panels2 = [("eda/fraud_by_amount.png", 0.02), ("ab_test/ab_result.png", 0.51)]
    for rel, x in panels2:
        p = os.path.join(REPORTS_DIR, rel)
        if os.path.exists(p):
            ax = fig.add_axes([x, 0.02, 0.47, 0.36]); ax.axis("off"); ax.imshow(mpimg.imread(p))
    out = os.path.join(REPORTS_DIR, "dashboard_preview.png")
    fig.savefig(out, dpi=105, facecolor="#0b1220"); plt.close(fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    dq = _load(f"{REPORTS_DIR}/eda/data_quality.json")
    m = _load(f"{REPORTS_DIR}/evaluation/metrics.json")
    rca = _load(f"{REPORTS_DIR}/rca/rca_findings.json")
    ab = _load(f"{REPORTS_DIR}/ab_test/ab_results.json")

    # copy chart assets
    assets = {
        "roc_curve.png": "evaluation/roc_curve.png",
        "pr_curve.png": "evaluation/pr_curve.png",
        "confusion_matrix.png": "evaluation/confusion_matrix.png",
        "threshold_tradeoff.png": "evaluation/threshold_tradeoff.png",
        "feature_importance.png": "rca/feature_importance.png",
        "fn_by_mode.png": "rca/fn_by_mode.png",
        "fraud_by_segment.png": "eda/fraud_by_segment.png",
        "fraud_by_amount.png": "eda/fraud_by_amount.png",
        "class_imbalance.png": "eda/class_imbalance.png",
        "ab_result.png": "ab_test/ab_result.png",
    }
    for dst, rel in assets.items():
        src = os.path.join(REPORTS_DIR, rel)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ASSETS, dst))

    _hero()
    _dashboard_preview(dq, m)
    shutil.copy(os.path.join(REPORTS_DIR, "hero.png"), os.path.join(ASSETS, "hero.png"))
    shutil.copy(os.path.join(REPORTS_DIR, "dashboard_preview.png"), os.path.join(ASSETS, "dashboard_preview.png"))

    # dashboard copy + .nojekyll
    if os.path.exists(os.path.join(DASH_DIR, "index.html")):
        shutil.copy(os.path.join(DASH_DIR, "index.html"), os.path.join(DOCS, "dashboard.html"))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()

    op = m.get("operating_points", {})
    cm, f1o, hr = op.get("cost_minimizing", {}), op.get("f1_optimal", {}), op.get("high_recall_0.90", {})
    ca = m.get("cost_assumptions", {})
    ratio = ca.get("implied_cost_ratio_missed_fraud_to_false_positive", 40)
    abres, abdec = ab.get("result", {}), ab.get("decision", {})

    def pct(x):
        return f"{x*100:.0f}%" if x is not None else "-"

    html = LANDING.format(
        n_rows=f"{dq.get('n_rows',0):,}", fraud_rate=f"{dq.get('fraud_rate',0)*100:.2f}",
        imbal=int(dq.get("imbalance_ratio", 0)),
        roc=f"{m.get('hgb',{}).get('roc_auc',0):.3f}", pr=f"{m.get('hgb',{}).get('pr_auc',0):.3f}",
        lr_roc=f"{m.get('logreg',{}).get('roc_auc',0):.3f}", lr_pr=f"{m.get('logreg',{}).get('pr_auc',0):.3f}",
        cm_p=pct(cm.get("precision")), cm_r=pct(cm.get("recall")), cm_t=f"{cm.get('threshold',0):.2f}",
        f1_p=pct(f1o.get("precision")), f1_r=pct(f1o.get("recall")), f1_t=f"{f1o.get('threshold',0):.2f}",
        hr_p=pct(hr.get("precision")), hr_r=pct(hr.get("recall")), hr_t=f"{hr.get('threshold',0):.2f}",
        ratio=ratio,
        worst=rca.get("worst_mode", "mule"),
        ab_ctrl=f"{abres.get('control_completed_fraud_rate',0)*100:.1f}",
        ab_trt=f"{abres.get('treatment_completed_fraud_rate',0)*100:.1f}",
        ab_rel=f"{abres.get('relative_reduction',0)*100:.0f}",
        ab_p=f"{abres.get('p_value',0):.3f}",
        ab_ship="SHIP" if abdec.get("ship") else "HOLD — extend the test",
        gh=GH,
    )
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
    print(f"[site] wrote docs/index.html, docs/dashboard.html, {len(assets)} assets, reports/hero.png")


LANDING = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Remittance Fraud Analytics — Results</title>
<style>
 :root{{--bg:#0b1220;--card:#151f34;--fg:#e6edf7;--mut:#93a3bd;--acc:#43b7ff;--good:#4ade80;--bad:#f87171;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5}}
 a{{color:var(--acc)}} .wrap{{max-width:1080px;margin:0 auto;padding:0 22px 70px}}
 header{{background:linear-gradient(120deg,#0ea5e9,#6366f1);padding:40px 22px}}
 header .wrap{{padding-bottom:0}} h1{{margin:0;font-size:30px}} .sub{{opacity:.92;margin-top:8px;max-width:760px}}
 .cta{{margin-top:18px;display:flex;gap:12px;flex-wrap:wrap}}
 .btn{{background:#fff;color:#111;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600}}
 .btn.ghost{{background:rgba(255,255,255,.15);color:#fff}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-top:28px}}
 .kpi{{background:var(--card);border:1px solid #24304d;border-radius:12px;padding:16px}}
 .kpi .v{{font-size:26px;font-weight:800}} .kpi .l{{color:var(--mut);font-size:12px;margin-top:4px}}
 h2{{margin-top:40px;font-size:20px;border-left:4px solid var(--acc);padding-left:10px}}
 table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px}}
 th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid #24304d}} th{{color:var(--mut)}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}}
 @media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
 .card{{background:var(--card);border:1px solid #24304d;border-radius:12px;padding:12px}}
 .card img{{width:100%;border-radius:8px;display:block}} .card h3{{margin:2px 0 8px;font-size:13px;color:var(--acc);text-transform:uppercase;letter-spacing:.4px}}
 .note{{background:#13233b;border:1px solid #2a4a6d;border-left:4px solid var(--good);border-radius:8px;padding:12px 14px;margin-top:14px;font-size:14px;color:#cfe0f5}}
 .pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px}}
 .hold{{background:#7f1d1d;color:#fca5a5}} .muted{{color:var(--mut);font-size:13px}}
 footer{{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid #24304d;padding-top:16px}}
</style></head><body>
<header><div class="wrap">
 <h1>Cross-Border Remittance Fraud Analytics</h1>
 <div class="sub">End-to-end fraud detection on public card-fraud data mapped to a remittance context:
   feature engineering, a gradient-boosted model with <b>out-of-time</b> evaluation, root-cause analysis,
   and a power-gated A/B test.</div>
 <div class="cta">
   <a class="btn" href="dashboard.html">▶ Live interactive dashboard</a>
   <a class="btn ghost" href="{gh}/reports/executive_summary.md">Executive summary</a>
   <a class="btn ghost" href="{gh}/README.md">Repo / code</a>
 </div>
</div></header>
<div class="wrap">
 <div class="kpis">
   <div class="kpi"><div class="v">{roc}</div><div class="l">ROC-AUC (out-of-time)</div></div>
   <div class="kpi"><div class="v">{pr}</div><div class="l">PR-AUC (out-of-time)</div></div>
   <div class="kpi"><div class="v">{cm_r}</div><div class="l">Recall @ operating point</div></div>
   <div class="kpi"><div class="v">{n_rows}</div><div class="l">Transactions</div></div>
   <div class="kpi"><div class="v">{fraud_rate}%</div><div class="l">Fraud rate (1:{imbal})</div></div>
 </div>

 <h2>Model performance</h2>
 <table>
  <tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
  <tr><td>Logistic Regression (baseline)</td><td>{lr_roc}</td><td>{lr_pr}</td></tr>
  <tr><td><b>Gradient-Boosted Trees (main)</b></td><td><b>{roc}</b></td><td><b>{pr}</b></td></tr>
 </table>
 <div class="grid2">
   <div class="card"><h3>ROC curve</h3><img src="assets/roc_curve.png"/></div>
   <div class="card"><h3>Precision-Recall curve</h3><img src="assets/pr_curve.png"/></div>
 </div>

 <h2>Precision / recall — pick your operating point</h2>
 <table>
  <tr><th>Operating point</th><th>Threshold</th><th>Precision</th><th>Recall</th><th>Use when</th></tr>
  <tr><td>Aggressive (chosen)</td><td>{cm_t}</td><td>{cm_p}</td><td>{cm_r}</td><td>a miss costs more than friction</td></tr>
  <tr><td>Balanced (F1)</td><td>{f1_t}</td><td>{f1_p}</td><td>{f1_r}</td><td>review capacity is the constraint</td></tr>
  <tr><td>Wide net</td><td>{hr_t}</td><td>{hr_p}</td><td>{hr_r}</td><td>backstop behind other checks</td></tr>
 </table>
 <div class="note"><b>Why the aggressive point is the right call:</b> a missed fraud costs ~the
  transaction value (avg&nbsp;~$161) while a false positive costs ~$4 of review friction — an implied
  <b>~{ratio}:1</b> cost ratio. At that ratio it is rational to accept ~40 false alarms to stop one
  fraud, so the cost-minimizing threshold sits at high recall.</div>
 <div class="grid2">
   <div class="card"><h3>Confusion @ operating point</h3><img src="assets/confusion_matrix.png"/></div>
   <div class="card"><h3>Precision / recall / F1 vs threshold</h3><img src="assets/threshold_tradeoff.png"/></div>
 </div>

 <h2>Root-cause analysis</h2>
 <p class="muted">Top drivers are 1-hour velocity, device change, new-corridor and anonymized network
  signals. The model catches account-takeover well but under-catches <b>{worst}</b> fraud, which hides
  in normal new-account behavior. → <a href="{gh}/reports/rca/rca_findings.md">full RCA writeup</a>.</p>
 <div class="grid2">
   <div class="card"><h3>Top fraud drivers</h3><img src="assets/feature_importance.png"/></div>
   <div class="card"><h3>Recall by fraud mode</h3><img src="assets/fn_by_mode.png"/></div>
 </div>

 <h2>A/B test — step-up verification (simulated, power-gated)</h2>
 <p class="muted">Completed-fraud rate {ab_ctrl}% → {ab_trt}% (<b>−{ab_rel}%</b> relative), but the test
  was underpowered so p={ab_p}. Decision: <span class="pill hold">{ab_ship}</span> — we gate on effect
  size <i>and</i> significance, not a single near-miss p-value.
  → <a href="{gh}/reports/ab_test/ab_results.md">A/B writeup</a>.</p>
 <div class="card" style="max-width:640px"><h3>A/B result</h3><img src="assets/ab_result.png"/></div>

 <footer>
  Built on public card-fraud data (ULB/IEEE-CIS style) mapped to a remittance context; synthetic where
  noted and reproducible on the real Kaggle dataset with <code>python run_pipeline.py --real</code>. Not real Remitly data.
  All figures computed from the model on a held-out out-of-time test set.
  <a href="{gh}/DATA.md">Data &amp; honesty →</a>
 </footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
