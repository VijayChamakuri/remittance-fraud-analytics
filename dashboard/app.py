"""
Streamlit dashboard (interactive alternative to the static dashboard/index.html).

Run:  streamlit run dashboard/app.py
Reads the same reports/*.json the pipeline produces.
"""
from __future__ import annotations
import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "reports")


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d or {}


st.set_page_config(page_title="Remittance Fraud Analytics", layout="wide")
dq = load(f"{R}/eda/data_quality.json")
eda = load(f"{R}/eda/eda_summary.json")
m = load(f"{R}/evaluation/metrics.json")
rca = load(f"{R}/rca/rca_findings.json")
ab = load(f"{R}/ab_test/ab_results.json")

st.title("Cross-Border Remittance Fraud Analytics")
src = dq.get("source", "synthetic")
st.info(f"Data source: {'REAL ULB' if src=='real_ulb' else 'synthetic ULB/IEEE-style stand-in'} — "
        "all metrics computed on a held-out out-of-time test set.")

op = m.get("operating_points", {}).get("cost_minimizing", {})
c = st.columns(6)
c[0].metric("Transactions", f"{dq.get('n_rows',0):,}")
c[1].metric("Fraud rate", f"{dq.get('fraud_rate',0)*100:.3f}%")
c[2].metric("ROC-AUC (GBT)", f"{m.get('hgb',{}).get('roc_auc',0):.3f}")
c[3].metric("PR-AUC (GBT)", f"{m.get('hgb',{}).get('pr_auc',0):.3f}")
c[4].metric("Recall @ op", f"{op.get('recall',0)*100:.1f}%")
c[5].metric("Precision @ op", f"{op.get('precision',0)*100:.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Fraud rate by amount band")
    ab_band = eda.get("fraud_rate_by_amount_band", {})
    st.bar_chart(pd.Series({k: v * 100 for k, v in ab_band.items()}, name="fraud_rate_%"))
    st.subheader("Precision / Recall vs threshold")
    sw = pd.DataFrame(m.get("threshold_sweep", []))
    if not sw.empty:
        st.line_chart(sw.set_index("threshold")[["precision", "recall", "f1"]])
with right:
    st.subheader("RCA — recall by fraud mode")
    rc = rca.get("recall_by_fraud_mode", {})
    if rc:
        st.bar_chart(pd.Series({k: v["recall"] * 100 for k, v in rc.items()}, name="recall_%"))
        st.caption(f"Hardest segment: {rca.get('worst_mode','n/a')}")
    st.subheader("Top fraud drivers")
    imp = rca.get("top_features_permutation", {})
    if imp:
        st.bar_chart(pd.Series(imp, name="importance"))

st.subheader("A/B test — step-up verification (simulated)")
res, dec = ab.get("result", {}), ab.get("decision", {})
if res:
    st.write(pd.DataFrame([
        {"arm": "control", "n": res["control_n"], "completed_fraud_rate": res["control_completed_fraud_rate"]},
        {"arm": "step-up", "n": res["treatment_n"], "completed_fraud_rate": res["treatment_completed_fraud_rate"]},
    ]))
    st.write(f"Relative reduction {res['relative_reduction']*100:.1f}%, p={res['p_value']:.3g}. "
             f"Decision: **{'SHIP' if dec.get('ship') else 'DO NOT SHIP'}** — {dec.get('rationale','')}")
