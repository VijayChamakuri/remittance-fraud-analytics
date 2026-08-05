"""
build_dashboard.py  (Pipeline step 9: DASHBOARD)
================================================
Builds a single self-contained dashboard/index.html (Chart.js from CDN) that
shows: fraud KPIs, model performance (ROC/PR + headline metrics), the
segment/RCA view, and the A/B test result. All numbers are injected from the
real pipeline JSON outputs -- open the file in any browser, no server needed.

(A Streamlit version is in dashboard/app.py for an interactive alternative.)
"""
from __future__ import annotations
import json
import os

from config import DASH_DIR, MODELS_DIR, REPORTS_DIR


def _load(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default or {}


def main():
    dq = _load(os.path.join(REPORTS_DIR, "eda", "data_quality.json"))
    eda = _load(os.path.join(REPORTS_DIR, "eda", "eda_summary.json"))
    metrics = _load(os.path.join(REPORTS_DIR, "evaluation", "metrics.json"))
    rca = _load(os.path.join(REPORTS_DIR, "rca", "rca_findings.json"))
    ab = _load(os.path.join(REPORTS_DIR, "ab_test", "ab_results.json"))
    split = _load(os.path.join(REPORTS_DIR, "evaluation", "split_meta.json"))

    payload = {"dq": dq, "eda": eda, "metrics": metrics, "rca": rca, "ab": ab, "split": split}
    data_js = json.dumps(payload)

    html = HTML_TEMPLATE.replace("__DATA__", data_js)
    out = os.path.join(DASH_DIR, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"[dashboard] wrote {out}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Remittance Fraud Analytics - Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--muted:#94a3b8;--fg:#e2e8f0;--accent:#38bdf8;
        --good:#4ade80;--bad:#f87171;--warn:#fbbf24;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
  header{padding:22px 28px;background:linear-gradient(90deg,#0ea5e9,#6366f1);color:#fff;}
  header h1{margin:0;font-size:22px} header p{margin:6px 0 0;opacity:.9;font-size:13px}
  .wrap{max-width:1200px;margin:0 auto;padding:20px 24px 60px}
  .banner{background:#422006;border:1px solid #a16207;color:#fde68a;padding:10px 14px;border-radius:8px;font-size:13px;margin:16px 0}
  .grid{display:grid;gap:16px}
  .kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
  .cols2{grid-template-columns:1fr 1fr}
  @media(max-width:800px){.cols2{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid #334155;border-radius:12px;padding:16px}
  .card h3{margin:0 0 10px;font-size:14px;color:var(--accent);letter-spacing:.3px;text-transform:uppercase}
  .kpi .v{font-size:26px;font-weight:700} .kpi .l{color:var(--muted);font-size:12px;margin-top:4px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:6px 8px;text-align:left;border-bottom:1px solid #334155}
  th{color:var(--muted);font-weight:600}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}
  .ship{background:#064e3b;color:#6ee7b7}.noship{background:#7f1d1d;color:#fca5a5}
  .sec{margin-top:26px} .muted{color:var(--muted);font-size:12px}
  canvas{max-height:280px}
</style>
</head>
<body>
<header>
  <h1>Cross-Border Remittance Fraud Analytics</h1>
  <p>Fraud-detection methodology on public card-fraud data, mapped to remittance risk (ATO, mules, stolen instruments, structuring).</p>
</header>
<div class="wrap">
  <div class="banner" id="banner"></div>

  <div class="grid kpis" id="kpis"></div>

  <div class="sec grid cols2">
    <div class="card"><h3>Model performance (out-of-time test)</h3><table id="perf"></table>
      <p class="muted" id="opnote"></p></div>
    <div class="card"><h3>Precision / Recall vs threshold</h3><canvas id="thrChart"></canvas></div>
  </div>

  <div class="sec grid cols2">
    <div class="card"><h3>Fraud rate by amount band</h3><canvas id="amtChart"></canvas></div>
    <div class="card"><h3>Fraud rate by segment</h3><canvas id="segChart"></canvas></div>
  </div>

  <div class="sec grid cols2">
    <div class="card"><h3>RCA - recall by fraud mode</h3><canvas id="rcaChart"></canvas>
      <p class="muted" id="rcanote"></p></div>
    <div class="card"><h3>Top fraud drivers</h3><canvas id="impChart"></canvas></div>
  </div>

  <div class="sec card"><h3>A/B test - step-up verification (simulated)</h3>
    <div id="abbox"></div>
  </div>

  <p class="sec muted">Generated from live pipeline outputs (reports/*.json). See README for the honest data-provenance note.</p>
</div>

<script>
const D = __DATA__;
const pct = x => (x*100).toFixed(2)+'%';
const f3 = x => (x==null?'-':Number(x).toFixed(3));

// Banner
const src = (D.dq.source||'synthetic');
document.getElementById('banner').innerHTML =
  '<b>Data source:</b> '+ (src==='real_ulb'?'REAL ULB creditcard.csv':'synthetic ULB/IEEE-style stand-in (offline build)') +
  ' &nbsp;•&nbsp; all metrics below are computed from real model predictions on a held-out out-of-time test set.';

// KPIs
const m = D.metrics, op = (m.operating_points||{}).cost_minimizing||{};
const kpis = [
  ['Transactions', (D.dq.n_rows||0).toLocaleString()],
  ['Fraud rate', pct(D.dq.fraud_rate||0)],
  ['Imbalance', '1 : '+Math.round(D.dq.imbalance_ratio||0)],
  ['ROC-AUC (GBT)', f3((m.hgb||{}).roc_auc)],
  ['PR-AUC (GBT)', f3((m.hgb||{}).pr_auc)],
  ['Recall @ op', op.recall!=null?pct(op.recall):'-'],
  ['Precision @ op', op.precision!=null?pct(op.precision):'-'],
];
document.getElementById('kpis').innerHTML = kpis.map(k=>
  `<div class="card kpi"><div class="v">${k[1]}</div><div class="l">${k[0]}</div></div>`).join('');

// Performance table
const rows = [['Logistic Regression','logreg'],['Gradient-Boosted Trees','hgb']];
document.getElementById('perf').innerHTML =
  '<tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th></tr>' +
  rows.map(r=>`<tr><td>${r[0]}</td><td>${f3((m[r[1]]||{}).roc_auc)}</td><td>${f3((m[r[1]]||{}).pr_auc)}</td></tr>`).join('');
document.getElementById('opnote').textContent =
  `Chosen operating point (cost-minimizing, threshold ${f3(op.threshold)}): `+
  `TP=${op.tp}, FP=${op.fp}, FN=${op.fn}, F1=${f3(op.f1)}.`;

// Threshold chart
const sw = m.threshold_sweep||[];
new Chart(thrChart,{type:'line',data:{labels:sw.map(s=>s.threshold),
  datasets:[
    {label:'Precision',data:sw.map(s=>s.precision),borderColor:'#38bdf8'},
    {label:'Recall',data:sw.map(s=>s.recall),borderColor:'#f87171'},
    {label:'F1',data:sw.map(s=>s.f1),borderColor:'#4ade80'}]},
  options:{plugins:{legend:{labels:{color:'#e2e8f0'}}},
    scales:{x:{ticks:{color:'#94a3b8'}},y:{ticks:{color:'#94a3b8'}}}}});

// Amount band chart
const ab = D.eda.fraud_rate_by_amount_band||{};
new Chart(amtChart,{type:'bar',data:{labels:Object.keys(ab),
  datasets:[{label:'Fraud rate',data:Object.values(ab).map(v=>v*100),backgroundColor:'#f87171'}]},
  options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'}},
    y:{ticks:{color:'#94a3b8',callback:v=>v+'%'}}}}});

// Segment chart (corridor)
const seg = D.eda.fraud_rate_by_corridor || D.eda.fraud_rate_by_device_type || {};
const segLabels = Object.keys(seg), segVals = segLabels.map(k=>seg[k].fraud_rate*100);
new Chart(segChart,{type:'bar',data:{labels:segLabels,
  datasets:[{label:'Fraud rate',data:segVals,backgroundColor:'#818cf8'}]},
  options:{indexAxis:'y',plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#94a3b8',callback:v=>v+'%'}},y:{ticks:{color:'#94a3b8'}}}}});

// RCA recall by mode
const rc = D.rca.recall_by_fraud_mode||{};
const rcL = Object.keys(rc);
new Chart(rcaChart,{type:'bar',data:{labels:rcL,
  datasets:[{label:'Recall',data:rcL.map(k=>rc[k].recall*100),backgroundColor:'#4ade80'}]},
  options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'}},
    y:{ticks:{color:'#94a3b8',callback:v=>v+'%'}}}}});
document.getElementById('rcanote').textContent = D.rca.worst_mode?
  ('Hardest segment: '+D.rca.worst_mode+' - lowest recall, prioritize for rules/step-up.'):'';

// Importance
const imp = D.rca.top_features_permutation||{};
const impK = Object.keys(imp).slice(0,10);
new Chart(impChart,{type:'bar',data:{labels:impK,
  datasets:[{label:'Importance',data:impK.map(k=>imp[k]),backgroundColor:'#38bdf8'}]},
  options:{indexAxis:'y',plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#94a3b8'}},y:{ticks:{color:'#94a3b8'}}}}});

// A/B box
const r = (D.ab.result)||{}, dec=(D.ab.decision)||{}, pw=(D.ab.power)||{};
document.getElementById('abbox').innerHTML =
  `<table>
    <tr><th>Arm</th><th>N</th><th>Completed fraud</th><th>Rate</th></tr>
    <tr><td>Control</td><td>${(r.control_n||0).toLocaleString()}</td><td>${r.control_completed_fraud}</td><td>${pct(r.control_completed_fraud_rate||0)}</td></tr>
    <tr><td>Step-up</td><td>${(r.treatment_n||0).toLocaleString()}</td><td>${r.treatment_completed_fraud}</td><td>${pct(r.treatment_completed_fraud_rate||0)}</td></tr>
   </table>
   <p style="margin-top:10px">Relative reduction <b>${((r.relative_reduction||0)*100).toFixed(1)}%</b>,
   z=${f3(r.z_stat)}, p=${(r.p_value==null?'-':Number(r.p_value).toPrecision(3))}.
   Powered per arm: ${pw.n_available_per_arm} vs required ${pw.n_required_per_arm} (${pw.adequately_powered?'OK':'underpowered'}).</p>
   <p>Decision: <span class="pill ${dec.ship?'ship':'noship'}">${dec.ship?'SHIP':'DO NOT SHIP'}</span>
   <span class="muted"> ${dec.rationale||''}</span></p>`;
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
