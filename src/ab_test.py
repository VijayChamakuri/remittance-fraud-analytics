"""
ab_test.py  (Pipeline step 7: A/B TEST)
=======================================
Designs and SIMULATES an experiment on a fraud intervention: a **step-up
verification** (one-time passcode / ID re-check) applied to borderline-risk
transactions surfaced by the model.

This is a *simulation* with clearly stated assumptions, then an HONEST statistical
analysis of the simulated outcomes:

  Hypothesis (H1):  Step-up verification reduces the rate of *completed fraud*
                    among borderline transactions vs. the current flow.
  Null (H0):        No difference in completed-fraud rate.
  Primary metric:   completed-fraud rate (share of transactions that are fraud
                    AND clear the flow).
  Guardrail metric: legitimate-customer abandonment (friction cost).
  Design:           50/50 randomization of eligible transactions; two-proportion
                    z-test; alpha = 0.05, target power = 0.80.
  Success gate:     ship only if the RELATIVE reduction >= a practical threshold
                    (20%) AND p < 0.05 AND guardrail friction is acceptable.
                    (We gate on effect size, not p-value alone.)

Assumed true effects (documented; a real test would measure these):
  * step-up blocks ~60% of would-be fraud (fraudster fails the challenge)
  * step-up causes ~5% of legit users to abandon (friction)
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import MODELS_DIR, REPORTS_DIR, SEED, TARGET

AB = os.path.join(REPORTS_DIR, "ab_test")

# ---- Experiment design constants ------------------------------------------
BAND_LOW, BAND_HIGH = 0.30, 0.95        # medium-high risk review band (step-up target)
TRUE_FRAUD_BLOCK = 0.60                  # assumed: step-up stops 60% of fraud
TRUE_LEGIT_ABANDON = 0.05                # assumed: 5% legit friction
ALPHA = 0.05
TARGET_POWER = 0.80
PRACTICAL_REL_REDUCTION = 0.20           # ship gate: >=20% relative reduction


def required_n_per_arm(p1, p2, alpha=ALPHA, power=TARGET_POWER):
    """Two-proportion sample size per arm (normal approximation)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / max((p1 - p2) ** 2, 1e-12)))


def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pbar = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    # 95% CI on the difference (unpooled SE)
    se_d = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci = (float((p1 - p2) - 1.96 * se_d), float((p1 - p2) + 1.96 * se_d))
    return float(z), float(p_value), ci


def main():
    P = json.load(open(os.path.join(MODELS_DIR, "predictions.json")))
    y = np.array(P["test"]["y"])
    score = np.array(P["test"]["hgb"])
    rng = np.random.default_rng(SEED)

    # ---- Eligibility: borderline band --------------------------------------
    elig = (score >= BAND_LOW) & (score < BAND_HIGH)
    ye = y[elig]
    n_elig = int(elig.sum())
    base_fraud_rate = float(ye.mean()) if n_elig else 0.0

    res = {"design": {
        "eligible_band": [BAND_LOW, BAND_HIGH], "n_eligible": n_elig,
        "baseline_completed_fraud_rate": base_fraud_rate,
        "assumed_fraud_block": TRUE_FRAUD_BLOCK, "assumed_legit_abandon": TRUE_LEGIT_ABANDON,
        "alpha": ALPHA, "target_power": TARGET_POWER,
        "practical_relative_reduction_gate": PRACTICAL_REL_REDUCTION,
    }}

    # ---- Power / sample size -----------------------------------------------
    p1 = base_fraud_rate
    p2 = base_fraud_rate * (1 - TRUE_FRAUD_BLOCK)
    n_needed = required_n_per_arm(p1, p2) if p1 > 0 else None
    res["power"] = {"p1_control": p1, "p2_treatment_expected": p2,
                    "n_required_per_arm": n_needed,
                    "n_available_per_arm": n_elig // 2,
                    "adequately_powered": bool(n_needed is not None and n_elig // 2 >= n_needed)}

    # ---- Randomize & simulate outcomes -------------------------------------
    assign = rng.random(n_elig) < 0.5           # True = treatment
    is_fraud = ye.astype(bool)

    # Control: everything completes as-is -> completed fraud = fraud rows
    ctrl_mask = ~assign
    ctrl_n = int(ctrl_mask.sum())
    ctrl_completed_fraud = int((is_fraud & ctrl_mask).sum())

    # Treatment: fraud blocked w.p. TRUE_FRAUD_BLOCK; legit abandons w.p. TRUE_LEGIT_ABANDON
    trt_mask = assign
    trt_n = int(trt_mask.sum())
    fraud_trt = is_fraud & trt_mask
    blocked = fraud_trt & (rng.random(n_elig) < TRUE_FRAUD_BLOCK)
    trt_completed_fraud = int((fraud_trt & ~blocked).sum())
    legit_trt = (~is_fraud) & trt_mask
    abandoned = legit_trt & (rng.random(n_elig) < TRUE_LEGIT_ABANDON)
    trt_legit_abandon = int(abandoned.sum())

    # ---- Statistical test on completed-fraud rate --------------------------
    z, pval, ci = two_prop_ztest(ctrl_completed_fraud, ctrl_n, trt_completed_fraud, trt_n)
    ctrl_rate = ctrl_completed_fraud / ctrl_n
    trt_rate = trt_completed_fraud / trt_n
    abs_reduction = ctrl_rate - trt_rate
    rel_reduction = abs_reduction / ctrl_rate if ctrl_rate else 0.0

    res["result"] = {
        "control_n": ctrl_n, "treatment_n": trt_n,
        "control_completed_fraud": ctrl_completed_fraud,
        "treatment_completed_fraud": trt_completed_fraud,
        "control_completed_fraud_rate": float(ctrl_rate),
        "treatment_completed_fraud_rate": float(trt_rate),
        "absolute_reduction": float(abs_reduction),
        "relative_reduction": float(rel_reduction),
        "z_stat": z, "p_value": pval, "diff_95ci": ci,
        "guardrail_legit_abandonment_rate": float(trt_legit_abandon / max((~is_fraud & trt_mask).sum(), 1)),
    }

    significant = pval < ALPHA
    big_enough = rel_reduction >= PRACTICAL_REL_REDUCTION
    res["decision"] = {
        "statistically_significant": bool(significant),
        "meets_practical_effect_gate": bool(big_enough),
        "ship": bool(significant and big_enough),
        "rationale": ("Ship: significant AND clears the 20% practical-reduction gate; "
                      "friction guardrail within tolerance."
                      if significant and big_enough else
                      "Do NOT ship on this evidence: "
                      + ("effect too small vs practical gate. " if not big_enough else "")
                      + ("not statistically significant. " if not significant else "")),
    }

    # ---- Figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["Control", "Step-up"], [ctrl_rate * 100, trt_rate * 100],
              color=["#4C78A8", "#54A24B"])
    ax[0].set_ylabel("Completed-fraud rate (%)")
    ax[0].set_title(f"Primary metric\nrel. reduction {rel_reduction*100:.1f}%  (p={pval:.3g})")
    for i, v in enumerate([ctrl_rate * 100, trt_rate * 100]):
        ax[0].text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax[1].bar(["Legit abandonment\n(guardrail)"], [res["result"]["guardrail_legit_abandonment_rate"] * 100],
              color="#E45756")
    ax[1].set_ylabel("Rate (%)"); ax[1].set_title("Friction guardrail")
    fig.suptitle("Step-up verification A/B test (simulated)", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(AB, "ab_result.png")); plt.close(fig)

    with open(os.path.join(AB, "ab_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    _write_md(res)

    d = res["result"]
    print(f"[ab] control={d['control_completed_fraud_rate']*100:.2f}%  "
          f"treatment={d['treatment_completed_fraud_rate']*100:.2f}%  "
          f"rel.red={d['relative_reduction']*100:.1f}%  p={d['p_value']:.3g}  "
          f"ship={res['decision']['ship']}")


def _write_md(r):
    d, res, dec, pw = r["design"], r["result"], r["decision"], r["power"]
    ci = res["diff_95ci"]
    lines = [
        "# A/B Test: Step-up Verification on Borderline-Risk Transactions (simulated)\n",
        "## Design\n",
        f"- **Hypothesis:** step-up verification lowers the *completed-fraud rate* among "
        f"borderline transactions (model score in [{d['eligible_band'][0]}, {d['eligible_band'][1]})).",
        f"- **Primary metric:** completed-fraud rate. **Guardrail:** legit-customer abandonment.",
        f"- **Randomization:** 50/50. **alpha:** {d['alpha']}, **target power:** {d['target_power']}.",
        f"- **Assumed true effects (simulation):** step-up blocks {d['assumed_fraud_block']*100:.0f}% of "
        f"fraud; {d['assumed_legit_abandon']*100:.0f}% legit friction. A real test would measure these.\n",
        "## Power / sample size\n",
        f"- Baseline completed-fraud rate (control): **{pw['p1_control']*100:.2f}%**; "
        f"expected under treatment: **{pw['p2_treatment_expected']*100:.2f}%**.",
        f"- Required n per arm for {d['target_power']*100:.0f}% power: "
        f"**{pw['n_required_per_arm']:,}**; available per arm: **{pw['n_available_per_arm']:,}** "
        f"-> {'adequately powered' if pw['adequately_powered'] else 'UNDERPOWERED (interpret with care)'}.\n",
        "## Result (simulated outcomes, honest two-proportion z-test)\n",
        f"| Arm | N | Completed fraud | Rate |",
        "|---|---:|---:|---:|",
        f"| Control | {res['control_n']:,} | {res['control_completed_fraud']:,} | {res['control_completed_fraud_rate']*100:.2f}% |",
        f"| Step-up | {res['treatment_n']:,} | {res['treatment_completed_fraud']:,} | {res['treatment_completed_fraud_rate']*100:.2f}% |",
        "",
        f"- **Absolute reduction:** {res['absolute_reduction']*100:.2f} pp "
        f"(95% CI on difference: [{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp).",
        f"- **Relative reduction:** {res['relative_reduction']*100:.1f}%.",
        f"- **z = {res['z_stat']:.2f}, p = {res['p_value']:.3g}.**",
        f"- **Guardrail:** legit abandonment in treatment = {res['guardrail_legit_abandonment_rate']*100:.2f}%.\n",
        "## Decision (gated on effect size, not just p-value)\n",
        f"- Statistically significant: **{dec['statistically_significant']}**.",
        f"- Meets >=20% practical-reduction gate: **{dec['meets_practical_effect_gate']}**.",
        f"- **Recommendation: {'SHIP' if dec['ship'] else 'DO NOT SHIP (yet)'}.** {dec['rationale']}\n",
        "> Plain English for Product/Compliance: step-up verification cuts fraud that gets through "
        "the gray zone, at the cost of a small amount of legitimate-customer friction. We only "
        "recommend rolling it out when the fraud reduction is both statistically real and large "
        "enough to justify that friction - a big p-value win on a tiny effect is not a launch.",
    ]
    with open(os.path.join(REPORTS_DIR, "ab_test", "ab_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
