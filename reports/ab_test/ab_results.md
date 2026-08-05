# A/B Test: Step-up Verification on Borderline-Risk Transactions (simulated)

## Design

- **Hypothesis:** step-up verification lowers the *completed-fraud rate* among borderline transactions (model score in [0.3, 0.95)).
- **Primary metric:** completed-fraud rate. **Guardrail:** legit-customer abandonment.
- **Randomization:** 50/50. **alpha:** 0.05, **target power:** 0.8.
- **Assumed true effects (simulation):** step-up blocks 60% of fraud; 5% legit friction. A real test would measure these.

## Power / sample size

- Baseline completed-fraud rate (control): **3.80%**; expected under treatment: **1.52%**.
- Required n per arm for 80% power: **781**; available per arm: **658** -> UNDERPOWERED (interpret with care).

## Result (simulated outcomes, honest two-proportion z-test)

| Arm | N | Completed fraud | Rate |
|---|---:|---:|---:|
| Control | 650 | 24 | 3.69% |
| Step-up | 666 | 13 | 1.95% |

- **Absolute reduction:** 1.74 pp (95% CI on difference: [-0.05, 3.53] pp).
- **Relative reduction:** 47.1%.
- **z = 1.91, p = 0.0562.**
- **Guardrail:** legit abandonment in treatment = 3.91%.

## Decision (gated on effect size, not just p-value)

- Statistically significant: **False**.
- Meets >=20% practical-reduction gate: **True**.
- **Recommendation: DO NOT SHIP (yet).** Do NOT ship on this evidence: not statistically significant. 

> Plain English for Product/Compliance: step-up verification cuts fraud that gets through the gray zone, at the cost of a small amount of legitimate-customer friction. We only recommend rolling it out when the fraud reduction is both statistically real and large enough to justify that friction - a big p-value win on a tiny effect is not a launch.
