# M5 and M6 — Evaluation Results

**Study:** CPCS683 Hajj Risk Study, with M4 Single-Agent baseline, M5 evaluation harness, and M6 controlled Multi-Agent comparison. **Data:** synthetic Hajj/Umrah crowd-management source.

**Design:** 30 source test cases × 3 fixed seeds per arm = 90 paired test executions, Single-Agent vs Multi-Agent, fixed gate `k=7`.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Issued decisions | 68/90 | 71/90 |
| Coverage | 75.56% | 78.89% |
| Wrong among issued | 26/68 | 39/71 |
| Selective error | 38.24% | 54.93% |

The 2,000-draw paired, stratified source-case bootstrap (21 clean and 9 altered cases) reported these **Multi minus Single** differences, as descriptive secondary analysis:

| Metric | Difference | 95% bootstrap interval |
|---|---:|---:|
| Coverage | +0.033333 | [-0.066667, +0.133333] |
| Selective error | +0.166943 | [+0.052817, +0.316667] |
| Trajectory safe completion | -0.144444 | [-0.266667, -0.044444] |

These are descriptive results from a limited, intentionally conditioned synthetic test set, not real operational validation. The paired bootstrap is a secondary analysis and should not be mistaken for preregistered primary inference.

Limitations: verification-aware ML+Gate baseline not completed; full manual semantic review of all trajectories not independently completed; private traces and Gold Labels are excluded from this public source-code repository while redistribution and security review is pending; exact score reproduction needs those protected artifacts. No outcome was changed after test inspection.
