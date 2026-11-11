# M5 and M6 — Evaluation Results

**Study:** CPCS683 Hajj Risk Study, with M4 Single-Agent baseline, M5 evaluation harness, and M6 controlled Multi-Agent comparison. **Data:** synthetic Hajj/Umrah crowd-management source.

**Design:** 30 source test cases × 3 fixed seeds (`11`, `42`, `77`) per arm = 90 primary executions per condition, Single-Agent vs Multi-Agent, fixed gate `k=7`. The 30 source cases are repeated across seeds; the 90 executions are **not** 90 independent source cases.

## Pooled primary-test counts and rates

The following percentages are **pooled across 90 executions**, not averages of seed-level percentages.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Issued decisions | 68/90 | 71/90 |
| Coverage (pooled) | 75.56% | 78.89% |
| Wrong among issued | 26/68 | 39/71 |
| Selective error (pooled) | 38.24% | 54.93% |

## Seed-wise results (mean ± sample SD)

The following values are **means and sample standard deviations across the three fixed seeds**, transcribed from the current [README M6 results](README.md#m6--multi-agent-experiment-and-results). They are not recalculated from private traces here. **pp** means percentage points.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Coverage | 75.56% ± 1.92 pp | 78.89% ± 1.92 pp |
| Selective error | 38.27% ± 3.15 pp | 54.89% ± 3.14 pp |
| Unsupported escalation | 38.27% ± 3.15 pp | 54.89% ± 3.14 pp |
| Under-escalation | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp |
| Trajectory safe completion | 71.11% ± 1.92 pp | 56.67% ± 3.33 pp |
| Logical agent calls / case | 1.46 ± 0.07 | 4.99 ± 0.12 |
| HTTP attempts / case | 1.64 ± 0.11 | 5.77 ± 0.33 |
| Full wall-clock seconds / case | 11.90 ± 1.20 | 43.56 ± 3.41 |

Pooled selective error and mean seed-wise selective error differ slightly because the numbers of issued decisions vary by seed. Do not substitute one for the other.

## Paired secondary analysis

The 2,000-draw paired, stratified source-case bootstrap (21 clean and 9 altered cases) reported these **Multi minus Single** differences, as descriptive secondary analysis:

| Metric | Difference | 95% bootstrap interval |
|---|---:|---:|
| Coverage | +0.033333 | [-0.066667, +0.133333] |
| Selective error | +0.166943 | [+0.052817, +0.316667] |
| Trajectory safe completion | -0.144444 | [-0.266667, -0.044444] |

These are descriptive results from a limited, intentionally conditioned synthetic test set, not real operational validation. The paired bootstrap is a secondary analysis and should not be mistaken for preregistered primary inference.

## Estimated API cost and evidence limits

The [README cost section](README.md#estimated-gemini-api-cost-in-usd-v2) reports **estimated USD per primary test run**, calculated from recorded token usage: approximately **$0.000679** for Single-Agent and **$0.001897** for Multi-Agent. These are **uncached standard-rate equivalents, not verified Google charges or complete actual costs**. Some HTTP attempts have missing usage metadata; API project/key and retry-policy changes also limit direct operational comparisons. No token-per-task mean ± SD is asserted here because it has not been independently verified from the protected V2 records.

## Limitations and interpretation

Neither condition met the **80% target coverage** (75.56% Single-Agent; 78.89% Multi-Agent), and the arms did not operate at equal observed coverage. Multi-Agent had higher observed selective error and lower trajectory safe completion in this held-out study; the hypothesis of improved safety was **not supported**. The Google API project/key and retry policy changed during Multi-Agent execution, limiting causal comparisons of operational overhead. The verification-aware ML+Gate baseline was not completed; full manual semantic review of all trajectories was not independently completed. Private traces and Gold Labels are excluded from this public source-code repository pending redistribution and security review; exact score reproduction requires those protected artifacts. No frozen outcome was changed after test inspection.
