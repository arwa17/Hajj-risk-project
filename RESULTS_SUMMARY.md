# M5 and M6 — Evaluation Results

**Study:** CPCS683 Hajj Risk Study, with M4 Single-Agent baseline, M5 evaluation harness, and M6 controlled Multi-Agent comparison. **Data:** synthetic Hajj/Umrah crowd-management source.

**Design:** 30 source test cases × 3 fixed seeds per arm = 90 paired test executions, Single-Agent vs Multi-Agent, fixed gate `k=7`.

## M5/M6 frozen primary results — mean ± sample SD

**Primary cases:** 30 test cases × 3 seeds = **90 runs per condition**. The following values are **mean ± sample SD across three seed-level per-task averages**, not pooled percentages or 90 independent cases. A task is one primary Test case/seed run including its recorded attempts; Clean References are excluded from the per-task values. Token figures include only reported usage; missing usage was not imputed.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Coverage | 75.56% ± 1.92 pp | 78.89% ± 1.92 pp |
| Selective error | 38.27% ± 3.15 pp | 54.89% ± 3.14 pp |
| Unsupported escalation | 38.27% ± 3.15 pp | 54.89% ± 3.14 pp |
| Under-escalation | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp |
| Trajectory safe completion | 71.11% ± 1.92 pp | 56.67% ± 3.33 pp |
| Logical agent calls / case | 1.46 ± 0.07 | 4.99 ± 0.12 |
| HTTP attempts / case | 1.64 ± 0.11 | 5.77 ± 0.33 |
| Recorded input tokens / primary Test task | 743.00 ± 30.83 | 2,010.40 ± 4.21 |
| Recorded output + thinking tokens / primary Test task | 182.60 ± 30.82 | 517.54 ± 25.67 |
| Recorded total tokens / primary Test task | 925.60 ± 60.37 | 2,527.94 ± 27.17 |
| Standard-rate USD equivalent / primary Test task | $0.00067940 ± $0.00008561 | $0.00189698 ± $0.00006455 |
| Full wall-clock seconds / case | 11.90 ± 1.20 | 43.56 ± 3.41 |

| Final outcome, all 90 runs | Single-Agent | Multi-Agent |
|---|---:|---:|
| Correct ISSUED | 42 | 32 |
| Incorrect ISSUED | 26 | 39 |
| ABSTAIN | 22 | 19 |
| Terminal FAILED after recovery | 0 | 0 |
| Pooled selective error | 26/68 = 38.24% | 39/71 = 54.93% |

The matching-case analysis recorded **31 both correct**, **11 Single-only correct**, **1 Multi-only correct**, and **47 neither issued a correct decision** (the last category includes abstentions and is not 47 errors). The paired corruption-flag rate was **6/9 per seed** in both conditions. For the proposal's conditional disrupted-case detection and clean-decision recovery analysis, Single-Agent has **N/A** because the eligible denominator is zero; the published Multi-Agent summary reports **16.67% ± 28.87 pp across seeds**, with **1/5 eligible paired observations** succeeding. These conditional measures must not be conflated with all-case accuracy or the 6/9 field-flag rate.

## Recorded token usage and estimated USD cost


The following amounts are **standard paid-tier, uncached-rate equivalents calculated from recorded token usage**, **not verified Google charges**. The study used `gemini-3.5-flash-lite`. Published Standard rates are **$0.30 per 1 million input tokens** and **$2.50 per 1 million output tokens, including thinking tokens**. The calculation is:

`estimated_USD = (recorded_prompt_tokens × 0.30 + (recorded_candidate_tokens + recorded_thinking_tokens) × 2.50) / 1,000,000`

Prompt tokens are priced at the uncached rate because the historical records do not separately identify cached-input usage. Candidate and thinking tokens are each counted once.

| Recorded-usage cost equivalent (USD) | Single-Agent | Multi-Agent |
|---|---:|---:|
| Primary Test runs (90 per condition) | $0.061146 | $0.170728 |
| Clean-reference runs (27 per condition) | $0.017415 | $0.044885 |
| **All recorded primary + reference runs** | **$0.078561** | **$0.215613** |
| Per primary Test run (mean ± sample SD across seeds) | $0.00067940 ± $0.00008561 | $0.00189698 ± $0.00006455 |

**Incomplete usage metadata:** The cost computation did not impute unreported tokens. Missing usage occurred in **19/148** Single primary HTTP attempts, **4/40** Single clean-reference attempts, **81/519** Multi primary attempts, and **19/139** Multi clean-reference attempts. As a result, these numbers are **not complete costs or a verified bill**, and are **not necessarily lower bounds on actual charges**: missing token usage could increase the calculated equivalent; caching, free-tier usage, or billing arrangements could reduce actual charges. Differences in API project and retry policy also limit the fairness of a direct monetary-cost comparison.

**Source and provenance:** [Google Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing), Standard rates for Gemini 3.5 Flash-Lite, checked 11 November 2026. Per-task token and USD figures were calculated offline from the original V2 G6/G7 run records and reconciled with the frozen V2 scores (**12/12 raw-to-frozen checks passed**). The computation did not impute missing usage, make new API calls, or change model outputs or source code. Private case records and API journals are not published here.

**Exact per-task provenance:** The token and USD means and sample standard deviations above were recomputed from **original V2 G6/G7 run records**, not sanitized aggregates, with **12/12 raw-to-frozen checks PASS**. Local source tests: **178 run, 0 failures, 0 errors, 1 skipped**; offline demo PASS; **no Gemini calls**. These are recorded-usage quantities, not complete billed totals.

**Interpretation:** Neither condition reached the planned 80% coverage target. The API project/key change and later increase in the Multi-Agent HTTP 503 retry policy limit comparisons of cost, attempts, and latency. Both systems had zero terminal execution failures after documented recovery, but **26 Single** and **39 Multi** issued incorrect decisions. Three seed repetitions do not turn 30 source cases into 90 independent cases.

## Descriptive paired bootstrap

The 2,000-draw paired, stratified source-case bootstrap (21 clean and 9 altered cases) reported these **Multi minus Single** differences, as descriptive secondary analysis:

| Metric | Difference | 95% bootstrap interval |
|---|---:|---:|
| Coverage | +0.033333 | [-0.066667, +0.133333] |
| Selective error | +0.166943 | [+0.052817, +0.316667] |
| Trajectory safe completion | -0.144444 | [-0.266667, -0.044444] |

These are descriptive results from a limited, intentionally conditioned synthetic test set, not real operational validation. The paired bootstrap is a secondary analysis and should not be mistaken for preregistered primary inference.

Limitations: verification-aware ML+Gate baseline not completed; full manual semantic review of all trajectories not independently completed; private traces and Gold Labels are excluded from this public source-code repository while redistribution and security review is pending; exact score reproduction needs those protected artifacts. No outcome was changed after test inspection.
