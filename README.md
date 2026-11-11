# CPCS683 — Controlled Single-Agent vs Multi-Agent Hajj Risk Study

**Source code:** https://github.com/arwa17/Hajj-risk-project  
**Course milestones:** M4 baseline, M5 evaluation harness, and M6 controlled comparison  
**Type of system:** Research prototype for synthetic Hajj crowd-risk assessment; **not** a deployed safety system.

## Start here — follow the project

| Step | What was implemented | Where to inspect | How to verify |
|---|---|---|---|
| **M4: Baseline** | Single Gemini generalist, structured reports, shared decision gate, one-domain verification | `hajj_study/generalist.py`, `gate.py`, `verify.py`, `schemas.py` | Run the safe offline demo in [Quick start](#quick-start-no-api-key-required) |
| **M5: Evaluation** | Private golden set, automated scorer, seed-wise metrics, component and trajectory diagnostics | `cases.py`, `score_runs.py`, `diagnose_m5.py`, `tests/study/` | See [M5](#m5--evaluation-harness-and-single-agent-results) and the private-data requirements |
| **M6: Comparison** | Three specialist agents and one coordinator; matched-case/seed experiment; cost, latency, and failure analysis | `specialists.py`, `coordinator.py`, `multi_agent.py`, `g7.py`, `m6_results.py` | See [M6](#m6--multi-agent-experiment-and-results) |

**Important distinction:** The repository publicly reproduces the **code tests and a scripted offline end-to-end demonstration**. Recomputing the **original Gemini experiment's numerical scores** requires the protected frozen test inputs and raw traces, which are not distributed publicly.

## 1. Research question and experimental design

**Question.** Does organizing a simulated crowd-risk decision task across three specialists and a coordinator improve decision reliability compared with a single generalist when observations may be missing, noisy, or conflicting?

**Hypothesis.** Specialist decomposition might improve detection of corrupted evidence and reduce incorrect or unsupported escalations, but may increase agent calls and latency. The hypothesis is **tested**, not assumed true.

**Manipulated factor:** Role/message organization. The recorded Gemini model was `gemini-3.5-flash-lite` (requested and observed version), with temperature `0`, top-p `1`, maximum output tokens `4096`, and fixed test seeds `11`, `42`, and `77`. Model outputs may still vary at temperature zero.

- **Single-Agent:** One Gemini generalist sees all seven permitted observations and the shared ML prediction.
- **Multi-Agent:** Crowd, Health, and Environment specialists see only their authorized observations. One coordinator receives their structured reports and the same shared ML prediction.
- **Held constant by design:** Source cases, three seeds (11, 42, 77), ML predictions, evidence threshold (`k=7`), reference labels, scoring code, and maximum one verification round.

The agents produce a LOW, MEDIUM, or HIGH risk proposal. A deterministic gate checks the report and decides **ISSUE**, **VERIFY** (once), or **ABSTAIN**. A technical failure is **FAILED**, never a safe abstention. Both systems are advisory only.

### Architecture and decision flow

```text
Case observations + shared frozen ML prediction
                   |
      +------------+-------------+
      |                          |
      v                          v
 SINGLE-AGENT              MULTI-AGENT
 Generalist               Crowd specialist
 (all 7 fields)           Health specialist
                         Environment specialist
                                |
                                v
                          Coordinator
      |                          |
      +------------+-------------+
                   v
         Validated JSON risk report
                   |
         Shared decision gate, k=7
                   |
          +--------+--------+
          |                 |
        ISSUE         Insufficient evidence
                            |
                     VERIFY one domain
                            |
                 Re-run applicable agent(s)
                            |
                    ISSUE or ABSTAIN
```

The specialists are separate model roles, called sequentially to respect rate pacing; the coordinator is a separate model call. A Multi-Agent case takes four logical calls initially, and two additional calls if verification occurs. A Single-Agent case takes one initial call and one additional call if verification occurs.

## 2. Quick start (no API key required)

Use **Python 3.12**. The following commands run from the repository root. Installation needs network access to download Python packages; the demo and offline tests do **not** call Gemini.

```bash
# Download the public source
git clone https://github.com/arwa17/Hajj-risk-project.git
cd Hajj-risk-project

# Isolated Python environment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

# Safe scripted demonstration: no keys, no Gemini network requests
python -B -m examples.offline_demo
```

**Expected offline demo:** Three scenarios — `clean`, `one_missing`, `two_missing` — exercise direct ISSUE, VERIFY → ISSUE, and VERIFY → ABSTAIN in both architectures. Successful output includes `DEMO_ASSERTIONS: PASS`, `PROVIDER: SCRIPTED_FIXTURE_ONLY`, `GEMINI_CALLS: ZERO`, and `FROZEN_EVALUATION_REPRODUCED: NO`.

Run the project tests separately:

```bash
# macOS: use a non-symlinked temporary directory for the protected WAL tests
mkdir -p "$HOME/.hajj-risk-test-tmp"
chmod 700 "$HOME/.hajj-risk-test-tmp"

# No live API credentials are needed
TMPDIR="$HOME/.hajj-risk-test-tmp" \
  env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY \
  python -B -m unittest discover -s tests/study -p 'test_*.py' -t tests/study
```

**Observed local verification (11 November 2026):** The reviewed delivery candidate ran **178 tests, 0 failures, 0 errors, 1 skipped** on macOS ARM64 with Python 3.12.5 and the pinned packages. The single skipped test requires private CSV, split and prediction files. All scripted offline demo assertions passed. This is a recorded local result, not a guarantee of identical behavior on every computer. If the temporary directory is a symlink, the attempt-journal safety checks intentionally refuse it; do **not** disable that protection.

### Code map

| File(s) | Purpose |
|---|---|
| `audit_data.py`, `proxy.py`, `split_data.py` | Validate original synthetic data; create deterministic proxy labels and a train/validation/test split |
| `train_model.py`, `feature_contract.py` | Train shared XGBoost model on train records only; save predictions and model hashes privately |
| `cases.py`, `corrupt.py` | Deterministic development/test case generation, disruptions, clean references and checksums |
| `generalist.py`, `prompts.py`, `schemas.py` | M4 agent interface, prompts and strict JSON validation |
| `gate.py`, `verify.py` | Shared evidence gate, bounded verification, abstention |
| `g5.py`, `g6.py`, `score_runs.py` | Frozen model/gate controls, Single-Agent recorded evaluation, M5 scorer |
| `specialists.py`, `coordinator.py`, `multi_agent.py` | M6 agent roles and coordinated decision flow |
| `m6_fairness.py`, `g7.py`, `m6_results.py` | Controlled-treatment checks, recorded Multi-Agent evaluation, M6 paired scoring |
| `provider.py`, `attempt_journal.py` | Gemini provider handling, bounded retries and private request/response journaling |
| `examples/offline_demo.py`, `tests/study/` | Public safe end-to-end demonstration and offline tests |

## 3. Data preparation and experimental controls

**Original source:** [Hajj and Umrah Crowd Management Dataset](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset/data), 10,000 synthetic records. The source has no directly supplied operational-risk target, so this study defines a fixed **proxy** risk label from seven fields: crowd density (0.25), temperature (0.20), fatigue (0.15), stress (0.15), queue time (0.10), emergency event (0.10), and distance between people (0.05). Thresholds are LOW below 1.8; MEDIUM from 1.8 through 2.4; HIGH above 2.4. These are **proxy labels, not observed incident outcomes**.

The source is stratified into **7,000 train / 1,000 validation / 2,000 test** source records (`split_data.py`, seed 42). The shared XGBoost + SMOTENC training path fits preprocessing/resampling on training data only. It excludes the seven label-construction fields from its allowed 17 predictors. Both agent conditions consume the **same frozen model outputs**, not separate ML models.

The study selected **40 development cases** before held-out testing. The **30 distinct held-out test cases** comprise:

| Case type | Test cases |
|---|---:|
| Typical clean | 18 |
| Missing observations | 3 |
| Noisy observations | 3 |
| Conflicting observations | 3 |
| Borderline clean | 3 |
| **Total** | **30** |

The test suite has 10 LOW, 10 MEDIUM, and 10 HIGH proxy labels. An additional **nine paired clean-reference cases** support corruption/recovery metrics. Across the three fixed seeds (11, 42, 77), each condition has **90 primary runs** (30 × 3) and **27 reference runs** (9 × 3). The 27 references are *excluded* from the primary 90-run denominator.

Case generation is deterministic, and original inputs, labels, private metadata, model, and frozen manifests are SHA-256 recorded in the protected study workspace. No evaluation gold label or clean reference is passed to an agent prompt; verification reveals only the selected domain's clean observations.

**Rebuilding private inputs:** `audit_data.py`, `split_data.py`, `train_model.py` and `cases.py` expose CLI entry points for authorized reconstruction. Re-creating new files is **not equivalent to reproducing the frozen experiment**. Never replace archived frozen inputs, rerun Gemini, or publish original CSV/labels without appropriate permission and license review.

## 4. M4 — Working Single-Agent baseline

The baseline is implemented by `hajj_study.generalist.execute_single_case`. It:

1. Validates the case (`case_id`, seven observations, frozen ML prediction).
2. Requests a schema-constrained report from the generalist.
3. Uses `SharedDecisionGate` to count sufficiently supported, unflagged fields (`7 − number_of_distinct_flagged_fields`).
4. Issues the proposed risk if evidence sufficiency reaches the frozen `k=7` threshold.
5. Otherwise verifies **one** affected domain, re-requests the report, and either issues or abstains.
6. Records exceptions as `FAILED`, not `ABSTAIN`.

**Runnable evidence:** `python -B -m examples.offline_demo` executes the real baseline and gate using a scripted provider, and `tests/study/test_generalist_mock.py` verifies issuance, verification, abstention, and provider-failure handling. These tests do not pretend to be fresh Gemini results.

## 5. M5 — Evaluation harness and Single-Agent results

### What was implemented

- **Golden set:** 30 held-out source cases with proxy ground truth, difficult-case types, and separate 40-case development set (`cases.py`). Gold labels and source-row metadata are **private**, not given to inference agents.
- **Automated scoring:** `score_runs.py` validates the run grid, model version, recorded reports, outcomes, case provenance, and metrics. It does not call Gemini.
- **End-to-end scoring:** Correct decisions, incorrect decisions, abstentions, failures and trajectory safe completion.
- **Component scoring:** Correct field/type corruption flags, false flags on clean cases, paired corruption-flag rate, verification use/budget, and structured-call validation.
- **Repeated measurements:** Three fixed seeds (11, 42, 77); seed-wise measurements summarized as **mean ± sample standard deviation**. The same 30 cases recur across seeds (not 90 independent cases).

### How M5 scoring is run with authorized frozen private artifacts

The public repository contains the scorer, **not** the protected original `g5`/`g6` files. If the assessor has authorized access to the original archived directories (with their unchanged manifests, source paths, gold labels and checkpoints), the scorer can be run **offline**:

```bash
# Replace these placeholders with authorized archived directories.
G5_DIR="/secure/path/to/frozen/g5"
G6_DIR="/secure/path/to/frozen/g6"

python -B -m hajj_study.score_runs \
  --g5-dir "$G5_DIR" --g6-dir "$G6_DIR" \
  --output "$G6_DIR/m5_scores_review.json"
```

This command **writes a new JSON result inside the supplied private directory**. Use a separately authorized writable private workspace, not a sealed original, if preservation is required. It makes **no Gemini calls**. It will refuse missing or inconsistent provenance; exact score regeneration cannot be verified from the public repository alone.

### Observed M5 baseline results

**Single-Agent, 90 primary test executions:** 42 correct ISSUED; 26 incorrect ISSUED; 22 ABSTAIN; **0 terminal FAILED after documented recovery**. Coverage is **68/90 = 75.56%**. Pooled selective error is **26/68 = 38.24%**. The mean of the *three per-seed selective error percentages* is **38.27% ± 3.15 percentage points**; it differs slightly from the pooled ratio because seeds have different issuance denominators.

| Metric (mean ± sample SD across seeds) | Single-Agent |
|---|---:|
| Coverage | 75.56% ± 1.92 pp |
| Selective error | 38.27% ± 3.15 pp |
| Unsupported escalation | 38.27% ± 3.15 pp |
| Under-escalation | 0.00% ± 0.00 pp |
| Trajectory safe completion | 71.11% ± 1.92 pp |

Paired corruption flag rate was **6/9 (66.67%) per seed**, but the proposal's *disrupted-case detection* and *clean-decision recovery* are **N/A** for Single-Agent because the eligible raw-decision-change denominator was zero. N/A **must not** be reported as zero performance.

## 6. M6 — Multi-Agent experiment and results

### Treatment implementation and comparison protocol

`specialists.py` contains domain-isolated specialist calls, `coordinator.py` combines validated specialist reports and the shared ML prediction, and `multi_agent.py` executes the bounded four-role workflow. `g7.py` records and resumes the private real-run experiment; `m6_fairness.py` checks shared controls and case pairing; `m6_results.py` performs paired scoring and operational comparisons.

The manipulated factor is **one generalist vs three specialist roles + one coordinator**. Both conditions used the **same 30 cases, three seeds, frozen common ML predictions, decision gate `k=7`, one-domain verification budget, and scorer**. The Multi-Agent condition has four initial logical model calls per case and two more after verification.

**Execution caveat:** The Google API project/key changed during part of the Multi-Agent experiment, and the Multi-Agent HTTP 503 retry limit was later raised from two to three attempts per role. Thus, operational overhead is **observed**, but API attempts, costs and latency are not perfectly controlled against the Single-Agent runs. No frozen outcome, gold label, prompt, gate or model version was retuned based on the held-out results.

### Reported results

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

### Estimated Gemini API cost in USD (V2)

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

### Systematic failure analysis

- **Terminal software execution failures:** 0 in the finalized run grids after documented recovery. This does not mean decisions were always correct.
- **Incorrect issued risk decisions:** 26 Single and 39 Multi; all were **over-escalations relative to the synthetic proxy**. **No under-escalations** were observed.
- **Stage of incorrect decisions:** Single 20 without verification and 6 after verification; Multi 27 without verification and 12 after verification.
- **Case type:** Typical clean cases contributed 23 Single errors and 28 Multi errors. Missing-observation cases contributed 3 and 6, respectively.
- **Coordinator/specialist attribution:** Of Multi's 39 wrong final decisions, 37 began as overestimates and two started at matching initial risk but ended wrong. The traces support these descriptions, **not** a definitive causal claim about one particular agent.

The frozen hypothesis was **not supported**: Multi-Agent coverage was slightly higher, but its selective error, safe-completion rate, call count and elapsed time were worse. Neither arm achieved the target **80% coverage**, and the achieved coverage differed between arms; this is **not** a fixed-equal-coverage demonstration.

### How M6 scoring is run with authorized frozen private artifacts

```bash
G5_DIR="/secure/path/to/frozen/g5"
G6_DIR="/secure/path/to/frozen/g6"
G7_DIR="/secure/path/to/frozen/g7"

# Fairness/provenance check (read-only, no Gemini)
python -B -m hajj_study.m6_fairness \
  --g5-dir "$G5_DIR" --g6-dir "$G6_DIR"

# Recompute documented comparison from archived real-run traces (no Gemini)
python -B -m hajj_study.m6_results \
  --g5-dir "$G5_DIR" --g6-dir "$G6_DIR" --g7-dir "$G7_DIR" \
  --output "$G7_DIR/m6_scores_review.json"
```

Like M5, the second command writes within the designated private output directory. Do not run it against a sealed archive if a new file is prohibited. **Do not run `g6.py run`, `g7.py run`, or G5 live smoke** as a method of checking these frozen results: those are provider-execution paths, not safe offline reproducibility commands.

The aggregate result tables are also available in [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## 7. Reproducibility — exactly what is and is not verified

| Level | Status and evidence |
|---|---|
| Install pinned Python packages | **PASS locally** with Python 3.12.5 on macOS ARM64; `pip check` found no broken requirements |
| Offline unit/integration tests | **PASS locally:** 178 run, 0 failures, 0 errors, 1 skipped (private fixtures unavailable) |
| Offline end-to-end system flow | **PASS locally:** scripted ISSUE, VERIFY → ISSUE, VERIFY → ABSTAIN for both agent architectures; no Gemini |
| Offline scorer implementation | Included, covered by mock/offline tests; full replay requires private frozen files |
| Independent reproduction of the original numerical Gemini results | **NOT PROVEN from the public repository alone**; original sealed inputs, checkpoints and traces are private |
| Exact journal linkage for every API attempt | **NOT PROVEN**: read-only audit found 884 of 894 recorded attempt slots matched exactly, with 6 missing same-slot WAL entries and 4 same-slot mismatches; some payloads were located in other WAL slots |

The read-only WAL review found no differing content among duplicated journal records, but full provider-invocation coverage cannot be claimed. The published repository should contain no API credentials, source CSV, gold labels, private prompts, raw response bodies, checkpoints, model binaries or private experiment artifacts.

## 8. Interpretation, limitations and responsible use

This is a **small, deliberately conditioned synthetic study**, not a real Hajj crowd-risk validation. Its labels are an explicitly reconstructed proxy, not actual incident outcomes. Three seeds repeat 30 source cases; they do not produce 90 independent samples. The target 80% coverage was not reached, and the arms did not operate at equal observed coverage. The Google API project/retry-policy changes limit causal interpretation of call, latency, and cost comparisons. The private trace audit and incomplete usage fields prevent claims of complete per-attempt provenance or exact USD cost. The verification-aware `ML+Gate` comparator was not fully completed; do not describe it as a finished benchmark.

Do not use the system to control real crowds, issue autonomous safety actions, or claim human-safety accuracy. Check the source dataset's terms before redistributing any original data. Keep credentials and private evaluation material outside GitHub.

## 9. Source attribution

- **Research foundation:** N. Samrgandi (2026), *Multi-Agent Crowd-Risk Prioritization for Large-Scale Events: A Hybrid ML–Agent Approach*, International Journal of Advanced Computer Science and Applications, 17(7). https://doi.org/10.14569/IJACSA.2026.0170738
- **Source data:** *Hajj and Umrah Crowd Management Dataset*, Kaggle, ziya07. https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset/data

This repository is an independent course experiment building on the cited research; it does not claim to reproduce the source paper's exact experiments. Attribution does **not** establish the separate dataset's redistribution license.