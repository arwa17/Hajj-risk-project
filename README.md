# CPCS683 — Controlled Single-Agent vs Multi-Agent Hajj Risk Study

## Research question

Does specialist-agent organization improve reliability for simulated Hajj crowd-risk
assessment with noisy, missing, or conflicting observations, compared with a
single generalist using the same evidence, shared ML model, and decision gate?
The compared factor is **role/message organization** (one generalist versus three
specialists plus a coordinator). The task is advisory, not a real Hajj safety system.

## Project milestones: M4, M5, and M6

- **M4 — Single-Agent baseline:** the generalist, shared machine-learning predictions, JSON contract, and decision/verification gate were implemented. The code is included; a formal in-class live demo is not independently established by this public repository.
- **M5 — Evaluation harness:** 30 held-out cases, three fixed seeds, automated offline scoring, component/trajectory checks, and reported baseline variation. Individual source cases and labels are private.
- **M6 — Multi-Agent comparison:** three domain specialists and one coordinator were evaluated on the same case/seed grid. Reported outcomes include variation, API attempts, call counts, elapsed time, and failure analysis with stated limitations.

This repository presents a **single course project** covering all three milestones. Its published code and aggregate results are the delivery artifacts; protected experimental inputs and traces are not included.

## Design and experimental protocol

- **Data:** 10,000 simulated crowd-management records. Evaluation risk labels
  are reconstructed from a fixed, seven-variable proxy, **not real-world harm labels**.
- **Development:** 40 cases. An 80% target coverage was used when calibrating
  the shared decision gate; **k = 7** was frozen before Fresh Test evaluation.
- **Fresh Test:** 30 held-out cases (21 clean, 9 disrupted). There are also 9 paired
  clean-reference cases. Seeds **11, 42, 77** yield **90 primary runs and 27
  clean-reference runs per condition** (117 run keys per condition).
- **Baseline:** one Generalist with all crowd, health, and environment evidence.
- **Treatment:** Crowd, Health, Environment Specialists -> Coordinator, using
  structured JSON. At most one verification round per run.
- **Shared controls:** same test cases, seeds, ML predictions, frozen decision gate,
  nominal model and observed model version (`gemini-3.5-flash-lite`), and scorer.
  Generation configuration includes temperature 0, top-p 1, max output tokens 4096.
- **Outcomes:** ISSUE, ABSTAIN, or FAILED; a FAILED execution is never counted as
  a safe abstention. Clean references are excluded from the 90-run primary denominator.

## M5 and M6 — Frozen evaluation results

Values below are **mean ± sample SD across three seeds**. Percent signs indicate
percentage points for the SD. The 30 source cases are repeated, not 90 distinct cases.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Coverage (%) | 75.56 ± 1.92 | 78.89 ± 1.92 |
| Selective error (%) | 38.27 ± 3.15 | 54.89 ± 3.14 |
| Unsupported escalation (%) | 38.27 ± 3.15 | 54.89 ± 3.14 |
| Under-escalation (%) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Trajectory safe completion (%) | 71.11 ± 1.92 | 56.67 ± 3.33 |
| Logical agent calls / primary case | 1.46 ± 0.07 | 4.99 ± 0.12 |
| HTTP attempts / primary case | 1.64 ± 0.11 | 5.77 ± 0.33 |
| Wall-clock time (seconds / primary case) | 11.90 ± 1.20 | 43.56 ± 3.41 |

### Primary outcomes (90 runs per condition)

| Outcome | Single-Agent | Multi-Agent |
|---|---:|---:|
| Correct ISSUED | 42 | 32 |
| Incorrect ISSUED | 26 | 39 |
| ABSTAINED | 22 | 19 |
| Terminal FAILED after documented recovery | 0 | 0 |

Paired primary outcomes: both correct **31**, Single-only correct **11**,
Multi-only correct **1**, neither issued a correct decision **47**. The last
category includes abstentions and is **not** equivalent to 47 errors.

### Corruption and recovery

- Paired corruption flag rate: **6/9 = 66.67%** for **each seed and both conditions**.
  This is a component flagging measure, not proof of changed-decision detection.
- Disrupted-case detection and clean-decision recovery: **Single = N/A** because
  each eligible denominator is zero (seeds 11/42/77). **Multi = 16.67% ± 28.87
  percentage points** per seed; across the three seeds, only 1/5 eligible paired
  observations succeeded. N/A is **not** 0%.

### Observed failures and trajectory analysis

Both conditions had zero terminal execution failures after recovery. Every
incorrect issued decision was an over-escalation relative to the simulated proxy;
under-escalations were not observed. Errors without/after verification were
**20/6** for Single and **27/12** for Multi. Typical clean cases accounted for
23/28 incorrect issued decisions, and missing-observation cases for 3/6,
respectively. In Multi, 37 of 39 wrong final decisions began as overestimates;
the remaining two began with matching initial risk and ended incorrect. These
patterns are descriptive and cannot identify a single agent as the cause.

### Interpretation and threats to validity

Single had lower selective error and higher trajectory safe completion, with
lower observed operational overhead. Multi achieved slightly higher coverage,
but both mean coverages fell below the **80% target** and achieved coverage was
**not equal between arms**. Fixed k does not imply equal achieved coverage.
The measured difference does **not** establish a universal safety advantage.

The simulated labels are a proxy and the 30-case holdout is small. Three seed
repetitions do not create 90 independent source cases. The Google API project/key
changed part-way through the Multi execution, and the Multi HTTP 503 retry policy
was later increased from two to three attempts per role. These changes could
confound direct API attempt, latency, and cost comparisons. They were logged;
no prompts, task data, gold labels, gate values, or model versions were retuned
based on Fresh Test performance. Model stochasticity is still possible at
`temperature=0`. Monetary cost must not be claimed without a complete validated
usage/pricing audit.

## Repository and reproducibility

**Required Python:** 3.12, based on the experiment's recorded environment.
Use a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository's `hajj_study/` package contains the Single and Multi systems,
JSON validation, shared gate, frozen experiment checks, offline scoring, and
M6 fairness verification. `tests/` holds unit/integration tests. To discover
and run the local test suite, after reviewing tests for their dependencies:

```bash
python -m unittest discover -s tests/study -p 'test_*.py' -t tests/study
```

**Safety:** This command corrects discovery only. The verified R03 audit also used an isolated temporary directory, removed API credentials, and blocked network calls. Do not run these tests with live credentials.


**Verified local clean-checkout test (11 November 2026):** On macOS ARM64,
Python 3.12.5, a newly created virtual environment with the pinned requirements
passed `pip check`; the original public test suite completed **177 tests, 0
failures, 0 errors, 1 skipped** with a network guard and no API credentials.
The skipped integration test requires protected CSV/split/model inputs.
This verifies source code and test execution in that environment, not the
regeneration of exact private evaluation results. See `EVIDENCE_M4_M5_M6.md`.

The offline paired scorer entry point is `python -m hajj_study.m6_results`.
Its options include `--g5-dir`, `--g6-dir`, `--g7-dir`, and `--output`, all of
which must point to an authorized **private** experiment workspace. The scorer
verifies the frozen G5 provenance; it requires the original G3 sources at their
recorded manifest paths and must not silently substitute or regenerate labels.

The published repository **must not contain** API keys, raw CSV data, gold
labels, Test inputs, private model outputs, HTTP request/response journals,
private test traces, or sealed private score files. The real-run evidence is
retained in a protected private archive; the aggregate results above come from
that sealed evaluation. Public code alone cannot reproduce the exact held-out
scores without authorized access to the frozen private inputs. Before any
redistribution of source data, its license and terms must be checked.

## Reproducible offline end-to-end demonstration

Run `python -m examples.offline_demo` from the repository root. This
uses **scripted synthetic provider outputs**, no Gemini, no API key and no
protected data. It exercises the **real** Single-Agent and Multi-Agent
validation, decision gate, bounded verification, and abstention logic.
Its three paths are direct ISSUE, VERIFY → ISSUE, and VERIFY → ABSTAIN.
**Local candidate verification (11 November 2026):** With the macOS test temporary directory set to a non-symlinked private audit path, the candidate completed **178 tests (0 failures, 0 errors, 1 skipped)** and all scripted end-to-end scenarios (`DEMO_ASSERTIONS: PASS`). These synthetic runs do **not** reproduce the frozen Gemini evaluation.

It is a functional source-code demonstration, **not** a replication of
the frozen empirical findings. See `EVIDENCE_M4_M5_M6.md`.

## API trace verification limitation

In the 11 November 2026 read-only provenance review, 884 of 894 recorded
attempt slots were linked exactly to their WAL entries. Six expected
slots were missing and four differed from their same-slot WAL entries;
some corresponding requests/responses were found in other slots.
Complete one-to-one API attempt provenance remains **unproven**.
Neither this caveat nor the offline demo changes the frozen test outputs.

## Responsible use

This software is a controlled class-project experiment. Do not interpret its
risk levels as validated live crowd-safety advice, deploy it on real Hajj
infrastructure, or let it issue automated operational actions. Credentials
must be supplied outside the repository and kept in an appropriate secret store.


## Research Credits and Dataset Attribution

### Original research

Najwa Samrgandi (2026), *Multi-Agent Crowd-Risk Prioritization for Large-Scale Events: A Hybrid ML–Agent Approach*, **International Journal of Advanced Computer Science and Applications (IJACSA)**, 17(7).

Original research paper: https://doi.org/10.14569/IJACSA.2026.0170738

We gratefully acknowledge Dr. Najwa Samrgandi's published research as a scientific foundation and reference for this course project. This repository contains an independent, controlled Single-Agent versus Multi-Agent comparison with evidence verification, abstention, and evaluation under altered observations. The project's implementation, evaluations and results are not the original paper's experiments or findings.

### Source dataset

**Hajj and Umrah Crowd Management Dataset**, Kaggle user **ziya07**.

Original dataset: https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset/data

The source is a synthetic crowd-management dataset with 10,000 rows. Attribution to the paper does not grant rights to redistribute the separate Kaggle dataset. Before adding the full source CSV to this public repository, confirm the dataset's own license and redistribution terms.

### Reproducibility and study limitations

- The evaluation uses 30 distinct test cases, three seeds (11, 42, 77), and a fixed evidence gate at `k=7`.
- This code release contains no real Gemini API keys, private prompt/response payloads, or raw test fixtures. Those inputs are **not** available in this source-only publication; reproducing exact frozen scores requires the protected artifacts.
- These comparisons were conducted on synthetic data and do not establish real-world Hajj safety performance.
- The Multi-Agent arm did not improve the measured selective error compared with Single-Agent in the frozen test results.
- The independently defined verification-aware `ML+Gate` baseline remains a limitation; do not mistake an exploratory ML-only wrapper for that baseline.
- See `RESULTS_SUMMARY.md` for descriptive frozen-test figures.
