# M4–M6 evidence, reproducibility, and limitations

**Project:** CPCS683 controlled Single-Agent vs Multi-Agent Hajj risk study.
**Scope:** Source-code delivery and protected real-run evidence. This file does not certify a classroom demonstration or claim that private data are public.
**Source version checked:** public commit `efa12da0615a418b3673609b48e43d00d2133911` (52 tracked files).
**Reference requirements:** Project grading rubric, C4.1–C6.4; Project Description and Milestones, M4–M6.

## How to run the public release safely

From the repository root on **Python 3.12**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.offline_demo
python -m unittest discover -s tests/study -p 'test_*.py' -t tests/study
```

The `examples.offline_demo` runner uses **fixed scripted synthetic responses** with the **actual Single-Agent, Multi-Agent, validation, verification, and gate functions**. It demonstrates three paths: direct ISSUE, VERIFY → ISSUE, and VERIFY → ABSTAIN. It **does not use Gemini** and **does not reproduce the held-out research evaluation or its accuracy**. Real Gemini execution requires separately authorized/private experiment artifacts and live API access; do not execute it solely to satisfy this checklist.

### Clean-checkout verification performed on 11 November 2026

- Clean public checkout at the commit above, in a separate Python 3.12.5 environment on macOS ARM64.
- All six direct pinned dependencies from `requirements.txt` installed at the specified versions; `pip check`: **No broken requirements found**.
- Published suite before adding this demo: **177 tests run, 0 failures, 0 errors, 1 skipped**; `CLEAN_REPRO_STATUS: PASS`, clean worktree.
- Skipped integration setup: `tests/study/test_cases.py`, because original protected CSV, splits, and ML predictions were unavailable in the source-only checkout.
- **Independent local delivery-candidate validation (11 November 2026):** **178 tests run, 0 failures, 0 errors, 1 skipped**, `CANDIDATE_TEST_STATUS: PASS`. The scripted offline end-to-end demo passed all three scenarios (`DEMO_ASSERTIONS: PASS`); network was denied and no Gemini requests were made. The initial candidate run encountered 11 temporary-directory symlink protection errors; rerunning with a real, non-symlinked `TMPDIR` passed without modifying the journal protections or frozen code.
- This is **code/test reproducibility**, not an independent reproduction of the exact 30-case LLM outputs or frozen scores.

## Evidence by grading criterion

| Requirement | Evidence and interpretation | Outstanding limitation |
|---|---|---|
| **C4.1** Single-Agent end-to-end function | `hajj_study/generalist.py` + `examples/offline_demo.py`; generated synthetic responses exercise the actual pipeline and policy. | A formal in-class live demonstration is not proved by code or repo. Slides/presentation are outside the user's requested scope. |
| **C4.2** Serious fair baseline | Generalist uses the same observations, frozen ML prediction, JSON contract, verification budget, sufficiency gate, nominal/observed model, and seed grid as treatment; see `CONTRACT.md`. | The observed coverage differs between arms. The 503 retry-policy change affects operational comparisons. |
| **C4.3** Runnable repository/README | Pinned Python requirements, command above, source, and offline tests. Clean clone was successfully tested independently as described above. | Proprietary/protected datasets and Gemini calls are not publicly reproducible from this source-only release. |
| **C5.1** Golden data ≥20 with defensible labels | Protected G3 manifest: 30 held-out source cases (18 typical clean, 3 missing, 3 noisy, 3 conflicting, 3 borderline clean); labels from unaltered synthetic source-row risk proxy, 10 cases per class; source split before selecting held-out cases; private hashes/manifests. | Synthetic proxy is *not* observed real-world incident ground truth. Original labels and raw rows must remain protected; third parties cannot independently verify them from this repository alone. |
| **C5.2** Reproducible automated scoring | `hajj_study/score_runs.py`, `hajj_study/m6_results.py`, schema/contracts, recorded manifest verification, frozen gate k=7, SHA checks, and offline unit tests. No extra provider calls are needed to score frozen traces. | Exact published scores cannot be recomputed without the original protected frozen G3/G5/G6/G7 inputs. Public unit tests do not establish that protected-data rerun. |
| **C5.3** End-to-end + component/trajectory | Coverage, selective error, unsupported escalation, under-escalation, verification, corruption-flag rate, conditional recovery, and trajectory safe completion, defined and scored in code. | Some rare component denominators are zero (correctly reported N/A); no claim of every trajectory manually semantically annotated. |
| **C5.4** Three-seed baseline variation | Single scores for seeds **11, 42, 77**, mean and sample SD in README (e.g., coverage 75.56 ± 1.92 percentage points and selective error 38.27 ± 3.15). | Three repetitions of 30 source cases are not 90 independent source samples. |
| **C6.1** Treatment and manipulated factor | Three separate Crowd/Health/Environment specialist roles plus Coordinator; structured fan-out/fan-in, domain separation, conditional one-domain verification, 4 or 6 logical calls; files `specialists.py`, `coordinator.py`, `multi_agent.py`. | This is a simulated advisory decision pipeline, not field-tested crowd safety. |
| **C6.2** Controlled comparison | Same 30 held-out source cases and seeds 11/42/77; 90 paired primary evaluations plus 27 clean-reference supplementary runs **per condition**. One frozen gate and shared ML predictions. | API project/key and one retry policy changed during Multi execution; do not treat latency/call-cost comparisons as perfectly controlled. |
| **C6.3** Variance, cost, latency | README reports means ± sample SD for both arms: logical calls/case **1.46 ± 0.07** vs **4.99 ± 0.12**; HTTP attempts/case **1.64 ± 0.11** vs **5.77 ± 0.33**; wall seconds/case **11.90 ± 1.20** vs **43.56 ± 3.41**. Token usage is logged but partially unknown. These are measured *operational cost proxies*, not USD fees. | A verified monetary invoice/fully observed token-cost total is unavailable and must not be fabricated. Retry policy and provider environment may confound comparison. |
| **C6.4** Failure analysis | Public README describes 26 incorrect issued Single decisions, 39 Multi, correct-issued 42 vs 32, 0 terminal execution FAIL after recovery, over-escalation patterns, and verification/non-verification splits. Failure and abstention are separate outcomes. | Descriptive patterns do not isolate which specialist caused an error, and not all trajectories have independent manual semantic review. |

## Frozen primary outcomes (not synthetic-demo results)

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Issued | 68/90 | 71/90 |
| Correct issued | 42/90 | 32/90 |
| Incorrect issued | 26/90 | 39/90 |
| Abstained | 22/90 | 19/90 |
| Terminal execution FAILED, after documented recovery | 0 | 0 |
| Coverage, mean ± sample SD across seeds | 75.56% ± 1.92 pp | 78.89% ± 1.92 pp |
| Selective error, mean ± sample SD across seeds | 38.27% ± 3.15 pp | 54.89% ± 3.14 pp |
| Trajectory safe completion, mean ± sample SD | 71.11% ± 1.92 pp | 56.67% ± 3.33 pp |

**Interpretation:** Multi had slightly higher coverage but substantially higher selective error in this 30-case synthetic holdout; neither met the 80% target. The selected metric must not be changed after seeing these results. These are not live Hajj safety outcomes.

## Protected-data repeatability: what is and is not provided

The protected original artifacts must stay out of the public repository: original CSV, train/validation/test splits, gold, corruption manifest, clean references, frozen ML model, G5/G6/G7 raw traces, API request/response journals, and credentials. Private manifests and SHA-256 checks can be reviewed in the authorized environment. The offline scorer entrypoint is:

```bash
python -m hajj_study.m6_results --g5-dir PRIVATE_G5_DIR --g6-dir PRIVATE_G6_DIR --g7-dir PRIVATE_G7_DIR --output NEW_PRIVATE_OUTPUT_PATH
```

This is a **documentation-only command example**, not an instruction to rerun or overwrite sealed files now. Use only if an authorized private-source replay is separately approved, with an output path outside frozen originals. **Never substitute fresh synthetic cases for the frozen test or silently invoke Gemini.**

## API trace provenance audit, 11 November 2026

A read-only audit of the protected workspace inspected **894 expected attempt slots** per identical candidate pair. **884** matched exactly. Of the remaining **10**, **6** had no WAL at the expected slot (4 requests/responses found in different slots, 2 still lacked such linkage); **4** had same-slot metadata mismatches, although corresponding responses were found in other WAL slots. The 13 conflicting historical expectation variants each had at least one exact WAL match. Duplicate WAL copies had no divergent content. This does **not** establish complete one-to-one attempt provenance, and an exact full universe of provider invocations is **not proven**. The same pair was inspected four times and must **not** be counted as four independent experiments.

**Status:** Code test reproducibility PASS; full public exact-score reproducibility NOT PROVEN; exact API trace coverage NOT PROVEN. These caveats do not, by themselves, overturn the recorded frozen outcomes. No API calls or experiments were repeated to prepare this note.

## Security and responsible use

- Never commit API keys, raw labels, source CSV, private logs, raw Gemini prompts/responses, clean references, model artifacts, or protected test rows.
- Synthetic offline demo contains no real cases, keys, source index IDs, private prompt/response traces, or real model responses.
- Respect the Kaggle dataset's independent license/redistribution terms; attribution is not redistribution permission.
- The study is academic and advisory only, not real crowd-control guidance. No automatic actuation.
- Check locally staged release candidates before approving any new GitHub commit; a successfully generated candidate is **not** an authorization to publish.