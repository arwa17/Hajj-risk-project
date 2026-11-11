# CPCS 683 — Controlled Hajj Risk Study Contract

**Status:** Implemented course project covering M4 (Single-Agent baseline), M5 (evaluation harness), and M6 (Multi-Agent experiment). The model configuration and shared decision threshold were frozen before held-out evaluation. The reported measurements are in `README.md` and `RESULTS_SUMMARY.md`; raw experimental evidence remains private.

## Source hierarchy
1. **Official:** `Project_Description_and_Milestones`, `Project grading rubric` — C4.1–C6.4.
2. **Approved research:** M2 Proposal pp. 2–7, M3 Related Work, Evaluation Plan — question, role decomposition, fixed comparison, 30 cases and seeds.
3. **Reference paper:** Samrgandi (2026), p. 426 for seven weights, thresholds, 17 predictors and p. 428 for reported model performance. Original raw-value risk encoding was not published; do **not** claim exact replication.
4. **Implementation choices:** This contract and `hajj_study/` modules, with passing acceptance tests. An implementation choice is not an official rubric requirement.

## Research comparison
- Only manipulated design factor: **agent-role and message organization**.
- Baseline: Gemini Generalist (one agent). Treatment: Crowd, Health, Environment specialists and one Gemini Coordinator (expert-team fan-out/fan-in).
- Identical source rows, shared ML predictions, gate and verification budget, frozen Gemini configuration, seeds `11,42,77`, metrics and test cases.
- The reported study uses the frozen 30-case held-out design and the fixed seeds `11`, `42`, and `77`.

## Ground truth and data protection
- Source: original synthetic Kaggle CSV, 10,000×30. SHA-256 `7c289edf5b3390b479c48ac5a493db566f6c46d4d587ae101b03d885bf948e8a`.
- Proxy from **clean** records: density `.25`, temperature `.20`, fatigue `.15`, stress `.15`, queue `.10`, emergency `.10`, distance `.05`; LOW `<1.8`, MEDIUM `1.8–2.4`, HIGH `>2.4`.
- Chosen reconstruction: categorical Low/Medium/High→1/2/3, numeric temperature & queue Min–Max to `[1,3]`, distance inverse scale `[3,1]`, emergency Yes→3/No→0. Expected class counts LOW=3337, MEDIUM=5755, HIGH=908; **not identical** to published counts.
- Source indices and dirty/clean records: 7,000 train / 1,000 validation / 2,000 test, stratified seed 42. ML trains on the 17 eligible predictors (explicit allowlist), never on seven label fields; preprocessing and SMOTENC fit only on train.
- Strictly private: original CSV, reconstructed source-row labels, clean references, corruption metadata, trained model, raw traces and API credentials pending dataset-license verification. Do not commit private gold or data to a public repo.

## Immutable case-suite decisions (G3)
- Test: 30 cases = 18 typical clean, 3 missing, 3 noisy, 3 conflicting, 3 borderline clean; 10 each class; 9 clean counterpart runs per seed/condition.
- Validation: 40 cases = 24 typical clean + 4 each missing/noisy/conflicting/borderline. Separate records and selection seeds.
- Three test corruptions and four validation corruptions span two domains. Deliberate stress-conditioned sampling is **not** an estimate of population incident prevalence.
- Conflicting readings alternate correct position in balanced, deterministic order; numeric types are identical. Agent input does not expose which is correct.
- Case IDs shuffled to avoid revealing category by sequence. All suite files are SHA-256 recorded privately.

## Information firewall (G4)
| Component | Allowed information | Not allowed |
|---|---|---|
| Generalist | 7 observed fields (clean/corrupted), identical frozen shared-ML prediction | proxy formula/weights/thresholds, reference label, hidden clean values, 17 raw ML features, metadata |
| Crowd specialist | density, queue, distance | health/environment evidence, ML-global prediction, labels |
| Health specialist | fatigue, stress, emergency | crowd/environment evidence, ML-global prediction, labels |
| Environment specialist | temperature | all other raw evidence, ML-global prediction, labels |
| Coordinator | 3 validated specialist reports, same global ML prediction as Generalist | raw 17 ML inputs, hidden reference/clean information |
| Shared gate | final report, its reported flags, integer `k`, verification state | scorer metadata, reference labels and risk-score formula |
| Scorer | private clean records, labels and actual traces | cannot alter inference or overwrite risk decisions |

- Initial corrupted observations have **no trusted corruption-quality flags**. Agents must infer flags from what they see. Missing is `null`, conflict is two unlabeled-as-clean readings, suspected noise is a silently modified value.
- Generalist and Coordinator final JSON: `case_id`, `proposed_risk_level`, `evidence_flags[{field,type,reason}]`, `supporting_evidence`. Specialists additionally report domain and `domain_risk_signal` and may flag only their own fields.
- Flag vocabulary: `missing`, `conflicting`, `suspected_noisy`. Severity priority: conflicting=3, missing=2, noisy=1. Tie priority: crowd > health > environment. False flags are **allowed and measured**; they are not auto-corrected from hidden metadata.
- Qualitative field ranges and LOW/MEDIUM/HIGH semantics are shared fairly. No proxy weights, thresholds or formula in prompts.

## Shared deterministic gate (G4)
1. Validate final JSON and `case_id`. A malformed provider response after the **same bounded retry policy** is FAILED, never ABSTAIN.
2. `sufficiency = 7 − number_of_distinct_flagged_fields`; this is only an operational completeness measure, not probability of correctness.
3. If `sufficiency >= k`, issue the **agent's** proposed risk level without re-scoring it.
4. Otherwise, choose the domain containing the strongest unresolved flag; tie crowd → health → environment; verify **one** complete domain maximum.
5. Replace that domain's observed values with clean values only; preserve observations in the other domains. Single: rerun Generalist. Multi: rerun selected specialist and Coordinator, preserve other reports.
6. Flags in unverified domains are inherited even if a rerun omits them. If post-verification sufficiency still `< k`, ABSTAIN; if malformed/failed, FAILED.
7. The gate never recomputes or substitutes the reference label. The verification oracle must not release any other domain or the gold label.

## M4 — Single-Agent baseline
- The implemented generalist receives the seven authorized observed inputs and the shared ML prediction; output validation and the common decision gate are implemented in `hajj_study/`.
- Source and unit tests are included. The actual experiment traces are protected, so this repository alone does not demonstrate that a formal classroom live presentation took place.

## M5 — Evaluation harness and baseline results
- Development used 40 cases, and the selected common sufficiency threshold was frozen at `k=7` before the held-out evaluation.
- The held-out test has 30 source cases evaluated with seeds `11`, `42`, and `77` (90 primary trials), plus 9 clean-reference source cases per seed (27 supplemental trials).
- The Single-Agent baseline issued 68/90 decisions, with 26/68 incorrect issued decisions. The scorer includes outcome, evidence-flag, verification, and trajectory measures; the corresponding private inputs are not part of this public source release.

## M6 — Multi-Agent controlled comparison and results
- The treatment uses Crowd, Health, and Environment specialists with one Coordinator, sharing the same cases, ML predictions, decision policy, and seed grid as the baseline.
- The Multi-Agent condition issued 71/90 decisions, with 39/71 incorrect issued decisions. Three-seed variation, calls, attempts, and elapsed times are reported in `README.md`; `RESULTS_SUMMARY.md` includes the descriptive paired bootstrap intervals.
- The API project and retry policy changed during execution, limiting direct operational-overhead comparisons. Some token-usage fields were unknown, so a verified total USD cost is not reported. The qualitative failure analysis is descriptive, not a causal attribution to a specific agent.
- Raw model traces, reconstructed gold labels, and the source CSV remain private until redistribution and security requirements are satisfied. The public source alone cannot independently reproduce the exact numerical scores.

## Interpretation
- The Single-Agent system achieved lower selective error than the Multi-Agent system in this held-out synthetic study. Neither condition achieved the 80% target coverage.
- The study does not validate safety for real Hajj operations. The experiment is documented as one completed course project, not an operational risk system.
