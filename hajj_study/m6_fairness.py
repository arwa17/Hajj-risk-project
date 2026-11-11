"""M6.2 read-only frozen-study parity and paired-key preflight.

This module does not call Gemini, expose Test gold, write private artifacts,
change G5/G6, or claim that G7 treatment results have been run. It produces a
strict M6.3 execution contract, checked again on each future G7 run/resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hajj_study import g5, g6
from hajj_study.schemas import validate_case


class FairnessError(ValueError):
    """Stop a treatment run instead of silently accepting non-matched evidence."""


SHARED_MODULES = ('prompts', 'schemas', 'provider', 'gate', 'verify',
                  'feature_contract')
TREATMENT_MODULES = ('specialists', 'coordinator', 'multi_agent')
TOPOLOGY = ('crowd', 'health', 'environment', 'coordinator')


def _fingerprint(data):
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _expected_cases(test, references, seeds):
    """Create exactly the key grid in G6 without importing scoring labels."""
    if list(seeds) != [11, 42, 77]:
        raise FairnessError('Seed policy differs from frozen M5 study')
    if len(test) != 30 or len(references) != 9:
        raise FairnessError('Expected 30 Test and 9 paired clean-reference cases')
    validated_test = [validate_case(row) for row in test]
    validated_refs = [validate_case(row) for row in references]
    t = {row['case_id']: row for row in validated_test}
    r = {row['case_id']: row for row in validated_refs}
    if len(t) != 30 or len(r) != 9 or not set(r).issubset(t):
        raise FairnessError('Repeated/unmatched Test or clean-reference IDs')
    if any(r[cid]['ml_prediction'] != t[cid]['ml_prediction'] for cid in r):
        raise FairnessError('Clean counterpart changed the shared ML prediction')
    result = {}
    for stage, cases in (('test', validated_test), ('clean_reference', validated_refs)):
        for case in cases:
            for seed in seeds:
                key = g6._run_key(stage, case['case_id'], seed)
                if key in result:
                    raise FairnessError('Duplicate pairing key')
                result[key] = (stage, case, seed)
    if len(result) != g6.EXPECTED_KEYS:
        raise FairnessError('Frozen pairing grid is not 117 keys')
    return result


def _assert_shared_sources(frozen, module_dir=None):
    """Ensure common model-facing code is byte-for-byte G5-frozen code."""
    module_dir = (Path(module_dir) if module_dir is not None
                  else Path(__file__).resolve().parent)
    hashes = {}
    for name in SHARED_MODULES:
        key = name
        expected = frozen['input_hashes'].get(key)
        if not expected or len(expected) != 64:
            raise FairnessError(f'Frozen {name} checksum missing')
        path = module_dir / (name + '.py')
        if not path.is_file():
            raise FairnessError(f'Shared module absent: {name}')
        actual = g5.sha256(path)
        if actual != expected:
            raise FairnessError(f'Shared {name} differs from frozen G5 code')
        hashes[name] = actual
    return hashes


def _treatment_code_hashes(module_dir=None):
    module_dir = (Path(module_dir) if module_dir is not None
                  else Path(__file__).resolve().parent)
    hashes = {}
    for name in TREATMENT_MODULES:
        path = module_dir / (name + '.py')
        if not path.is_file():
            raise FairnessError(f'Missing M6.1 module: {name}')
        hashes[name] = g5.sha256(path)
    return hashes


def _check_baseline_manifest(frozen, frozen_sha, reference_sha, baseline):
    """G6 runner manifest is the reference; do not infer settings from scores."""
    required = {
        'stage': 'G6_SINGLE_PENDING_SCORING',
        'g5_frozen_manifest_sha256': frozen_sha,
        'model_id': frozen['model_id'],
        'observed_model_version': frozen['observed_model_version'],
        'generation': frozen['generation'],
        'prompt_version': frozen['prompt_version'],
        'seeds': frozen['seeds'],
        'shared_gate_k': frozen['shared_gate_k'],
        'test_cases_sha256': frozen['input_hashes']['test_cases'],
        'test_clean_references_sha256': reference_sha,
    }
    if not isinstance(baseline, dict) or any(baseline.get(k) != v for k, v in required.items()):
        raise FairnessError('G6 baseline run manifest differs from G5 frozen study')
    interval = baseline.get('min_seconds_between_logical_calls')
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 0 <= interval < float('inf'):
        raise FairnessError('Invalid frozen G6 pacing policy')


def _check_baseline_rows(expected, rows, frozen, frozen_sha):
    """Verify every saved M5 row is the same frozen input and a real DONE row."""
    if len(rows) != len(expected):
        raise FairnessError('M5 baseline does not contain 117 unique paired keys')
    if set(rows) != set(expected):
        raise FairnessError('M5 baseline key mismatch, extra keys or missing keys')
    for key, (stage, case, seed) in expected.items():
        row = rows[key]
        if row.get('frozen_manifest_sha256') != frozen_sha:
            raise FairnessError('M5 baseline fingerprint mismatch')
        try:
            g6._validate_prior(row, stage, case, seed, frozen)
        except (g6.G6Error, KeyError, TypeError, ValueError) as exc:
            raise FairnessError(f'M5 frozen row or shared ML mismatch for {key}') from exc
        if row.get('event') != 'DONE' or row.get('final_status') not in ('ISSUE', 'ABSTAIN'):
            raise FairnessError('M5 paired run has not completed successfully')


def make_treatment_manifest(frozen, frozen_sha, reference_sha, baseline,
                            expected, shared_hashes, treatment_hashes):
    """Exact read-only specification that the future G7 runner must match."""
    _check_baseline_manifest(frozen, frozen_sha, reference_sha, baseline)
    if len(expected) != g6.EXPECTED_KEYS:
        raise FairnessError('Cannot freeze a partial M6 pairing grid')
    return {
        'stage': 'G7_MULTI_PENDING_SCORING',
        'condition': 'multi',
        'g5_frozen_manifest_sha256': frozen_sha,
        'model_id': frozen['model_id'],
        'observed_model_version': frozen['observed_model_version'],
        'generation': frozen['generation'],
        'prompt_version': frozen['prompt_version'],
        'seeds': frozen['seeds'],
        'shared_gate_k': frozen['shared_gate_k'],
        'verification_budget': 1,
        'topology': list(TOPOLOGY),
        'test_cases_sha256': frozen['input_hashes']['test_cases'],
        'test_clean_references_sha256': reference_sha,
        'paired_keys_sha256': _fingerprint(sorted(expected)),
        'shared_code_sha256': shared_hashes,
        'treatment_code_sha256': treatment_hashes,
        'min_seconds_between_logical_calls': baseline['min_seconds_between_logical_calls'],
    }


def assert_treatment_manifest(candidate, expected_manifest):
    """A G7 run/resume must use the exact reviewed conditions and code hashes."""
    if candidate != expected_manifest:
        raise FairnessError('G7 treatment manifest differs from frozen fair-comparison plan')
    return True


def assert_multi_rows_paired(expected, multi_rows, planned, require_complete=False):
    """Validate supplied G7 paired rows; no hidden labels are read or sent onward.

    A draft G7 row must carry generation settings to prevent per-row parameter
    drift. This is a contract for the forthcoming M6.3 runner, not a claim that
    real treatment traces already exist.
    """
    if not isinstance(multi_rows, dict) or set(multi_rows) - set(expected):
        raise FairnessError('G7 contains unexpected or duplicate pairing keys')
    if require_complete and set(multi_rows) != set(expected):
        raise FairnessError('G7 is missing matched Case/Seed pairs')
    for key, row in multi_rows.items():
        stage, case, seed = expected[key]
        fields = {
            'run_key': key, 'condition': 'multi', 'stage': stage,
            'case_id': case['case_id'], 'seed': seed,
            'input_observations': case['observations'],
            'ml_prediction': case['ml_prediction'],
            'frozen_manifest_sha256': planned['g5_frozen_manifest_sha256'],
            'model_id': planned['model_id'],
            'model_version_frozen': planned['observed_model_version'],
            'generation': planned['generation'],
            'shared_gate_k': planned['shared_gate_k'],
            'prompt_version': planned['prompt_version'],
            'treatment_code_sha256': planned['treatment_code_sha256'],
        }
        if not isinstance(row, dict) or any(row.get(k) != v for k, v in fields.items()):
            raise FairnessError('G7 Case/Seed, ML, observations or frozen config drifted')
        if row.get('event') not in ('DONE', 'FAILED'):
            raise FairnessError('G7 row lacks an explicit DONE/FAILED event')
        if require_complete:
            if row['event'] == 'DONE' and row.get('final_status') not in ('ISSUE', 'ABSTAIN'):
                raise FairnessError('DONE paired row lacks a valid terminal decision')
            if row['event'] == 'FAILED' and (
                    row.get('final_status') != 'FAILED' or
                    row.get('final_risk_level') is not None or
                    not isinstance(row.get('failure'), dict)):
                raise FairnessError('FAILED paired row must retain its explicit failure')
    return True


def inspect(g5_dir='artifacts/g5', g6_dir='artifacts/g6'):
    """Read-only on-Mac preflight. Returns sanitized report, not case data."""
    try:
        frozen, pre, frozen_sha = g6._frozen(g5_dir)
        if type(frozen.get('shared_gate_k')) is not int or not 0 <= frozen['shared_gate_k'] <= 7:
            raise FairnessError('Frozen shared gate k must be an integer in [0, 7]')
        test, refs, _, ref_sha = g6._test_suite(pre, frozen)
        expected = _expected_cases(test, refs, frozen['seeds'])
        shared = _assert_shared_sources(frozen)
        treatment = _treatment_code_hashes()
        baseline_path = Path(g6_dir) / 'run_manifest.json'
        if not baseline_path.is_file():
            raise FairnessError('Missing original G6 baseline manifest')
        baseline = g6._read_json(baseline_path)
        _check_baseline_manifest(frozen, frozen_sha, ref_sha, baseline)
        rows = g6._existing(Path(g6_dir) / 'single_runs.jsonl', frozen_sha)
        _check_baseline_rows(expected, rows, frozen, frozen_sha)
        proposed = make_treatment_manifest(frozen, frozen_sha, ref_sha, baseline,
                                           expected, shared, treatment)
    except (g5.G5Error, g6.G6Error, OSError, KeyError, TypeError, ValueError) as exc:
        raise FairnessError(f'M6.2 preflight cannot accept frozen evidence: {exc}') from exc
    report = {
        'status': 'READY_FOR_M6_3_IMPLEMENTATION_NOT_EXECUTION',
        'frozen_manifest_sha256': frozen_sha,
        'test_cases': len(test), 'clean_references': len(refs),
        'expected_paired_keys': len(expected),
        'baseline_done': len(rows),
        'shared_gate_k': frozen['shared_gate_k'],
        'seeds': frozen['seeds'], 'model_version': frozen['observed_model_version'],
        'paired_keys_sha256': proposed['paired_keys_sha256'],
        'multi_manifest_sha256': _fingerprint(proposed),
        'has_private_gold_in_manifest': False,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='M6.2 read-only fairness preflight; NO Gemini calls')
    parser.add_argument('--g5-dir', default='artifacts/g5')
    parser.add_argument('--g6-dir', default='artifacts/g6')
    args = parser.parse_args(argv)
    try:
        report = inspect(args.g5_dir, args.g6_dir)
        print('M6.2 FAIRNESS PREFLIGHT VERIFIED:', json.dumps(report, sort_keys=True))
    except FairnessError as exc:
        parser.exit(2, f'M6.2 FAIRNESS PREFLIGHT FAILED: {exc}\n')


if __name__ == '__main__':
    main()
