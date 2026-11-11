"""G6 / M5 single-agent held-out runner; consumes the immutable G5 manifest.

Private artifacts only. No gold labels are passed to Gemini. This does not
score the experiment; scoring follows the full, independently audited trace.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hajj_study import g5
from hajj_study.gate import SharedDecisionGate
from hajj_study.provider import CallFailure
from hajj_study.schemas import ContractError, validate_case, validate_final_report
from hajj_study.verify import VerificationOracle


class G6Error(RuntimeError):
    pass


STAGES = ('test', 'clean_reference')
EXPECTED_KEYS = 3 * (30 + 9)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _frozen(private_g5):
    """Checks COMPLETE G5 from original private files. Does not call Gemini."""
    private_g5 = Path(private_g5).expanduser().resolve()
    path = private_g5 / 'manifest_frozen.json'
    if not path.is_file():
        raise G6Error('G5 frozen manifest missing; do not access held-out Test')
    frozen = _read_json(path)
    if frozen.get('stage') != 'G5_FROZEN_AFTER_REAL_GEMINI':
        raise G6Error('G5 freeze stage not accepted')
    # Re-verify preflight inputs, Smoke, Validation, calibration and manifest.
    # freeze() is idempotent when manifest_frozen.json already exists.
    try:
        actual = g5.freeze(private_g5)
    except (g5.G5Error, ValueError, OSError) as exc:
        raise G6Error(f'G5 frozen provenance check failed: {exc}') from exc
    if frozen != actual:
        raise G6Error('G5 frozen manifest changed after calibration')
    if frozen.get('model_id') != frozen.get('observed_model_version'):
        # An observed version may differ from the endpoint ID for some models;
        # this particular G5 experiment is validated against its observed value.
        if not frozen.get('observed_model_version'):
            raise G6Error('Missing frozen observed modelVersion')
    if frozen['shared_gate_k'] not in range(8):
        raise G6Error('Invalid frozen shared gate threshold')
    return frozen, g5.read_pre(private_g5), g5.sha256(path)


def _test_suite(pre, frozen):
    test_path = Path(pre['files']['test_cases']['path'])
    test = [validate_case(x) for x in g5._jsonl(test_path)]
    if len(test) != 30 or len({x['case_id'] for x in test}) != 30:
        raise G6Error('Expected 30 unique frozen Test cases')
    if g5.sha256(test_path) != frozen['input_hashes']['test_cases']:
        raise G6Error('Test suite SHA-256 changed')
    case_manifest_path = Path(pre['files']['case_manifest']['path'])
    manifest = _read_json(case_manifest_path)
    if g5.sha256(case_manifest_path) != frozen['input_hashes']['case_manifest']:
        raise G6Error('G3 manifest changed')
    ref_spec = manifest['suite_files']['test']['clean_references']
    refs_path = test_path.parent / ref_spec['file'].split('/')[-1]
    if not refs_path.is_file() or g5.sha256(refs_path) != ref_spec['sha256']:
        raise G6Error('Test clean-reference SHA-256 does not match frozen G3 manifest')
    refs = [validate_case(x) for x in g5._jsonl(refs_path)]
    if len(refs) != 9 or len({x['case_id'] for x in refs}) != 9:
        raise G6Error('Expected exactly 9 unique clean-reference cases')
    base = {x['case_id']: x for x in test}
    if any(r['case_id'] not in base for r in refs):
        raise G6Error('Clean-reference ID not in frozen Test')
    if any(r['ml_prediction'] != base[r['case_id']]['ml_prediction'] for r in refs):
        raise G6Error('Clean-reference ML prediction differs from corresponding Test case')
    # Only the private oracle can read clean references; the agents see
    # initial corrupted case observations, or a planned clean counterpart run.
    clean = {c['case_id']: c['observations'] for c in test}
    clean.update({r['case_id']: r['observations'] for r in refs})
    oracle = VerificationOracle(clean)
    return test, refs, oracle, ref_spec['sha256']


def _entries(test, refs, seeds):
    for seed in seeds:
        for case in test:
            yield 'test', case, seed
        for case in refs:
            yield 'clean_reference', case, seed


def _run_key(stage, cid, seed):
    if stage not in STAGES:
        raise G6Error('Illegal G6 stage')
    return f'{stage}|{cid}|{seed}'


def _existing(path, fingerprint):
    result = {}
    for row in g5._jsonl(path):
        key = row.get('run_key')
        if not key or row.get('frozen_manifest_sha256') != fingerprint:
            raise G6Error('Existing G6 record has incompatible or missing provenance')
        if key not in result or row.get('event') == 'DONE' or row.get('event') == 'FAILED':
            result[key] = row
    return result


class Pacer:
    """Conservative interval between logical calls; retries remain provider-owned."""
    def __init__(self, interval=8.0, sleeper=time.sleep, clock=time.monotonic):
        if interval < 0:
            raise ValueError('Invalid pacing interval')
        self.interval = interval
        self.sleeper, self.clock, self.last = sleeper, clock, None

    def wait(self):
        now = self.clock()
        if self.last is not None:
            remaining = self.interval - (now - self.last)
            if remaining > 0:
                self.sleeper(remaining)
        self.last = self.clock()


def _success_check(call, model_version):
    if not g5._accepted_call(call):
        raise G6Error('Missing STOP, usage metadata, or model version')
    if call['model_version'] != model_version:
        raise G6Error('Gemini modelVersion drifted after G5 freeze; halt study')


def _one(stage, case, seed, *, oracle, frozen, frozen_sha, prior=None, pacer=None):
    cid = case['case_id']
    key = _run_key(stage, cid, seed)
    result = {'run_key': key, 'event': 'FAILED', 'case_id': cid, 'stage': stage,
              'condition': 'single', 'seed': seed, 'model_id': frozen['model_id'],
              'model_version_frozen': frozen['observed_model_version'],
              'shared_gate_k': frozen['shared_gate_k'],
              'frozen_manifest_sha256': frozen_sha,
              'prompt_version': frozen['prompt_version'],
              'input_observations': case['observations'],
              'ml_prediction': case['ml_prediction'],
              'initial': None, 'initial_gate': None, 'verification_domain': None,
              'verification_count': 0, 'verified_observations': None,
              'post': None, 'post_gate': None, 'final_status': 'FAILED',
              'final_risk_level': None, 'failure': None}
    # Preserve actual failed API attempts when a case is resumed. Otherwise
    # the offline scorer would silently undercount total calls and cost.
    prior_failures = list(prior.get('previous_failures', [])) if prior else []
    if prior and prior.get('failure') is not None:
        prior_failures.append(prior['failure'])
    result['previous_failures'] = prior_failures
    previous_active_ms = float(prior.get('active_elapsed_ms', 0.0)) if prior else 0.0
    previous_wall_ms = float(prior.get('wall_elapsed_ms', previous_active_ms)) if prior else 0.0
    if not 0 <= previous_active_ms < float('inf') or not 0 <= previous_wall_ms < float('inf'):
        raise G6Error('Invalid prior active/wall latency')
    started = time.perf_counter()
    pacing_seconds = 0.0

    def pace():
        nonlocal pacing_seconds
        if pacer:
            tick = time.perf_counter()
            pacer.wait()
            pacing_seconds += time.perf_counter() - tick

    def finish():
        active_ms = max(0.0, (time.perf_counter() - started - pacing_seconds) * 1000.0)
        # Cumulative active task time includes earlier failed attempts across
        # resumes; excludes deliberate inter-request pacing / user downtime.
        result['active_elapsed_ms'] = previous_active_ms + active_ms
        # Sum measured execution segments; exclude idle time between resumes.
        result['wall_elapsed_ms'] = previous_wall_ms + max(0.0, (time.perf_counter() - started) * 1000.0)
        return result

    gate = SharedDecisionGate(frozen['shared_gate_k'])
    provider = g5._provider(frozen['generation'], seed)
    try:
        if prior and prior.get('initial') is not None:
            # Recover a validated initial call if a post-verification request
            # previously failed. This prevents repeating successful API calls.
            previous_initial = prior['initial']
            _success_check(previous_initial, frozen['observed_model_version'])
            result['initial'] = previous_initial
        else:
            pace()
            result['initial'] = g5._call(case, provider,
                                              audit_scope=f'G6|SINGLE|{frozen_sha}|{key}|initial')
            _success_check(result['initial'], frozen['observed_model_version'])
        initial_report = validate_final_report(result['initial']['report'], cid)
        initial = gate.initial(initial_report, cid)
        result['initial_gate'] = {'status': initial.status, 'risk_level': initial.risk_level,
                                  'evidence_sufficiency': initial.evidence_sufficiency}
        if initial.status == 'ISSUE':
            result['final_status'] = 'ISSUE'
            result['final_risk_level'] = initial.risk_level
            result['event'] = 'DONE'
            return finish()
        domain = initial.verification_domain
        result['verification_domain'] = domain
        session = oracle.new_session(cid)
        updated = session.verify_domain(case['observations'], domain, case_id=cid)
        result['verified_observations'] = updated
        result['verification_count'] = 1
        pace()
        result['post'] = g5._call({**case, 'observations': updated}, provider,
                                  verified_domain=domain,
                                  audit_scope=f'G6|SINGLE|{frozen_sha}|{key}|post|{domain}')
        _success_check(result['post'], frozen['observed_model_version'])
        post = gate.post_verification(initial_report, result['post']['report'], cid, domain,
                                      verification_used=session.used)
        result['post_gate'] = {'status': post.status, 'risk_level': post.risk_level,
                               'evidence_sufficiency': post.evidence_sufficiency,
                               'merged_flags': list(post.evidence_flags)}
        result['final_status'], result['final_risk_level'] = post.status, post.risk_level
        result['event'] = 'DONE'
    except CallFailure as exc:
        result['failure'] = {'category': exc.category, 'http_status': exc.status_code,
                             'attempts': exc.attempts}
    except G6Error as exc:
        result['failure'] = {'category': 'G6_CONTRACT', 'reason': str(exc)}
    except ContractError as exc:
        result['failure'] = {'category': 'CONTRACT', 'reason': str(exc)}
    return finish()


def _validate_prior(row, stage, case, seed, frozen):
    if (row['stage'] != stage or row['case_id'] != case['case_id'] or
            row['seed'] != seed or row['condition'] != 'single' or
            row['model_id'] != frozen['model_id'] or
            row['model_version_frozen'] != frozen['observed_model_version'] or
            row['prompt_version'] != frozen['prompt_version'] or
            row['shared_gate_k'] != frozen['shared_gate_k'] or
            row['input_observations'] != case['observations'] or
            row['ml_prediction'] != case['ml_prediction']):
        raise G6Error('Existing G6 run differs from frozen input or config')
    if row.get('event') == 'DONE':
        if row.get('final_status') not in ('ISSUE', 'ABSTAIN'):
            raise G6Error('Existing DONE outcome is invalid')
        _success_check(row['initial'], frozen['observed_model_version'])
        if row['verification_count'] == 1:
            _success_check(row['post'], frozen['observed_model_version'])
    return row


def inspect(private_g5, output):
    frozen, pre, frozen_sha = _frozen(private_g5)
    test, refs, oracle, ref_sha = _test_suite(pre, frozen)
    output = Path(output)
    records = _existing(output / 'single_runs.jsonl', frozen_sha)
    expected = {_run_key(stage, case['case_id'], seed) for stage, case, seed in
                _entries(test, refs, frozen['seeds'])}
    if len(expected) != EXPECTED_KEYS:
        raise G6Error('Expected 117 distinct G6 run keys')
    if set(records) - expected:
        raise G6Error('Unexpected G6 run keys')
    cases = {_run_key(stage, case['case_id'], seed): (stage, case, seed) for stage, case, seed in
             _entries(test, refs, frozen['seeds'])}
    for key, row in records.items():
        _validate_prior(row, *cases[key], frozen)
    completed = sum(row.get('event') == 'DONE' for row in records.values())
    failed = sum(row.get('event') == 'FAILED' for row in records.values())
    return {'g5_frozen_manifest_sha256': frozen_sha, 'model_version': frozen['observed_model_version'],
            'k': frozen['shared_gate_k'], 'test_cases': len(test), 'clean_reference_cases': len(refs),
            'clean_reference_sha256': ref_sha, 'run_keys': len(expected),
            'done': completed, 'failed': failed, 'remaining': len(expected)-completed}


def run(private_g5, output, *, interval=8.0):
    if not os.getenv('GEMINI_API_KEY'):
        raise G6Error('GEMINI_API_KEY is missing; no live call attempted')
    summary = inspect(private_g5, output)
    frozen, pre, frozen_sha = _frozen(private_g5)
    test, refs, oracle, _ = _test_suite(pre, frozen)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'stage': 'G6_SINGLE_PENDING_SCORING', 'g5_frozen_manifest_sha256': frozen_sha,
                'model_id': frozen['model_id'], 'observed_model_version': frozen['observed_model_version'],
                'generation': frozen['generation'], 'prompt_version': frozen['prompt_version'],
                'seeds': frozen['seeds'], 'shared_gate_k': frozen['shared_gate_k'],
                'test_cases_sha256': frozen['input_hashes']['test_cases'],
                'test_clean_references_sha256': summary['clean_reference_sha256'],
                'min_seconds_between_logical_calls': interval}
    manifest_path = output / 'run_manifest.json'
    if manifest_path.is_file() and _read_json(manifest_path) != manifest:
        raise G6Error('G6 execution manifest changed; refuse to resume')
    if not manifest_path.is_file():
        g5._save_json(manifest_path, manifest)
    log_path = output / 'single_runs.jsonl'
    rows = _existing(log_path, frozen_sha)
    pacer = Pacer(interval)
    for stage, case, seed in _entries(test, refs, frozen['seeds']):
        key = _run_key(stage, case['case_id'], seed)
        prior = rows.get(key)
        if prior:
            _validate_prior(prior, stage, case, seed, frozen)
            if prior['event'] == 'DONE':
                continue
            if prior.get('failure', {}).get('category') == 'G6_CONTRACT':
                raise G6Error(f'Previous contract failure: {key}; do not blindly retry')
        record = _one(stage, case, seed, oracle=oracle, frozen=frozen,
                      frozen_sha=frozen_sha, prior=prior, pacer=pacer)
        g5._append(log_path, record)
        rows[key] = record
        if record['event'] != 'DONE':
            raise G6Error(f'G6 stopped at {key}; failure recorded privately')
    return inspect(private_g5, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Private frozen G6 / M5 Single-Agent runner')
    parser.add_argument('command', choices=('inspect', 'run'))
    parser.add_argument('--g5-dir', default='artifacts/g5')
    parser.add_argument('--output', default='artifacts/g6')
    parser.add_argument('--min-interval', type=float, default=8.0)
    args = parser.parse_args(argv)
    try:
        summary = inspect(args.g5_dir, args.output) if args.command == 'inspect' else run(
            args.g5_dir, args.output, interval=args.min_interval)
        print('G6', args.command.upper(), json.dumps(summary, sort_keys=True))
    except (G6Error, g5.G5Error, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'G6 NOT COMPLETE: {exc}\n')


if __name__ == '__main__':
    main()
