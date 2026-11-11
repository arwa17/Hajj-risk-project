"""G7 / M6 Multi-Agent: validation smoke, frozen paired runner, resumable traces.

All output belongs under private artifacts/. Never reads Test gold or changes G5/G6.
The controlled treatment has 4 initial Gemini role requests, optionally 2 post-
verification requests. Completed requests are checkpointed before the next call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from hajj_study import g5, g6, m6_fairness as fair
from hajj_study.gate import SharedDecisionGate
from hajj_study.multi_agent import execute_multi_case
from hajj_study.provider import CallFailure, GeminiProvider


class G7Error(RuntimeError):
    pass


def _hash(payload):
    return fair._fingerprint(payload)


def _safe_name(key):
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def _plan(g5_dir, g6_dir):
    frozen, pre, frozen_sha = g6._frozen(g5_dir)
    if type(frozen.get('shared_gate_k')) is not int or not 0 <= frozen['shared_gate_k'] <= 7:
        raise G7Error('Frozen shared gate k must be an integer in [0, 7]')
    test, refs, oracle, ref_sha = g6._test_suite(pre, frozen)
    expected = fair._expected_cases(test, refs, frozen['seeds'])
    baseline_path = Path(g6_dir) / 'run_manifest.json'
    if not baseline_path.is_file():
        raise G7Error('Original M5 run manifest is missing')
    baseline = g6._read_json(baseline_path)
    fair._check_baseline_manifest(frozen, frozen_sha, ref_sha, baseline)
    rows = g6._existing(Path(g6_dir) / 'single_runs.jsonl', frozen_sha)
    fair._check_baseline_rows(expected, rows, frozen, frozen_sha)
    plan = fair.make_treatment_manifest(frozen, frozen_sha, ref_sha, baseline,
              expected, fair._assert_shared_sources(frozen), fair._treatment_code_hashes())
    return frozen, pre, test, refs, oracle, expected, plan


def _dev_smoke_plan(g5_dir):
    """Bind Multi Development Smoke to G5 only, never G6/Test run results.

    Requires the validated Single-Agent Development calibration to be frozen,
    but DOES NOT require any M5 held-out Test inference or baseline manifest.
    """
    root = Path(g5_dir)
    frozen_file = root / 'manifest_frozen.json'
    if not frozen_file.is_file():
        raise G7Error('G5 Development calibration must be frozen before Multi smoke')
    frozen = g5.freeze(root)
    if g6._read_json(frozen_file) != frozen:
        raise G7Error('Frozen G5 manifest changed; refuse Development smoke')
    k = frozen.get('shared_gate_k')
    if type(k) is not int or not 0 <= k <= 7:
        raise G7Error('Development smoke requires frozen integer k in [0, 7]')
    pre = g5.read_pre(root)
    if pre['files']['validation_cases']['sha256'] != frozen['input_hashes']['validation_cases']:
        raise G7Error('Development cases changed after G5 freeze')
    if pre['files']['validation_references']['sha256'] != frozen['input_hashes']['validation_references']:
        raise G7Error('Development clean references changed after G5 freeze')
    shared_hashes = fair._assert_shared_sources(frozen)
    treatment_hashes = fair._treatment_code_hashes()
    plan = {
        'stage': 'G7_DEV_SMOKE_AFTER_G5_FREEZE_BEFORE_G6_TEST',
        'condition': 'multi',
        'g5_frozen_manifest_sha256': g5.sha256(frozen_file),
        'validation_cases_sha256': frozen['input_hashes']['validation_cases'],
        'validation_references_sha256': frozen['input_hashes']['validation_references'],
        'model_id': frozen['model_id'],
        'observed_model_version': frozen['observed_model_version'],
        'generation': frozen['generation'],
        'prompt_version': frozen['prompt_version'],
        'shared_gate_k': k,
        'seed': 42,
        'selection': 'first_of_40_deterministic_development_cases',
        'verification_budget': 1,
        'shared_code_sha256': shared_hashes,
        'treatment_code_sha256': treatment_hashes,
        'min_seconds_between_logical_calls': 8.0,
    }
    return frozen, pre, plan


class CheckpointProvider:
    """Validate cached successful role calls against request bytes before replay.

    Checkpoints store private raw model outputs and API usage immediately after
    every successful Gemini request. Failed attempts are retained for accounting.
    """
    def __init__(self, path, key, manifest_sha, provider, model_version,
                 interval, pacer=None):
        self.path = Path(path)
        self.key = key
        self.identity = {'run_key': key, 'manifest_sha256': manifest_sha}
        self.provider = provider
        self.model_version = model_version
        self.pacer = pacer if pacer is not None else g6.Pacer(interval)
        self.data = {'identity': self.identity, 'calls': [], 'failures': [],
                     'active_elapsed_ms': 0.0, 'wall_elapsed_ms': 0.0}
        if self.path.is_file():
            old = g6._read_json(self.path)
            if old.get('identity') != self.identity or not isinstance(old.get('calls'), list) or not isinstance(old.get('failures'), list):
                raise G7Error('Incompatible checkpoint; refuse to resume')
            self.data = old
        self.cursor = 0
        self.started = time.perf_counter()
        self.pacing_ms = 0.0

    def _save(self):
        g5._save_json(self.path, self.data)

    def generate_validated(self, *, system_prompt, user_prompt, response_schema, validator):
        fingerprint = _hash({'system': system_prompt, 'user': user_prompt,
                             'schema': response_schema})
        if self.cursor < len(self.data['calls']):
            stored = self.data['calls'][self.cursor]
            if stored.get('request_sha256') != fingerprint:
                raise G7Error('Cached Gemini prompt/schema changed; refusing replay')
            if stored.get('model_version') != self.model_version or not g5._accepted_call(stored):
                raise G7Error('Cached Gemini STOP/usage/modelVersion is invalid')
            try:
                checked = validator(stored['report'])
            except (ValueError, KeyError, TypeError) as exc:
                raise G7Error('Cached Gemini report failed current schema') from exc
            self.cursor += 1
            return SimpleNamespace(report=checked, attempts=stored['attempts'],
                    model_version=stored['model_version'], output_text=stored['raw_response_private'],
                    raw_http_response_b64_private=stored.get('raw_http_response_b64_private'),
                    raw_http_sha256=stored.get('raw_http_sha256'),
                    total_elapsed_ms=stored['elapsed_ms'])
        if self.cursor != len(self.data['calls']):
            raise G7Error('Checkpoint invocation count drift')
        if any(f.get('at_call_index') == self.cursor for f in self.data['failures']):
            # Real Gemini calls are never silently re-sent after a failed checkpoint.
            # Injected offline fake providers may replay for legacy mock recovery tests.
            if isinstance(self.provider, GeminiProvider):
                raise G7Error('Prior failed role at this checkpoint: no automatic replay')
        if isinstance(self.provider, GeminiProvider):
            self.provider.audit_scope = (
                f'G7|MULTI|{self.identity["manifest_sha256"]}|{self.key}|call-{self.cursor}')
        start = time.perf_counter()
        self.pacer.wait()
        self.pacing_ms += (time.perf_counter() - start) * 1000.0
        try:
            result = self.provider.generate_validated(system_prompt=system_prompt,
                user_prompt=user_prompt, response_schema=response_schema, validator=validator)
        except CallFailure as exc:
            self.data['failures'].append({'at_call_index': self.cursor,
                'category': exc.category, 'http_status': exc.status_code,
                'attempts': list(exc.attempts)})
            self._save()
            raise
        stored = {'request_sha256': fingerprint, 'report': validator(result.report),
                  'attempts': list(result.attempts), 'model_version': result.model_version,
                  'raw_response_private': result.output_text,
                  'raw_http_response_b64_private': getattr(result, 'raw_http_response_b64_private', None),
                  'raw_http_sha256': getattr(result, 'raw_http_sha256', None),
                  'elapsed_ms': result.total_elapsed_ms}
        if stored['model_version'] != self.model_version or not g5._accepted_call(stored):
            raise G7Error('Gemini response modelVersion, STOP or usage drifted; stop study')
        self.data['calls'].append(stored)
        self.cursor += 1
        self._save()
        return result

    def active_elapsed_ms(self):
        elapsed = max(0, (time.perf_counter() - self.started) * 1000 - self.pacing_ms)
        return self.data['active_elapsed_ms'] + elapsed

    def finish(self):
        # Measure THIS execution segment, not offline time between resumes.
        segment_wall_ms = max(0.0, (time.perf_counter() - self.started) * 1000.0)
        segment_active_ms = max(0.0, segment_wall_ms - self.pacing_ms)
        previous_active_ms = self.data['active_elapsed_ms']
        previous_wall_ms = self.data.get('wall_elapsed_ms', previous_active_ms)
        active = previous_active_ms + segment_active_ms
        wall = previous_wall_ms + segment_wall_ms
        self.data['active_elapsed_ms'] = active
        self.data['wall_elapsed_ms'] = wall
        self.started = time.perf_counter()
        self.pacing_ms = 0.0
        self._save()
        return active


def _one(stage, case, seed, oracle, frozen, planned, out, *, pacer=None,
         provider=None):
    key = (f'validation|{case["case_id"]}|{seed}' if stage == 'validation'
           else g6._run_key(stage, case['case_id'], seed))
    checkpoint = Path(out) / 'checkpoints' / (_safe_name(key) + '.json')
    wrapper = CheckpointProvider(checkpoint, key, _hash(planned),
                    provider if provider is not None else g5._provider(frozen['generation'], seed),
                    frozen['observed_model_version'],
                    planned['min_seconds_between_logical_calls'], pacer=pacer)
    outcome = execute_multi_case(case, provider=wrapper,
        gate=SharedDecisionGate(frozen['shared_gate_k']), oracle=oracle,
        expected_model_version=frozen['observed_model_version'])
    row = {'run_key': key, 'event': 'DONE' if outcome.status in ('ISSUE', 'ABSTAIN') else 'FAILED',
        'condition': 'multi', 'stage': stage, 'case_id': case['case_id'], 'seed': seed,
        'frozen_manifest_sha256': planned['g5_frozen_manifest_sha256'],
        'model_id': planned['model_id'], 'model_version_frozen': planned['observed_model_version'],
        'generation': planned['generation'], 'shared_gate_k': planned['shared_gate_k'],
        'prompt_version': planned['prompt_version'],
        'treatment_code_sha256': planned['treatment_code_sha256'],
        'input_observations': case['observations'], 'ml_prediction': case['ml_prediction'],
        'initial': None, 'post': None,
        'initial_gate': outcome.trace['initial_gate'], 'post_gate': outcome.trace['post_gate'],
        'verification_domain': outcome.trace['verification_domain'],
        'verification_count': outcome.trace['verification_count'],
        'verified_observations': outcome.trace['verified_observations'],
        'final_status': outcome.status, 'final_risk_level': outcome.risk_level,
        'multi_trace': outcome.trace, 'failure': outcome.trace['failure'],
        'previous_failures': wrapper.data['failures'],
        'call_log': [{k: v for k, v in c.items() if k not in ('raw_response_private', 'raw_http_response_b64_private')}
                     for c in wrapper.data['calls']],
        'private_checkpoint_sha256': None}
    calls = wrapper.data['calls']
    if len(calls) >= 4:
        row['initial'] = {k: calls[3][k] for k in ('report', 'attempts', 'model_version', 'elapsed_ms')}
    if len(calls) == 6:
        row['post'] = {k: calls[5][k] for k in ('report', 'attempts', 'model_version', 'elapsed_ms')}
    row['active_elapsed_ms'] = wrapper.finish()
    row['wall_elapsed_ms'] = wrapper.data['wall_elapsed_ms']
    row['private_checkpoint_sha256'] = g5.sha256(checkpoint)
    if outcome.status != 'FAILED' and len(calls) != (4 if row['verification_count'] == 0 else 6):
        raise G7Error('Unexpected number of completed real Gemini role calls')
    return row


def _existing(out, expected, planned):
    rows = g6._existing(Path(out) / 'multi_runs.jsonl', planned['g5_frozen_manifest_sha256'])
    fair.assert_multi_rows_paired(expected, rows, planned)
    return rows


def _is_rate_limited(row):
    """HTTP 429 is provider capacity exhaustion, not an agent risk decision."""
    failure = (row or {}).get('failure') or {}
    return failure.get('category') == 'HTTP' and failure.get('http_status') == 429


def inspect(g5_dir='artifacts/g5', g6_dir='artifacts/g6', out='artifacts/g7'):
    _, _, _, _, _, expected, planned = _plan(g5_dir, g6_dir)
    out = Path(out)
    manifest_file = out / 'run_manifest.json'
    if manifest_file.is_file():
        fair.assert_treatment_manifest(g6._read_json(manifest_file), planned)
    rows = _existing(out, expected, planned)
    done = sum(x['event'] == 'DONE' for x in rows.values())
    failed = sum(x['event'] == 'FAILED' for x in rows.values())
    rate_limited = sum(_is_rate_limited(x) for x in rows.values())
    unattempted = len(expected) - done - failed
    status = ('G7_DONE_UNSCORED' if done == len(expected) else
              'G7_TERMINAL_WITH_FAILURES' if unattempted == 0 else
              'G7_IN_PROGRESS')
    return {'status': status,
            'expected': len(expected), 'done': done, 'failed': failed,
            'unattempted': unattempted, 'remaining': len(expected) - done,
            'http_429_pending_retry': rate_limited,
            'non_429_failed': failed - rate_limited,
            'model_version': planned['observed_model_version'],
            'frozen_sha256': planned['g5_frozen_manifest_sha256'],
            'manifest_sha256': _hash(planned)}


def smoke(g5_dir='artifacts/g5', g6_dir='artifacts/g6', out='artifacts/g7'):
    """Live Development-only smoke; requires explicit key and frozen G5, not G6.

    The g6_dir parameter is accepted for backward CLI compatibility, but is
    NOT accessed. This function makes no held-out Test requests/decisions.
    """
    if not os.getenv('GEMINI_API_KEY'):
        raise G7Error('No GEMINI_API_KEY; no live validation smoke attempted')
    frozen, pre, planned = _dev_smoke_plan(g5_dir)
    out = Path(out)
    record_path = out / 'validation_smoke.json'
    if record_path.exists():
        prior = g6._read_json(record_path)
        if prior.get('manifest_sha256') != _hash(planned):
            raise G7Error('Validation smoke belongs to different G5/code provenance')
        if prior.get('event') != 'DONE':
            raise G7Error('Previous Development smoke failed; inspect private checkpoint; no automatic retry')
        if prior.get('calls') not in (4, 6) or prior.get('model_version') != frozen['observed_model_version']:
            raise G7Error('Existing Development smoke proof is invalid')
        return {'status': 'VALIDATION_SMOKE_VERIFIED', 'calls': prior['calls'],
                'model_version': prior['model_version']}
    cases = [g5.validate_case(x) for x in g5._jsonl(pre['files']['validation_cases']['path'])]
    refs = g5._jsonl(pre['files']['validation_references']['path'])
    if len(cases) != 40 or len({c['case_id'] for c in cases}) != 40 or len(refs) != 12:
        raise G7Error('Development validation suite must contain 40 cases and 12 clean references')
    oracle = g5._validation_oracle(cases, refs)
    case = cases[0]
    result = _one('validation', case, 42, oracle, frozen, planned,
                  out / 'validation_only', pacer=g6.Pacer(planned['min_seconds_between_logical_calls']))
    calls = result['call_log']
    record = {'manifest_sha256': _hash(planned), 'event': result['event'],
              'status': result['final_status'], 'model_version': frozen['observed_model_version'],
              'case_id': case['case_id'], 'calls': len(calls),
              'verified_roles': ['crowd', 'health', 'environment', 'coordinator'],
              'verification_count': result['verification_count'], 'failure': result['failure']}
    g5._save_json(record_path, record)
    if record['event'] != 'DONE' or len(calls) not in (4, 6):
        raise G7Error('M6.1 live Development smoke failed; no Test run permitted')
    return {'status': 'VALIDATION_SMOKE_VERIFIED', 'calls': len(calls),
            'model_version': frozen['observed_model_version']}




def run(g5_dir='artifacts/g5', g6_dir='artifacts/g6', out='artifacts/g7', *,
        max_cases=1, continue_on_failure=False, retry_http_429=False):
    if not os.getenv('GEMINI_API_KEY'):
        raise G7Error('No GEMINI_API_KEY; no live Test run permitted')
    if max_cases < 1 or max_cases > 117:
        raise G7Error('max_cases must be 1..117')
    if retry_http_429 and not continue_on_failure:
        raise G7Error('--retry-http-429 requires --continue-on-failure')
    # Validate Development smoke BEFORE opening or planning held-out Test.
    out = Path(out)
    smoke_file = out / 'validation_smoke.json'
    if not smoke_file.is_file():
        raise G7Error('Run M6.1 live Development smoke before touching held-out Test')
    _, _, dev_plan = _dev_smoke_plan(g5_dir)
    proof = g6._read_json(smoke_file)
    if (proof.get('manifest_sha256') != _hash(dev_plan) or proof.get('event') != 'DONE'
            or proof.get('calls') not in (4, 6)
            or proof.get('model_version') != dev_plan['observed_model_version']):
        raise G7Error('Development smoke does not authenticate current frozen G5/treatment')
    # Held-out Test is now permitted, but remains strictly paired to G6 M5.
    frozen, _, test, refs, oracle, expected, planned = _plan(g5_dir, g6_dir)
    for binding in ('g5_frozen_manifest_sha256', 'model_id', 'observed_model_version',
                    'generation', 'prompt_version', 'shared_gate_k',
                    'shared_code_sha256', 'treatment_code_sha256',
                    'min_seconds_between_logical_calls'):
        if planned.get(binding) != dev_plan.get(binding):
            raise G7Error('G6 comparison plan differs from independently checked Development smoke: ' + binding)
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / 'run_manifest.json'
    if manifest_file.exists():
        fair.assert_treatment_manifest(g6._read_json(manifest_file), planned)
    else:
        g5._save_json(manifest_file, planned)
    rows = _existing(out, expected, planned)
    pacer = g6.Pacer(planned['min_seconds_between_logical_calls'])
    executed = 0
    for stage, case, seed in g6._entries(test, refs, frozen['seeds']):
        key = g6._run_key(stage, case['case_id'], seed)
        previous = rows.get(key)
        if previous and previous['event'] == 'DONE':
            continue
        if previous and (previous.get('failure') or {}).get('category') == 'CONTRACT':
            raise G7Error('Previous contract failure; do not silently retry')
        if previous and previous['event'] == 'FAILED' and continue_on_failure:
            # Only provider HTTP 429 can be resumed explicitly after quota clears.
            # Other failures stay terminal; never relabel them as ABSTAIN.
            if not (retry_http_429 and _is_rate_limited(previous)):
                continue
        row = _one(stage, case, seed, oracle, frozen, planned, out, pacer=pacer)
        fair.assert_multi_rows_paired(expected, {key: row}, planned)
        g5._append(out / 'multi_runs.jsonl', row)
        rows[key] = row
        executed += 1
        if row['event'] != 'DONE':
            if _is_rate_limited(row):
                # Do not burn an entire batch on a depleted API quota.
                # Preserve failed attempts and already successful role checkpoints.
                raise G7Error(f'HTTP 429 at {key}; checkpoint saved. '
                              'Stop API calls; retry after quota is available.')
            if ((row.get('failure') or {}).get('category') == 'CONTRACT'
                    or not continue_on_failure):
                raise G7Error(f'Live role failed at {key}; checkpoint and failure saved privately')
        if executed >= max_cases:
            break
    return inspect(g5_dir, g6_dir, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description='M6 Gemini role runner; all data private')
    parser.add_argument('command', choices=('inspect', 'smoke', 'run'))
    parser.add_argument('--g5-dir', default='artifacts/g5')
    parser.add_argument('--g6-dir', default='artifacts/g6')
    parser.add_argument('--output', default='artifacts/g7')
    parser.add_argument('--max-cases', type=int, default=1)
    parser.add_argument('--continue-on-failure', action='store_true',
                        help='Preserve non-quota FAILED outcomes and continue untouched pairs')
    parser.add_argument('--retry-http-429', action='store_true',
                        help='Resume only saved HTTP 429 failures; requires --continue-on-failure')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            report = inspect(args.g5_dir, args.g6_dir, args.output)
        elif args.command == 'smoke':
            report = smoke(args.g5_dir, args.g6_dir, args.output)
        else:
            report = run(args.g5_dir, args.g6_dir, args.output,
                         max_cases=args.max_cases,
                         continue_on_failure=args.continue_on_failure,
                         retry_http_429=args.retry_http_429)
        print('M6', args.command.upper(), json.dumps(report, sort_keys=True))
    except (G7Error, fair.FairnessError, g6.G6Error, g5.G5Error, ValueError,
            KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'M6 NOT COMPLETE: {exc}\n')


if __name__ == '__main__':
    main()
