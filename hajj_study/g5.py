"""G5: preflight, live Gemini smoke, validation-only calibration and freeze.

No API credentials, private cases, raw responses, reference labels, or model
artifacts belong in the public repository. G5 is incomplete until actual
Gemini smoke AND validation records satisfy the acceptance checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from hajj_study.gate import SharedDecisionGate
from hajj_study.prompts import PROMPT_VERSION, generalist_messages
from hajj_study.provider import CallFailure, GeminiProvider
from hajj_study.schemas import final_response_schema, validate_case, validate_final_report
from hajj_study.verify import VerificationOracle

SOURCE_SHA = '7c289edf5b3390b479c48ac5a493db566f6c46d4d587ae101b03d885bf948e8a'
SEEDS = (11, 42, 77)
EXPECTED_VALIDATION = 40
SMOKE_PROBES = ((0, 42), (5, 42), (10, 42), (15, 42), (20, 42),
                (25, 42), (30, 42), (35, 42), (0, 11), (0, 77))


class G5Error(ValueError):
    """G5 failed a gate; no manifest may be frozen."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def _append(path, entry):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, sort_keys=True, separators=(',', ':')) + '\n')


def _paths(snapshot_root, source_csv):
    r = Path(snapshot_root).resolve()
    return {
        'raw_csv': Path(source_csv).resolve(),
        'proxy_code': r/'hajj_study/proxy.py',
        'split': r/'artifacts/row_splits.csv',
        'feature_contract': r/'hajj_study/feature_contract.py',
        'model': r/'artifacts/model/shared_model.joblib',
        'predictions': r/'artifacts/model/predictions.jsonl',
        'test_cases': r/'artifacts/g3_cases/test/cases.jsonl',
        'test_gold': r/'artifacts/g3_cases/test/gold_labels.jsonl',
        'validation_cases': r/'artifacts/g3_cases/validation/cases.jsonl',
        'validation_references': r/'artifacts/g3_cases/validation/clean_references.jsonl',
        'validation_gold': r/'artifacts/g3_cases/validation/gold_labels.jsonl',
        'case_manifest': r/'artifacts/g3_cases/case_manifest.json',
        'prompts': r/'hajj_study/prompts.py',
        'schemas': r/'hajj_study/schemas.py',
        'provider': r/'hajj_study/provider.py',
        'gate': r/'hajj_study/gate.py',
        'verify': r/'hajj_study/verify.py',
        'generalist': r/'hajj_study/generalist.py',
        'g5_code': Path(__file__).resolve(),
        'requirements': r/'requirements-study.txt',
        'contract': r/'CONTRACT.md',
    }


def _check_suite(pre):
    vpath = pre['files']['validation_cases']['path']
    tpath = pre['files']['test_cases']['path']
    validation = [validate_case(c) for c in _jsonl(vpath)]
    test = [validate_case(c) for c in _jsonl(tpath)]
    vi = [c['case_id'] for c in validation]
    ti = [c['case_id'] for c in test]
    if len(vi) != 40 or len(ti) != 30 or len(set(vi)) != 40 or len(set(ti)) != 30 or set(vi)&set(ti):
        raise G5Error('Frozen Test/Validation suite count, uniqueness or isolation mismatch')
    refs = _jsonl(pre['files']['validation_references']['path'])
    if len(refs) != 12 or len({r['case_id'] for r in refs}) != 12:
        raise G5Error('Frozen validation clean-reference set must contain 12 cases')
    if not {r['case_id'] for r in refs}.issubset(set(vi)):
        raise G5Error('Validation clean references contain unknown case')
    case_manifest = json.loads(Path(pre['files']['case_manifest']['path']).read_text())
    for group in ('test', 'validation'):
        for item, name in (('cases',group+'_cases'), ('gold_labels',group+'_gold')):
            if case_manifest['suite_files'][group][item]['sha256'] != pre['files'][name]['sha256']:
                raise G5Error(f'G3 case-manifest checksum mismatch: {name}')
    for group in ('test', 'validation'):
        ref_name = group + '_references'
        if group == 'validation' and case_manifest['suite_files'][group]['clean_references']['sha256'] != pre['files'][ref_name]['sha256']:
            raise G5Error('G3 validation reference checksum mismatch')
    if case_manifest['input_hashes']['raw_csv'] != SOURCE_SHA:
        raise G5Error('G3 references a different source CSV')
    for reference_key, file_key in (('row_splits','split'), ('shared_predictions','predictions')):
        if case_manifest['input_hashes'][reference_key] != pre['files'][file_key]['sha256']:
            raise G5Error(f'G3 input checksum mismatch: {reference_key}')
    return validation


def create_preflight(*, snapshot_root, source_csv, outdir, model_id,
                     temperature=0.0, thinking_budget=None, max_output_tokens=4096,
                     top_p=1.0):
    if not isinstance(model_id, str) or not model_id:
        raise G5Error('Model candidate must be explicit')
    # The CLI may run from a fresh GitHub checkout while the G4 snapshot is
    # extracted separately. Refuse code drift between active imports and G4.
    active = Path(__file__).resolve().parent
    frozen = Path(snapshot_root).resolve() / 'hajj_study'
    for module in ('proxy', 'feature_contract', 'prompts', 'schemas',
                   'provider', 'gate', 'verify', 'generalist'):
        if sha256(active / (module + '.py')) != sha256(frozen / (module + '.py')):
            raise G5Error(f'Active GitHub code differs from frozen G4: {module}')
    files = {name: {'path': str(p), 'sha256': sha256(p)}
             for name, p in _paths(snapshot_root, source_csv).items()}
    if files['raw_csv']['sha256'] != SOURCE_SHA:
        raise G5Error('Original dataset SHA-256 does not match the frozen source')
    config = {'model_id': model_id, 'temperature': temperature, 'top_p': top_p,
              'thinking_budget': thinking_budget, 'max_output_tokens': max_output_tokens,
              'seed_policy': list(SEEDS), 'calibration_seed': 42}
    # Constructor also checks bounds; it does not call Gemini.
    GeminiProvider(model_id=model_id, temperature=temperature, thinking_budget=thinking_budget,
                   max_output_tokens=max_output_tokens, top_p=top_p)
    pre = {'stage': 'G5_PRE__NOT_FROZEN', 'prompt_version': PROMPT_VERSION,
           'files': files, 'candidate_config': config,
           'no_live_results_claimed': True}
    _check_suite(pre)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest_frozen.json').exists() or (out/'smoke.jsonl').exists() or (out/'validation_runs.jsonl').exists():
        raise G5Error('Output contains existing runs; use a new private directory to change preflight config')
    _save_json(out/'manifest_pre.json', pre)
    return pre


def read_pre(outdir):
    path = Path(outdir)/'manifest_pre.json'
    pre = json.loads(path.read_text(encoding='utf-8'))
    if pre.get('stage') != 'G5_PRE__NOT_FROZEN' or pre.get('prompt_version') != PROMPT_VERSION:
        raise G5Error('Wrong G5 pre-manifest or prompt version')
    for name, spec in pre['files'].items():
        if sha256(spec['path']) != spec['sha256']:
            raise G5Error(f'Frozen input changed since preflight: {name}')
    _check_suite(pre)
    return pre


def _provider(config, seed):
    return GeminiProvider(model_id=config['model_id'], temperature=config['temperature'],
                          top_p=config['top_p'],
                          thinking_budget=config['thinking_budget'],
                          max_output_tokens=config['max_output_tokens'], seed=seed)


def _call(case, provider, *, verified_domain=None, audit_scope=None):
    system, user = generalist_messages(case, verified_domain=verified_domain)
    if audit_scope is not None:
        provider.audit_scope = audit_scope
    result = provider.generate_validated(system_prompt=system, user_prompt=user,
              response_schema=final_response_schema(),
              validator=lambda item: validate_final_report(item, case['case_id']))
    return {'report': result.report, 'attempts': result.attempts,
            'raw_response_private': result.output_text,
            'raw_http_response_b64_private': result.raw_http_response_b64_private,
            'raw_http_sha256': result.raw_http_sha256,
            'model_version': result.model_version, 'elapsed_ms': result.total_elapsed_ms}


def _accepted_call(call):
    if not call.get('model_version') or not call.get('attempts'):
        return False
    last = call['attempts'][-1]
    usage = last.get('usage') or {}
    return (last.get('status') == 'STOP' and last.get('finish_reason') == 'STOP'
            and isinstance(usage.get('prompt_tokens'), int)
            and isinstance(usage.get('candidates_tokens'), int))


def _last_by_id(records, key):
    result = {}
    for record in records:
        result[record[key]] = record
    return result


def check_smoke(outdir, pre):
    records = _last_by_id(_jsonl(Path(outdir)/'smoke.jsonl'), 'probe_id')
    if len(records) != 10:
        raise G5Error(f'Need 10 distinct real smoke probes, found {len(records)}')
    for n in range(10):
        entry = records.get(f'smoke-{n+1:02d}')
        if not entry or entry.get('pre_sha256') != sha256(Path(outdir)/'manifest_pre.json'):
            raise G5Error('Smoke provenance/preflight mismatch')
        if not entry.get('accepted') or not _accepted_call(entry['call']):
            raise G5Error('Smoke contains unsuccessful or unverified provider request')
    versions = {r['call']['model_version'] for r in records.values()}
    if len(versions) != 1:
        raise G5Error('Smoke responses have inconsistent modelVersion values')
    return records, next(iter(versions))


def smoke(outdir):
    pre = read_pre(outdir)
    if not os.getenv('GEMINI_API_KEY'):
        raise G5Error('No GEMINI_API_KEY configured: live smoke was NOT attempted')
    out = Path(outdir)
    records = _last_by_id(_jsonl(out/'smoke.jsonl'), 'probe_id')
    cases = _check_suite(pre)
    pre_sha = sha256(out/'manifest_pre.json')
    for n, (i, seed) in enumerate(SMOKE_PROBES, start=1):
        probe = f'smoke-{n:02d}'
        if probe in records and records[probe].get('accepted'):
            if records[probe].get('pre_sha256') != pre_sha:
                raise G5Error('Cannot resume smoke under changed preflight')
            continue
        case = cases[i]
        entry = {'probe_id': probe, 'pre_sha256': pre_sha,
                 'case_id': case['case_id'], 'seed': seed, 'accepted': False}
        try:
            entry['call'] = _call(case, _provider(pre['candidate_config'], seed),
                                  audit_scope=f'G5|SMOKE|{pre_sha}|{probe}|{seed}|initial')
            entry['accepted'] = _accepted_call(entry['call'])
            if not entry['accepted']:
                entry['failure'] = 'Missing STOP, model version, or usage counters'
        except CallFailure as exc:
            entry['failure'] = {'category': exc.category, 'http_status': exc.status_code,
                                'attempts': exc.attempts}
        _append(out/'smoke.jsonl', entry)
        if not entry['accepted']:
            raise G5Error(f'Live smoke failed at {probe}; details saved privately')
    return check_smoke(out, pre)[1]


def _validation_oracle(cases, refs):
    reference = {r['case_id']: validate_case(r)['observations'] for r in refs}
    if len(reference) != 12:
        raise G5Error('Corruption reference count mismatch')
    clean = {}
    for case in cases:
        cid = case['case_id']
        if cid not in reference:
            clean[cid] = case['observations']  # clean/borderline: observed values are original
        else:
            clean[cid] = reference[cid]
    return VerificationOracle(clean)


def validation(outdir):
    pre = read_pre(outdir)
    _, version = check_smoke(outdir, pre)
    if not os.getenv('GEMINI_API_KEY'):
        raise G5Error('No GEMINI_API_KEY configured: live validation was NOT attempted')
    out = Path(outdir)
    records = _last_by_id(_jsonl(out/'validation_runs.jsonl'), 'case_id')
    cases = _check_suite(pre)
    oracle = _validation_oracle(cases, _jsonl(pre['files']['validation_references']['path']))
    pre_sha = sha256(out/'manifest_pre.json')
    for case in cases:
        cid = case['case_id']
        if cid in records and records[cid].get('status') == 'OK':
            if records[cid].get('pre_sha256') != pre_sha:
                raise G5Error('Cannot resume validation under changed preflight')
            continue
        entry = {'case_id': cid, 'pre_sha256': pre_sha, 'seed': 42,
                 'status': 'FAILED', 'initial': None, 'post': None,
                 'verification_domain': None, 'verification_count': 0}
        provider = _provider(pre['candidate_config'], seed=42)
        try:
            entry['initial'] = _call(case, provider,
                                         audit_scope=f'G5|DEV|{pre_sha}|{cid}|42|initial')
            if not _accepted_call(entry['initial']) or entry['initial']['model_version'] != version:
                raise G5Error('Validation initial call missing metadata or model version changed')
            initial = entry['initial']['report']
            domain = SharedDecisionGate.choose_domain(initial)
            if domain is not None:
                entry['verification_domain'] = domain
                s = oracle.new_session(cid)
                observation = s.verify_domain(case['observations'], domain, case_id=cid)
                entry['verification_count'] = 1
                updated = {**case, 'observations': observation}
                entry['post'] = _call(updated, provider, verified_domain=domain,
                                          audit_scope=f'G5|DEV|{pre_sha}|{cid}|42|post|{domain}')
                if not _accepted_call(entry['post']) or entry['post']['model_version'] != version:
                    raise G5Error('Validation post-call missing metadata or model version changed')
            entry['status'] = 'OK'
        except CallFailure as exc:
            entry['failure'] = {'category': exc.category, 'http_status': exc.status_code,
                                'attempts': exc.attempts}
        except G5Error as exc:
            entry['failure'] = str(exc)
        _append(out/'validation_runs.jsonl', entry)
        if entry['status'] != 'OK':
            raise G5Error(f'Live validation stopped at {cid}; saved failure privately')
    return calibrate_records(_last_by_id(_jsonl(out/'validation_runs.jsonl'), 'case_id').values())


def calibrate_records(records):
    """Replay the single-condition G4 gate for every integer k using live val reports."""
    rows = list(records)
    ids = [row['case_id'] for row in rows]
    if len(ids) != EXPECTED_VALIDATION or len(set(ids)) != EXPECTED_VALIDATION:
        raise G5Error('Calibration requires exactly 40 unique validation records')
    if any(row.get('status') != 'OK' or row.get('seed') != 42 for row in rows):
        raise G5Error('Calibration cannot use mock/failed/wrong-seed records')
    result = []
    for k in range(8):
        gate = SharedDecisionGate(k)
        issues = abstains = verified = 0
        for row in rows:
            cid = row['case_id']
            initial_report = validate_final_report(row['initial']['report'], cid)
            first = gate.initial(initial_report, cid)
            if first.status == 'ISSUE':
                issues += 1
                continue
            if row.get('verification_count') != 1 or row.get('verification_domain') != first.verification_domain or not row.get('post'):
                raise G5Error(f'Missing or inconsistent validation verification: {cid}')
            after = validate_final_report(row['post']['report'], cid)
            outcome = gate.post_verification(initial_report, after, cid, first.verification_domain,
                                             verification_used=True)
            verified += 1
            if outcome.status == 'ISSUE':
                issues += 1
            elif outcome.status == 'ABSTAIN':
                abstains += 1
            else:
                raise G5Error('Unexpected post-gate state')
        result.append({'k': k, 'total': len(rows), 'issues': issues,
                       'abstains': abstains, 'verifications': verified,
                       'coverage': issues/len(rows)})
    # Target 0.80; deterministic tie-breaker chooses the larger k.
    chosen = min(result, key=lambda r: (abs(r['coverage'] - 0.8), -r['k']))
    return {'k': chosen['k'], 'target': 0.8, 'observed_validation_coverage': chosen['coverage'],
            'table': result, 'method': '40 live Single-Agent validation cases, seed 42; offline k replay'}


def freeze(outdir):
    pre = read_pre(outdir)
    _, version = check_smoke(outdir, pre)
    out = Path(outdir)
    records = _last_by_id(_jsonl(out/'validation_runs.jsonl'), 'case_id')
    expected_ids = {c['case_id'] for c in _check_suite(pre)}
    if set(records) != expected_ids:
        raise G5Error('Validation suite does not exactly match 40 frozen IDs')
    if any(r.get('pre_sha256') != sha256(out/'manifest_pre.json') for r in records.values()):
        raise G5Error('Validation provenance changed')
    for r in records.values():
        for phase in ['initial', 'post']:
            if r.get(phase) is not None and (not _accepted_call(r[phase]) or r[phase]['model_version'] != version):
                raise G5Error(f'Unverified validation {phase} call for {r["case_id"]}')
    calibration = calibrate_records(records.values())
    frozen = {'stage': 'G5_FROZEN_AFTER_REAL_GEMINI',
              'manifest_pre_sha256': sha256(out/'manifest_pre.json'),
              'smoke_sha256': sha256(out/'smoke.jsonl'),
              'validation_runs_sha256': sha256(out/'validation_runs.jsonl'),
              'input_hashes': {name: entry['sha256'] for name, entry in pre['files'].items()},
              'model_id': pre['candidate_config']['model_id'],
              'observed_model_version': version,
              'generation': pre['candidate_config'],
              'prompt_version': pre['prompt_version'],
              'seeds': list(SEEDS),
              'shared_gate_k': calibration['k'],
              'calibration': calibration,
              'scope': 'Single-Agent and three-specialist-plus-coordinator study, same cases and gate'}
    fpath = out/'manifest_frozen.json'
    if fpath.exists():
        if json.loads(fpath.read_text()) != frozen:
            raise G5Error('Frozen manifest already exists and differs; do not overwrite it')
        return frozen
    _save_json(fpath, frozen)
    return frozen


def main(argv=None):
    ap = argparse.ArgumentParser(description='G5. Private output dir: never commit data or traces.')
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--snapshot-root', required=True)
    p.add_argument('--source-csv', required=True)
    p.add_argument('--private-output', required=True)
    p.add_argument('--model-id', default='gemini-3.5-flash-lite')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--thinking-budget', type=int)
    p.add_argument('--max-output-tokens', type=int, default=4096)
    for name in ('smoke', 'validation', 'freeze'):
        sub.add_parser(name).add_argument('--private-output', required=True)
    args = ap.parse_args(argv)
    try:
        if args.step == 'preflight':
            pre = create_preflight(snapshot_root=args.snapshot_root, source_csv=args.source_csv,
                  outdir=args.private_output, model_id=args.model_id,
                  temperature=args.temperature, thinking_budget=args.thinking_budget,
                  max_output_tokens=args.max_output_tokens, top_p=args.top_p)
            print('G5 preflight verified, candidate model:', pre['candidate_config']['model_id'],
                  '(NOT frozen)')
        elif args.step == 'smoke':
            print('G5 live smoke modelVersion:', smoke(args.private_output))
        elif args.step == 'validation':
            print('G5 VALIDATION ONLY calibration:', json.dumps(validation(args.private_output)))
        else:
            frozen = freeze(args.private_output)
            print('G5 frozen k:', frozen['shared_gate_k'])
    except (G5Error, FileNotFoundError, json.JSONDecodeError) as exc:
        ap.exit(2, f'G5 NOT COMPLETE: {exc}\n')


if __name__ == '__main__':
    main()
