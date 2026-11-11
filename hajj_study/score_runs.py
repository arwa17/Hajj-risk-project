"""Offline, gold-isolated M5 Single-Agent scorer. Never invokes Gemini.

The scorer reads private G6 test traces only after inference has finished.
Clean-reference raw proposals are matched to corrupted cases by case ID+seed.
Zero-denominator rates are N/A, never silently 0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from hajj_study import g5
from hajj_study.schemas import validate_final_report

RANK = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
CORRUPTION_TYPES = {'missing': 'missing', 'noisy': 'suspected_noisy',
                    'conflicting': 'conflicting'}
CLEAN_GROUPS = {'typical_clean', 'borderline_clean'}
SEEDS = (11, 42, 77)


class ScoreError(ValueError):
    """A scoring input or denominator violates the frozen study contract."""


def ratio(numerator: int, denominator: int) -> dict:
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise ScoreError('Metric counts must be integers')
    if not 0 <= numerator <= denominator:
        raise ScoreError('Invalid numerator/denominator')
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def _index(items, name):
    result = {}
    for x in items:
        if not isinstance(x, dict) or not isinstance(x.get('case_id'), str):
            raise ScoreError(f'Invalid {name} row')
        if x['case_id'] in result:
            raise ScoreError(f'Duplicate {name} case ID')
        result[x['case_id']] = x
    return result


def _latest(rows):
    latest = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get('run_key'):
            raise ScoreError('Unkeyed G6 run row')
        latest[row['run_key']] = row
    return latest


def _first_report(row, expected_version):
    call = row.get('initial')
    if call is None:
        return None
    if call.get('model_version') != expected_version:
        raise ScoreError('Actual modelVersion differs from G5 frozen version')
    if not g5._accepted_call(call):
        raise ScoreError('Unaccepted initial provider call in scoring trace')
    return validate_final_report(call.get('report'), row['case_id'])


def _check_run(row, stage, case_id, seed, fingerprint, version, condition):
    expected_key = f'{stage}|{case_id}|{seed}'
    if (row.get('run_key') != expected_key or row.get('case_id') != case_id or
            row.get('seed') != seed or row.get('stage') != stage or
            row.get('condition') != condition or
            row.get('frozen_manifest_sha256') != fingerprint or
            row.get('model_version_frozen') != version):
        raise ScoreError(f'Run provenance mismatch: {expected_key}')
    if row.get('event') not in ('DONE', 'FAILED'):
        raise ScoreError('Run has no terminal event')
    status = row.get('final_status')
    if row['event'] == 'DONE':
        if status not in ('ISSUE', 'ABSTAIN'):
            raise ScoreError('DONE must ISSUE or ABSTAIN')
        _first_report(row, version)
        if status == 'ISSUE' and row.get('final_risk_level') not in RANK:
            raise ScoreError('Issued decision requires a valid risk level')
        if status == 'ABSTAIN' and row.get('final_risk_level') is not None:
            raise ScoreError('ABSTAIN must not return a risk level')
    elif status != 'FAILED' or row.get('final_risk_level') is not None:
        raise ScoreError('FAILED cannot be scored as a safe abstention')
    if row.get('verification_count') not in (0, 1):
        raise ScoreError('Verification budget exceeded')
    for phase in ('initial', 'post'):
        call = row.get(phase)
        if call is not None:
            if call.get('model_version') != version or not g5._accepted_call(call):
                raise ScoreError('Unaccepted/mismatched provider call')
            validate_final_report(call.get('report'), case_id)
    if row.get('post') is not None and row['verification_count'] != 1:
        raise ScoreError('Post report without verification')
    if row['event'] == 'DONE' and row['verification_count'] == 1 and row.get('post') is None:
        raise ScoreError('Verified DONE without post report')


def _attempts_and_usage(row):
    attempts = []
    logical = 0
    for phase in ('initial', 'post'):
        call = row.get(phase)
        if call is not None:
            logical += 1
            attempts.extend(call.get('attempts', []))
    # Append-only resumes may contain earlier FAILED requests for the same
    # case. Count those HTTP/model attempts, but not successful initial calls
    # repeated as metadata in the latest resumed DONE record.
    previous_failures = row.get('previous_failures', [])
    if not isinstance(previous_failures, list):
        raise ScoreError('Invalid previous-failure history')
    for failure in [row.get('failure'), *previous_failures]:
        if failure is None:
            continue
        if not isinstance(failure, dict):
            raise ScoreError('Invalid provider failure history')
        prior_attempts = failure.get('attempts', [])
        if not isinstance(prior_attempts, list):
            raise ScoreError('Invalid provider attempt history')
        if prior_attempts:
            logical += 1
            attempts.extend(prior_attempts)
    tokens = {'prompt_tokens': 0, 'candidates_tokens': 0,
              'thoughts_tokens': 0, 'total_tokens': 0}
    missing_usage = 0
    known_service_ms = 0.0
    for item in attempts:
        usage = item.get('usage') or {}
        if not isinstance(usage.get('prompt_tokens'), int) or not isinstance(usage.get('candidates_tokens'), int):
            missing_usage += 1
        for key in tokens:
            if isinstance(usage.get(key), int):
                tokens[key] += usage[key]
        ms = item.get('elapsed_ms')
        if isinstance(ms, (int, float)) and math.isfinite(ms) and ms >= 0:
            known_service_ms += ms
    return logical, len(attempts), tokens, missing_usage, known_service_ms


def _summarize_seed(rows, clean_rows, gold, metadata, version):
    counts = defaultdict(int)
    groups = defaultdict(lambda: {'cases': 0, 'issued': 0, 'abstained': 0, 'failed': 0,
                                  'correct_issued': 0})
    raw_changed = 0
    changed_detected = 0
    recovery = 0
    observable_pairs = 0
    pairs_with_correct_flag = 0
    unobservable_pairs = 0
    true_flags = predicted_flags = altered_fields_seen = 0
    clean_false_flag_cases = clean_observable_cases = 0
    attempted_calls = logical_calls = unknown_usage_attempts = 0
    total_tokens = defaultdict(int)
    service_ms = 0.0
    valid_structured_calls = 0
    logical_reference = attempts_reference = 0
    reference_tokens = defaultdict(int)
    missing_reference_usage = 0
    wall_ms = []
    examples = {'missed_high': 0, 'high_abstain': 0, 'false_high': 0}
    for cid, row in rows.items():
        status, prediction = row['final_status'], row.get('final_risk_level')
        truth = gold[cid]['risk_level']
        group = metadata[cid]['case_group']
        bucket = groups[group]
        bucket['cases'] += 1
        counts['total'] += 1
        if status == 'ISSUE':
            counts['issued'] += 1
            bucket['issued'] += 1
            if prediction == truth:
                counts['correct_issued'] += 1
                bucket['correct_issued'] += 1
            else:
                counts['incorrect_issued'] += 1
            if RANK[prediction] > RANK[truth]:
                counts['over'] += 1
            if RANK[prediction] < RANK[truth]:
                counts['under'] += 1
            if truth == 'HIGH' and prediction != 'HIGH':
                examples['missed_high'] += 1
            if prediction == 'HIGH' and truth != 'HIGH':
                examples['false_high'] += 1
        elif status == 'ABSTAIN':
            counts['abstained'] += 1
            bucket['abstained'] += 1
            if truth == 'HIGH':
                examples['high_abstain'] += 1
        else:
            counts['failed'] += 1
            bucket['failed'] += 1
        logical, attempts, tokens, missing_usage, provider_ms = _attempts_and_usage(row)
        logical_calls += logical
        attempted_calls += attempts
        unknown_usage_attempts += missing_usage
        valid_structured_calls += int(row.get('initial') is not None) + int(row.get('post') is not None)
        service_ms += provider_ms
        for k, v in tokens.items():
            total_tokens[k] += v
        elapsed = row.get('wall_elapsed_ms')
        if isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= 0:
            wall_ms.append(elapsed)
        initial = _first_report(row, version)
        if group in CLEAN_GROUPS:
            if initial is not None:
                clean_observable_cases += 1
                if initial['evidence_flags']:
                    clean_false_flag_cases += 1
            continue
        if group not in CORRUPTION_TYPES:
            raise ScoreError(f'Unknown case group: {group}')
        info = metadata[cid].get('corruption_metadata') or {}
        if info.get('kind') != group or not isinstance(info.get('altered_fields'), list):
            raise ScoreError('Missing private corruption annotations')
        altered = set(info['altered_fields'])
        if not altered:
            raise ScoreError('Corruption annotations contain no affected fields')
        if initial is not None:
            flags = initial['evidence_flags']
            # Attribute flags to correct FIELD + TYPE, not just any alarm.
            unique_flags = {(x['field'], x['type']) for x in flags}
            predicted_flags += len(unique_flags)
            true_flags += sum((field, CORRUPTION_TYPES[group]) in unique_flags for field in altered)
            altered_fields_seen += len(altered)
        ref = clean_rows.get(cid)
        if ref is None:
            raise ScoreError('Corrupted case missing its paired clean reference')
        clean_first = _first_report(ref, version)
        if initial is None or clean_first is None:
            unobservable_pairs += 1
            continue
        observable_pairs += 1
        # Diagnostic independent of risk-decision changes: did the model flag
        # a truly altered field with the corresponding corruption type?
        if any(f['field'] in altered and f['type'] == CORRUPTION_TYPES[group]
               for f in initial['evidence_flags']):
            pairs_with_correct_flag += 1
        if initial['proposed_risk_level'] != clean_first['proposed_risk_level']:
            raw_changed += 1
            if any(f['field'] in altered and f['type'] == CORRUPTION_TYPES[group]
                   for f in initial['evidence_flags']):
                changed_detected += 1
            if status == 'ISSUE' and prediction == clean_first['proposed_risk_level']:
                recovery += 1
    # Reference runs are evaluation overhead, not normal task decision cost.
    for row in clean_rows.values():
        l, a, t, missing, _ = _attempts_and_usage(row)
        logical_reference += l
        attempts_reference += a
        missing_reference_usage += missing
        for k, v in t.items():
            reference_tokens[k] += v
    if counts['total'] != len(gold):
        raise ScoreError('Seed did not contain all frozen test cases')
    n = counts['total']
    metrics = {
        'coverage': ratio(counts['issued'], n),
        'selective_error': ratio(counts['incorrect_issued'], counts['issued']),
        'unsupported_escalation': ratio(counts['over'], counts['issued']),
        'under_escalation': ratio(counts['under'], counts['issued']),
        'disrupted_case_detection': ratio(changed_detected, raw_changed),
        'clean_decision_recovery': ratio(recovery, raw_changed),
        'trajectory_safe_completion': ratio(counts['correct_issued'] + counts['abstained'], n),
        'field_flag_precision': ratio(true_flags, predicted_flags),
        'field_flag_recall': ratio(true_flags, altered_fields_seen),
        'clean_false_flag_case_rate': ratio(clean_false_flag_cases, clean_observable_cases),
        'failed_case_rate': ratio(counts['failed'], n),
        'abstain_rate': ratio(counts['abstained'], n),
        'validated_structured_call_rate': ratio(valid_structured_calls, logical_calls),
    }
    cost = {'mean_logical_agent_calls_per_case': logical_calls / n,
            'mean_api_attempts_per_case': attempted_calls / n,
            'mean_known_provider_service_ms_per_case': service_ms / n,
            'mean_full_wall_ms_per_case': sum(wall_ms) / n if len(wall_ms) == n else None,
            'full_wall_measurements_available': len(wall_ms),
            'total_logical_calls': logical_calls, 'total_api_attempts': attempted_calls,
            'unknown_usage_attempts': unknown_usage_attempts,
            'known_tokens': dict(total_tokens),
            'tokens_are_complete': unknown_usage_attempts == 0,
            'clean_reference_overhead': {'logical_calls': logical_reference,
                                         'api_attempts': attempts_reference,
                                         'unknown_usage_attempts': missing_reference_usage,
                                         'known_tokens': dict(reference_tokens)}}
    diagnostics = {'raw_changed_pairs': raw_changed, 'observable_pairs': observable_pairs,
                   'unobservable_pairs': unobservable_pairs,
                   # Exploratory component diagnostics, NOT replacements for
                   # the M2 metrics conditional on a changed raw decision.
                   'raw_decision_change_rate': ratio(raw_changed, observable_pairs),
                   'pairs_with_correct_field_flag': pairs_with_correct_flag,
                   'paired_corruption_flag_rate': ratio(pairs_with_correct_flag, observable_pairs),
                   'missed_high_issued': examples['missed_high'],
                   'high_abstained': examples['high_abstain'],
                   'false_high_issued': examples['false_high'],
                   'verified': sum(x['verification_count'] for x in rows.values()),
                   'within_verification_budget': all(x['verification_count'] <= 1 for x in rows.values())}
    return {'cases': n, 'counts': dict(counts), 'metrics': metrics, 'cost_latency': cost,
            'components': diagnostics, 'by_case_group': dict(groups)}


def score_condition(run_records, gold_records, metadata_records, *,
                    fingerprint, expected_version, condition='single',
                    seeds=SEEDS, expected_test_cases=30, expected_clean_refs=9):
    """Strict full-grid scoring; never turns missing Test calls into results."""
    gold = _index(gold_records, 'gold')
    metadata = _index(metadata_records, 'private metadata')
    if len(gold) != expected_test_cases or set(metadata) != set(gold):
        raise ScoreError('Golden set size or private metadata IDs mismatch')
    if any(x.get('risk_level') not in RANK for x in gold.values()):
        raise ScoreError('Invalid frozen gold label')
    if len(set(seeds)) != 3 or any(not isinstance(s, int) for s in seeds):
        raise ScoreError('Three unique experimental seeds are required')
    paired = {cid for cid, meta in metadata.items() if meta.get('case_group') in CORRUPTION_TYPES}
    if len(paired) != expected_clean_refs:
        raise ScoreError('Clean-reference pair count disagrees with frozen G3 suite')
    latest = _latest(run_records)
    expected = {f'test|{cid}|{seed}' for seed in seeds for cid in gold}
    expected |= {f'clean_reference|{cid}|{seed}' for seed in seeds for cid in paired}
    if set(latest) != expected:
        missing, extra = expected - set(latest), set(latest) - expected
        raise ScoreError(f'Incomplete or unexpected run grid: missing={len(missing)}, extra={len(extra)}')
    per_seed = {}
    for seed in seeds:
        test = {cid: latest[f'test|{cid}|{seed}'] for cid in gold}
        refs = {cid: latest[f'clean_reference|{cid}|{seed}'] for cid in paired}
        for cid, row in test.items():
            _check_run(row, 'test', cid, seed, fingerprint, expected_version, condition)
        for cid, row in refs.items():
            _check_run(row, 'clean_reference', cid, seed, fingerprint, expected_version, condition)
        per_seed[str(seed)] = _summarize_seed(test, refs, gold, metadata, expected_version)
    names = list(next(iter(per_seed.values()))['metrics'])
    aggregates = {}
    for name in names:
        values = [per_seed[str(seed)]['metrics'][name]['value'] for seed in seeds]
        valid = [x for x in values if x is not None]
        aggregates[name] = {'mean': statistics.mean(valid) if valid else None,
                            'sample_sd': statistics.stdev(valid) if len(valid) >= 2 else None,
                            'valid_seeds': len(valid), 'seed_values': dict(zip(map(str,seeds), values))}
    for key in ('mean_logical_agent_calls_per_case','mean_api_attempts_per_case',
                'mean_known_provider_service_ms_per_case','mean_full_wall_ms_per_case'):
        valid = [per_seed[str(seed)]['cost_latency'][key] for seed in seeds]
        measured = [x for x in valid if x is not None]
        aggregates[key] = {'mean': statistics.mean(measured) if measured else None,
                           'sample_sd': statistics.stdev(measured) if len(measured) >= 2 else None,
                           'valid_seeds': len(measured),
                           'seed_values': dict(zip(map(str,seeds),valid))}
    return {'status': 'OFFLINE_SCORED_FROM_ACTUAL_TRACES', 'condition': condition,
            'frozen_manifest_sha256': fingerprint, 'model_version': expected_version,
            'unique_run_keys': len(latest), 'expected_test_cases': len(gold),
            'expected_clean_refs': len(paired), 'seeds': list(seeds),
            'per_seed': per_seed, 'summary': aggregates,
            'notes': ['Raw-change pairs use matched clean initial proposal at the SAME seed.',
                      'N/A denotes a zero denominator, never a measured zero.',
                      'Exploratory component diagnostics report raw-decision change rate and correctly flagged corrupted pairs, independent of the conditional primary metrics.',
                      'Failures remain in all-case denominators; failures are not safe abstentions.',
                      'Clean-reference calls are evaluation overhead, separately reported from decision-path cost.',
                      'Provider service time excludes unmeasured failed-request intervals; active case latency includes previous failed attempts on resume and excludes intentional request pacing.',
                      'Synthetic, stress-selected cases are not a population risk prevalence estimate.']}


def _verify_frozen_inputs(run_records, test_cases, clean_references, seeds):
    """Prove G6 traces used the EXACT frozen case observations and ML inputs.

    This operates locally on private evidence and returns only counts.
    No raw observations, gold labels or case IDs go into the score output.
    """
    test = _index(test_cases, 'frozen test cases')
    refs = _index(clean_references, 'frozen clean references')
    if set(refs) - set(test):
        raise ScoreError('Clean-reference case absent from frozen Test suite')
    if not refs:
        raise ScoreError('Frozen Test has no clean reference pairs')
    for cid, reference in refs.items():
        corrupted = test[cid]
        if corrupted.get('observations') == reference.get('observations'):
            raise ScoreError('Frozen clean and corrupted observations are identical')
        if corrupted.get('ml_prediction') != reference.get('ml_prediction'):
            raise ScoreError('Shared ML prediction changed within frozen pair')

    latest = _latest(run_records)
    expected = {}
    for seed in seeds:
        for cid, case in test.items():
            expected[f'test|{cid}|{seed}'] = case
        for cid, case in refs.items():
            expected[f'clean_reference|{cid}|{seed}'] = case
    if set(latest) != set(expected):
        raise ScoreError('Frozen trace grid has missing or unexpected run keys')
    for key, case in expected.items():
        row = latest[key]
        if row.get('input_observations') != case.get('observations'):
            raise ScoreError('Trace observations differ from frozen G3 suite')
        if row.get('ml_prediction') != case.get('ml_prediction'):
            raise ScoreError('Trace ML prediction differs from frozen G3 suite')
    return {'frozen_test_cases': len(test), 'frozen_pairs': len(refs),
            'validated_run_keys': len(expected)}


def score_private_directory(g5_dir: Path, g6_dir: Path):
    """Read all gold/metadata only at scoring time, after G5 has frozen."""
    g5_dir = Path(g5_dir).resolve()
    frozen = g5.freeze(g5_dir)  # offline provenance verification, no network
    pre = g5.read_pre(g5_dir)
    frozen_path = g5_dir / 'manifest_frozen.json'
    fingerprint = g5.sha256(frozen_path)
    source = Path(pre['files']['test_cases']['path']).parent
    manifest = json.loads(Path(pre['files']['case_manifest']['path']).read_text())
    def protected(name):
        spec = manifest['suite_files']['test'][name]
        path = source / Path(spec['file']).name
        if g5.sha256(path) != spec['sha256']:
            raise ScoreError(f'Frozen G3 {name} source differs from checksum')
        rows = g5._jsonl(path)
        if len(rows) != spec['n']:
            raise ScoreError(f'Frozen G3 {name} case count mismatch')
        return rows
    gold = protected('gold_labels')
    metadata = protected('private_metadata')
    frozen_cases = protected('cases')
    frozen_refs = protected('clean_references')
    log_path = Path(g6_dir) / 'single_runs.jsonl'
    run_records = g5._jsonl(log_path)
    _verify_frozen_inputs(run_records, frozen_cases, frozen_refs, tuple(frozen['seeds']))
    return score_condition(run_records, gold, metadata,
                           fingerprint=fingerprint,
                           expected_version=frozen['observed_model_version'],
                           seeds=tuple(frozen['seeds']))


def main(argv=None):
    parser = argparse.ArgumentParser(description='M5 offline scorer: no Gemini network calls')
    parser.add_argument('--g5-dir', default='artifacts/g5')
    parser.add_argument('--g6-dir', default='artifacts/g6')
    parser.add_argument('--output', default='artifacts/g6/m5_scores.json')
    args = parser.parse_args(argv)
    try:
        result = score_private_directory(args.g5_dir, args.g6_dir)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        g5._save_json(output, result)
        print('M5 OFFLINE SCORED:', json.dumps({'run_keys': result['unique_run_keys'],
              'seeds': result['seeds'], 'output': str(output)}))
    except (ScoreError, g5.G5Error, ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'M5 SCORING NOT COMPLETE: {exc}\n')


if __name__ == '__main__':
    main()
