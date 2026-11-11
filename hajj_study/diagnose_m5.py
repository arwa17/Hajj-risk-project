"""Read-only M5 attribution audit, based on frozen single-agent Test traces.

No model calls and no raw inputs or individual case IDs in printed/saved output.
Exploratory attribution counts do not establish a causal explanation for Gemini.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hajj_study import g5
from hajj_study.g6 import _frozen, _test_suite
from hajj_study.score_runs import CORRUPTION_TYPES, RANK, ScoreError, score_private_directory

LEVELS = tuple(RANK)


def _matrix(counter):
    return {actual: {predicted: counter[actual, predicted] for predicted in LEVELS}
            for actual in LEVELS}


def analyze(cases, gold, metadata, records, seeds):
    """Pure aggregate calculation; caller separately verifies frozen provenance."""
    case_index = {x['case_id']: x for x in cases}
    gold_index = {x['case_id']: x['risk_level'] for x in gold}
    meta_index = {x['case_id']: x for x in metadata}
    if (len(case_index) != len(cases) or len(gold_index) != len(gold) or len(meta_index) != len(metadata)
            or set(case_index) != set(gold_index) or set(case_index) != set(meta_index)):
        raise ScoreError('Test IDs do not match between cases/gold/metadata')
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ScoreError('Three distinct frozen seeds required')
    latest = {}
    for row in records:
        if not isinstance(row, dict) or 'run_key' not in row:
            raise ScoreError('Malformed G6 record')
        latest[row['run_key']] = row
    corrupted = {cid for cid, m in meta_index.items() if m['case_group'] in CORRUPTION_TYPES}
    expected = {f'test|{cid}|{seed}' for cid in case_index for seed in seeds}
    expected |= {f'clean_reference|{cid}|{seed}' for cid in corrupted for seed in seeds}
    if set(latest) != expected:
        raise ScoreError('Missing or unexpected M5 Test/clean-reference run keys')

    ml, raw, issued, ml_vs_raw = (Counter() for _ in range(4))
    totals, by_group, by_seed, by_truth = Counter(), Counter(), {}, Counter()
    pair_counts, pair_by_type = Counter(), Counter()
    for cid, case in case_index.items():
        truth = gold_index[cid]
        shared_ml = case['ml_prediction']['predicted_risk']
        group = meta_index[cid]['case_group']
        if truth not in RANK or shared_ml not in RANK:
            raise ScoreError('Unknown risk class')
        ml[truth, shared_ml] += 1
        totals['unique_test_cases'] += 1
        for seed in seeds:
            key = f'test|{cid}|{seed}'
            row = latest[key]
            if row.get('event') != 'DONE' or row.get('final_status') not in ('ISSUE','ABSTAIN'):
                raise ScoreError('M5 audit requires completed G6 trials')
            first = row.get('initial')
            if not first or not first.get('report'):
                raise ScoreError('Missing initial Gemini report')
            initial = first['report']['proposed_risk_level']
            if initial not in RANK:
                raise ScoreError('Unknown initial risk class')
            raw[truth, initial] += 1
            ml_vs_raw[shared_ml, initial] += 1
            totals['test_case_seed_runs'] += 1
            by_truth[truth, 'total'] += 1
            by_group[group, 'total'] += 1
            stat = by_seed.setdefault(str(seed), Counter())
            stat['cases'] += 1
            if initial != truth:
                stat['wrong_raw'] += 1
                totals['wrong_raw'] += 1
            if initial != shared_ml:
                totals['raw_disagrees_with_ml'] += 1
            if row['final_status'] == 'ABSTAIN':
                stat['abstained'] += 1
                totals['abstained'] += 1
                by_group[group, 'abstained'] += 1
                by_truth[truth, 'abstained'] += 1
            else:
                final = row.get('final_risk_level')
                if final not in RANK:
                    raise ScoreError('Issued case missing valid final risk')
                issued[truth, final] += 1
                stat['issued'] += 1
                totals['issued'] += 1
                by_truth[truth, 'issued'] += 1
                by_group[group, 'issued'] += 1
                if final != initial:
                    totals['issued_changed_risk_after_gate_or_verification'] += 1
                if final != truth:
                    totals['wrong_issued'] += 1
                    stat['wrong_issued'] += 1
                    by_group[group, 'wrong_issued'] += 1
                    by_truth[truth, 'wrong_issued'] += 1
                    if RANK[final] > RANK[truth]:
                        totals['wrong_over_escalated'] += 1
                    else:
                        totals['wrong_under_escalated'] += 1
                    if shared_ml == truth:
                        totals['wrong_issued_when_ml_correct'] += 1
                    if shared_ml == final:
                        totals['wrong_issued_matching_ml'] += 1
                    if initial != final:
                        totals['wrong_issued_changed_after_verification'] += 1
            if cid in corrupted:
                reference = latest[f'clean_reference|{cid}|{seed}']
                if reference.get('event') != 'DONE' or reference.get('initial') is None:
                    raise ScoreError('Missing paired clean initial response')
                clean_first = reference['initial']['report']['proposed_risk_level']
                kind = meta_index[cid]['corruption_metadata']['kind']
                altered = set(meta_index[cid]['corruption_metadata']['altered_fields'])
                flags = first['report']['evidence_flags']
                correct_flag = any(f['field'] in altered and f['type'] == CORRUPTION_TYPES[kind] for f in flags)
                pair_counts['paired_observations'] += 1
                pair_by_type[kind, 'pairs'] += 1
                if initial != clean_first:
                    pair_counts['raw_risk_changed'] += 1
                    pair_by_type[kind, 'raw_risk_changed'] += 1
                if correct_flag:
                    pair_counts['correct_field_and_type_flag'] += 1
                    pair_by_type[kind, 'correct_field_and_type_flag'] += 1

    if totals['issued'] + totals['abstained'] != totals['test_case_seed_runs']:
        raise ScoreError('Invalid issued/abstained sum')
    return {
        'study_stage': 'M5_READ_ONLY_DIAGNOSTIC',
        'note': 'Proxy is a reconstructed reference, not clinical/operational truth. Association is not causation.',
        'seeds': list(seeds),
        'counts': dict(sorted(totals.items())),
        'confusion': {
            'shared_ml_vs_proxy_30_cases': _matrix(ml),
            'gemini_initial_vs_proxy_case_seed': _matrix(raw),
            'gemini_initial_vs_shared_ml_case_seed': _matrix(ml_vs_raw),
            'issued_final_vs_proxy_case_seed': _matrix(issued),
        },
        'per_seed': {k: dict(sorted(v.items())) for k,v in sorted(by_seed.items())},
        'by_case_group': {group: dict(sorted({k[1]: v for k, v in by_group.items() if k[0] == group}.items()))
                          for group in sorted({k[0] for k in by_group})},
        'by_true_class': {label: dict(sorted({k[1]: v for k,v in by_truth.items() if k[0] == label}.items()))
                          for label in LEVELS},
        'corruption_pair_diagnostics': dict(sorted(pair_counts.items())),
        'pair_diagnostics_by_type': {kind: dict(sorted({k[1]:v for k,v in pair_by_type.items() if k[0] == kind}.items()))
                                     for kind in sorted({k[0] for k in pair_by_type})},
    }


def _validate_against_official(result, scored, seeds):
    """Cross-check diagnostic counts against the official scorer's field names.

    The official scorer uses `incorrect_issued`; the diagnosis uses the
    semantically equivalent label `wrong_issued`. A former same-name lookup
    falsely reported a mismatch whenever any issued decision was wrong.
    """
    key_map = {'issued': 'issued', 'abstained': 'abstained',
               'wrong_issued': 'incorrect_issued'}
    for seed in seeds:
        stats = scored['per_seed'][str(seed)]['counts']
        found = result['per_seed'][str(seed)]
        for diagnostic_key, scorer_key in key_map.items():
            if found.get(diagnostic_key, 0) != stats.get(scorer_key, 0):
                raise ScoreError(
                    f'Diagnostic {diagnostic_key} differs from official '
                    f'{scorer_key} for seed {seed}'
                )


def diagnose(g5_dir='artifacts/g5', g6_dir='artifacts/g6'):
    frozen, pre, _ = _frozen(g5_dir)
    scored = score_private_directory(Path(g5_dir), Path(g6_dir))
    if scored['unique_run_keys'] != 117 or scored['seeds'] != [11,42,77]:
        raise ScoreError('Expected complete frozen 117-key, three-seed M5 study')
    cases, _, _, _ = _test_suite(pre, frozen)
    root = Path(pre['files']['test_cases']['path']).parent
    manifest = json.loads(Path(pre['files']['case_manifest']['path']).read_text(encoding='utf-8'))
    def protected(name):
        spec = manifest['suite_files']['test'][name]
        path = root / Path(spec['file']).name
        if g5.sha256(path) != spec['sha256']:
            raise ScoreError(f'Frozen {name} hash mismatch')
        return g5._jsonl(path)
    result = analyze(cases, protected('gold_labels'), protected('private_metadata'),
                     g5._jsonl(Path(g6_dir)/'single_runs.jsonl'), tuple(frozen['seeds']))
    result['frozen_manifest_sha256'] = g5.sha256(Path(g5_dir)/'manifest_frozen.json')
    _validate_against_official(result, scored, tuple(frozen['seeds']))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g5-dir', default='artifacts/g5')
    parser.add_argument('--g6-dir', default='artifacts/g6')
    parser.add_argument('--output', default='artifacts/g6/m5_diagnosis.json')
    args=parser.parse_args(argv)
    try:
        report=diagnose(args.g5_dir,args.g6_dir)
        g5._save_json(args.output,report)
        print('M5 DIAGNOSTIC VERIFIED:',json.dumps(report,sort_keys=True))
    except (ScoreError, ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f'M5 DIAGNOSTIC FAILED: {exc}\n')


if __name__=='__main__':
    main()
