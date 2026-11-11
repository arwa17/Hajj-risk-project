"""M6.4--M6.6 private offline paired scoring, cost audit and failure examples.

Creates a machine-readable PRIVATE result artifact, never a presentation or report
in GitHub. All risk labels and case IDs remain local to the researcher's Mac.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from hajj_study import g5, g6, g7, m6_fairness as fair, score_runs as score


class M6ResultsError(ValueError):
    pass


def _measured(row):
    calls = row.get('call_log')
    done = row.get('event', 'DONE') == 'DONE'
    expected_calls = 4 if row['verification_count'] == 0 else 6
    if (not isinstance(calls, list) or
            (done and len(calls) != expected_calls) or
            (not done and (row.get('event') != 'FAILED' or
                           not 0 <= len(calls) <= expected_calls))):
        raise M6ResultsError('Multi run has incorrect number of role calls')
    failures = row.get('previous_failures', [])
    if not isinstance(failures, list):
        raise M6ResultsError('Missing/invalid prior attempt history')
    attempts = [a for c in calls for a in c['attempts']]
    if any(c.get('model_version') != row['model_version_frozen'] or not g5._accepted_call(c)
           for c in calls):
        raise M6ResultsError('Unverified role modelVersion/STOP/usage')
    for failure in failures:
        attempts.extend(failure.get('attempts', []))
    known_tokens = {key: sum((a.get('usage') or {}).get(key, 0) or 0 for a in attempts)
                    for key in ('prompt_tokens', 'candidates_tokens', 'thoughts_tokens', 'total_tokens')}
    missing = sum(not isinstance((a.get('usage') or {}).get('prompt_tokens'), int)
                  or not isinstance((a.get('usage') or {}).get('candidates_tokens'), int)
                  for a in attempts)
    service_ms = sum(a.get('elapsed_ms', 0) for a in attempts
                     if isinstance(a.get('elapsed_ms'), (int, float)) and
                     math.isfinite(a['elapsed_ms']) and a['elapsed_ms'] >= 0)
    return {'logical_calls': len(calls) + len(failures), 'api_attempts': len(attempts),
            'known_tokens': known_tokens, 'unknown_usage_attempts': missing,
            'service_ms': service_ms, 'valid_calls': len(calls)}


def _cost(rows, refs):
    measurements = [_measured(x) for x in rows]
    ref = [_measured(x) for x in refs]
    n = len(rows)
    if n == 0:
        raise M6ResultsError('No primary Test records')
    agg = lambda name: sum(x[name] for x in measurements)
    ref_agg = lambda name: sum(x[name] for x in ref)
    token_keys = ('prompt_tokens', 'candidates_tokens', 'thoughts_tokens', 'total_tokens')
    times = [x['wall_elapsed_ms'] for x in rows]
    return {
      'mean_logical_agent_calls_per_case': agg('logical_calls') / n,
      'mean_api_attempts_per_case': agg('api_attempts') / n,
      'mean_known_provider_service_ms_per_case': agg('service_ms') / n,
      'mean_full_wall_ms_per_case': sum(times)/n if all(isinstance(t, (int,float)) and math.isfinite(t) and t>=0 for t in times) else None,
      'full_wall_measurements_available': sum(isinstance(t, (int,float)) and math.isfinite(t) and t>=0 for t in times),
      'total_logical_calls': agg('logical_calls'),
      'total_api_attempts': agg('api_attempts'),
      'unknown_usage_attempts': agg('unknown_usage_attempts'),
      'known_tokens': {k:sum(v['known_tokens'][k] for v in measurements) for k in token_keys},
      'tokens_are_complete': agg('unknown_usage_attempts') == 0,
      'clean_reference_overhead': {'logical_calls': ref_agg('logical_calls'),
         'api_attempts': ref_agg('api_attempts'),
         'unknown_usage_attempts': ref_agg('unknown_usage_attempts'),
         'known_tokens': {k:sum(v['known_tokens'][k] for v in ref) for k in token_keys}}
    }


def _recompute_multi_cost(scored, rows, seeds):
    for seed in seeds:
        s = str(seed)
        primary = [r for r in rows.values() if r['stage']=='test' and r['seed']==seed]
        reference = [r for r in rows.values() if r['stage']=='clean_reference' and r['seed']==seed]
        if len(primary) != 30 or len(reference) != 9:
            raise M6ResultsError('Per-seed Multi run count mismatch')
        scored['per_seed'][s]['cost_latency'] = _cost(primary, reference)
        total = sum(_measured(x)['logical_calls'] for x in primary)
        valid = sum(_measured(x)['valid_calls'] for x in primary)
        scored['per_seed'][s]['metrics']['validated_structured_call_rate'] = score.ratio(valid, total)
    for key in ('mean_logical_agent_calls_per_case','mean_api_attempts_per_case',
                'mean_known_provider_service_ms_per_case','mean_full_wall_ms_per_case'):
        per = {str(s):scored['per_seed'][str(s)]['cost_latency'][key] for s in seeds}
        nums = [v for v in per.values() if v is not None]
        scored['summary'][key] = {'mean':statistics.mean(nums) if nums else None,
            'sample_sd':statistics.stdev(nums) if len(nums)>=2 else None,
            'valid_seeds':len(nums),'seed_values':per}
    key='validated_structured_call_rate'
    per={str(s):scored['per_seed'][str(s)]['metrics'][key]['value'] for s in seeds}
    nums=[v for v in per.values() if v is not None]
    scored['summary'][key]={'mean':statistics.mean(nums) if nums else None,
        'sample_sd':statistics.stdev(nums) if len(nums)>=2 else None,
        'valid_seeds':len(nums),'seed_values':per}
    scored['notes'].append('Multi cost and validation rates include ALL specialist and coordinator calls, not only coordinator final reports.')
    return scored


def _intersection(single, multi, gold, metadata, seeds):
    result={}
    for seed in seeds:
        shared=[]
        for cid in gold:
            key=f'test|{cid}|{seed}'
            a,b=single[key],multi[key]
            if a['final_status']=='ISSUE' and b['final_status']=='ISSUE':
                shared.append((cid,a['final_risk_level'],b['final_risk_level']))
        mismatches=lambda position: sum(r[position]!=gold[r[0]]['risk_level'] for r in shared)
        over=lambda position:sum(score.RANK[r[position]]>score.RANK[gold[r[0]]['risk_level']] for r in shared)
        result[str(seed)]={'common_issued':len(shared), 'total_test':len(gold),
            'single_errors':mismatches(1), 'multi_errors':mismatches(2),
            'single_over_escalations':over(1), 'multi_over_escalations':over(2),
            'single_conditional_error':score.ratio(mismatches(1),len(shared)),
            'multi_conditional_error':score.ratio(mismatches(2),len(shared))}
    return result


def _actual_examples(single,multi,gold,metadata,seeds,limit=12):
    cases=[]
    for seed in seeds:
        for cid in gold:
            key=f'test|{cid}|{seed}'
            a,b=single[key],multi[key]
            truth=gold[cid]['risk_level']
            a_wrong=a['final_status']=='ISSUE' and a['final_risk_level']!=truth
            b_wrong=b['final_status']=='ISSUE' and b['final_risk_level']!=truth
            multi_failed = b['final_status'] == 'FAILED'
            if not (a_wrong or b_wrong or multi_failed):
                continue
            cases.append({'run_key':key, 'group':metadata[cid]['case_group'],
                'proxy_risk':truth, 'single_status':a['final_status'],
                'single_risk':a['final_risk_level'], 'multi_status':b['final_status'],
                'multi_risk':b['final_risk_level'], 'single_wrong_issued':a_wrong,
                'multi_wrong_issued':b_wrong,
                'multi_failed':multi_failed,
                'multi_failure_category':(b.get('failure') or {}).get('category') if multi_failed else None,
                'multi_failure_role':(b.get('failure') or {}).get('role') if multi_failed else None,
                'single_initial_flags':a.get('initial',{}).get('report',{}).get('evidence_flags',[]) if a.get('initial') else [],
                'multi_initial_flags':b.get('initial',{}).get('report',{}).get('evidence_flags',[]) if b.get('initial') else [],
                'multi_verification_domain':b.get('verification_domain'),
                'multi_roles':len(b.get('call_log',[]))})
    # Inspect both improvements and harms; no invented examples or cherry-picked conclusions.
    cases.sort(key=lambda x:(not x['multi_failed'],
                             x['single_wrong_issued']==x['multi_wrong_issued'],
                             x['group'],x['run_key']))
    return cases[:limit]


def _operational_comparison(single, multi, original, treatment, gold, seeds):
    """Direct paired outcome from ALL original held-out Test case/seed keys.

    A wrong ISSUE, ABSTAIN or FAILED cannot count as a correct issued decision.
    Equal-weight paired cases make the primary percentage transparent; failure
    counts stay separate to avoid portraying model failure as safe abstention.
    """
    n = len(gold) * len(seeds)
    def aggregate(scored):
        per = [scored['per_seed'][str(seed)]['counts'] for seed in seeds]
        result = {name: sum(c.get(name, 0) for c in per) for name in (
            'issued', 'correct_issued', 'incorrect_issued', 'abstained', 'failed')}
        if result['issued'] != result['correct_issued'] + result['incorrect_issued']:
            raise M6ResultsError('Issued decision count is inconsistent')
        if result['issued'] + result['abstained'] + result['failed'] != n:
            raise M6ResultsError('All-case decision count is inconsistent')
        result['correct_issued_pct_of_all_test_runs'] = 100.0 * result['correct_issued'] / n
        return result

    a = aggregate(single)
    b = aggregate(multi)
    paired = {'both_correct': 0, 'single_only_correct': 0,
              'multi_only_correct': 0, 'neither_correct': 0}
    for seed in seeds:
        for cid in gold:
            key = f'test|{cid}|{seed}'
            one, two = original[key], treatment[key]
            correct_a = (one['final_status'] == 'ISSUE' and
                         one['final_risk_level'] == gold[cid]['risk_level'])
            correct_b = (two['final_status'] == 'ISSUE' and
                         two['final_risk_level'] == gold[cid]['risk_level'])
            field = ('both_correct' if correct_a and correct_b else
                     'single_only_correct' if correct_a else
                     'multi_only_correct' if correct_b else 'neither_correct')
            paired[field] += 1
    if sum(paired.values()) != n:
        raise M6ResultsError('Paired outcome count mismatch')
    if a['correct_issued'] != paired['both_correct'] + paired['single_only_correct']:
        raise M6ResultsError('Single outcome and scored decisions disagree')
    if b['correct_issued'] != paired['both_correct'] + paired['multi_only_correct']:
        raise M6ResultsError('Multi outcome and scored decisions disagree')
    winner = ('single' if a['correct_issued'] > b['correct_issued'] else
              'multi' if b['correct_issued'] > a['correct_issued'] else 'tie')
    return {'primary_metric': 'correct_issued_pct_of_all_test_runs',
            'n_test_run_keys': n, 'single': a, 'multi': b,
            'paired_correctness': paired,
            'winner_on_primary_metric': winner,
            'interpretation': ('Winner is on correct issued decisions from all frozen test runs, '
                               'not proof of universally better safety performance. '
                               'Wrong decisions, abstentions, failures and costs remain separate.')}


def _require_quota_recovery(rows):
    """Do not attribute HTTP 429 provider-capacity failures to the agent model."""
    pending = [key for key, row in rows.items() if g7._is_rate_limited(row)]
    if pending:
        raise M6ResultsError(
            f'{len(pending)} HTTP 429 cases still need provider retry; '
            'complete quota recovery before the final comparison')


def score_private(g5_dir='artifacts/g5',g6_dir='artifacts/g6',g7_dir='artifacts/g7'):
    frozen, pre, test, refs, _, expected, plan = g7._plan(g5_dir,g6_dir)
    g7_dir=Path(g7_dir)
    planned_file=g7_dir/'run_manifest.json'
    if not planned_file.is_file():
        raise M6ResultsError('G7 frozen treatment run manifest missing')
    fair.assert_treatment_manifest(g6._read_json(planned_file), plan)
    rows=g7._existing(g7_dir,expected,plan)
    fair.assert_multi_rows_paired(expected,rows,plan,require_complete=True)
    _require_quota_recovery(rows)
    for key,row in rows.items():
        checkpoint=g7_dir/'checkpoints'/(g7._safe_name(key)+'.json')
        if not checkpoint.is_file() or g5.sha256(checkpoint)!=row.get('private_checkpoint_sha256'):
            raise M6ResultsError('Missing/modified private Gemini role checkpoint')
        data=g6._read_json(checkpoint)
        if len(data.get('calls',[]))!=len(row.get('call_log',[])):
            raise M6ResultsError('Checkpoint role count differs from saved G7 row')
        expected_calls = 4 if row['verification_count'] == 0 else 6
        if row['event'] == 'DONE':
            if len(row['call_log']) != expected_calls:
                raise M6ResultsError('Multi topology/verification call count differs')
        elif row['event'] == 'FAILED':
            failure = row.get('failure')
            if (row.get('final_status') != 'FAILED' or
                    row.get('final_risk_level') is not None or
                    not isinstance(failure, dict)):
                raise M6ResultsError('Unrecorded/inconsistent Multi failure')
            category = failure.get('category')
            contract_failure = category in ('CONTRACT', 'G6_CONTRACT')
            if (len(row['call_log']) > expected_calls or
                    (not contract_failure and
                     (len(row['call_log']) >= expected_calls or
                      not row.get('previous_failures')))):
                raise M6ResultsError('Unrecorded/inconsistent Multi failure')
        else:
            raise M6ResultsError('Unexpected Multi event')
        _measured(row)
    # Read labels only NOW, after all G7 inference records are complete.
    cases_dir=Path(pre['files']['test_cases']['path']).parent
    case_manifest=g6._read_json(pre['files']['case_manifest']['path'])
    def protected(name):
        spec=case_manifest['suite_files']['test'][name]
        path=cases_dir/Path(spec['file']).name
        if g5.sha256(path)!=spec['sha256']:
            raise M6ResultsError('Gold/metadata frozen checksum mismatch')
        return g5._jsonl(path)
    gold_list=protected('gold_labels')
    metadata_list=protected('private_metadata')
    original=g6._existing(Path(g6_dir)/'single_runs.jsonl',plan['g5_frozen_manifest_sha256'])
    single_scores=score.score_private_directory(Path(g5_dir),Path(g6_dir))
    multi_scores=score.score_condition(list(rows.values()),gold_list,metadata_list,
        fingerprint=plan['g5_frozen_manifest_sha256'],
        expected_version=plan['observed_model_version'],condition='multi',
        seeds=tuple(plan['seeds']))
    multi_scores=_recompute_multi_cost(multi_scores,rows,tuple(plan['seeds']))
    summary={}
    for metric in single_scores['summary']:
        if metric not in multi_scores['summary']:
            continue
        a=single_scores['summary'][metric]
        b=multi_scores['summary'][metric]
        diffs={str(seed):(b['seed_values'][str(seed)]-a['seed_values'][str(seed)]
            if b['seed_values'][str(seed)] is not None and a['seed_values'][str(seed)] is not None
            else None) for seed in plan['seeds']}
        valid=[v for v in diffs.values() if v is not None]
        summary[metric]={'single_mean':a['mean'],'single_sd':a['sample_sd'],
            'multi_mean':b['mean'],'multi_sd':b['sample_sd'],
            'multi_minus_single_mean':statistics.mean(valid) if valid else None,
            'paired_diff_sample_sd':statistics.stdev(valid) if len(valid)>=2 else None,
            'paired_delta_per_seed':diffs}
    gold={x['case_id']:x for x in gold_list}
    metadata={x['case_id']:x for x in metadata_list}
    intersection=_intersection(original,rows,gold,metadata,plan['seeds'])
    operational=_operational_comparison(single_scores,multi_scores,original,rows,gold,plan['seeds'])
    a_cov=single_scores['summary']['coverage']['mean']
    b_cov=multi_scores['summary']['coverage']['mean']
    return {'status':'M6_OFFLINE_PAIRED_SCORE_FROM_REAL_TRACES',
            'frozen_manifest_sha256':plan['g5_frozen_manifest_sha256'],
            'multi_manifest_sha256':fair._fingerprint(plan),
            'n_test_per_seed':30,'n_reference_per_seed':9,'seeds':plan['seeds'],
            'coverage_statement': ('both_conditions_meet_80_percent' if a_cov>=.8 and b_cov>=.8 else
                                   'at_least_one_condition_below_80_percent'),
            'coverage_equal_observed':a_cov==b_cov,
            'paired_comparison':summary,
            'operational_comparison':operational,
            'single_scores':single_scores,'multi_scores':multi_scores,
            'common_issued_sensitivity':intersection,
            'actual_failure_trace_examples_PRIVATE':_actual_examples(original,rows,gold,metadata,plan['seeds']),
            'limitations':['Simulation and programmatic proxy are not true Hajj safety ground truth.',
                'Common-issued intersection is selected on both model outputs; not a fixed-coverage experiment.',
                'Three seed repeats on the same cases are not 90 independent cases.',
                'No 80% fixed-coverage claim unless actual reported coverage supports it.',
                'No confidence interval or significance test inferred from these records.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Private M6.4-M6.6 offline analysis; no Gemini calls')
    parser.add_argument('--g5-dir',default='artifacts/g5')
    parser.add_argument('--g6-dir',default='artifacts/g6')
    parser.add_argument('--g7-dir',default='artifacts/g7')
    parser.add_argument('--output',default='artifacts/g7/m6_scores.json')
    args=parser.parse_args(argv)
    try:
        result=score_private(args.g5_dir,args.g6_dir,args.g7_dir)
        g5._save_json(args.output,result)
        coverage=result['paired_comparison']['coverage']
        primary=result['operational_comparison']
        print('M6 PAIRED SCORING VERIFIED:',json.dumps({'output':args.output,
            'expected_paired_runs':117,
            'primary_winner':primary['winner_on_primary_metric'],
            'single_correct_issued':primary['single']['correct_issued'],
            'multi_correct_issued':primary['multi']['correct_issued'],
            'multi_failed':primary['multi']['failed'],
            'single_coverage':coverage['single_mean'],
            'multi_coverage':coverage['multi_mean'],
            'coverage_statement':result['coverage_statement']},sort_keys=True))
    except (M6ResultsError, g7.G7Error, fair.FairnessError, g6.G6Error,
            g5.G5Error, score.ScoreError, ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2,f'M6 ANALYSIS NOT COMPLETE: {exc}\n')

if __name__=='__main__':
    main()
