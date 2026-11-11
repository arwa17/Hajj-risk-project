"""Offline score contract checks; synthetic measurements are NOT project results."""
import copy
import unittest

from hajj_study import m6_results as m


def attempt():
    return {'attempt': 1, 'status': 'STOP', 'finish_reason': 'STOP',
            'usage': {'prompt_tokens': 10, 'candidates_tokens': 5,
                      'thoughts_tokens': 1, 'total_tokens': 16}, 'elapsed_ms': 7.5}


def row(verify=False):
    calls = 6 if verify else 4
    return {'verification_count': int(verify), 'model_version_frozen': 'synthetic-model',
            'call_log': [{'model_version':'synthetic-model', 'attempts':[attempt()]}
                         for _ in range(calls)],
            'previous_failures': [], 'wall_elapsed_ms': 200.0 if verify else 120.0}


class M6ResultTests(unittest.TestCase):
    def test_all_four_role_calls_counted_not_coordinator_only(self):
        measured=m._measured(row())
        self.assertEqual(measured['logical_calls'],4)
        self.assertEqual(measured['api_attempts'],4)
        self.assertEqual(measured['known_tokens']['total_tokens'],64)
        self.assertEqual(measured['service_ms'],30.)
        self.assertEqual(measured['valid_calls'],4)

    def test_six_role_calls_after_single_verification(self):
        self.assertEqual(m._measured(row(True))['logical_calls'],6)
        self.assertEqual(m._measured(row(True))['api_attempts'],6)

    def test_failed_previous_requests_preserve_cost_and_denom(self):
        changed=row()
        changed['previous_failures']=[{'category':'HTTP','attempts':[
            {'attempt':1,'status':'HTTP','http_status':429}]}]
        measured=m._measured(changed)
        self.assertEqual(measured['logical_calls'],5)
        self.assertEqual(measured['api_attempts'],5)
        self.assertEqual(measured['unknown_usage_attempts'],1)

    def test_terminal_failed_role_remains_failure_and_counts_attempts(self):
        failed=row()
        failed['event']='FAILED'
        failed['final_status']='FAILED'
        failed['call_log']=failed['call_log'][:3]
        failed['previous_failures']=[{'category':'INVALID_REPORT','attempts':[
            {'attempt':1,'status':'INVALID_REPORT','usage':{'prompt_tokens':7,'candidates_tokens':3}},
            {'attempt':2,'status':'INVALID_REPORT','usage':{'prompt_tokens':7,'candidates_tokens':3}}]}]
        x=m._measured(failed)
        self.assertEqual(x['valid_calls'],3)
        self.assertEqual(x['logical_calls'],4)
        self.assertEqual(x['api_attempts'],5)
        rejected=row()
        rejected['call_log']=rejected['call_log'][:3]
        with self.assertRaisesRegex(m.M6ResultsError,'role calls'):
            m._measured(rejected)

    def test_primary_comparison_counts_failed_as_not_correct(self):
        def results(counts):
            return {'per_seed':{'11':{'counts':counts}}}
        a=results({'issued':2,'correct_issued':1,'incorrect_issued':1,
                   'abstained':0,'failed':0})
        b=results({'issued':1,'correct_issued':1,'incorrect_issued':0,
                   'abstained':0,'failed':1})
        gold={'A':{'risk_level':'LOW'}, 'B':{'risk_level':'HIGH'}}
        single={'test|A|11':{'final_status':'ISSUE','final_risk_level':'LOW'},
                'test|B|11':{'final_status':'ISSUE','final_risk_level':'LOW'}}
        multi={'test|A|11':{'final_status':'FAILED','final_risk_level':None},
               'test|B|11':{'final_status':'ISSUE','final_risk_level':'HIGH'}}
        out=m._operational_comparison(a,b,single,multi,gold,[11])
        self.assertEqual(out['winner_on_primary_metric'],'tie')
        self.assertEqual(out['multi']['failed'],1)
        self.assertEqual(out['multi']['correct_issued_pct_of_all_test_runs'],50.)
        self.assertEqual(out['paired_correctness']['single_only_correct'],1)
        self.assertEqual(out['paired_correctness']['multi_only_correct'],1)

    def test_quota_failures_must_be_retried_before_final_scoring(self):
        rows = {'a':{'event':'DONE', 'failure':None},
                'b':{'event':'FAILED',
                     'failure':{'category':'HTTP','http_status':429}}}
        with self.assertRaisesRegex(m.M6ResultsError, 'HTTP 429 cases'):
            m._require_quota_recovery(rows)
        rows['b'] = {'event':'FAILED','failure':{'category':'INVALID_REPORT'}}
        m._require_quota_recovery(rows)

    def test_cost_is_per_test_case_and_refs_are_separate_overhead(self):
        c=m._cost([row(),row(True)], [row()])
        self.assertEqual(c['mean_logical_agent_calls_per_case'],5)
        self.assertEqual(c['mean_api_attempts_per_case'],5)
        self.assertEqual(c['mean_full_wall_ms_per_case'],160.)
        self.assertEqual(c['clean_reference_overhead']['logical_calls'],4)
        self.assertEqual(c['total_logical_calls'],10)

    def test_missing_specialist_role_is_rejected(self):
        bad=row()
        bad['call_log'].pop()
        with self.assertRaisesRegex(m.M6ResultsError,'number of role calls'):
            m._measured(bad)

    def test_wrong_model_version_is_rejected(self):
        bad=row()
        bad['call_log'][0]['model_version']='different'
        with self.assertRaisesRegex(m.M6ResultsError,'Unverified'):
            m._measured(bad)

    def test_common_issued_intersection_excludes_abstain(self):
        gold={'A':{'risk_level':'LOW'},'B':{'risk_level':'MEDIUM'},'C':{'risk_level':'HIGH'}}
        def item(status,risk):
            return {'final_status':status,'final_risk_level':risk}
        first={}; second={}
        for seed in (11,42,77):
            first.update({f'test|A|{seed}':item('ISSUE','LOW'),
                          f'test|B|{seed}':item('ISSUE','HIGH'),
                          f'test|C|{seed}':item('ABSTAIN',None)})
            second.update({f'test|A|{seed}':item('ISSUE','HIGH'),
                           f'test|B|{seed}':item('ISSUE','MEDIUM'),
                           f'test|C|{seed}':item('ISSUE','HIGH')})
        out=m._intersection(first,second,gold,{},(11,42,77))
        self.assertEqual(out['11']['common_issued'],2)
        self.assertEqual(out['11']['single_errors'],1)
        self.assertEqual(out['11']['multi_errors'],1)
        self.assertEqual(out['11']['single_conditional_error']['value'],.5)
        self.assertEqual(out['11']['multi_conditional_error']['value'],.5)

    def test_empty_common_issued_returns_na_not_zero(self):
        gold={'A':{'risk_level':'HIGH'}}
        first={f'test|A|{s}':{'final_status':'ABSTAIN','final_risk_level':None} for s in (11,42,77)}
        second={f'test|A|{s}':{'final_status':'ISSUE','final_risk_level':'HIGH'} for s in (11,42,77)}
        out=m._intersection(first,second,gold,{},(11,42,77))
        self.assertIsNone(out['11']['multi_conditional_error']['value'])
        self.assertEqual(out['11']['common_issued'],0)

    def test_failure_examples_come_from_actual_input_records_only(self):
        gold={'A':{'risk_level':'LOW'}}
        metadata={'A':{'case_group':'typical_clean'}}
        one={};two={}
        for seed in (11,42,77):
            k=f'test|A|{seed}'
            one[k]={'final_status':'ISSUE','final_risk_level':'LOW',
                    'initial':{'report':{'evidence_flags':[]}}}
            two[k]={'final_status':'ISSUE','final_risk_level':'HIGH',
                    'initial':{'report':{'evidence_flags':[]}},
                    'call_log':[1,2,3,4], 'verification_domain':None}
        examples=m._actual_examples(one,two,gold,metadata,(11,42,77))
        self.assertEqual(len(examples),3)
        self.assertEqual(examples[0]['proxy_risk'],'LOW')
        self.assertEqual(examples[0]['multi_risk'],'HIGH')
        self.assertTrue(examples[0]['multi_wrong_issued'])

if __name__=='__main__':
    unittest.main()
