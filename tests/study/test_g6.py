"""Offline acceptance tests for G6. Never contact Gemini or use real Test gold."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hajj_study import g6
from hajj_study.provider import CallFailure, ProviderError
from hajj_study.verify import VerificationOracle


CID = 'SYNTHETIC001'
CLEAN = {'Crowd_Density': 'Low', 'Temperature': 39.,
         'Fatigue_Level': 'High', 'Stress_Level': 'Medium',
         'Queue_Time_minutes': 50., 'Emergency_Event': 'No',
         'Distance_Between_People_m': 1.2}
PRED = {'predicted_risk': 'MEDIUM',
        'probabilities': {'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}
FROZEN = {'model_id': 'test-frozen-model', 'observed_model_version': 'test-frozen-model',
          'generation': {'model_id': 'test-frozen-model', 'temperature': 0.,
                         'max_output_tokens': 4096, 'thinking_budget': None,
                         'seed_policy': [11, 42, 77], 'calibration_seed': 42},
          'prompt_version': 'test-frozen-prompt', 'shared_gate_k': 7}


def case():
    return {'case_id': CID, 'observations': dict(CLEAN), 'ml_prediction': dict(PRED)}


def report(flags=None, level='HIGH'):
    return {'case_id': CID, 'proposed_risk_level': level,
            'evidence_flags': flags or [], 'supporting_evidence': ['Test synthetic evidence.']}


def flag(field='Temperature'):
    return {'field': field, 'type': 'missing', 'reason': 'Missing field.'}


def call(r=None, model='test-frozen-model'):
    return {'report': r or report(), 'model_version': model,
            'raw_response_private': json.dumps(r or report()), 'elapsed_ms': 12.,
            'attempts': [{'attempt': 1, 'status': 'STOP', 'finish_reason': 'STOP',
                          'model_version': model,
                          'usage': {'prompt_tokens': 5, 'candidates_tokens': 8,
                                    'thoughts_tokens': 0, 'total_tokens': 13},
                          'elapsed_ms': 12., 'response_sha256': 'synthetic'}]}


class PacerTest(unittest.TestCase):
    def test_pace_without_first_delay(self):
        t = [10.]
        waits = []
        def sleep(sec):
            waits.append(sec)
            t[0] += sec
        p = g6.Pacer(8., sleeper=sleep, clock=lambda: t[0])
        p.wait()
        self.assertEqual(waits, [])
        t[0] += 2
        p.wait()
        self.assertEqual(waits, [6.])

    def test_negative_interval_rejected(self):
        with self.assertRaises(ValueError):
            g6.Pacer(-1)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.oracle = VerificationOracle({CID: CLEAN})

    def test_run_key_and_expected_grid(self):
        cases = [dict(case(), case_id=f'SYNT{i:03d}') for i in range(30)]
        refs = cases[:9]
        keys = [g6._run_key(s, c['case_id'], seed)
                for s, c, seed in g6._entries(cases, refs, [11,42,77])]
        self.assertEqual(len(keys), 117)
        self.assertEqual(len(set(keys)), 117)
        self.assertEqual(keys[0], 'test|SYNT000|11')
        self.assertEqual(keys[30], 'clean_reference|SYNT000|11')

    def test_single_issue_no_verification(self):
        with patch('hajj_study.g6.g5._provider', return_value=object()), \
             patch('hajj_study.g6.g5._call', return_value=call()) as invoke:
            row = g6._one('test', case(), 42, oracle=self.oracle,
                          frozen=FROZEN, frozen_sha='private_hash')
        self.assertEqual((row['event'],row['final_status'],row['final_risk_level']),
                         ('DONE', 'ISSUE','HIGH'))
        self.assertEqual(row['verification_count'],0)
        self.assertEqual(invoke.call_count,1)
        self.assertEqual(row['input_observations'],CLEAN)

    def test_verification_is_only_one_domain(self):
        obs = dict(CLEAN)
        obs['Temperature'] = None
        initial = call(report([flag()]))
        post = call(report([], 'MEDIUM'))
        with patch('hajj_study.g6.g5._provider', return_value=object()), \
             patch('hajj_study.g6.g5._call', side_effect=[initial,post]) as invoke:
            row = g6._one('test', {**case(),'observations':obs},11,
                          oracle=self.oracle, frozen=FROZEN, frozen_sha='hash')
        self.assertEqual((row['event'],row['final_status'],row['verification_count']),
                         ('DONE','ISSUE',1))
        self.assertEqual(row['verification_domain'],'environment')
        self.assertEqual(row['verified_observations']['Temperature'],39.)
        self.assertEqual(invoke.call_count,2)

    def test_failed_post_resumes_without_second_initial(self):
        obs = dict(CLEAN)
        obs['Temperature'] = None
        initial = call(report([flag()]))
        failure = CallFailure(ProviderError('HTTP 429', category='HTTP',status_code=429,retryable=True),
                              [{'status':'HTTP','http_status':429,'attempt':1}])
        with patch('hajj_study.g6.g5._provider', return_value=object()), \
             patch('hajj_study.g6.g5._call', side_effect=[initial,failure]) as invoke:
            first = g6._one('test', {**case(),'observations':obs},42,
                            oracle=self.oracle,frozen=FROZEN,frozen_sha='hash')
        self.assertEqual(first['event'],'FAILED')
        self.assertEqual(first['failure']['http_status'],429)
        self.assertEqual(first['verification_count'],1)
        with patch('hajj_study.g6.g5._provider', return_value=object()), \
             patch('hajj_study.g6.g5._call', return_value=call(report([], 'LOW'))) as invoke:
            second = g6._one('test', {**case(),'observations':obs},42,
                             oracle=self.oracle,frozen=FROZEN,frozen_sha='hash',prior=first)
        self.assertEqual(second['event'],'DONE')
        self.assertEqual(second['final_risk_level'],'LOW')
        self.assertEqual(invoke.call_count,1)  # only post request repeated
        self.assertEqual(second['previous_failures'], [first['failure']])
        self.assertGreaterEqual(second['active_elapsed_ms'], first['active_elapsed_ms'])
        # Wall time includes elapsed pacing, but active time excludes it.
        # Both measurements accumulate execution segments, not idle time between resumes.
        self.assertGreaterEqual(second['wall_elapsed_ms'], second['active_elapsed_ms'])
        self.assertGreaterEqual(second['wall_elapsed_ms'], first['wall_elapsed_ms'])
        # Both the original failed post request and the successful resume count.
        from hajj_study.score_runs import _attempts_and_usage
        logical, attempted, *_ = _attempts_and_usage(second)
        self.assertEqual((logical, attempted), (3, 3))

    def test_model_version_drift_halts_without_success(self):
        with patch('hajj_study.g6.g5._provider', return_value=object()), \
             patch('hajj_study.g6.g5._call', return_value=call(model='different')):
            r=g6._one('test',case(),77,oracle=self.oracle,frozen=FROZEN,frozen_sha='hash')
        self.assertEqual((r['event'],r['final_status']),('FAILED','FAILED'))
        self.assertEqual(r['failure']['category'],'G6_CONTRACT')

    def test_existing_provenance_refuses_altered_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'records.jsonl'
            p.write_text(json.dumps({'run_key':'test|SYNTHETIC001|42',
                                     'frozen_manifest_sha256':'wrong'})+'\n')
            with self.assertRaisesRegex(g6.G6Error,'incompatible'):
                g6._existing(p,'correct')

    def test_existing_records_never_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'records.jsonl'
            p.write_text('')
            row={'run_key':'test|SYNTHETIC001|42','event':'DONE',
                 'frozen_manifest_sha256':'h','final_status':'ISSUE'}
            with p.open('a') as f:
                f.write(json.dumps(row)+'\n')
                f.write(json.dumps({**row,'event':'FAILED','final_status':'FAILED'})+'\n')
            self.assertEqual(g6._existing(p,'h')[row['run_key']]['event'],'FAILED')

    def test_read_only_inspect_rejects_missing_freeze(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(g6.G6Error,'frozen manifest missing'):
                g6.inspect(t, str(Path(t)/'g6'))

    def test_validate_prior_catches_input_drift(self):
        prior={'stage':'test','case_id':CID,'seed':42,'model_id':FROZEN['model_id'],
               'model_version_frozen':FROZEN['observed_model_version'],
               'prompt_version': FROZEN['prompt_version'], 'condition': 'single',
               'shared_gate_k':7, 'input_observations':dict(CLEAN),
               'ml_prediction':PRED,'event':'FAILED'}
        modified=case()
        modified['observations']['Temperature']=40.
        with self.assertRaisesRegex(g6.G6Error,'frozen input'):
            g6._validate_prior(prior,'test',modified,42,FROZEN)


if __name__=='__main__':
    unittest.main()
