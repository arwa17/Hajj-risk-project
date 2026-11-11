"""Offline safety tests for M6.3 runner; never contact Gemini or use Test gold."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hajj_study import g7
from hajj_study.gate import SharedDecisionGate
from hajj_study.verify import VerificationOracle
from hajj_study.provider import CallFailure, ProviderError

CID = 'SYNTH-G7'
CLEAN = {'Crowd_Density': 'Low', 'Queue_Time_minutes': 50.,
         'Distance_Between_People_m': 1.2, 'Fatigue_Level': 'High',
         'Stress_Level': 'Medium', 'Emergency_Event': 'No', 'Temperature': 39.}
ML = {'predicted_risk': 'MEDIUM', 'probabilities': {'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}


def case(obs=None):
    return {'case_id': CID, 'observations': copy.deepcopy(CLEAN if obs is None else obs),
            'ml_prediction': copy.deepcopy(ML)}


def specialist(domain):
    return {'case_id': CID, 'domain': domain, 'domain_risk_signal': 'MEDIUM',
            'evidence_flags': [], 'supporting_evidence': ['Synthetic domain evidence.']}


def decision(flags=None, level='MEDIUM'):
    return {'case_id': CID, 'proposed_risk_level': level, 'evidence_flags': flags or [],
            'supporting_evidence': ['Synthetic specialist evidence.']}


class FakeProvider:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate_validated(self, *, system_prompt, user_prompt, response_schema, validator):
        self.prompts.append(json.loads(user_prompt))
        next_value = self.answers.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return SimpleNamespace(report=validator(next_value), output_text=json.dumps(next_value),
            model_version='mock-frozen-model', total_elapsed_ms=7.,
            attempts=[{'attempt': 1, 'status': 'STOP', 'finish_reason': 'STOP',
                'model_version': 'mock-frozen-model', 'usage': {'prompt_tokens': 12,
                'candidates_tokens': 8, 'thoughts_tokens': 0, 'total_tokens': 20},
                'elapsed_ms': 7.}])


def failure():
    return CallFailure(ProviderError('HTTP 429', category='HTTP', retryable=True,
        status_code=429), [{'attempt': 1, 'status': 'HTTP', 'http_status': 429}])


def answers(initial=None):
    return [specialist('crowd'), specialist('health'), specialist('environment'),
            initial or decision()]


PLAN = {'g5_frozen_manifest_sha256': 'a'*64, 'model_id': 'mock-frozen-model',
        'observed_model_version': 'mock-frozen-model', 'generation': {'temperature': 0.0},
        'shared_gate_k': 7, 'prompt_version': 'g4-candidate-v1',
        'treatment_code_sha256': {'multi_agent': 'b'*64},
        'min_seconds_between_logical_calls': 0.0}
DEV_SMOKE_PLAN = {**PLAN, 'stage': 'synthetic_independent_development_smoke'}

FROZEN = {'shared_gate_k': 7, 'observed_model_version': 'mock-frozen-model',
          'generation': {'temperature': 0.0}}


class G7OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.oracle = VerificationOracle({CID: CLEAN})

    def one(self, provider, *, obs=None):
        return g7._one('test', case(obs), 42, self.oracle, FROZEN, PLAN,
                       self.root, provider=provider, pacer=g7.g6.Pacer(0))

    def test_four_real_provider_calls_are_recorded_with_usage(self):
        provider = FakeProvider(answers())
        row = self.one(provider)
        self.assertEqual((row['event'], row['final_status']), ('DONE', 'ISSUE'))
        self.assertEqual(len(provider.prompts), 4)
        self.assertEqual(len(row['call_log']), 4)
        self.assertEqual(len(row['multi_trace']['api_attempts']), 4)
        self.assertEqual(row['initial']['report']['proposed_risk_level'], 'MEDIUM')
        self.assertEqual(row['verification_count'], 0)
        self.assertIn('private_checkpoint_sha256', row)
        self.assertNotIn('raw_response_private', json.dumps(row))
        self.assertTrue(list((self.root / 'checkpoints').glob('*.json')))
        self.assertIn('raw_response_private', list((self.root / 'checkpoints').glob('*.json'))[0].read_text())

    def test_six_calls_only_when_domain_verification_required(self):
        obs = copy.deepcopy(CLEAN)
        obs['Temperature'] = None
        first = decision([{'field': 'Temperature', 'type': 'missing', 'reason': 'Missing.'}])
        provider = FakeProvider(answers(first)+[specialist('environment'), decision([], 'HIGH')])
        row = self.one(provider, obs=obs)
        self.assertEqual(row['event'], 'DONE')
        self.assertEqual(row['verification_domain'], 'environment')
        self.assertEqual(row['verification_count'], 1)
        self.assertEqual(row['final_risk_level'], 'HIGH')
        self.assertEqual(len(row['call_log']), 6)
        self.assertEqual(len(provider.prompts), 6)
        self.assertIsNone(row['initial']['report'].get('gold_label'))

    def test_successful_calls_replayed_with_no_second_network_calls(self):
        original = FakeProvider(answers())
        first = self.one(original)
        second = self.one(FakeProvider([]))
        self.assertEqual(len(original.prompts), 4)
        self.assertEqual(len(second['call_log']), 4)
        self.assertEqual(second['final_status'], 'ISSUE')
        self.assertEqual(len(second['call_log']), 4)

    def test_partial_failure_checkpoints_and_resume_exact_role(self):
        attempted = FakeProvider([specialist('crowd'), failure()])
        row = self.one(attempted)
        self.assertEqual(row['event'], 'FAILED')
        self.assertEqual(len(row['call_log']), 1)
        self.assertEqual(len(row['previous_failures']), 1)
        self.assertEqual(row['previous_failures'][0]['attempts'][0]['http_status'], 429)
        recovered = FakeProvider([specialist('health'), specialist('environment'), decision()])
        after = self.one(recovered)
        self.assertEqual(after['event'], 'DONE')
        self.assertEqual(len(recovered.prompts), 3)
        self.assertEqual(len(after['call_log']), 4)
        self.assertEqual(len(after['previous_failures']), 1)
        self.assertGreaterEqual(after['active_elapsed_ms'], row['active_elapsed_ms'])

    def test_continue_flag_skips_existing_failed_key_without_recalling_it(self):
        first = case()
        second = case()
        second['case_id'] = 'SYNTH-SECOND'
        expected = {
            'test|SYNTH-G7|42': ('test', first, 42),
            'test|SYNTH-SECOND|42': ('test', second, 42)}
        plan_sha = g7._hash(DEV_SMOKE_PLAN)
        g7.g5._save_json(self.root / 'validation_smoke.json',
                         {'manifest_sha256': plan_sha, 'event': 'DONE',
                          'calls': 4, 'model_version': PLAN['observed_model_version']})
        old = {'run_key':'test|SYNTH-G7|42','event':'FAILED',
               'failure':{'category':'INVALID_REPORT'}}
        new = {'run_key':'test|SYNTH-SECOND|42','event':'DONE'}
        # The test fixture must include the same seed policy read by g7.run.
        # This is synthetic test setup only, not the frozen study manifest.
        fake_frozen = {**FROZEN, 'seeds': [11, 42, 77]}
        with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}), \
             patch('hajj_study.g7._dev_smoke_plan', return_value=(FROZEN, {}, DEV_SMOKE_PLAN)), \
             patch('hajj_study.g7._plan',
                   return_value=(fake_frozen, {}, [first, second], [], self.oracle, expected, PLAN)), \
             patch('hajj_study.g7._existing',return_value={old['run_key']:old}), \
             patch('hajj_study.g7.g6._entries',
                   return_value=[('test',first,42),('test',second,42)]), \
             patch('hajj_study.g7._one', return_value=new) as execute, \
             patch('hajj_study.g7.fair.assert_multi_rows_paired'), \
             patch('hajj_study.g7.g5._append') as save, \
             patch('hajj_study.g7.inspect',
                   return_value={'status':'G7_IN_PROGRESS'}) as inspect:
            got = g7.run(out=self.root, max_cases=1, continue_on_failure=True)
        self.assertEqual(got['status'],'G7_IN_PROGRESS')
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[1]['case_id'], 'SYNTH-SECOND')
        save.assert_called_once()
        inspect.assert_called_once()

    def test_http_429_recovery_requires_explicit_flag_and_replays_saved_key(self):
        first = case()
        key = 'test|SYNTH-G7|42'
        expected = {key: ('test', first, 42)}
        old = {'run_key':key, 'event':'FAILED',
               'failure':{'category':'HTTP', 'http_status':429}}
        completed = {'run_key':key, 'event':'DONE', 'failure':None}
        frozen = {**FROZEN, 'seeds':[11,42,77]}
        g7.g5._save_json(self.root / 'validation_smoke.json',
                         {'manifest_sha256':g7._hash(DEV_SMOKE_PLAN), 'event':'DONE',
                          'calls':4, 'model_version':PLAN['observed_model_version']})
        with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}), \
             patch('hajj_study.g7._dev_smoke_plan', return_value=(FROZEN, {}, DEV_SMOKE_PLAN)), \
             patch('hajj_study.g7._plan',
                   return_value=(frozen,{},[first],[],self.oracle,expected,PLAN)), \
             patch('hajj_study.g7._existing',return_value={key:old}), \
             patch('hajj_study.g7.g6._entries',return_value=[('test',first,42)]), \
             patch('hajj_study.g7._one',return_value=completed) as execute, \
             patch('hajj_study.g7.fair.assert_multi_rows_paired'), \
             patch('hajj_study.g7.g5._append'), \
             patch('hajj_study.g7.inspect',return_value={'status':'G7_IN_PROGRESS'}):
            g7.run(out=self.root,max_cases=1,continue_on_failure=True)
            execute.assert_not_called()
            g7.run(out=self.root,max_cases=1,continue_on_failure=True,retry_http_429=True)
            execute.assert_called_once()
        self.assertTrue(g7._is_rate_limited(old))
        self.assertFalse(g7._is_rate_limited(completed))

    def test_new_http_429_stops_batch_even_with_continue_on_failure(self):
        first = case()
        key = 'test|SYNTH-G7|42'
        expected = {key: ('test', first, 42)}
        limited = {'run_key':key, 'event':'FAILED',
                   'failure':{'category':'HTTP', 'http_status':429}}
        frozen = {**FROZEN, 'seeds':[11,42,77]}
        g7.g5._save_json(self.root / 'validation_smoke.json',
                         {'manifest_sha256':g7._hash(DEV_SMOKE_PLAN), 'event':'DONE',
                          'calls':4, 'model_version':PLAN['observed_model_version']})
        with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}), \
             patch('hajj_study.g7._dev_smoke_plan', return_value=(FROZEN, {}, DEV_SMOKE_PLAN)), \
             patch('hajj_study.g7._plan',
                   return_value=(frozen,{},[first],[],self.oracle,expected,PLAN)), \
             patch('hajj_study.g7._existing',return_value={}), \
             patch('hajj_study.g7.g6._entries',return_value=[('test',first,42)]), \
             patch('hajj_study.g7._one',return_value=limited), \
             patch('hajj_study.g7.fair.assert_multi_rows_paired'), \
             patch('hajj_study.g7.g5._append') as persist:
            with self.assertRaisesRegex(g7.G7Error, 'HTTP 429'):
                g7.run(out=self.root,max_cases=20,continue_on_failure=True)
            persist.assert_called_once()

    def test_retry_rate_limit_flag_without_continuation_is_rejected(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}):
            with self.assertRaisesRegex(g7.G7Error, 'requires --continue-on-failure'):
                g7.run(max_cases=1, retry_http_429=True)

    def test_version_drift_causes_no_done_record(self):
        other = FakeProvider(answers())
        other.generate_validated = lambda **k: SimpleNamespace(
            report=k['validator'](specialist('crowd')), output_text='{}',
            model_version='wrong-model', total_elapsed_ms=10.,
            attempts=[{'status':'STOP','finish_reason':'STOP','usage':
                       {'prompt_tokens':12,'candidates_tokens':8}}])
        with self.assertRaisesRegex(g7.G7Error, 'modelVersion'):
            self.one(other)

    def test_replay_refuses_different_request_bytes(self):
        p = FakeProvider([specialist('crowd')])
        key = 'test|SYNTH-G7|42'
        path = self.root / 'checkpoint.json'
        c = g7.CheckpointProvider(path, key, 'hash', p, 'mock-frozen-model', 0.)
        c.generate_validated(system_prompt='sys',user_prompt='{"message": "same"}',response_schema={'type':'OBJECT'},
                             validator=lambda x:x)
        r = g7.CheckpointProvider(path, key, 'hash', FakeProvider([]), 'mock-frozen-model', 0.)
        with self.assertRaisesRegex(g7.G7Error, 'prompt/schema changed'):
            r.generate_validated(system_prompt='sys',user_prompt='{"message": "changed"}',
                                 response_schema={'type':'OBJECT'},validator=lambda x:x)

    def test_refuse_incompatible_checkpoint_provenance(self):
        path = self.root / 'cp.json'
        g7.CheckpointProvider(path, 'key', 'manifest1', FakeProvider([]), 'mock-frozen-model', 0.)._save()
        with self.assertRaisesRegex(g7.G7Error, 'Incompatible checkpoint'):
            g7.CheckpointProvider(path, 'key', 'manifest2', FakeProvider([]), 'mock-frozen-model', 0.)

    def test_test_runner_cannot_start_without_live_validation_smoke(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}), \
             patch('hajj_study.g7._plan', return_value=(FROZEN, {}, [case()], [], self.oracle, {}, PLAN)):
            with self.assertRaisesRegex(g7.G7Error, 'Development smoke'):
                g7.run('g5', 'g6', self.root)

    def test_read_only_inspect_requires_real_frozen(self):
        with self.assertRaisesRegex(g7.g6.G6Error, 'frozen manifest missing'):
            g7.inspect(self.root / 'not-real-g5', self.root / 'not-real-g6', self.root)

    def test_max_cases_validation_no_live_calls(self):
        with self.assertRaisesRegex(g7.G7Error, 'max_cases must'):
            with patch.dict('os.environ', {'GEMINI_API_KEY':'fake'}):
                g7.run(max_cases=118)

    def test_no_api_key_smoke_blocks_before_provider_call(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(g7.G7Error, 'GEMINI_API_KEY'):
                g7.smoke('not-real-g5', 'not-real-g6', self.root)


if __name__ == '__main__':
    unittest.main()
