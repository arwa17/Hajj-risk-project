"""Synthetic offline proof that G7 Development smoke does not require G6 Test.

No real case data and no network. Uses only mocked G5 validation, one synthetic
multi result, and a strictly synthetic G6 pairing plan.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hajj_study import g7

G5_FROZEN = {
    'stage': 'G5_FROZEN_AFTER_REAL_GEMINI',
    'seeds': [11, 42, 77],  # synthetic Test-grid fixture, never real Test data
    'shared_gate_k': 3,
    'input_hashes': {'validation_cases': 'a'*64, 'validation_references': 'b'*64},
    'model_id': 'synthetic-model', 'observed_model_version': 'synthetic-model',
    'generation': {'model_id':'synthetic-model', 'temperature':0.0},
    'prompt_version': 'synthetic-prompt-v1',
}

DEV_PLAN = {
    'stage': 'G7_DEV_SMOKE_AFTER_G5_FREEZE_BEFORE_G6_TEST',
    'condition': 'multi', 'g5_frozen_manifest_sha256':'synthetic-frozen',
    'validation_cases_sha256':'a'*64, 'validation_references_sha256':'b'*64,
    'model_id':'synthetic-model','observed_model_version':'synthetic-model',
    'generation': {'model_id':'synthetic-model', 'temperature':0.0},
    'prompt_version':'synthetic-prompt-v1','shared_gate_k':3,
    'seed':42,'selection':'first_of_40_deterministic_development_cases',
    'verification_budget':1,
    'shared_code_sha256':{'proxy':'z'*64},
    'treatment_code_sha256':{'g7':'c'*64},
    'min_seconds_between_logical_calls':8.0,
}


def dev_cases():
    return [{'case_id':f'DEV-SYN-{n:03d}', 'observations':{}, 'ml_prediction':{}}
            for n in range(40)]


class DevSmokeDecouplingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.out=self.root/'g7'
        self.pre={'files':{
            'validation_cases':{'path':str(self.root/'dev_cases.jsonl'),'sha256':'a'*64},
            'validation_references':{'path':str(self.root/'dev_refs.jsonl'),'sha256':'b'*64},
        }}
        (self.root/'manifest_frozen.json').write_text('{}')

    def patched_plan(self):
        return patch('hajj_study.g7._dev_smoke_plan',
                     return_value=(G5_FROZEN,self.pre,DEV_PLAN))

    def record(self,event='DONE', calls=4, model='synthetic-model', manifest=None):
        self.out.mkdir(exist_ok=True)
        rec={'manifest_sha256':manifest if manifest is not None else g7._hash(DEV_PLAN),
             'event':event,'calls':calls,'model_version':model}
        (self.out/'validation_smoke.json').write_text(json.dumps(rec))

    def test_dev_plan_only_requires_frozen_g5_not_g6_test_results(self):
        with patch('hajj_study.g7.g5.freeze',return_value=G5_FROZEN) as freeze, \
             patch('hajj_study.g7.g5.read_pre',return_value=self.pre), \
             patch('hajj_study.g7.g6._read_json',return_value=G5_FROZEN), \
             patch('hajj_study.g7.fair._assert_shared_sources',return_value={'proxy':'z'*64}), \
             patch('hajj_study.g7.fair._treatment_code_hashes',return_value={'g7':'c'*64}), \
             patch('hajj_study.g7.g6._test_suite',side_effect=AssertionError('NO TEST ALLOWED')), \
             patch('hajj_study.g7.fair._check_baseline_rows',side_effect=AssertionError('NO G6 BASELINE')):
            frozen,pre,plan=g7._dev_smoke_plan(self.root)
        freeze.assert_called_once()
        self.assertIs(pre,self.pre)
        self.assertEqual(plan['shared_gate_k'],3)
        self.assertEqual(plan['validation_cases_sha256'],'a'*64)
        self.assertNotIn('test_cases_sha256',plan)
        self.assertNotIn('test_clean_references_sha256',plan)

    def test_dev_smoke_uses_first_dev_case_and_never_calls_g6_plan(self):
        cases=dev_cases()
        fake_result={'event':'DONE','final_status':'ISSUE','call_log':[{}]*4,
                     'verification_count':0,'failure':None}
        with patch.dict('os.environ',{'GEMINI_API_KEY':'SYNTHETIC_NO_NETWORK'}), \
             self.patched_plan(), \
             patch('hajj_study.g7._plan',side_effect=AssertionError('NO G6 TEST')), \
             patch('hajj_study.g7.g5._jsonl',side_effect=[cases,[{}]*12]), \
             patch('hajj_study.g7.g5.validate_case',side_effect=lambda c:c), \
             patch('hajj_study.g7.g5._validation_oracle',return_value=object()), \
             patch('hajj_study.g7._one',return_value=fake_result) as one:
            result=g7.smoke('unused-g5','ABSENT-G6-TEST-MANIFEST',self.out)
        self.assertEqual(result['status'],'VALIDATION_SMOKE_VERIFIED')
        self.assertEqual(one.call_args.args[0],'validation')
        self.assertEqual(one.call_args.args[1]['case_id'],'DEV-SYN-000')
        self.assertEqual(one.call_args.args[2],42)
        saved=json.loads((self.out/'validation_smoke.json').read_text())
        self.assertEqual(saved['manifest_sha256'],g7._hash(DEV_PLAN))

    def test_failed_dev_smoke_is_not_silently_retried(self):
        self.record(event='FAILED')
        with patch.dict('os.environ',{'GEMINI_API_KEY':'SYNTHETIC_NO_NETWORK'}), \
             self.patched_plan(), patch('hajj_study.g7._one') as one:
            with self.assertRaisesRegex(g7.G7Error,'no automatic retry'):
                g7.smoke('unused','ABSENT-G6-TEST',self.out)
            one.assert_not_called()

    def test_run_rejects_legacy_treatment_manifest_smoke_proof(self):
        self.record(manifest='old-manifest-hash')
        with patch.dict('os.environ',{'GEMINI_API_KEY':'SYNTHETIC_NO_NETWORK'}), \
             self.patched_plan(), \
             patch('hajj_study.g7._plan',return_value=(G5_FROZEN,{},[],[],object(),{},DEV_PLAN)), \
             patch('hajj_study.g7._one') as one:
            with self.assertRaisesRegex(g7.G7Error,'Development smoke does not authenticate'):
                g7.run('g5','g6',self.out)
            one.assert_not_called()

    def test_run_accepts_independent_dev_smoke_but_keeps_g6_pairing(self):
        self.record()
        fake_plan={**DEV_PLAN}
        with patch.dict('os.environ',{'GEMINI_API_KEY':'SYNTHETIC_NO_NETWORK'}), \
             self.patched_plan(), \
             patch('hajj_study.g7._plan',return_value=(G5_FROZEN,{},[],[],object(),{},fake_plan)) as check_g6, \
             patch('hajj_study.g7._existing',return_value={}), \
             patch('hajj_study.g7.g6._entries',return_value=iter(())), \
             patch('hajj_study.g7.inspect',return_value={'status':'G7_IN_PROGRESS'}), \
             patch('hajj_study.g7._one') as one:
            result=g7.run('frozen-g5','paired-g6',self.out)
        self.assertEqual(result['status'],'G7_IN_PROGRESS')
        check_g6.assert_called_once_with('frozen-g5','paired-g6')
        one.assert_not_called()

    def test_run_rejects_pacing_drift_before_any_test_call(self):
        self.record()
        bad_plan={**DEV_PLAN,'min_seconds_between_logical_calls':6.0}
        with patch.dict('os.environ',{'GEMINI_API_KEY':'SYNTHETIC_NO_NETWORK'}), \
             self.patched_plan(), \
             patch('hajj_study.g7._plan',return_value=(G5_FROZEN,{},[],[],object(),{},bad_plan)), \
             patch('hajj_study.g7._one') as one:
            with self.assertRaisesRegex(g7.G7Error,'differs from independently checked Development smoke'):
                g7.run('g5','g6',self.out)
            one.assert_not_called()

    def test_no_api_key_means_no_g5_or_g6_or_calls(self):
        with patch.dict('os.environ',{},clear=True), \
             patch('hajj_study.g7._dev_smoke_plan') as prep, \
             patch('hajj_study.g7._one') as one:
            with self.assertRaisesRegex(g7.G7Error,'GEMINI_API_KEY'):
                g7.smoke(out=self.out)
            prep.assert_not_called()
            one.assert_not_called()


if __name__=='__main__':
    unittest.main()
