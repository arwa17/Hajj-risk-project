"""M6.2 strict read-only pairing tests using SYNTHETIC cases only."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hajj_study import m6_fairness as fair


OBS = {'Crowd_Density': 'Low', 'Temperature': 39.,
       'Fatigue_Level': 'High', 'Stress_Level': 'Medium',
       'Queue_Time_minutes': 50., 'Emergency_Event': 'No',
       'Distance_Between_People_m': 1.2}
PRED = {'predicted_risk': 'MEDIUM',
        'probabilities': {'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}
SEEDS = [11, 42, 77]
GENERATION = {'model_id': 'mock-frozen-model', 'temperature': 0.0,
              'thinking_budget': None, 'max_output_tokens': 4096,
              'seed_policy': SEEDS, 'calibration_seed': 42}
FAKE_FINGERPRINT = 'a' * 64
REF_SHA = 'b' * 64


def fixture_cases():
    cases = [{'case_id': f'SYNTH{i:03d}', 'observations': copy.deepcopy(OBS),
              'ml_prediction': copy.deepcopy(PRED)} for i in range(30)]
    refs = copy.deepcopy(cases[0:9])
    refs[0]['observations']['Temperature'] = 40.0
    return cases, refs


def frozen():
    return {'model_id': 'mock-frozen-model',
            'observed_model_version': 'mock-frozen-model',
            'generation': copy.deepcopy(GENERATION),
            'prompt_version': 'g4-candidate-v1',
            'seeds': list(SEEDS), 'shared_gate_k': 7,
            'input_hashes': {'test_cases': 'c' * 64,
                             **{name: 'd' * 64 for name in fair.SHARED_MODULES}}}


def baseline_manifest(f=None):
    f = frozen() if f is None else f
    return {'stage': 'G6_SINGLE_PENDING_SCORING',
            'g5_frozen_manifest_sha256': FAKE_FINGERPRINT,
            'model_id': f['model_id'],
            'observed_model_version': f['observed_model_version'],
            'generation': copy.deepcopy(f['generation']),
            'prompt_version': f['prompt_version'],
            'seeds': list(f['seeds']),
            'shared_gate_k': f['shared_gate_k'],
            'test_cases_sha256': f['input_hashes']['test_cases'],
            'test_clean_references_sha256': REF_SHA,
            'min_seconds_between_logical_calls': 8.0}


def accepted_call():
    return {'model_version': 'mock-frozen-model', 'attempts': [
        {'attempt': 1, 'status': 'STOP', 'finish_reason': 'STOP',
         'usage': {'prompt_tokens': 4, 'candidates_tokens': 5}}]}


def baseline_rows(expected, f=None):
    f = frozen() if f is None else f
    return {key: {'run_key': key, 'event': 'DONE',
                  'stage': stage, 'case_id': case['case_id'],
                  'condition': 'single', 'seed': seed,
                  'model_id': f['model_id'],
                  'model_version_frozen': f['observed_model_version'],
                  'prompt_version': f['prompt_version'],
                  'shared_gate_k': f['shared_gate_k'],
                  'frozen_manifest_sha256': FAKE_FINGERPRINT,
                  'input_observations': copy.deepcopy(case['observations']),
                  'ml_prediction': copy.deepcopy(case['ml_prediction']),
                  'initial': accepted_call(), 'verification_count': 0,
                  'final_status': 'ISSUE'}
            for key, (stage, case, seed) in expected.items()}


def fake_m6_row(key, matching, plan):
    stage, case, seed = matching
    return {'run_key': key, 'condition': 'multi', 'stage': stage,
            'case_id': case['case_id'], 'seed': seed,
            'input_observations': copy.deepcopy(case['observations']),
            'ml_prediction': copy.deepcopy(case['ml_prediction']),
            'frozen_manifest_sha256': plan['g5_frozen_manifest_sha256'],
            'model_id': plan['model_id'],
            'model_version_frozen': plan['observed_model_version'],
            'generation': copy.deepcopy(plan['generation']),
            'shared_gate_k': plan['shared_gate_k'],
            'prompt_version': plan['prompt_version'],
            'treatment_code_sha256': copy.deepcopy(plan['treatment_code_sha256']),
            'event': 'DONE', 'final_status': 'ISSUE'}


class M62FairnessTests(unittest.TestCase):
    def setUp(self):
        self.f = frozen()
        self.test, self.refs = fixture_cases()
        self.expected = fair._expected_cases(self.test, self.refs, SEEDS)
        self.baseline = baseline_manifest(self.f)
        self.rows = baseline_rows(self.expected, self.f)
        self.shared = {name: 'd' * 64 for name in fair.SHARED_MODULES}
        self.treatment = {name: 'e' * 64 for name in fair.TREATMENT_MODULES}
        self.planned = fair.make_treatment_manifest(
            self.f, FAKE_FINGERPRINT, REF_SHA, self.baseline,
            self.expected, self.shared, self.treatment)

    def test_all_117_test_and_reference_keys_are_paired(self):
        self.assertEqual(len(self.expected), 117)
        self.assertEqual(sum(k.startswith('test|') for k in self.expected), 90)
        self.assertEqual(sum(k.startswith('clean_reference|') for k in self.expected), 27)
        self.assertEqual(self.planned['paired_keys_sha256'], fair._fingerprint(sorted(self.expected)))
        self.assertNotIn('gold_label', json.dumps(self.planned))
        self.assertNotIn('observations', json.dumps(self.planned))

    def test_duplicate_or_unknown_reference_is_refused(self):
        refs = copy.deepcopy(self.refs)
        refs[-1]['case_id'] = 'NOT-IN-TEST'
        with self.assertRaisesRegex(fair.FairnessError, 'Repeated/unmatched'):
            fair._expected_cases(self.test, refs, SEEDS)
        refs = copy.deepcopy(self.refs)
        refs[1]['case_id'] = refs[0]['case_id']
        with self.assertRaisesRegex(fair.FairnessError, 'Repeated/unmatched'):
            fair._expected_cases(self.test, refs, SEEDS)

    def test_clean_reference_cannot_change_shared_ml(self):
        refs = copy.deepcopy(self.refs)
        refs[0]['ml_prediction']['predicted_risk'] = 'HIGH'
        with self.assertRaisesRegex(fair.FairnessError, 'shared ML'):
            fair._expected_cases(self.test, refs, SEEDS)

    def test_seed_order_or_identity_drift_fails(self):
        with self.assertRaisesRegex(fair.FairnessError, 'Seed policy'):
            fair._expected_cases(self.test, self.refs, [11, 42, 99])
        with self.assertRaisesRegex(fair.FairnessError, 'Seed policy'):
            fair._expected_cases(self.test, self.refs, [42, 11, 77])

    def test_original_baseline_is_validated_and_missing_run_fails(self):
        fair._check_baseline_rows(self.expected, self.rows, self.f, FAKE_FINGERPRINT)
        rows = copy.deepcopy(self.rows)
        del rows[next(iter(rows))]
        with self.assertRaisesRegex(fair.FairnessError, '117 unique'):
            fair._check_baseline_rows(self.expected, rows, self.f, FAKE_FINGERPRINT)

    def test_original_baseline_ml_or_observation_mismatch_fails(self):
        for column, change in (
            ('ml_prediction', lambda x: x['probabilities'].update({'LOW': .3, 'MEDIUM': .5})),
            ('input_observations', lambda x: x.update({'Temperature': 40.})),
        ):
            rows = copy.deepcopy(self.rows)
            key = next(iter(rows))
            change(rows[key][column])
            with self.assertRaisesRegex(fair.FairnessError, 'frozen row'):
                fair._check_baseline_rows(self.expected, rows, self.f, FAKE_FINGERPRINT)

    def test_g6_manifest_model_temperature_gate_and_hash_drift_fail(self):
        fair._check_baseline_manifest(self.f, FAKE_FINGERPRINT, REF_SHA, self.baseline)
        for key, newval in [('shared_gate_k', 6), ('model_id', 'other-model'),
                             ('test_cases_sha256', '1' * 64),
                             ('test_clean_references_sha256', '2' * 64)]:
            row = copy.deepcopy(self.baseline)
            row[key] = newval
            with self.assertRaises(fair.FairnessError):
                fair._check_baseline_manifest(self.f, FAKE_FINGERPRINT, REF_SHA, row)
        row = copy.deepcopy(self.baseline)
        row['generation']['temperature'] = 0.5
        with self.assertRaises(fair.FairnessError):
            fair._check_baseline_manifest(self.f, FAKE_FINGERPRINT, REF_SHA, row)

    def test_byte_level_shared_code_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            f = copy.deepcopy(self.f)
            for name in fair.SHARED_MODULES:
                path = folder / f'{name}.py'
                content = (name + '\n').encode()
                path.write_bytes(content)
                f['input_hashes'][name] = hashlib.sha256(content).hexdigest()
            hashes = fair._assert_shared_sources(f, folder)
            self.assertEqual(set(hashes), set(fair.SHARED_MODULES))
            (folder / 'prompts.py').write_bytes(b'changed')
            with self.assertRaisesRegex(fair.FairnessError, 'prompts differs'):
                fair._assert_shared_sources(f, folder)

    def test_treatment_source_hashes_change_if_implementation_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in fair.TREATMENT_MODULES:
                (folder / f'{name}.py').write_text('original', encoding='utf-8')
            before = fair._treatment_code_hashes(folder)
            (folder / 'coordinator.py').write_text('changed', encoding='utf-8')
            after = fair._treatment_code_hashes(folder)
            self.assertNotEqual(before['coordinator'], after['coordinator'])
            self.assertEqual(before['specialists'], after['specialists'])

    def test_multi_manifest_guard_blocks_any_reconfiguration(self):
        fair.assert_treatment_manifest(copy.deepcopy(self.planned), self.planned)
        changes = [('shared_gate_k', 5), ('condition', 'single'),
                   ('paired_keys_sha256', 'changed')]
        for field, val in changes:
            draft = copy.deepcopy(self.planned)
            draft[field] = val
            with self.assertRaisesRegex(fair.FairnessError, 'treatment manifest'):
                fair.assert_treatment_manifest(draft, self.planned)
        draft = copy.deepcopy(self.planned)
        draft['generation']['max_output_tokens'] = 100
        with self.assertRaises(fair.FairnessError):
            fair.assert_treatment_manifest(draft, self.planned)
        draft = copy.deepcopy(self.planned)
        draft['treatment_code_sha256']['coordinator'] = 'changed'
        with self.assertRaises(fair.FairnessError):
            fair.assert_treatment_manifest(draft, self.planned)

    def test_future_multi_rows_must_match_shared_inputs(self):
        key = next(iter(self.expected))
        row = fake_m6_row(key, self.expected[key], self.planned)
        self.assertTrue(fair.assert_multi_rows_paired(self.expected, {key: row}, self.planned))
        for field in ('ml_prediction', 'input_observations', 'seed',
                      'generation', 'model_version_frozen', 'shared_gate_k'):
            bad = copy.deepcopy(row)
            if field == 'ml_prediction':
                bad[field]['predicted_risk'] = 'HIGH'
            elif field == 'input_observations':
                bad[field]['Temperature'] = 42.0
            elif field == 'generation':
                bad[field]['temperature'] = 0.7
            else:
                bad[field] = 'wrong'
            with self.assertRaisesRegex(fair.FairnessError, 'drifted'):
                fair.assert_multi_rows_paired(self.expected, {key: bad}, self.planned)

    def test_missing_multi_pair_cannot_be_reported_complete(self):
        key = next(iter(self.expected))
        row = fake_m6_row(key, self.expected[key], self.planned)
        with self.assertRaisesRegex(fair.FairnessError, 'missing matched'):
            fair.assert_multi_rows_paired(self.expected, {key: row}, self.planned,
                                          require_complete=True)
        with self.assertRaisesRegex(fair.FairnessError, 'unexpected'):
            fair.assert_multi_rows_paired(self.expected, {'unexpected': row}, self.planned)

    def test_terminal_failed_pair_is_accepted_without_faking_abstention(self):
        key=next(iter(self.expected))
        row=fake_m6_row(key,self.expected[key],self.planned)
        row.update({'event':'FAILED','final_status':'FAILED',
                    'final_risk_level':None,
                    'failure':{'category':'INVALID_REPORT','role':'coordinator'}})
        self.assertTrue(fair.assert_multi_rows_paired(
            {key:self.expected[key]},{key:row},self.planned,require_complete=True))
        row['final_status']='ABSTAIN'
        with self.assertRaisesRegex(fair.FairnessError,'explicit failure'):
            fair.assert_multi_rows_paired(
                {key:self.expected[key]},{key:row},self.planned,require_complete=True)

    def test_read_only_inspect_uses_private_frozen_checks_but_no_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'run_manifest.json').write_text(json.dumps(self.baseline))
            with patch('hajj_study.m6_fairness.g6._frozen',
                       return_value=(self.f, {}, FAKE_FINGERPRINT)), \
                 patch('hajj_study.m6_fairness.g6._test_suite',
                       return_value=(self.test, self.refs, None, REF_SHA)), \
                 patch('hajj_study.m6_fairness._assert_shared_sources', return_value=self.shared), \
                 patch('hajj_study.m6_fairness._treatment_code_hashes', return_value=self.treatment), \
                 patch('hajj_study.m6_fairness.g6._existing', return_value=self.rows), \
                 patch('hajj_study.m6_fairness.g5._provider', side_effect=AssertionError('No Gemini allowed')):
                result = fair.inspect('not-real-g5', root)
            self.assertEqual(result['baseline_done'], 117)
            self.assertEqual(result['expected_paired_keys'], 117)
            self.assertEqual(result['status'], 'READY_FOR_M6_3_IMPLEMENTATION_NOT_EXECUTION')
            self.assertFalse(result['has_private_gold_in_manifest'])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['run_manifest.json'])


if __name__ == '__main__':
    unittest.main()
