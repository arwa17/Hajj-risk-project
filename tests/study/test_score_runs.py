"""Offline synthetic fixtures; no Gemini, private test labels or user dataset."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from hajj_study import score_runs as s

MODEL = 'fixture-gemini'
FINGERPRINT = 'fixture-sha'
GOLD = [{'case_id': 'A', 'risk_level': 'LOW'},
        {'case_id': 'B', 'risk_level': 'HIGH'},
        {'case_id': 'C', 'risk_level': 'MEDIUM'}]
META = [{'case_id': 'A', 'case_group': 'typical_clean'},
        {'case_id': 'B', 'case_group': 'noisy',
         'corruption_metadata': {'kind': 'noisy', 'altered_fields': ['Temperature']}},
        {'case_id': 'C', 'case_group': 'borderline_clean'}]


def call(cid, level, flags=()):
    report = {'case_id': cid, 'proposed_risk_level': level,
              'evidence_flags': list(flags), 'supporting_evidence': ['Synthetic']}
    return {'report': report, 'model_version': MODEL,
            'attempts': [{'status': 'STOP', 'finish_reason': 'STOP',
                          'model_version': MODEL,
                          'usage': {'prompt_tokens': 3, 'candidates_tokens': 4,
                                    'thoughts_tokens': 0, 'total_tokens': 7},
                          'elapsed_ms': 5.0}]}


def row(stage, cid, seed, final='ISSUE', level=None, initial=None, post=None, count=0):
    return {'run_key': f'{stage}|{cid}|{seed}', 'stage': stage, 'case_id': cid,
            'condition': 'single', 'seed': seed, 'frozen_manifest_sha256': FINGERPRINT,
            'model_version_frozen': MODEL, 'event': 'FAILED' if final == 'FAILED' else 'DONE',
            'final_status': final, 'final_risk_level': level if final == 'ISSUE' else None,
            'initial': initial, 'post': post, 'verification_count': count}


def sample_rows():
    result=[]
    flag = {'field': 'Temperature', 'type': 'suspected_noisy', 'reason': 'Unexpected value'}
    for seed in (11,42,77):
        result.append(row('test', 'A', seed, level='LOW', initial=call('A','LOW')))
        if seed == 77:
            fail = row('test','B',seed,final='FAILED')
            fail['failure']={'category':'HTTP','http_status':429,
                             'attempts':[{'status':'HTTP','http_status':429,'usage':{},'elapsed_ms':None}]}
            result.append(fail)
        else:
            level = 'HIGH' if seed == 11 else 'MEDIUM'
            result.append(row('test','B',seed, level=level,
                              initial=call('B','MEDIUM', [flag]),post=call('B',level),count=1))
        result.append(row('test','C',seed,final='ABSTAIN',initial=call('C','MEDIUM'),
                          post=call('C','MEDIUM', [{'field':'Temperature','type':'missing','reason':'Synthetic'}]),count=1))
        result.append(row('clean_reference','B',seed,level='HIGH',initial=call('B','HIGH')))
    return result


class M5ScoreContractTests(unittest.TestCase):
    def score(self, rows=None):
        return s.score_condition(rows if rows is not None else sample_rows(), GOLD, META,
                                 fingerprint=FINGERPRINT, expected_version=MODEL,
                                 expected_test_cases=3, expected_clean_refs=1)

    def test_all_eight_metrics_correct_denominators(self):
        result = self.score()
        x = result['per_seed']['11']
        m = x['metrics']
        self.assertEqual(m['coverage'],s.ratio(2,3))
        self.assertEqual(m['selective_error'],s.ratio(0,2))
        self.assertEqual(m['unsupported_escalation'],s.ratio(0,2))
        self.assertEqual(m['under_escalation'],s.ratio(0,2))
        self.assertEqual(m['disrupted_case_detection'],s.ratio(1,1))
        self.assertEqual(m['clean_decision_recovery'],s.ratio(1,1))
        self.assertEqual(m['trajectory_safe_completion'],s.ratio(3,3))
        self.assertEqual(x['cost_latency']['mean_logical_agent_calls_per_case'],5/3)
        self.assertEqual(x['cost_latency']['mean_api_attempts_per_case'],5/3)
        self.assertIsNone(x['cost_latency']['mean_full_wall_ms_per_case'])

    def test_sample_sd_three_seeds_and_failed_not_safe(self):
        result=self.score()
        self.assertEqual(result['per_seed']['42']['metrics']['under_escalation'],s.ratio(1,2))
        self.assertEqual(result['per_seed']['77']['metrics']['coverage'],s.ratio(1,3))
        self.assertEqual(result['per_seed']['77']['metrics']['trajectory_safe_completion'],s.ratio(2,3))
        self.assertEqual(result['per_seed']['77']['components']['unobservable_pairs'],1)
        self.assertIsNone(result['per_seed']['77']['metrics']['disrupted_case_detection']['value'])
        self.assertEqual(result['summary']['coverage']['valid_seeds'],3)
        self.assertAlmostEqual(result['summary']['coverage']['sample_sd'],0.19245008972987526)

    def test_no_raw_change_stays_na_but_flag_detection_is_measurable(self):
        rows = sample_rows()
        for item in rows:
            if item['stage'] == 'clean_reference':
                item['initial']['report']['proposed_risk_level'] = 'MEDIUM'
        result = self.score(rows)
        diag = result['per_seed']['11']['components']
        self.assertEqual(diag['raw_decision_change_rate'], s.ratio(0, 1))
        self.assertEqual(diag['paired_corruption_flag_rate'], s.ratio(1, 1))
        self.assertEqual(diag['pairs_with_correct_field_flag'], 1)
        self.assertEqual(result['per_seed']['11']['metrics']['disrupted_case_detection'], s.ratio(0, 0))
        self.assertEqual(result['per_seed']['11']['metrics']['clean_decision_recovery'], s.ratio(0, 0))

    def test_trace_inputs_must_match_immutable_frozen_pair_and_shared_ml(self):
        cases = [dict(case_id='A', observations={'x': 1}, ml_prediction={'risk': 'LOW'}),
                 dict(case_id='B', observations={'x': None}, ml_prediction={'risk': 'MEDIUM'}),
                 dict(case_id='C', observations={'x': 3}, ml_prediction={'risk': 'MEDIUM'})]
        refs = [dict(case_id='B', observations={'x': 2}, ml_prediction={'risk': 'MEDIUM'})]
        rows = sample_rows()
        indexed = {r['case_id']:r for r in cases}
        for r in rows:
            case = refs[0] if r['stage'] == 'clean_reference' else indexed[r['case_id']]
            r['input_observations'] = case['observations']
            r['ml_prediction'] = case['ml_prediction']
        ok = s._verify_frozen_inputs(rows, cases, refs, (11, 42, 77))
        self.assertEqual(ok, {'frozen_test_cases': 3, 'frozen_pairs': 1, 'validated_run_keys': 12})
        altered = copy.deepcopy(rows)
        altered[0]['input_observations'] = {'x': 'tampered'}
        with self.assertRaisesRegex(s.ScoreError, 'Trace observations differ'):
            s._verify_frozen_inputs(altered, cases, refs, (11, 42, 77))
        wrong_refs = copy.deepcopy(refs)
        wrong_refs[0]['observations'] = {'x': None}
        with self.assertRaisesRegex(s.ScoreError, 'identical'):
            s._verify_frozen_inputs(rows, cases, wrong_refs, (11, 42, 77))
        wrong_refs = copy.deepcopy(refs)
        wrong_refs[0]['ml_prediction'] = {'risk': 'HIGH'}
        with self.assertRaisesRegex(s.ScoreError, 'Shared ML prediction'):
            s._verify_frozen_inputs(rows, cases, wrong_refs, (11, 42, 77))

    def test_zero_denominator_is_na_not_zero(self):
        self.assertIsNone(s.ratio(0,0)['value'])
        self.assertEqual(s.ratio(0,0)['denominator'],0)
        with self.assertRaises(s.ScoreError):s.ratio(2,1)

    def test_detect_only_correct_corrupted_field_and_type(self):
        rows=sample_rows()
        rows[1]['initial']['report']['evidence_flags'] = [
            {'field':'Temperature','type':'missing','reason':'wrong flag type'}]
        result=self.score(rows)
        m=result['per_seed']['11']['metrics']
        self.assertEqual(m['disrupted_case_detection'],s.ratio(0,1))
        self.assertEqual(m['field_flag_precision'],s.ratio(0,1))

    def test_clean_false_flags_are_measured(self):
        rows=sample_rows()
        rows[0]['initial']['report']['evidence_flags']=[
            {'field':'Crowd_Density','type':'suspected_noisy','reason':'false alarm'}]
        m=self.score(rows)['per_seed']['11']['metrics']
        self.assertEqual(m['clean_false_flag_case_rate'],s.ratio(1,2))

    def test_missing_case_or_reference_refused(self):
        with self.assertRaisesRegex(s.ScoreError,'Incomplete or unexpected'):
            self.score(sample_rows()[:-1])

    def test_hash_and_model_version_drift_rejected(self):
        rows=sample_rows()
        rows[0]['frozen_manifest_sha256']='wrong'
        with self.assertRaisesRegex(s.ScoreError,'provenance'):
            self.score(rows)
        rows=sample_rows()
        rows[0]['initial']['model_version']='changed'
        with self.assertRaisesRegex(s.ScoreError,'modelVersion'):
            self.score(rows)

    def test_unexpected_gold_case_and_metadata_rejected(self):
        with self.assertRaisesRegex(s.ScoreError,'Golden set size'):
            s.score_condition(sample_rows(),GOLD,META[:-1],fingerprint=FINGERPRINT,
                              expected_version=MODEL,expected_test_cases=3,expected_clean_refs=1)

    def test_verified_runs_must_have_post(self):
        rows=sample_rows()
        rows[1]['post']=None
        with self.assertRaisesRegex(s.ScoreError,'without post'):
            self.score(rows)

    def test_verification_budget_enforced(self):
        rows=sample_rows()
        rows[1]['verification_count']=2
        with self.assertRaisesRegex(s.ScoreError,'Verification budget'):
            self.score(rows)

    def test_append_only_resume_uses_last_record(self):
        rows=sample_rows()
        old=copy.deepcopy(rows[1]);old['event']='FAILED';old['final_status']='FAILED';old['final_risk_level']=None
        self.assertEqual(self.score([old]+rows)['unique_run_keys'],12)

    def test_synthetic_unit_tests_cannot_be_scored_as_live_without_valid_fingerprint(self):
        with self.assertRaisesRegex(s.ScoreError,'provenance'):
            s.score_condition(sample_rows(),GOLD,META,fingerprint='real-file-sha',
                              expected_version=MODEL,expected_test_cases=3,expected_clean_refs=1)

    def test_no_raw_response_or_gold_in_aggregate(self):
        out=self.score()
        self.assertNotIn('raw_response_private',json.dumps(out))
        self.assertNotIn('observations',json.dumps(out))
        self.assertNotIn('risk_level',json.dumps(out))


if __name__=='__main__':unittest.main()
