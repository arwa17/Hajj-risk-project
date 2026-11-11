"""Synthetic, private-data-free M5 attribution diagnostic tests."""
import copy
import unittest

from hajj_study import diagnose_m5 as d
from hajj_study.score_runs import ScoreError


def fixtures():
    cases = [
        {'case_id':'A','ml_prediction':{'predicted_risk':'LOW'}},
        {'case_id':'B','ml_prediction':{'predicted_risk':'MEDIUM'}},
    ]
    gold = [{'case_id':'A','risk_level':'LOW'}, {'case_id':'B','risk_level':'MEDIUM'}]
    meta = [{'case_id':'A','case_group':'typical_clean'},
            {'case_id':'B','case_group':'noisy',
             'corruption_metadata':{'kind':'noisy','altered_fields':['Temperature']}}]
    flags=[{'field':'Temperature','type':'suspected_noisy','reason':'synthetic'}]
    rows=[]
    for seed in (11,42,77):
        rows.extend([
            {'run_key':f'test|A|{seed}','event':'DONE','final_status':'ISSUE',
             'final_risk_level':'MEDIUM','initial':{'report':{'proposed_risk_level':'MEDIUM','evidence_flags':[]}}},
            {'run_key':f'test|B|{seed}','event':'DONE','final_status':'ISSUE',
             'final_risk_level':'HIGH','initial':{'report':{'proposed_risk_level':'HIGH','evidence_flags':flags}}},
            {'run_key':f'clean_reference|B|{seed}','event':'DONE','final_status':'ISSUE',
             'final_risk_level':'HIGH','initial':{'report':{'proposed_risk_level':'HIGH','evidence_flags':[]}}},
        ])
    return cases, gold, meta, rows


class DiagnoseTests(unittest.TestCase):
    def test_synthetic_overescalation_and_no_pair_change(self):
        result=d.analyze(*fixtures(),seeds=(11,42,77))
        counts=result['counts']
        self.assertEqual(counts['unique_test_cases'],2)
        self.assertEqual(counts['test_case_seed_runs'],6)
        self.assertEqual(counts['issued'],6)
        self.assertEqual(counts['wrong_issued'],6)
        self.assertEqual(counts['wrong_over_escalated'],6)
        self.assertEqual(counts['wrong_issued_when_ml_correct'],6)
        self.assertEqual(counts.get('wrong_issued_matching_ml',0),0)
        self.assertEqual(result['corruption_pair_diagnostics']['paired_observations'],3)
        self.assertEqual(result['corruption_pair_diagnostics'].get('raw_risk_changed',0),0)
        self.assertEqual(result['corruption_pair_diagnostics']['correct_field_and_type_flag'],3)
        self.assertEqual(result['confusion']['shared_ml_vs_proxy_30_cases']['LOW']['LOW'],1)
        self.assertNotIn('case_id',str(result))
        self.assertNotIn('input_observations',str(result))

    def test_official_count_mapping_accepts_realistic_nonzero_errors(self):
        diagnosis = d.analyze(*fixtures(), seeds=(11, 42, 77))
        official = {'per_seed': {str(seed): {'counts': {
            'issued': 2, 'abstained': 0, 'incorrect_issued': 2,
            'correct_issued': 0, 'total': 2,
        }} for seed in (11, 42, 77)}}
        d._validate_against_official(diagnosis, official, (11, 42, 77))
        # A real divergence must still fail; do not disable the cross-check.
        official['per_seed']['42']['counts']['incorrect_issued'] = 1
        with self.assertRaisesRegex(ScoreError, 'wrong_issued.*incorrect_issued'):
            d._validate_against_official(diagnosis, official, (11, 42, 77))
        official['per_seed']['42']['counts']['incorrect_issued'] = 2
        official['per_seed']['11']['counts']['issued'] = 1
        with self.assertRaisesRegex(ScoreError, 'issued.*issued'):
            d._validate_against_official(diagnosis, official, (11, 42, 77))

    def test_reject_incomplete_and_failed_run(self):
        cases,gold,meta,rows=fixtures()
        with self.assertRaisesRegex(ScoreError,'run keys'):
            d.analyze(cases,gold,meta,rows[:-1],(11,42,77))
        rows=copy.deepcopy(rows)
        rows[0]['event']='FAILED'
        with self.assertRaisesRegex(ScoreError,'completed G6'):
            d.analyze(cases,gold,meta,rows,(11,42,77))

    def test_after_verification_changed_risk_tracked(self):
        cases,gold,meta,rows=fixtures()
        rows=copy.deepcopy(rows)
        rows[0]['final_risk_level']='HIGH'
        result=d.analyze(cases,gold,meta,rows,(11,42,77))
        self.assertEqual(result['counts']['issued_changed_risk_after_gate_or_verification'],1)
        self.assertEqual(result['counts']['wrong_issued_changed_after_verification'],1)

    def test_reject_mismatched_ids(self):
        cases,gold,meta,rows=fixtures()
        with self.assertRaisesRegex(ScoreError,'IDs'):
            d.analyze(cases,gold[:-1],meta,rows,(11,42,77))


if __name__ == '__main__':
    unittest.main()
