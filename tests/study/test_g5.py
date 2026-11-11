"""G5 offline acceptance tests. Synthetic provider fixtures are NOT Gemini results."""
import json
import tempfile
import unittest
from pathlib import Path

from hajj_study.g5 import (G5Error, calibrate_records, create_preflight,
                           read_pre, check_smoke, freeze, _accepted_call,
                           _validation_oracle, SMOKE_PROBES)
from hajj_study.gate import SharedDecisionGate


def report(cid, fields=()):
    return {'case_id': cid, 'proposed_risk_level': 'MEDIUM',
            'evidence_flags': [{'field': f, 'type': 'missing', 'reason': 'Missing value'} for f in fields],
            'supporting_evidence': ['Synthetic fixture for unit testing.']}


def record(i, flags=(), post=None):
    cid = f'HJ-V{i:03d}'
    return {'case_id': cid, 'seed': 42, 'status': 'OK',
            'initial': {'report': report(cid, flags)},
            'verification_domain': 'crowd' if flags else None,
            'verification_count': 1 if flags else 0,
            'post': {'report': report(cid, post or ())} if flags else None}


class G5Tests(unittest.TestCase):
    def test_probe_design_contains_repeats_and_three_seeds(self):
        self.assertEqual(len(SMOKE_PROBES), 10)
        self.assertEqual({seed for _, seed in SMOKE_PROBES}, {11,42,77})
        self.assertEqual([index for index, _ in SMOKE_PROBES].count(0), 3)

    def test_calibration_all_clean_prefers_stricter_k(self):
        calibrated = calibrate_records(record(i) for i in range(1, 41))
        self.assertEqual(calibrated['k'], 7)
        self.assertEqual([r['coverage'] for r in calibrated['table']], [1.0] * 8)

    def test_calibration_replay_uses_post_verification(self):
        rows = [record(i, ['Crowd_Density'], ['Crowd_Density']) if i <= 8
                else record(i) for i in range(1,41)]
        calibrated = calibrate_records(rows)
        self.assertEqual(calibrated['table'][-1]['coverage'], 0.8)
        self.assertEqual(calibrated['table'][-1]['verifications'], 8)
        self.assertEqual(calibrated['k'], 7)

    def test_calibration_tie_chooses_larger_k(self):
        rows = [record(i, ['Crowd_Density'], ['Crowd_Density']) if i <= 10
                else record(i) for i in range(1,41)]
        self.assertEqual(calibrate_records(rows)['k'], 7)  # 75% closer than 100%

    def test_calibration_rejects_missing_run(self):
        with self.assertRaises(G5Error):
            calibrate_records(record(i) for i in range(1,40))

    def test_calibration_rejects_mock_failure_or_wrong_seed(self):
        for key, value in [('status','FAILED'), ('seed',11)]:
            rows=[record(i) for i in range(1,41)]
            rows[0][key]=value
            with self.assertRaises(G5Error):calibrate_records(rows)

    def test_calibration_rejects_missing_verify(self):
        rows=[record(i) for i in range(1,41)]
        rows[0]=record(1, ['Crowd_Density'])
        rows[0]['post']=None
        with self.assertRaises(G5Error):calibrate_records(rows)

    def test_calibration_rejects_wrong_verify_domain(self):
        rows=[record(i) for i in range(1,41)]
        rows[0]=record(1, ['Crowd_Density'])
        rows[0]['verification_domain']='health'
        with self.assertRaises(G5Error):calibrate_records(rows)

    def test_calibration_uses_inherited_cross_domain_flags(self):
        rows=[record(i) for i in range(1,41)]
        rows[0]=record(1, ['Crowd_Density','Temperature'], [])
        self.assertEqual(calibrate_records(rows)['table'][-1]['coverage'], 39/40)

    def test_accepted_call_requires_stop_and_real_usage_fields(self):
        good={'model_version':'gemini-X','attempts':[{'status':'STOP','finish_reason':'STOP',
                    'usage':{'prompt_tokens':45,'candidates_tokens':20,'thoughts_tokens':0}}]}
        self.assertTrue(_accepted_call(good))
        for altered in [dict(good,model_version=None),dict(good,attempts=[]),
                        dict(good,attempts=[{'status':'STOP','finish_reason':'MAX_TOKENS','usage':good['attempts'][0]['usage']}]),
                        dict(good,attempts=[{'status':'STOP','finish_reason':'STOP','usage':{}}])]:
            self.assertFalse(_accepted_call(altered))

    def test_smoke_or_freeze_cannot_pass_without_live_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'manifest_pre.json').write_text('{}')
            with self.assertRaises(G5Error):check_smoke(p, {})

    def test_threshold_keeps_g4_verification_budget(self):
        before=report('HJ-V001',['Crowd_Density','Temperature'])
        after=report('HJ-V001',[])
        gate=SharedDecisionGate(7)
        first=gate.initial(before,'HJ-V001')
        self.assertEqual(first.verification_domain, 'crowd')
        post=gate.post_verification(before,after,'HJ-V001','crowd',verification_used=True)
        self.assertEqual(post.status,'ABSTAIN')

if __name__=='__main__': unittest.main()
