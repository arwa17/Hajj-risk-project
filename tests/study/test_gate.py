import unittest
from hajj_study.gate import SharedDecisionGate
from hajj_study.schemas import ContractError
from hajj_study.verify import VerificationOracle

CID = 'HJ-V001'


def report(flags=None, risk='HIGH'):
    return {'case_id': CID, 'proposed_risk_level': risk,
            'evidence_flags': flags or [], 'supporting_evidence': ['Observed crowd and health signals.']}


def flag(field, kind='missing'):
    return {'field': field, 'type': kind, 'reason': 'Visible unresolved evidence problem'}


CLEAN = {'Crowd_Density': 'Medium', 'Temperature': 37., 'Fatigue_Level': 'Low',
         'Stress_Level': 'Low', 'Queue_Time_minutes': 19., 'Emergency_Event': 'No',
         'Distance_Between_People_m': 1.5}


class GateTests(unittest.TestCase):
    def test_agent_proposed_risk_is_not_overwritten(self):
        gate = SharedDecisionGate(7)
        self.assertEqual(gate.initial(report(), CID).risk_level, 'HIGH')
        self.assertEqual(gate.initial(report(risk='LOW'), CID).risk_level, 'LOW')

    def test_quality_flag_from_agent_controls_threshold(self):
        gate = SharedDecisionGate(7)
        x = gate.initial(report([flag('Temperature')]), CID)
        self.assertEqual((x.status, x.evidence_sufficiency, x.verification_domain),
                         ('VERIFY', 6, 'environment'))
        self.assertEqual(SharedDecisionGate(6).initial(report([flag('Temperature')]), CID).status, 'ISSUE')

    def test_severity_then_domain_tiebreak(self):
        gate = SharedDecisionGate(7)
        r = report([flag('Temperature', 'conflicting'), flag('Crowd_Density', 'conflicting')])
        self.assertEqual(gate.initial(r, CID).verification_domain, 'crowd')
        r2 = report([flag('Crowd_Density', 'missing'), flag('Temperature', 'conflicting')])
        self.assertEqual(gate.initial(r2, CID).verification_domain, 'environment')
        r3 = report([flag('Fatigue_Level', 'missing'), flag('Crowd_Density', 'missing')])
        self.assertEqual(gate.initial(r3, CID).verification_domain, 'crowd')

    def test_inheritance_prevents_unverified_domain_from_disappearing(self):
        g = SharedDecisionGate(7)
        before = report([flag('Crowd_Density', 'conflicting'), flag('Temperature')])
        after = report([], risk='LOW')
        r = g.post_verification(before, after, CID, 'crowd', verification_used=True)
        self.assertEqual((r.status, r.risk_level, r.evidence_sufficiency), ('ABSTAIN', None, 6))
        self.assertEqual([x['field'] for x in r.evidence_flags], ['Temperature'])

    def test_same_risk_survives_post_verification(self):
        g = SharedDecisionGate(7)
        before = report([flag('Crowd_Density')], risk='HIGH')
        after = report([], risk='LOW')
        outcome = g.post_verification(before, after, CID, 'crowd', verification_used=True)
        self.assertEqual((outcome.status, outcome.risk_level), ('ISSUE', 'LOW'))

    def test_invalid_k_or_unverified_post_rejected(self):
        for k in (-1, 8, 3.0, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                SharedDecisionGate(k)
        with self.assertRaises(ContractError):
            SharedDecisionGate(7).post_verification(report(), report(), CID, 'crowd', verification_used=False)

    def test_verification_releases_only_selected_domain_once(self):
        oracle = VerificationOracle({CID: CLEAN})
        session = oracle.new_session(CID)
        altered = dict(CLEAN)
        altered['Crowd_Density'] = None
        altered['Temperature'] = 43.
        post = session.verify_domain(altered, 'crowd', case_id=CID)
        self.assertEqual(post['Crowd_Density'], CLEAN['Crowd_Density'])
        self.assertEqual(post['Temperature'], 43.)
        self.assertNotIn('risk_label', post)
        with self.assertRaises(ContractError):
            session.verify_domain(post, 'environment', case_id=CID)

    def test_reject_cross_case_verification(self):
        session = VerificationOracle({CID:CLEAN}).new_session(CID)
        with self.assertRaises(ContractError):
            session.verify_domain(dict(CLEAN), 'crowd', case_id='HJ-V999')

if __name__ == '__main__':
    unittest.main()
