import unittest
import json

from hajj_study.schemas import (
    ContractError, validate_case, validate_final_report, validate_specialist_report,
    final_response_schema, specialist_response_schema,
)
from hajj_study.prompts import generalist_messages, specialist_messages, coordinator_messages

CID = 'HJ-V002'
OBS = {'Crowd_Density': 'Low', 'Temperature': 40., 'Fatigue_Level': 'Low',
       'Stress_Level': 'Medium', 'Queue_Time_minutes': 20., 'Emergency_Event': 'No',
       'Distance_Between_People_m': 1.2}
CASE = {'case_id': CID, 'observations': OBS,
        'ml_prediction': {'predicted_risk': 'MEDIUM', 'probabilities': {
            'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}}


def report():
    return {'case_id': CID, 'proposed_risk_level': 'LOW', 'evidence_flags': [],
            'supporting_evidence': ['Low crowd density and moderate wait time.']}


def specialist(domain):
    return {'case_id': CID, 'domain': domain, 'domain_risk_signal': 'MEDIUM',
            'evidence_flags': [], 'supporting_evidence': ['Observations assessed in assigned domain.']}

class SchemaLeakageTests(unittest.TestCase):
    def test_valid_case_and_report(self):
        self.assertEqual(validate_case(CASE), CASE)
        self.assertEqual(validate_final_report(report(), CID), report())
        self.assertIn('proposed_risk_level', final_response_schema()['required'])
        self.assertIn('domain', specialist_response_schema('crowd')['required'])

    def test_case_does_not_accept_gold_or_score(self):
        for prohibited in ('gold', 'Risk_Label', 'source_row_index', 'corruption_metadata'):
            with self.subTest(key=prohibited), self.assertRaises(ContractError):
                validate_case({**CASE, prohibited: 'HIGH'})

    def test_numeric_conflict_must_preserve_type_and_role(self):
        bad = json.loads(json.dumps(CASE))
        bad['observations']['Temperature'] = [
            {'source':'reading_a', 'value':'40'}, {'source':'reading_b', 'value':42.0}]
        with self.assertRaises(ContractError):validate_case(bad)
        bad['observations']['Temperature'][0]['value'] = 40.0
        validate_case(bad)
        bad['observations']['Temperature'][0]['source'] = 'clean_first'
        with self.assertRaises(ContractError):validate_case(bad)

    def test_validation_does_not_auto_reject_false_flags(self):
        r = report()
        r['evidence_flags'] = [{'field':'Stress_Level','type':'suspected_noisy','reason':'Potential anomaly'}]
        self.assertEqual(len(validate_final_report(r, CID)['evidence_flags']), 1)

    def test_reject_cross_domain_specialist_flags(self):
        r = specialist('crowd')
        r['evidence_flags'] = [{'field':'Temperature','type':'missing','reason':'Outside scope'}]
        with self.assertRaises(ContractError): validate_specialist_report(r, CID, 'crowd')

    def test_specialist_cannot_name_other_domain_fields_in_evidence(self):
        r = specialist('crowd')
        r['supporting_evidence'] = ['Temperature changed unexpectedly']
        with self.assertRaises(ContractError):
            validate_specialist_report(r, CID, 'crowd')

    def test_generalist_only_seven_inputs_and_ml(self):
        system, user = generalist_messages(CASE)
        payload = json.loads(user)
        self.assertEqual(set(payload), {'case_id','observations','shared_ml_prediction'})
        self.assertEqual(set(payload['observations']), set(OBS))
        for token in ('gold_label','Risk_Label','proxy_weights','Operational_Risk_Score',
                      'weighted proxy','0.25','1.8','2.4','Health_Condition','Incident_Type'):
            self.assertNotIn(token, system + user)

    def test_specialists_do_not_receive_ml_or_other_domains(self):
        _, user = specialist_messages(CASE, 'environment')
        payload = json.loads(user)
        self.assertEqual(set(payload['observations']), {'Temperature'})
        self.assertNotIn('ml_prediction', user)
        self.assertNotIn('Stress_Level', user)

    def test_coordinator_sees_reports_not_raw_fields_or_gold(self):
        reports = {d:specialist(d) for d in ('crowd','health','environment')}
        _, user = coordinator_messages(CID, reports, CASE['ml_prediction'])
        payload = json.loads(user)
        self.assertEqual(set(payload), {'case_id', 'shared_ml_prediction','specialist_reports'})
        self.assertNotIn('observations', user)
        self.assertNotIn('Risk_Label', user)

    def test_hidden_raw_features_cannot_enter_case(self):
        bad = json.loads(json.dumps(CASE))
        bad['observations']['Health_Condition'] = 'Heatstroke'
        with self.assertRaises(ContractError): validate_case(bad)

if __name__ == '__main__':
    unittest.main()
