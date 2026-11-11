"""M6.1 offline topology and isolation tests: no network, Test labels or keys."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

from hajj_study.gate import SharedDecisionGate
from hajj_study.coordinator import request_coordinator
from hajj_study.schemas import ContractError
from hajj_study.multi_agent import execute_multi_case
from hajj_study.provider import CallFailure, ProviderError
from hajj_study.verify import VerificationOracle


CID = 'SYNTH-M61'
CLEAN = {'Crowd_Density': 'Low', 'Queue_Time_minutes': 50.,
         'Distance_Between_People_m': 1.2, 'Fatigue_Level': 'High',
         'Stress_Level': 'Medium', 'Emergency_Event': 'No', 'Temperature': 39.}
ML = {'predicted_risk': 'MEDIUM',
      'probabilities': {'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}


def case(observations=None):
    return {'case_id': CID, 'observations': dict(CLEAN if observations is None else observations),
            'ml_prediction': deepcopy(ML)}


def specialist(domain, flags=None):
    return {'case_id': CID, 'domain': domain, 'domain_risk_signal': 'MEDIUM',
            'evidence_flags': flags or [], 'supporting_evidence': ['Relevant domain evidence.']}


def coordinator(flags=None, level='LOW'):
    return {'case_id': CID, 'proposed_risk_level': level, 'evidence_flags': flags or [],
            'supporting_evidence': ['Specialist evidence jointly evaluated.']}


def flag(field, kind='missing'):
    return {'field': field, 'type': kind, 'reason': 'Synthetic uncertain reading.'}


class FakeGemini:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def generate_validated(self, *, system_prompt, user_prompt, response_schema, validator):
        payload = json.loads(user_prompt)
        self.requests.append((system_prompt, payload, response_schema))
        if not self.answers:
            raise AssertionError('Unexpected model call')
        response = self.answers.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(report=validator(response),
                               model_version='synthetic-frozen-model',
                               total_elapsed_ms=12.5,
                               attempts=[{'attempt': 1, 'status': 'STOP',
                                          'model_version': 'synthetic-frozen-model',
                                          'usage': {'total_tokens': 25}, 'elapsed_ms': 12.5}])


def failure():
    cause = ProviderError('HTTP 429', category='HTTP', retryable=True, status_code=429)
    return CallFailure(cause, [{'attempt': 1, 'status': 'HTTP', 'http_status': 429}])


class MultiAgentTests(unittest.TestCase):
    def oracle(self):
        return VerificationOracle({CID: CLEAN})

    def answers(self, first=None):
        return [specialist('crowd'), specialist('health'), specialist('environment'),
                first or coordinator()]

    def test_four_distinct_real_roles_on_issue_and_isolated_messages(self):
        provider = FakeGemini(self.answers())
        result = execute_multi_case(case(), provider=provider, gate=SharedDecisionGate(7),
                                    oracle=self.oracle())
        self.assertEqual((result.status, result.risk_level), ('ISSUE', 'LOW'))
        self.assertEqual(result.trace['logical_agent_calls'], 4)
        self.assertEqual(result.trace['verification_count'], 0)
        self.assertEqual([x['role'] for x in result.trace['call_trace']],
                         ['specialist', 'specialist', 'specialist', 'coordinator'])
        for index, fields in enumerate(({'Crowd_Density', 'Queue_Time_minutes', 'Distance_Between_People_m'},
                                        {'Fatigue_Level', 'Stress_Level', 'Emergency_Event'},
                                        {'Temperature'})):
            msg = provider.requests[index][1]
            self.assertEqual(set(msg['observations']), fields)
            self.assertNotIn('shared_ml_prediction', msg)
            self.assertNotIn('gold_label', json.dumps(msg))
        coord_payload = provider.requests[3][1]
        self.assertEqual(coord_payload['shared_ml_prediction'], ML)
        self.assertNotIn('observations', coord_payload)
        self.assertEqual(set(coord_payload['specialist_reports']), {'crowd', 'health', 'environment'})
        self.assertEqual(len(result.trace['api_attempts']), 4)

    def test_verify_only_selected_domain_then_recoordinator(self):
        observed = dict(CLEAN)
        observed['Temperature'] = None
        first = coordinator([flag('Temperature')], 'MEDIUM')
        provider = FakeGemini(self.answers(first) +
                              [specialist('environment'), coordinator([], 'HIGH')])
        result = execute_multi_case(case(observed), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual((result.status, result.risk_level), ('ISSUE', 'HIGH'))
        self.assertEqual(result.trace['verification_count'], 1)
        self.assertEqual(result.trace['verification_domain'], 'environment')
        self.assertEqual(result.trace['logical_agent_calls'], 6)
        self.assertEqual(provider.requests[4][1]['domain'], 'environment')
        self.assertEqual(provider.requests[4][1]['observations'], {'Temperature': 39.})
        self.assertEqual(provider.requests[4][1]['verified_domain'], 'environment')
        self.assertNotIn('observations', provider.requests[5][1])
        self.assertEqual(provider.requests[5][1]['specialist_reports']['crowd'],
                         provider.requests[3][1]['specialist_reports']['crowd'])
        self.assertEqual(provider.requests[5][1]['specialist_reports']['health'],
                         provider.requests[3][1]['specialist_reports']['health'])
        self.assertEqual([x['phase'] for x in result.trace['call_trace']],
                         ['initial'] * 4 + ['post', 'post'])

    def test_abstain_if_other_domain_flag_remains_after_one_verification(self):
        observed = dict(CLEAN)
        observed['Temperature'] = None
        observed['Crowd_Density'] = None
        first = coordinator([flag('Crowd_Density', 'conflicting'), flag('Temperature')])
        provider = FakeGemini(self.answers(first) +
                              [specialist('crowd'), coordinator([])])
        result = execute_multi_case(case(observed), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual((result.status, result.risk_level), ('ABSTAIN', None))
        self.assertEqual(result.trace['verification_domain'], 'crowd')
        self.assertEqual(result.trace['verification_count'], 1)
        self.assertEqual([f['field'] for f in result.trace['post_gate']['merged_flags']],
                         ['Temperature'])
        self.assertEqual(result.trace['logical_agent_calls'], 6)

    def test_specialist_failure_stops_without_coordinator_fallback(self):
        provider = FakeGemini([specialist('crowd'), failure()])
        result = execute_multi_case(case(), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual((result.status, result.risk_level), ('FAILED', None))
        self.assertEqual(result.trace['logical_agent_calls'], 2)
        self.assertEqual(result.trace['failure']['category'], 'HTTP')
        self.assertEqual(result.trace['failure']['role'], 'specialist')
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(len(result.trace['api_attempts']), 2)

    def test_coordinator_failure_is_failed_not_abstain(self):
        provider = FakeGemini(self.answers()[:3] + [failure()])
        result = execute_multi_case(case(), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual((result.status, result.risk_level), ('FAILED', None))
        self.assertEqual(result.trace['logical_agent_calls'], 4)
        self.assertEqual(result.trace['failure']['role'], 'coordinator')

    def test_report_domain_leak_fails_without_final_decision(self):
        wrong = specialist('crowd')
        wrong['evidence_flags'] = [flag('Temperature')]
        provider = FakeGemini([wrong])
        result = execute_multi_case(case(), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual(result.status, 'FAILED')
        self.assertEqual(result.trace['failure']['category'], 'CONTRACT')
        self.assertEqual(result.trace['logical_agent_calls'], 1)

    def test_report_bad_case_id_fails(self):
        wrong = specialist('crowd')
        wrong['case_id'] = 'OTHER'
        result = execute_multi_case(case(), provider=FakeGemini([wrong]),
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual(result.status, 'FAILED')
        self.assertEqual(result.trace['failure']['category'], 'CONTRACT')

    def test_post_agent_failure_is_failed_not_abstain(self):
        initial = coordinator([flag('Temperature')])
        provider = FakeGemini(self.answers(initial) + [failure()])
        result = execute_multi_case(case(), provider=provider, gate=SharedDecisionGate(7),
                                    oracle=self.oracle())
        self.assertEqual(result.status, 'FAILED')
        self.assertEqual(result.trace['verification_count'], 1)
        self.assertEqual(result.trace['logical_agent_calls'], 5)
        self.assertEqual(result.trace['failure']['role'], 'specialist')

    def test_same_pacing_hook_applies_to_every_role(self):
        ticks = []
        provider = FakeGemini(self.answers())
        result = execute_multi_case(case(), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle(),
                                    before_call=lambda: ticks.append('paced'))
        self.assertEqual(result.status, 'ISSUE')
        self.assertEqual(len(ticks), 4)

    def test_model_version_drift_fails_with_recorded_attempts(self):
        provider = FakeGemini(self.answers())
        result = execute_multi_case(case(), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle(),
                                    expected_model_version='other-frozen-model')
        self.assertEqual(result.status, 'FAILED')
        self.assertEqual(result.trace['failure']['category'], 'CONTRACT')
        self.assertEqual(result.trace['logical_agent_calls'], 1)
        self.assertEqual(len(result.trace['api_attempts']), 1)

    def test_coordinator_standalone_rejects_gold_leak_without_model_call(self):
        malicious = case()
        malicious['gold_label'] = 'HIGH'
        reports = {d: specialist(d) for d in ('crowd', 'health', 'environment')}
        provider = FakeGemini([coordinator()])
        with self.assertRaises(ContractError):
            request_coordinator(malicious, reports, provider=provider)
        self.assertEqual(provider.requests, [])

    def test_coordinator_requires_three_valid_reports_before_call(self):
        provider = FakeGemini([coordinator()])
        with self.assertRaises(ContractError):
            request_coordinator(case(), {'crowd': specialist('crowd')}, provider=provider)
        self.assertEqual(provider.requests, [])

    def test_verified_domain_is_not_disclosed_to_other_roles(self):
        observed = dict(CLEAN)
        observed['Crowd_Density'] = None
        provider = FakeGemini(self.answers(coordinator([flag('Crowd_Density')])) +
                              [specialist('crowd'), coordinator([])])
        result = execute_multi_case(case(observed), provider=provider,
                                    gate=SharedDecisionGate(7), oracle=self.oracle())
        self.assertEqual(result.status, 'ISSUE')
        self.assertEqual(provider.requests[4][1]['verified_domain'], 'crowd')
        self.assertEqual(provider.requests[4][1]['observations']['Crowd_Density'], 'Low')
        self.assertNotIn('observations', provider.requests[5][1])


if __name__ == '__main__':
    unittest.main()
