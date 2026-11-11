"""Targeted regression suite. Synthetic-only; no model calls or private labels."""
from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from hajj_study import g7, m6_results
from hajj_study.corrupt import DOMAINS
from hajj_study.gate import SharedDecisionGate
from hajj_study.provider import GeminiProvider
from hajj_study.schemas import (ContractError, final_response_schema,
                                specialist_response_schema, validate_final_report)
from hajj_study.verify import VerificationOracle

CID = 'SYNTH-REGRESSION'
CLEAN = {'Crowd_Density': 'Low', 'Queue_Time_minutes': 50.0,
         'Distance_Between_People_m': 1.2, 'Fatigue_Level': 'High',
         'Stress_Level': 'Medium', 'Emergency_Event': 'No', 'Temperature': 39.0}
ML = {'predicted_risk': 'MEDIUM', 'probabilities': {'LOW': .2, 'MEDIUM': .6, 'HIGH': .2}}
CASE = {'case_id': CID, 'observations': CLEAN, 'ml_prediction': ML}
VERSION = 'mock-frozen-model'
PLAN = {'g5_frozen_manifest_sha256': 'a' * 64, 'model_id': VERSION,
        'observed_model_version': VERSION, 'generation': {'temperature': 0.0},
        'shared_gate_k': 7, 'prompt_version': 'g4-candidate-v1',
        'treatment_code_sha256': {'multi_agent': 'b' * 64},
        'min_seconds_between_logical_calls': 0.0}
FROZEN = {'shared_gate_k': 7, 'observed_model_version': VERSION,
          'generation': {'temperature': 0.0}}


def flag(field, kind='suspected_noisy'):
    return {'field': field, 'type': kind, 'reason': 'Synthetic flag.'}


def final(flags=None):
    return {'case_id': CID, 'proposed_risk_level': 'MEDIUM',
            'evidence_flags': list(flags or []), 'supporting_evidence': ['Synthetic evidence.']}


def specialist(domain):
    return {'case_id': CID, 'domain': domain, 'domain_risk_signal': 'MEDIUM',
            'evidence_flags': [], 'supporting_evidence': ['Synthetic domain evidence.']}


class SyntheticProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_validated(self, *, system_prompt, user_prompt, response_schema, validator):
        result = validator(self.replies[self.calls])
        self.calls += 1
        return SimpleNamespace(report=result, output_text=json.dumps(result),
            model_version=VERSION, total_elapsed_ms=7.0,
            attempts=[{'attempt': 1, 'status': 'STOP', 'finish_reason': 'STOP',
                'model_version': VERSION, 'usage': {'prompt_tokens': 12,
                'candidates_tokens': 8, 'thoughts_tokens': 0, 'total_tokens': 20},
                'elapsed_ms': 7.0}])


class SlowPacer:
    def wait(self):
        time.sleep(0.023)


class RegressionTests(unittest.TestCase):
    def test_schema_bounds_include_both_types(self):
        schemas = [final_response_schema()] + [specialist_response_schema(x) for x in DOMAINS]
        for schema in schemas:
            props = schema['properties']
            self.assertEqual(props['evidence_flags']['maxItems'], 7)
            arr = props['supporting_evidence']
            self.assertEqual((arr['minItems'], arr['maxItems']), (1, 7))
            self.assertEqual((arr['items']['minLength'], arr['items']['maxLength']), (1, 450))
            reason = props['evidence_flags']['items']['properties']['reason']
            self.assertEqual((reason['minLength'], reason['maxLength']), (1, 400))

    def test_protected_unverified_flag_takes_precedence(self):
        before = final([flag('Crowd_Density', 'missing'), flag('Temperature', 'conflicting')])
        after = final([flag('Temperature', 'suspected_noisy')])
        merged = SharedDecisionGate.merge_post_flags(before, after, 'crowd')
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['field'], 'Temperature')
        self.assertEqual(merged[0]['type'], 'conflicting')

    def test_verified_field_only_uses_revised_report(self):
        before = final([flag('Crowd_Density', 'missing'), flag('Temperature', 'missing')])
        after = final([flag('Crowd_Density', 'suspected_noisy')])
        merged = SharedDecisionGate.merge_post_flags(before, after, 'crowd')
        flags = {x['field']: x['type'] for x in merged}
        self.assertEqual(flags['Crowd_Density'], 'suspected_noisy')
        self.assertEqual(flags['Temperature'], 'missing')

    def test_post_merge_canonicalizes_eight_flags(self):
        before = final([flag('Crowd_Density', 'conflicting'), flag('Temperature', 'missing')])
        after = final([flag(f) for f in CLEAN])
        result = SharedDecisionGate(7).post_verification(
            before, after, CID, 'crowd', verification_used=True)
        self.assertEqual(result.status, 'ABSTAIN')
        self.assertEqual(len(result.evidence_flags), 7)
        self.assertEqual(len({x['field'] for x in result.evidence_flags}), 7)
        self.assertEqual(next(x for x in result.evidence_flags
                              if x['field'] == 'Temperature')['type'], 'missing')

    def test_multirole_contract_failure_full_calls_measured(self):
        case = copy.deepcopy(CASE)
        altered = copy.deepcopy(CLEAN)
        altered['Crowd_Density'] = None
        case['observations'] = altered
        provider = SyntheticProvider([
            specialist('crowd'), specialist('health'), specialist('environment'),
            final([flag('Crowd_Density', 'missing')]),
            specialist('crowd'), final([])])
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(SharedDecisionGate, 'post_verification',
                              side_effect=ContractError('Synthetic injected gate failure')):
                row = g7._one('test', case, 42, VerificationOracle({CID: CLEAN}),
                              FROZEN, PLAN, temp, provider=provider, pacer=SlowPacer())
        self.assertEqual(row['event'], 'FAILED')
        self.assertEqual(row['failure']['category'], 'CONTRACT')
        self.assertEqual(provider.calls, 6)
        self.assertEqual(m6_results._measured(row)['logical_calls'], 6)
        self.assertGreater(row['wall_elapsed_ms'], row['active_elapsed_ms'] + 100)

    def test_g7_wall_includes_pacing_and_resume_does_not_count_idle(self):
        with tempfile.TemporaryDirectory() as temp:
            provider = SyntheticProvider([
                specialist('crowd'), specialist('health'), specialist('environment'), final([])])
            first = g7._one('test', copy.deepcopy(CASE), 42,
                             VerificationOracle({CID: CLEAN}), FROZEN, PLAN,
                             temp, provider=provider, pacer=SlowPacer())
            self.assertEqual(first['event'], 'DONE')
            self.assertEqual(provider.calls, 4)
            self.assertGreater(first['wall_elapsed_ms'] - first['active_elapsed_ms'], 70)
            time.sleep(0.12)
            second = g7._one('test', copy.deepcopy(CASE), 42,
                              VerificationOracle({CID: CLEAN}), FROZEN, PLAN,
                              temp, provider=SyntheticProvider([]), pacer=SlowPacer())
            self.assertEqual(second['event'], 'DONE')
            self.assertGreaterEqual(second['wall_elapsed_ms'], first['wall_elapsed_ms'])
            self.assertLess(second['wall_elapsed_ms'] - first['wall_elapsed_ms'], 100)
            self.assertGreaterEqual(second['active_elapsed_ms'], first['active_elapsed_ms'])

    def test_provider_http_error_attempt_has_elapsed_time(self):
        calls = []
        def transport(req, timeout=None):
            calls.append(True)
            raise HTTPError(req.full_url, 503, 'synthetic', {}, None)
        provider = GeminiProvider(model_id='synthetic', api_key='SYNTHETIC_ONLY',
                                  transport=transport, sleeper=lambda _: None)
        from hajj_study.provider import CallFailure
        with self.assertRaises(CallFailure) as context:
            provider.generate_validated(system_prompt='system', user_prompt='user',
                response_schema={'type': 'object'}, validator=lambda v: v)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(isinstance(a['elapsed_ms'], float) and a['elapsed_ms'] >= 0
                            for a in context.exception.attempts))

    def test_contract_failure_measurement_with_zero_successes(self):
        base = {'event': 'FAILED', 'verification_count': 0,
                'model_version_frozen': VERSION, 'call_log': [],
                'previous_failures': [], 'failure': {'category': 'CONTRACT'}}
        measured = m6_results._measured(base)
        self.assertEqual(measured['logical_calls'], 0)
        self.assertEqual(measured['valid_calls'], 0)

    def test_validator_rejects_empty_evidence_even_when_schema_bounded(self):
        bad = final([])
        bad['supporting_evidence'] = []
        with self.assertRaises(ContractError):
            validate_final_report(bad, CID)

if __name__ == '__main__':
    unittest.main()
