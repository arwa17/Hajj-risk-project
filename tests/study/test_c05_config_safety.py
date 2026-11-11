"""Offline-only regression of explicit requested sampling config.

No credentials, requests, source rows or held-out Gold are read here.
"""
import unittest

from hajj_study.provider import GeminiProvider
from hajj_study import g5


class C05ConfigSafety(unittest.TestCase):
    def test_explicit_requested_top_p_reaches_actual_request(self):
        p = g5._provider({
            'model_id': 'fixture-only', 'temperature': 0.0, 'top_p': 1.0,
            'thinking_budget': None, 'max_output_tokens': 4096,
        }, 42)
        body = p.request_data('system fixture', 'user fixture', {'type': 'OBJECT'})
        self.assertEqual(body['generationConfig']['topP'], 1.0)
        self.assertEqual(body['generationConfig']['temperature'], 0.0)
        self.assertEqual(body['generationConfig']['seed'], 42)
        self.assertEqual(body['generationConfig']['maxOutputTokens'], 4096)

    def test_invalid_top_p_fail_before_any_call(self):
        for value in (0, -0.1, 1.01, True, '1.0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GeminiProvider(model_id='fixture-only', top_p=value)

    def test_standalone_provider_omits_unrequested_top_p(self):
        p = GeminiProvider(model_id='fixture-only', transport=lambda *a, **k: None)
        body = p.request_data('system fixture', 'user fixture', {'type': 'OBJECT'})
        self.assertNotIn('topP', body['generationConfig'])

    def test_preflight_function_has_explicit_protocol_default(self):
        import inspect
        parameter = inspect.signature(g5.create_preflight).parameters['top_p']
        self.assertEqual(parameter.default, 1.0)
