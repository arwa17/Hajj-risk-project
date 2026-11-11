import io
import json
import unittest
from urllib import error

from hajj_study.provider import GeminiProvider, ProviderError, CallFailure
from hajj_study.schemas import final_response_schema

CID='HJ-V003'
REPORT={'case_id':CID,'proposed_risk_level':'MEDIUM','evidence_flags':[],
        'supporting_evidence':['Observed fields support an intermediate risk assessment.']}

class FakeResponse:
    def __init__(self,obj):self.body=json.dumps(obj).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.body


def gemini_response(finish='STOP',report=REPORT):
    return {'candidates':[{'finishReason':finish, 'content':{'parts':[
        {'text':'hidden internal analysis','thought':True},
        {'text':json.dumps(report)}]}}],
        'modelVersion':'gemini-test-version',
        'usageMetadata':{'promptTokenCount':37,'candidatesTokenCount':45,
                         'thoughtsTokenCount':121,'totalTokenCount':203}}


class MockTests(unittest.TestCase):
    def provider(self,responses):
        seq=list(responses)
        def transport(req,timeout):
            assert timeout == 12
            assert req.headers.get('X-goog-api-key') == 'testkey'
            item=seq.pop(0)
            if isinstance(item,Exception):raise item
            return FakeResponse(item)
        return GeminiProvider(model_id='gemini-test',api_key='testkey',seed=42,
                              temperature=0.0,max_output_tokens=4096,timeout_s=12,
                              transport=transport,sleeper=lambda t:None)

    def test_thinking_tokens_finish_reason_and_json_schema(self):
        p=self.provider([gemini_response()])
        result=p.generate_once('sys','user',final_response_schema())
        self.assertEqual(result.finish_reason,'STOP')
        self.assertEqual(result.model_version,'gemini-test-version')
        self.assertEqual(result.usage['thoughts_tokens'],121)
        self.assertEqual(result.usage['total_tokens'],203)
        self.assertNotIn('hidden internal analysis',result.text)
        self.assertEqual(json.loads(result.text),REPORT)

    def test_max_tokens_fail_explicitly(self):
        p=self.provider([gemini_response(finish='MAX_TOKENS')])
        with self.assertRaises(ProviderError) as caught:
            p.generate_once('sys','user',final_response_schema())
        self.assertEqual(caught.exception.category,'MAX_TOKENS')
        self.assertEqual(caught.exception.usage['thoughts_tokens'],121)
        self.assertEqual(caught.exception.finish_reason,'MAX_TOKENS')

    def test_invalid_json_retry_once_with_attempt_record(self):
        malformed=gemini_response()
        malformed['candidates'][0]['content']['parts'][1]['text']='not-json'
        p=self.provider([malformed,gemini_response()])
        x=p.generate_validated(system_prompt='sys',user_prompt='user',
            response_schema=final_response_schema(),validator=lambda d:d)
        self.assertEqual(x.report,REPORT)
        self.assertEqual([a['status'] for a in x.attempts],['INVALID_REPORT','STOP'])

    def test_http_429_stops_without_immediate_retry(self):
        # C05 protocol: a 429 is not retried immediately; original legacy test is retained in C04.
        from io import BytesIO
        from urllib.error import HTTPError
        from hajj_study.provider import GeminiProvider, CallFailure
        class Fake429:
            calls = 0
            def __call__(self, req, timeout):
                self.calls += 1
                raise HTTPError('https://example.invalid', 429, 'quota', {}, BytesIO(b'{"error":"quota"}'))
        transport = Fake429()
        p = GeminiProvider(model_id='fixture-model', api_key='FAKE_TEST_ONLY', transport=transport,
                           sleeper=lambda _: None)
        with self.assertRaises(CallFailure) as raised:
            p.generate_validated(system_prompt='system', user_prompt='user',
                response_schema={'type':'OBJECT'}, validator=lambda r:r)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.category, 'HTTP')
        self.assertEqual(len(raised.exception.attempts), 1)
        self.assertEqual(transport.calls, 1)

    def test_non_retryable_http_400_fails_fast(self):
        http=error.HTTPError('https://example.test',400,'bad request',{},io.BytesIO(b''))
        p=self.provider([http])
        with self.assertRaises(CallFailure) as caught:
            p.generate_validated(system_prompt='sys',user_prompt='user',
                response_schema=final_response_schema(),validator=lambda d:d)
        self.assertEqual(len(caught.exception.attempts),1)

    def test_config_is_explicit_and_no_api_key_in_body(self):
        p=self.provider([])
        req=p.request_data('sys','user',final_response_schema())
        gc=req['generationConfig']
        self.assertEqual(gc['responseMimeType'],'application/json')
        self.assertEqual(gc['maxOutputTokens'],4096)
        self.assertEqual(gc['temperature'],0.)
        self.assertEqual(gc['seed'],42)
        self.assertNotIn('testkey',json.dumps(req))
        q=GeminiProvider(model_id='gemini-test',api_key='secret',temperature=None,seed=None)
        self.assertNotIn('temperature',q.request_data('s','u',final_response_schema())['generationConfig'])
        self.assertNotIn('seed',q.request_data('s','u',final_response_schema())['generationConfig'])

if __name__ == '__main__':unittest.main()
