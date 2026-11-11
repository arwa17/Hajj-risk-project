"""No-key no-network mock regression for V2 fail-closed attempt journaling."""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from hajj_study.attempt_journal import AttemptJournal, AttemptJournalError
from hajj_study.provider import GeminiProvider, CallFailure


class Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def read(self):
        return self.body
    def getcode(self):
        return 200


def good(body=None):
    if body is None:
        body = {"case_id": "fixture", "risk_level": "LOW"}
    return json.dumps({
        'modelVersion':'fixture-model-v1',
        'usageMetadata':{'promptTokenCount':12,'candidatesTokenCount':8,'totalTokenCount':20},
        'candidates':[{'finishReason':'STOP', 'content':{'parts':[{'text':json.dumps(body)}]}}],
        'extra_provider_field':'keep-raw',
    }, sort_keys=True).encode('utf-8')


class FakeTransport:
    def __init__(self, replies):
        self.calls = 0
        self.replies = list(replies)
        self.requests = []
    def __call__(self, req, timeout):
        self.calls += 1
        self.requests.append(req)
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return Response(value)


class TestAttemptJournal(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.scope = 'DEV|fixture|seed42|single|initial'
    def provider(self, transport, scope=None):
        return GeminiProvider(model_id='fixture-model', api_key='FAKE_TEST_KEY_ONLY',
                              temperature=0, seed=42, transport=transport,
                              journal_dir=self.path, audit_scope=scope or self.scope, sleeper=lambda _: None)
    def invoke(self, p, system='fixture system', schema=None):
        return p.generate_validated(system_prompt=system, user_prompt='fixture user',
                                    response_schema=schema or {'type':'OBJECT'},
                                    validator=lambda x: x if x['case_id'] == 'fixture' else ValueError('bad'))
    def entry(self):
        files=list(self.path.glob('*.json'))
        self.assertEqual(len(files),1)
        return json.loads(files[0].read_text())

    def test_raw_request_response_and_replay_without_transport(self):
        raw = good()
        transport = FakeTransport([raw])
        p = self.provider(transport)
        one=self.invoke(p)
        self.assertEqual(one.model_version,'fixture-model-v1')
        self.assertEqual(transport.calls,1)
        entry=self.entry()
        self.assertEqual(entry['state'],'RECEIVED')
        self.assertEqual(base64.b64decode(entry['raw_response_base64']),raw)
        self.assertEqual(entry['raw_response_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(entry['identity']['request_sha256'],hashlib.sha256(transport.requests[0].data).hexdigest())
        self.assertEqual(entry['identity']['request_body_utf8'].encode(),transport.requests[0].data)
        self.assertNotIn('FAKE_TEST_KEY_ONLY',json.dumps(entry))
        self.assertEqual(one.raw_http_response_b64_private,entry['raw_response_base64'])
        events = [json.loads(line) for line in next(self.path.glob('*.events.jsonl')).read_text().splitlines()]
        self.assertEqual(events[0]['type'],'STARTED')
        self.assertEqual(events[1]['type'],'RECEIVED')
        self.assertEqual(events[2]['type'],'ANNOTATION')
        # Replay must not invoke the transport; preserve original token and latency evidence.
        replay=self.invoke(p)
        self.assertEqual(transport.calls,1)
        self.assertEqual(replay.attempts[-1]['replayed_from_journal'],True)
        self.assertEqual(replay.attempts[-1]['usage']['total_tokens'],20)

    def test_unknown_prior_send_fail_closed(self):
        a=AttemptJournal(self.path, allow_test_directory=True)
        scope=self.scope
        body=b'{"test":"fixture"}'
        a.start_or_replay(scope,1,'fixture-model',body,'https://example.invalid/v1')
        transport=FakeTransport([good()])
        # journal collision at scope+attempt stops even when requested payload differs
        with self.assertRaises(AttemptJournalError):
            self.invoke(self.provider(transport))
        self.assertEqual(transport.calls,0)

    def test_changed_payload_same_scope_must_refuse(self):
        t=FakeTransport([good()])
        p=self.provider(t)
        self.invoke(p)
        with self.assertRaises(AttemptJournalError):
            self.invoke(p,system='DIFFERENT SYSTEM')
        self.assertEqual(t.calls,1)

    def test_missing_complete_response_never_retries(self):
        from urllib.error import URLError
        transport=FakeTransport([URLError('connection reset'), good()])
        p=self.provider(transport)
        with self.assertRaises(CallFailure) as c:
            self.invoke(p)
        self.assertEqual(c.exception.category,'UNKNOWN_DELIVERY')
        self.assertEqual(transport.calls,1)
        self.assertEqual(self.entry()['state'],'UNKNOWN_DELIVERY')
        with self.assertRaises(AttemptJournalError):
            self.invoke(p)
        self.assertEqual(transport.calls,1)

    def test_429_no_immediate_retry_and_body_retained(self):
        from io import BytesIO
        err=HTTPError('https://example.invalid',429,'quota',{},BytesIO(b'{"message":"rate limited"}'))
        t=FakeTransport([err,good()])
        with self.assertRaises(CallFailure) as e:
            self.invoke(self.provider(t))
        self.assertEqual(e.exception.status_code,429)
        self.assertEqual(t.calls,1)
        self.assertEqual(base64.b64decode(self.entry()['raw_response_base64']),b'{"message":"rate limited"}')

    def test_known_503_one_bounded_retry(self):
        from io import BytesIO
        t=FakeTransport([HTTPError('https://example.invalid',503,'temporary',{},BytesIO(b'{"error":503}')),good()])
        result=self.invoke(self.provider(t))
        self.assertEqual(t.calls,2)
        self.assertEqual(len(result.attempts),2)
        self.assertEqual(result.attempts[0]['http_status'],503)
        self.assertEqual(result.attempts[1]['status'],'STOP')
        self.assertEqual(len(list(self.path.glob('*.json'))),2)
        result2=self.invoke(self.provider(t))
        self.assertEqual(t.calls,2)
        self.assertEqual(len(result2.attempts),2)

    def test_invalid_json_retained_no_network_retry(self):
        t=FakeTransport([b'{broken'])
        with self.assertRaises(CallFailure) as exc:
            self.invoke(self.provider(t))
        self.assertEqual(exc.exception.category,'INVALID_PROVIDER_JSON')
        self.assertEqual(t.calls,1)
        self.assertEqual(base64.b64decode(self.entry()['raw_response_base64']),b'{broken')

    def test_same_payload_different_authorized_scope_is_separate(self):
        t=FakeTransport([good(),good()])
        self.invoke(self.provider(t,scope='dev-pass-1|case|42|initial'))
        self.invoke(self.provider(t,scope='dev-pass-2|case|42|initial'))
        self.assertEqual(t.calls,2)
        self.assertEqual(len(list(self.path.glob('*.json'))),2)

    def test_changed_model_seed_in_same_scope_refuses(self):
        t=FakeTransport([good()])
        self.invoke(self.provider(t))
        changed=GeminiProvider(model_id='fixture-model', api_key='FAKE_TEST_KEY_ONLY',
                               temperature=0, seed=77, transport=t, journal_dir=self.path,
                               audit_scope=self.scope, sleeper=lambda _:None)
        with self.assertRaises(AttemptJournalError):
            self.invoke(changed)
        self.assertEqual(t.calls,1)

    def test_corrupt_saved_response_stops_before_transport(self):
        t=FakeTransport([good()])
        self.invoke(self.provider(t))
        file=next(self.path.glob('*.json'))
        entry=json.loads(file.read_text())
        entry['raw_response_sha256']='00'*32
        file.write_text(json.dumps(entry))
        with self.assertRaises(AttemptJournalError):
            self.invoke(self.provider(t))
        self.assertEqual(t.calls,1)

    def test_schema_retry_does_not_overwrite_first_raw(self):
        bad=good({'case_id':'other'})
        t=FakeTransport([bad,good()])
        p=self.provider(t)
        # validator explicitly raises on mismatch
        result=p.generate_validated(system_prompt='sys',user_prompt='usr',
           response_schema={'type':'OBJECT'},
           validator=lambda x: x if x['case_id']=='fixture' else (_ for _ in ()).throw(ValueError('bad')))
        self.assertEqual(t.calls,2)
        self.assertEqual(result.attempts[0]['status'],'INVALID_REPORT')
        self.assertEqual(len(list(self.path.glob('*.json'))),2)


if __name__=='__main__':
    unittest.main()
