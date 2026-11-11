"""G4 Gemini generateContent adapter with mockable transport and auditable token usage.

This module does not access the network during tests. Run the G5 live smoke
before declaring any model ID, temperature, seed or thinking setting frozen.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from urllib import error, request
from urllib.parse import quote

from hajj_study.attempt_journal import AttemptJournal, AttemptJournalError, utc_now


class ProviderError(RuntimeError):
    def __init__(self, reason: str, *, category='PROVIDER_ERROR', retryable=False,
                 usage=None, finish_reason=None, status_code=None, elapsed_ms=None):
        super().__init__(reason)
        self.category = category
        self.retryable = retryable
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.status_code = status_code
        self.elapsed_ms = elapsed_ms


class CallFailure(ProviderError):
    def __init__(self, original: ProviderError, attempts: list):
        super().__init__(str(original), category=original.category, retryable=original.retryable,
                         usage=original.usage, finish_reason=original.finish_reason,
                         status_code=original.status_code, elapsed_ms=original.elapsed_ms)
        self.attempts = attempts


@dataclass(frozen=True)
class ProviderResult:
    text: str
    finish_reason: str
    model_version: str | None
    usage: dict
    elapsed_ms: float
    response_sha256: str
    raw_http_response_b64_private: str | None = None
    raw_http_sha256: str | None = None
    replayed_from_journal: bool = False


@dataclass(frozen=True)
class ValidatedCall:
    report: dict
    attempts: list[dict]
    model_version: str | None
    output_text: str
    total_elapsed_ms: float
    raw_http_response_b64_private: str | None = None
    raw_http_sha256: str | None = None


class GeminiProvider:
    """One shared provider and retry policy for single/treatment conditions."""
    def __init__(self, *, model_id: str, api_key: str | None = None,
                 temperature: float | None = None, seed: int | None = None,
                 thinking_budget: int | None = None, max_output_tokens: int = 4096,
                 top_p: float | None = None,
                 timeout_s: float = 120.0, base_url='https://generativelanguage.googleapis.com/v1beta',
                 transport=None, sleeper=None, journal_dir=None, audit_scope=None):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('Explicit model_id required before smoke')
        if temperature is not None and (isinstance(temperature, bool) or
            not isinstance(temperature, (float, int)) or not 0 <= temperature <= 2):
            raise ValueError('Temperature must be omitted or in [0,2]')
        if top_p is not None and (isinstance(top_p, bool) or
            not isinstance(top_p, (float, int)) or not 0 < top_p <= 1):
            raise ValueError('top_p must be omitted or within (0,1]')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647):
            raise ValueError('seed out of API range')
        if thinking_budget is not None and (not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool)):
            raise ValueError('thinking_budget must be a supported integer; verify it during smoke')
        if not isinstance(max_output_tokens, int) or max_output_tokens < 256:
            raise ValueError('max_output_tokens too small for structured reports')
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.model_id = model_id
        self.api_key = api_key if api_key is not None else os.getenv('GEMINI_API_KEY', '')
        self.temperature = float(temperature) if temperature is not None else None
        self.seed = seed
        self.top_p = float(top_p) if top_p is not None else None
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip('/')
        self.live_transport = transport is None
        self.transport = transport or request.urlopen
        self.sleeper = sleeper or time.sleep
        self.journal_dir = journal_dir
        self.audit_scope = audit_scope
        self._journal = None
        self._current_wal_path = None
        self._last_received = False
        self._last_request_sha256 = None

    def _mark_validation(self, status):
        if self._journal is not None and self._last_received:
            self._journal.annotate(self._current_wal_path,
                                   {'validation_status': status, 'validated_utc': utc_now()})

    def request_data(self, system_prompt: str, user_prompt: str, response_schema: dict) -> dict:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError('Missing system prompt')
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError('Missing user prompt')
        config = {
            'responseMimeType': 'application/json',
            'responseSchema': response_schema,
            'maxOutputTokens': self.max_output_tokens,
        }
        if self.temperature is not None:
            config['temperature'] = self.temperature
        if self.seed is not None:
            config['seed'] = self.seed
        if self.top_p is not None:
            config['topP'] = self.top_p
        if self.thinking_budget is not None:
            config['thinkingConfig'] = {'thinkingBudget': self.thinking_budget}
        return {
            'systemInstruction': {'parts': [{'text': system_prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
            'generationConfig': config,
        }

    def generate_once(self, system_prompt: str, user_prompt: str,
                      response_schema: dict, *, attempt_no: int = 1) -> ProviderResult:
        if not self.api_key:
            raise ProviderError('GEMINI_API_KEY is not configured', category='NO_API_KEY')
        payload = self.request_data(system_prompt, user_prompt, response_schema)
        payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
        endpoint = self.base_url + '/models/' + quote(self.model_id, safe='-_.') + ':generateContent'
        req = request.Request(endpoint, data=payload_bytes, method='POST',
                              headers={'Content-Type': 'application/json', 'x-goog-api-key': self.api_key})
        # All real-network requests require durable WAL, before bytes are sent.
        directory = self.journal_dir or os.getenv('CPCS683_V2_PRIVATE_ATTEMPT_LOG_DIR')
        if self.live_transport and not directory:
            raise AttemptJournalError('STOP: private attempt log not configured; no Gemini request sent')
        journal = AttemptJournal(directory, allow_test_directory=not self.live_transport) if directory else None
        self._journal = journal
        self._current_wal_path = None
        self._last_received = False
        self._last_request_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        if self.live_transport and not self.audit_scope:
            raise AttemptJournalError('STOP: logical invocation scope missing; no Gemini request sent')
        wal_path = entry = None
        replayed = False
        if journal:
            wal_path, entry, replayed = journal.start_or_replay(
                self.audit_scope, attempt_no, self.model_id, payload_bytes, endpoint)
            self._current_wal_path = wal_path
        started = time.perf_counter()
        try:
            if replayed:
                raw = base64.b64decode(entry['raw_response_base64'], validate=True)
                http_status = entry['http_status']
            else:
                try:
                    with self.transport(req, timeout=self.timeout_s) as response:
                        raw = response.read()
                        http_status = int(response.getcode() or 200) if hasattr(response, 'getcode') else 200
                except error.HTTPError as exc:
                    raw = exc.read()  # capture the exact provider error body privately
                    http_status = int(exc.code)
                except (error.URLError, OSError, TimeoutError) as exc:
                    # Delivery can be UNKNOWN after a network error. Never retry automatically.
                    if journal:
                        journal.unknown(wal_path, entry, type(exc).__name__)
                    raise ProviderError('Unknown Gemini delivery after transport error',
                                        category='UNKNOWN_DELIVERY', retryable=False,
                                        elapsed_ms=max(0.0, (time.perf_counter()-started)*1000.0)) from exc
                if journal:
                    journal.received(wal_path, entry, raw, http_status,
                                     max(0.0, (time.perf_counter() - started) * 1000.0))
            self._last_received = bool(journal)
            elapsed_ms = (float(entry['provider_elapsed_ms']) if replayed
                          else max(0.0, (time.perf_counter() - started) * 1000.0))
            raw_digest = hashlib.sha256(raw).hexdigest()
            raw_b64 = base64.b64encode(raw).decode('ascii')
            if http_status != 200:
                raise ProviderError(f'HTTP {http_status}', category='HTTP',
                                    # 429 should stop for quota recovery, not burn a second request.
                                    retryable=(500 <= http_status <= 599),
                                    status_code=http_status, elapsed_ms=elapsed_ms)
            try:
                obj = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProviderError('Malformed Gemini transport JSON', category='INVALID_PROVIDER_JSON',
                                    elapsed_ms=elapsed_ms) from exc
            usage = obj.get('usageMetadata') if isinstance(obj, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            token_use = {
                'prompt_tokens': usage.get('promptTokenCount'),
                'candidates_tokens': usage.get('candidatesTokenCount'),
                'thoughts_tokens': usage.get('thoughtsTokenCount'),
                'total_tokens': usage.get('totalTokenCount'),
            }
            try:
                candidate = obj['candidates'][0]
                reason = candidate.get('finishReason')
                parts = candidate.get('content', {}).get('parts', [])
                if not isinstance(parts, list):
                    raise TypeError('parts is not a list')
                visible = ''.join(part.get('text', '') for part in parts
                                  if isinstance(part, dict) and not part.get('thought', False))
                model_version = obj.get('modelVersion')
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                raise ProviderError('No valid Gemini candidate', category='MISSING_CANDIDATE',
                                    usage=token_use, elapsed_ms=elapsed_ms) from exc
            if journal:
                journal.annotate(wal_path, {'observed_model_version': model_version,
                                            'finish_reason': reason, 'usage': token_use})
            if reason != 'STOP':
                raise ProviderError(f'Gemini candidate finished with {reason!r}',
                                    category='MAX_TOKENS' if reason == 'MAX_TOKENS' else 'NON_STOP',
                                    usage=token_use, finish_reason=reason, elapsed_ms=elapsed_ms)
            if not visible.strip():
                raise ProviderError('Empty visible Gemini response', category='EMPTY_TEXT',
                                    usage=token_use, finish_reason=reason, elapsed_ms=elapsed_ms)
            return ProviderResult(text=visible, finish_reason=reason, model_version=model_version,
                                  usage=token_use, elapsed_ms=elapsed_ms,
                                  response_sha256=hashlib.sha256(visible.encode('utf-8')).hexdigest(),
                                  raw_http_response_b64_private=raw_b64,
                                  raw_http_sha256=raw_digest,
                                  replayed_from_journal=replayed)
        except AttemptJournalError:
            # If persistence fails, do not silently run another API call.
            raise

    def generate_validated(self, *, system_prompt: str, user_prompt: str,
                           response_schema: dict, validator) -> ValidatedCall:
        """Bounded attempt policy: one retry for schema JSON and known HTTP 5xx.

        A retry is an API attempt, not another verification operation. Failures
        must be surfaced as FAILED by the outer experiment runner.
        """
        attempts = []
        call_started = time.perf_counter()
        for n in (1, 2):
            try:
                result = self.generate_once(system_prompt, user_prompt, response_schema, attempt_no=n)
                try:
                    report = validator(json.loads(result.text))
                except (ValueError, TypeError, KeyError) as exc:
                    attempts.append({'attempt': n, 'status': 'INVALID_REPORT',
                                     'finish_reason': result.finish_reason,
                                     'model_version': result.model_version,
                                     'usage': result.usage, 'elapsed_ms': result.elapsed_ms,
                                     'response_sha256': result.response_sha256,
                                     'raw_http_sha256': result.raw_http_sha256,
                                     'request_sha256': self._last_request_sha256,
                                     'replayed_from_journal': result.replayed_from_journal})
                    self._mark_validation('INVALID_REPORT')
                    failure = ProviderError('Malformed/invalid structured report',
                                            category='INVALID_REPORT', retryable=True,
                                            usage=result.usage, finish_reason=result.finish_reason)
                    if n == 2:
                        raise CallFailure(failure, attempts) from exc
                    continue
                self._mark_validation('VALID_REPORT')
                attempts.append({'attempt': n, 'status': 'STOP',
                                 'finish_reason': result.finish_reason,
                                 'model_version': result.model_version, 'usage': result.usage,
                                 'elapsed_ms': result.elapsed_ms, 'response_sha256': result.response_sha256,
                                 'raw_http_sha256': result.raw_http_sha256,
                                 'request_sha256': self._last_request_sha256,
                                 'replayed_from_journal': result.replayed_from_journal})
                # Recover original provider service latency from the journal on replay,
                # rather than incorrectly recording a cached call as zero API time.
                observed_ms = (time.perf_counter()-call_started)*1000
                service_ms = sum(float(a.get('elapsed_ms') or 0.0) for a in attempts)
                return ValidatedCall(report=report, attempts=attempts,
                                     model_version=result.model_version, output_text=result.text,
                                     total_elapsed_ms=max(observed_ms, service_ms),
                                     raw_http_response_b64_private=result.raw_http_response_b64_private,
                                     raw_http_sha256=result.raw_http_sha256)
            except CallFailure:
                raise
            except ProviderError as exc:
                self._mark_validation('PROVIDER_' + exc.category)
                attempts.append({'attempt': n, 'status': exc.category,
                                 'finish_reason': exc.finish_reason, 'usage': exc.usage,
                                 'http_status': exc.status_code, 'elapsed_ms': exc.elapsed_ms})
                if not exc.retryable or n == 2:
                    raise CallFailure(exc, attempts) from exc
                self.sleeper(1.0)  # recorded as elapsed total, excluded from per-call service timing
        raise AssertionError('unreachable')
