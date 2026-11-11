"""G4 real-agent interface for M4 and the M5 Single-Agent baseline.

The implementation is transport-independent for offline/mocked acceptance tests.
G5 supplies the real GeminiProvider after live smoke and config freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hajj_study.gate import GateOutcome, SharedDecisionGate
from hajj_study.prompts import PROMPT_VERSION, generalist_messages
from hajj_study.schemas import ContractError, final_response_schema, validate_case, validate_final_report
from hajj_study.provider import CallFailure
from hajj_study.verify import VerificationOracle


@dataclass
class SingleCaseResult:
    case_id: str
    status: str
    risk_level: str | None
    trace: dict


def _safe_attempts(calls):
    return [dict(attempt) for call in calls for attempt in call.attempts]


def execute_single_case(case: dict, *, provider: Any, gate: SharedDecisionGate,
                        oracle: VerificationOracle) -> SingleCaseResult:
    """Single initial proposal -> ISSUE, or VERIFY once -> ISSUE/ABSTAIN; errors FAILED."""
    case = validate_case(case)
    case_id = case['case_id']
    session = oracle.new_session(case_id)
    calls = []
    trace = {'case_id': case_id, 'condition': 'single', 'prompt_version': PROMPT_VERSION,
             'initial_report': None, 'initial_gate': None, 'verification_domain': None,
             'post_report': None, 'post_gate': None, 'api_attempts': [],
             'logical_agent_calls': 0, 'verification_count': 0, 'failure': None}

    def call_agent(observations, verified_domain=None):
        trace['logical_agent_calls'] += 1
        current = {'case_id': case_id, 'observations': observations,
                   'ml_prediction': case['ml_prediction']}
        system, user = generalist_messages(current, verified_domain=verified_domain)
        try:
            result = provider.generate_validated(
                system_prompt=system, user_prompt=user, response_schema=final_response_schema(),
                validator=lambda report: validate_final_report(report, case_id),
            )
        except CallFailure as exc:
            trace['api_attempts'].extend(exc.attempts)
            raise
        calls.append(result)
        trace['api_attempts'].extend(result.attempts)
        return result.report

    try:
        initial_report = call_agent(case['observations'])
        trace['initial_report'] = initial_report
        initial_gate = gate.initial(initial_report, case_id)
        trace['initial_gate'] = {'status': initial_gate.status,
                                 'risk_level': initial_gate.risk_level,
                                 'evidence_sufficiency': initial_gate.evidence_sufficiency}
        if initial_gate.status == 'ISSUE':
            return SingleCaseResult(case_id, 'ISSUE', initial_gate.risk_level, trace)
        domain = initial_gate.verification_domain
        trace['verification_domain'] = domain
        updated = session.verify_domain(case['observations'], domain, case_id=case_id)
        trace['verification_count'] = 1
        after_report = call_agent(updated, verified_domain=domain)
        trace['post_report'] = after_report
        post = gate.post_verification(initial_report, after_report, case_id, domain,
                                      verification_used=session.used)
        trace['post_gate'] = {'status': post.status, 'risk_level': post.risk_level,
                              'evidence_sufficiency': post.evidence_sufficiency,
                              'merged_flags': list(post.evidence_flags)}
        return SingleCaseResult(case_id, post.status, post.risk_level, trace)
    except CallFailure as exc:
        trace['failure'] = {'category': exc.category, 'reason': str(exc),
                            'finish_reason': exc.finish_reason}
        return SingleCaseResult(case_id, 'FAILED', None, trace)
    except ContractError as exc:
        trace['failure'] = {'category': 'CONTRACT', 'reason': str(exc)}
        return SingleCaseResult(case_id, 'FAILED', None, trace)
