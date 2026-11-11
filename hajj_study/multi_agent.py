"""M6.1: bounded expert-team case topology for offline integration testing.

Initial: Crowd -> Health -> Environment -> Gemini coordinator (4 calls).
If gate requires one verification, only the affected specialist is re-run,
followed by coordinator (2 additional calls). This is sequential fan-out/fan-in
for safe rate pacing; roles are independent requests, not one split response.

No live batch runner, frozen Test access, or scoring is implemented here:
those are distinct M6.2--M6.4 gates requiring review before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hajj_study import coordinator, specialists
from hajj_study.corrupt import DOMAINS
from hajj_study.gate import SharedDecisionGate
from hajj_study.prompts import PROMPT_VERSION
from hajj_study.provider import CallFailure
from hajj_study.schemas import ContractError, validate_case
from hajj_study.verify import VerificationOracle


@dataclass
class MultiCaseResult:
    case_id: str
    status: str
    risk_level: str | None
    trace: dict


def execute_multi_case(case: dict, *, provider: Any, gate: SharedDecisionGate,
                       oracle: VerificationOracle, before_call=None,
                       expected_model_version: str | None = None) -> MultiCaseResult:
    """Execute one controlled case, never creating fake fallbacks for failed agents."""
    case = validate_case(case)
    cid = case['case_id']
    trace = {
        'case_id': cid, 'condition': 'multi', 'prompt_version': PROMPT_VERSION,
        'topology': 'three_specialists_then_gemini_coordinator',
        'initial_specialists': {}, 'initial_coordinator': None, 'initial_gate': None,
        'verification_domain': None, 'verification_count': 0,
        'verified_observations': None, 'post_specialist': None,
        'post_coordinator': None, 'post_gate': None,
        'logical_agent_calls': 0, 'api_attempts': [], 'call_trace': [],
        'failure': None,
    }
    active_role = None

    def record_call(role, phase, domain, invoke):
        nonlocal active_role
        active_role = role
        trace['logical_agent_calls'] += 1
        record = {'role': role, 'domain': domain, 'phase': phase}
        if before_call is not None:
            before_call()  # M6.3 supplies the same rate pacer as G6.
        try:
            call = invoke()
        except CallFailure as exc:
            record['status'] = 'FAILED'
            record['attempts'] = list(exc.attempts)
            trace['call_trace'].append(record)
            trace['api_attempts'].extend(dict(a, role=role, domain=domain, phase=phase)
                                         for a in exc.attempts)
            raise
        record['status'] = 'STOP'
        record['attempts'] = list(call.attempts)
        record['model_version'] = call.model_version
        record['elapsed_ms'] = getattr(call, 'total_elapsed_ms', None)
        trace['call_trace'].append(record)
        trace['api_attempts'].extend(dict(a, role=role, domain=domain, phase=phase)
                                     for a in call.attempts)
        if (expected_model_version is not None and
                call.model_version != expected_model_version):
            raise ContractError('Gemini modelVersion differs from frozen G5')
        return call.report

    def run_specialist(domain, observed_case, phase, verified=None):
        return record_call(
            'specialist', phase, domain,
            lambda: specialists.request_specialist(observed_case, domain, provider=provider,
                                                    verified_domain=verified),
        )

    def run_coordinator(reports, phase, observed_case):
        return record_call(
            'coordinator', phase, None,
            lambda: coordinator.request_coordinator(observed_case, reports, provider=provider),
        )

    try:
        reports = {}
        for domain in DOMAINS:
            reports[domain] = run_specialist(domain, case, 'initial')
            trace['initial_specialists'][domain] = reports[domain]
        first = run_coordinator(reports, 'initial', case)
        trace['initial_coordinator'] = first
        initial = gate.initial(first, cid)
        trace['initial_gate'] = {
            'status': initial.status, 'risk_level': initial.risk_level,
            'evidence_sufficiency': initial.evidence_sufficiency,
        }
        if initial.status == 'ISSUE':
            return MultiCaseResult(cid, 'ISSUE', initial.risk_level, trace)
        if initial.status != 'VERIFY':
            raise ContractError('Initial multi-agent gate must ISSUE or VERIFY')
        domain = initial.verification_domain
        trace['verification_domain'] = domain
        session = oracle.new_session(cid)
        updated = session.verify_domain(case['observations'], domain, case_id=cid)
        trace['verification_count'] = 1
        trace['verified_observations'] = updated
        revised_case = {**case, 'observations': updated}
        revised = dict(reports)  # Keep the other two reports intact.
        revised[domain] = run_specialist(domain, revised_case, 'post', verified=domain)
        trace['post_specialist'] = revised[domain]
        after = run_coordinator(revised, 'post', revised_case)
        trace['post_coordinator'] = after
        post = gate.post_verification(first, after, cid, domain,
                                      verification_used=session.used)
        trace['post_gate'] = {
            'status': post.status, 'risk_level': post.risk_level,
            'evidence_sufficiency': post.evidence_sufficiency,
            'merged_flags': list(post.evidence_flags),
        }
        return MultiCaseResult(cid, post.status, post.risk_level, trace)
    except CallFailure as exc:
        trace['failure'] = {
            'role': active_role, 'category': exc.category,
            'finish_reason': exc.finish_reason, 'http_status': exc.status_code,
        }
    except ContractError as exc:
        trace['failure'] = {'role': active_role, 'category': 'CONTRACT',
                            'reason': str(exc)}
    return MultiCaseResult(cid, 'FAILED', None, trace)
