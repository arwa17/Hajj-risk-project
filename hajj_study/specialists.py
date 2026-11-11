"""M6.1: domain-isolated specialist Gemini calls (one role per request).

The shared frozen provider owns model settings/retries. Specialists never see
shared ML output, other-domain observations, or any scorer/proxy reference.
"""
from __future__ import annotations

from hajj_study.corrupt import DOMAINS
from hajj_study.prompts import specialist_messages
from hajj_study.schemas import (ContractError, specialist_response_schema,
                                validate_case, validate_specialist_report)


def request_specialist(case: dict, domain: str, *, provider, verified_domain=None):
    """Make one structured specialist logical call; return provider call metadata."""
    case = validate_case(case)
    if domain not in DOMAINS:
        raise ContractError('Unknown specialist domain')
    system, user = specialist_messages(case, domain, verified_domain=verified_domain)
    call = provider.generate_validated(
        system_prompt=system,
        user_prompt=user,
        response_schema=specialist_response_schema(domain),
        validator=lambda response: validate_specialist_report(response, case['case_id'], domain),
    )
    # Also guard against test/mock providers ignoring their validator.
    validate_specialist_report(call.report, case['case_id'], domain)
    return call
