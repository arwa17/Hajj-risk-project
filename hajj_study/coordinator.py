"""M6.1: real Gemini coordinator: receives only structured reports + frozen ML.

No direct access to raw seven observations, clean values, or scorer labels.
The coordinator has the SAME final-report contract as the generalist.
"""
from __future__ import annotations

from hajj_study.corrupt import DOMAINS
from hajj_study.prompts import coordinator_messages
from hajj_study.schemas import (ContractError, final_response_schema,
                                validate_case, validate_final_report,
                                validate_specialist_report)


def request_coordinator(case: dict, specialist_reports: dict, *, provider):
    """Synthesize a final risk proposal with an independent Gemini request."""
    case = validate_case(case)  # Contract boundary even for standalone use.
    cid = case['case_id']
    if not isinstance(specialist_reports, dict) or set(specialist_reports) != set(DOMAINS):
        raise ContractError('Coordinator requires exactly the three specialist reports')
    checked = {domain: validate_specialist_report(specialist_reports[domain], cid, domain)
               for domain in DOMAINS}
    system, user = coordinator_messages(cid, checked, case['ml_prediction'])
    call = provider.generate_validated(
        system_prompt=system,
        user_prompt=user,
        response_schema=final_response_schema(),
        validator=lambda response: validate_final_report(response, cid),
    )
    validate_final_report(call.report, cid)
    return call
