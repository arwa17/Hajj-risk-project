"""Frozen-candidate G4 prompts; change only using validation evidence before test.

No proxy equation, weight, threshold, reference answer, raw ML-only features,
or hidden corruption metadata can enter a prompt.
"""
from __future__ import annotations

import json
from hajj_study.corrupt import DOMAINS
from hajj_study.schemas import validate_case, validate_specialist_report, ContractError

PROMPT_VERSION = 'g4-candidate-v1'  # frozen only after live smoke and validation

FIELD_GUIDE = {
    'Crowd_Density': 'Low / Medium / High; more crowding tends to increase concern',
    'Temperature': 'Degrees C, typically 30–45; higher heat tends to increase concern',
    'Fatigue_Level': 'Low / Medium / High; more fatigue tends to increase concern',
    'Stress_Level': 'Low / Medium / High; more stress tends to increase concern',
    'Queue_Time_minutes': 'Minutes, typically 0–60; longer waits tend to increase concern',
    'Emergency_Event': 'Yes / No; an emergency report tends to increase concern',
    'Distance_Between_People_m': 'Metres, typically 0.5–2.5; shorter distance tends to increase concern',
}
RISK_GUIDE = ('LOW = limited concern based on available evidence; '
              'MEDIUM = notable concern requiring human attention; '
              'HIGH = urgent concern requiring prompt human review. '
              'These are operational descriptions, not mathematical score boundaries.')
FLAG_GUIDE = ('Use a field-level evidence flag only when warranted by observed evidence: '
              'missing (null), conflicting (two differing readings of one field), '
              'suspected_noisy (a plausible but doubtful reading supported by explicit evidence). '
              'A single unexpected value is not proof of corruption. If evidence is sufficient, '
              'report an empty evidence_flags list. Do not invent a hidden clean reading.')


def _json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _guidance(fields):
    return '\n'.join(f'- {name}: {FIELD_GUIDE[name]}' for name in fields)


def generalist_messages(case, *, verified_domain=None):
    case = validate_case(case)
    if verified_domain is not None and verified_domain not in DOMAINS:
        raise ContractError('Unknown verified domain')
    system = ('You are the SINGLE GENERALIST for a synthetic Hajj crowd-risk research study. '
              'Analyze the observed Crowd, Health and Environment evidence jointly. '
              'Propose LOW, MEDIUM or HIGH based on available evidence and uncertainty. '
              'The same risk concepts are used by the treatment coordinator.\n'
              + RISK_GUIDE + '\n' + FLAG_GUIDE + '\n'
              'Field descriptions:\n' + _guidance(FIELD_GUIDE) + '\n'
              'Return strictly the final JSON report: case_id, proposed_risk_level, '
              'evidence_flags (field,type,reason), supporting_evidence (list of concise statements). '
              'You propose a risk level; a separate fixed policy decides whether to issue it, '
              'verify one domain, or abstain. Do not perform arithmetic label reconstruction.')
    user_payload = {'case_id': case['case_id'], 'observations': case['observations'],
                    'shared_ml_prediction': case['ml_prediction']}
    if verified_domain is not None:
        user_payload['verified_domain'] = verified_domain
        user_payload['verification_notice'] = 'Observations from this domain have been checked; other domains are unchanged.'
    return system, _json(user_payload)


def specialist_messages(case, domain, *, verified_domain=None):
    case = validate_case(case)
    if domain not in DOMAINS:
        raise ContractError('Unknown specialist domain')
    if verified_domain is not None and verified_domain != domain:
        raise ContractError('A specialist may be re-run only for its verified domain')
    fields = DOMAINS[domain]
    system = (f'You are the {domain.upper()} SPECIALIST in a synthetic Hajj crowd-risk study. '
              'Review only your authorized domain. Send evidence and a domain risk signal '
              'to a coordinator; you do not issue the overall system decision.\n'
              + RISK_GUIDE + '\n' + FLAG_GUIDE + '\n'
              'Authorized field descriptions:\n' + _guidance(fields) + '\n'
              'Return strictly JSON with case_id, domain, domain_risk_signal, '
              'evidence_flags (field,type,reason), supporting_evidence. '
              'Do not refer to fields or observations from other domains.')
    payload = {'case_id': case['case_id'], 'domain': domain,
               'observations': {field: case['observations'][field] for field in fields}}
    if verified_domain:
        payload['verified_domain'] = verified_domain
        payload['verification_notice'] = 'These domain observations have been checked.'
    return system, _json(payload)


def coordinator_messages(case_id, specialist_reports, shared_ml_prediction):
    if set(specialist_reports) != set(DOMAINS):
        raise ContractError('Coordinator needs exactly Crowd, Health and Environment reports')
    for domain, report in specialist_reports.items():
        validate_specialist_report(report, case_id, domain)
    system = ('You are the TREATMENT COORDINATOR for a synthetic Hajj crowd-risk study. '
              'Combine the three specialist reports into one overall proposed risk level. '
              'You receive the same global ML prediction that the generalist receives, but '
              'you do not receive hidden source observations or scoring rules.\n'
              + RISK_GUIDE + '\n' + FLAG_GUIDE + '\n'
              'Field descriptions (use only what is reported by specialists):\n'
              + _guidance(FIELD_GUIDE) + '\n'
              'Return strictly the same final JSON schema as the generalist: case_id, '
              'proposed_risk_level, evidence_flags (field,type,reason), supporting_evidence. '
              'Use evidence supplied by specialists; do not invent observations. '
              'A separate fixed policy handles final issue, verification or abstention.')
    payload = {'case_id': case_id, 'specialist_reports': specialist_reports,
               'shared_ml_prediction': shared_ml_prediction}
    return system, _json(payload)
