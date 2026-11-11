"""G4: trusted boundary validation for observed cases and agent JSON reports.

The scorer's clean references, ground-truth corruption metadata and weighted
proxy formula must never be included in any agent input or API request.
"""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any

from hajj_study.corrupt import DOMAINS, FIELD_DOMAIN, NUMERIC_PARAMS
from hajj_study.feature_contract import ALLOWED_MODEL_FEATURES
from hajj_study.proxy import PROXY_FIELDS

RISK_LEVELS = ('LOW', 'MEDIUM', 'HIGH')
FLAG_TYPES = ('missing', 'conflicting', 'suspected_noisy')
CASE_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
FORBIDDEN_AGENT_TERMS = tuple(ALLOWED_MODEL_FEATURES) + ('Risk_Label', 'Operational_Risk_Score', 'golden_label',
                         'gold_label', 'proxy_score', 'weighted_proxy',
                         'proxy_weights', 'source_row_index', 'private_metadata')

class ContractError(ValueError):
    """Malformed or out-of-contract case/report; never silently repair it."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_value(field: str, value: Any) -> None:
    if field in NUMERIC_PARAMS:
        lo, hi, *_ = NUMERIC_PARAMS[field]
        if not _is_number(value) or not lo <= value <= hi:
            raise ContractError(f'{field} must be a finite number in [{lo},{hi}]')
    elif field == 'Emergency_Event':
        if value not in ('No', 'Yes'):
            raise ContractError(f'{field} must be No or Yes')
    else:
        if value not in ('Low', 'Medium', 'High'):
            raise ContractError(f'{field} must be Low, Medium or High')


def _assert_no_forbidden_strings(payload: Any) -> None:
    txt = json.dumps(payload, ensure_ascii=False)
    for token in FORBIDDEN_AGENT_TERMS:
        if re.search(r'(?<!\w)' + re.escape(token) + r'(?!\w)', txt, flags=re.I):
            raise ContractError(f'Forbidden reference/scorer term in agent content: {token}')


def validate_observations(observations: Any, allowed_fields=PROXY_FIELDS) -> dict:
    allowed = set(allowed_fields)
    if not isinstance(observations, dict) or set(observations) != allowed:
        raise ContractError(f'Expected precisely these observation fields: {sorted(allowed)}')
    for field, observation in observations.items():
        if observation is None:
            continue
        if isinstance(observation, list):
            if len(observation) != 2 or not all(isinstance(x, dict) for x in observation):
                raise ContractError(f'{field}: conflict needs exactly 2 readings')
            if [x.get('source') for x in observation] != ['reading_a', 'reading_b']:
                raise ContractError(f'{field}: unknown or revealing reading source labels')
            if any(set(x) != {'source', 'value'} for x in observation):
                raise ContractError(f'{field}: unexpected conflict metadata')
            a, b = (observation[0]['value'], observation[1]['value'])
            if type(a) is not type(b) or a == b:
                raise ContractError(f'{field}: conflicts must differ with identical value types')
            _validate_value(field, a)
            _validate_value(field, b)
        else:
            _validate_value(field, observation)
    return deepcopy(observations)


def validate_case(case: Any) -> dict:
    if not isinstance(case, dict) or set(case) != {'case_id', 'observations', 'ml_prediction'}:
        raise ContractError('Case permits only case_id, observations, ml_prediction')
    case_id = case['case_id']
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ContractError('Invalid case ID')
    observed = validate_observations(case['observations'])
    prediction = case['ml_prediction']
    if not isinstance(prediction, dict) or set(prediction) != {'predicted_risk', 'probabilities'}:
        raise ContractError('Invalid frozen shared ML report')
    if prediction['predicted_risk'] not in RISK_LEVELS:
        raise ContractError('Unknown shared ML predicted class')
    probs = prediction['probabilities']
    if not isinstance(probs, dict) or set(probs) != set(RISK_LEVELS):
        raise ContractError('Shared ML needs all three class probabilities')
    if not all(_is_number(value) and 0 <= value <= 1 for value in probs.values()):
        raise ContractError('Shared ML has invalid probability values')
    if not abs(sum(probs.values()) - 1) <= 1e-5:
        raise ContractError('Shared ML probabilities must sum to 1')
    validated = {'case_id': case_id, 'observations': observed, 'ml_prediction': deepcopy(prediction)}
    _assert_no_forbidden_strings(validated)
    return validated


def _validate_flags(flags: Any, allowed_fields: set[str]) -> list[dict]:
    if not isinstance(flags, list) or len(flags) > 7:
        raise ContractError('evidence_flags must be a list with at most seven entries')
    seen = set()
    final = []
    for flag in flags:
        if not isinstance(flag, dict) or set(flag) != {'field', 'type', 'reason'}:
            raise ContractError('Each flag must have field, type, reason only')
        if flag['field'] not in allowed_fields or flag['type'] not in FLAG_TYPES:
            raise ContractError('Out-of-scope flag field or unknown type')
        if not isinstance(flag['reason'], str) or not 1 <= len(flag['reason'].strip()) <= 400:
            raise ContractError('A flag needs a short explicit reason')
        key = (flag['field'], flag['type'])
        if key in seen:
            raise ContractError('Duplicate flag; cannot inflate evidence insufficiency')
        seen.add(key)
        final.append({'field': flag['field'], 'type': flag['type'], 'reason': flag['reason'].strip()})
    return final


def _validate_evidence(evidence: Any) -> list[str]:
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 7:
        raise ContractError('supporting_evidence must be a nonempty list (up to seven statements)')
    if not all(isinstance(value, str) and 1 <= len(value.strip()) <= 450 for value in evidence):
        raise ContractError('Each evidence statement must be a short nonempty string')
    return [value.strip() for value in evidence]


def validate_final_report(report: Any, case_id: str) -> dict:
    required = {'case_id', 'proposed_risk_level', 'evidence_flags', 'supporting_evidence'}
    if not isinstance(report, dict) or set(report) != required or report.get('case_id') != case_id:
        raise ContractError('Final report schema or case_id mismatch')
    if report['proposed_risk_level'] not in RISK_LEVELS:
        raise ContractError('Generalist/Coordinator must propose a known risk level')
    validated = {
        'case_id': case_id,
        'proposed_risk_level': report['proposed_risk_level'],
        'evidence_flags': _validate_flags(report['evidence_flags'], set(PROXY_FIELDS)),
        'supporting_evidence': _validate_evidence(report['supporting_evidence']),
    }
    _assert_no_forbidden_strings(validated)
    return validated


def _reject_cross_domain_mentions(payload: dict, domain: str) -> None:
    """Reject literal other-domain field names in specialist free-text reports."""
    text = json.dumps({'supporting_evidence': payload['supporting_evidence'],
                       'evidence_flags': payload['evidence_flags']}, ensure_ascii=False)
    disallowed = set(PROXY_FIELDS) - set(DOMAINS[domain])
    for name in disallowed:
        if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, flags=re.I):
            raise ContractError('Specialist mentioned a field outside its assigned domain')


def validate_specialist_report(report: Any, case_id: str, domain: str) -> dict:
    required = {'case_id', 'domain', 'domain_risk_signal', 'evidence_flags', 'supporting_evidence'}
    if domain not in DOMAINS:
        raise ContractError('Unknown specialist domain')
    if not isinstance(report, dict) or set(report) != required or report.get('case_id') != case_id or report.get('domain') != domain:
        raise ContractError('Specialist schema, case_id or domain mismatch')
    if report.get('domain_risk_signal') not in RISK_LEVELS:
        raise ContractError('Invalid specialist risk signal')
    validated = {
        'case_id': case_id, 'domain': domain,
        'domain_risk_signal': report['domain_risk_signal'],
        'evidence_flags': _validate_flags(report['evidence_flags'], set(DOMAINS[domain])),
        'supporting_evidence': _validate_evidence(report['supporting_evidence']),
    }
    _assert_no_forbidden_strings(validated)
    _reject_cross_domain_mentions(validated, domain)
    return validated


def final_response_schema() -> dict:
    return {
        'type': 'object',
        'properties': {
            'case_id': {'type': 'string'},
            'proposed_risk_level': {'type': 'string', 'enum': list(RISK_LEVELS)},
            'evidence_flags': {'type': 'array', 'maxItems': 7, 'items': {
                'type': 'object',
                'properties': {'field': {'type': 'string', 'enum': list(PROXY_FIELDS)},
                               'type': {'type': 'string', 'enum': list(FLAG_TYPES)},
                               'reason': {'type': 'string', 'minLength': 1, 'maxLength': 400}},
                'required': ['field', 'type', 'reason'],
            }},
            'supporting_evidence': {'type': 'array', 'minItems': 1, 'maxItems': 7, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 450}},
        },
        'required': ['case_id', 'proposed_risk_level', 'evidence_flags', 'supporting_evidence'],
    }


def specialist_response_schema(domain: str) -> dict:
    if domain not in DOMAINS:
        raise ContractError('Unknown specialist domain')
    return {
        'type': 'object',
        'properties': {
            'case_id': {'type': 'string'},
            'domain': {'type': 'string', 'enum': [domain]},
            'domain_risk_signal': {'type': 'string', 'enum': list(RISK_LEVELS)},
            'evidence_flags': {'type': 'array', 'maxItems': 7, 'items': {
                'type': 'object',
                'properties': {'field': {'type': 'string', 'enum': list(DOMAINS[domain])},
                               'type': {'type': 'string', 'enum': list(FLAG_TYPES)},
                               'reason': {'type': 'string', 'minLength': 1, 'maxLength': 400}},
                'required': ['field', 'type', 'reason'],
            }},
            'supporting_evidence': {'type': 'array', 'minItems': 1, 'maxItems': 7, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 450}},
        },
        'required': ['case_id', 'domain', 'domain_risk_signal', 'evidence_flags', 'supporting_evidence'],
    }
