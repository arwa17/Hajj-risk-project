"""G4 unified deterministic policy shared by Single-Agent and Multi-Agent.

The gate cannot access proxy labels, clean scorer references, or model-fitting
features. Its only inputs are validated agent reports, flags, k and the
one-verification state.
"""
from __future__ import annotations

from dataclasses import dataclass

from hajj_study.corrupt import DOMAINS, FIELD_DOMAIN
from hajj_study.schemas import ContractError, validate_final_report

FLAG_SEVERITY = {'conflicting': 3, 'missing': 2, 'suspected_noisy': 1}
DOMAIN_TIE_ORDER = {'crowd': 0, 'health': 1, 'environment': 2}


@dataclass(frozen=True)
class GateOutcome:
    status: str  # ISSUE / VERIFY / ABSTAIN
    risk_level: str | None
    evidence_sufficiency: int
    verification_domain: str | None
    evidence_flags: tuple


class SharedDecisionGate:
    def __init__(self, k: int):
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 7:
            raise ValueError('Shared acceptance k must be an integer in [0, 7]')
        self.k = k

    @staticmethod
    def unique_flagged_fields(report) -> set[str]:
        return {flag['field'] for flag in report['evidence_flags']}

    @classmethod
    def sufficiency(cls, report) -> int:
        result = 7 - len(cls.unique_flagged_fields(report))
        if not 0 <= result <= 7:
            raise ContractError('Illegal number of flagged fields')
        return result

    @staticmethod
    def choose_domain(report) -> str | None:
        flags = report['evidence_flags']
        if not flags:
            return None
        ranked = sorted(
            flags,
            key=lambda f: (-FLAG_SEVERITY[f['type']],
                           DOMAIN_TIE_ORDER[FIELD_DOMAIN[f['field']]],
                           f['field']),
        )
        return FIELD_DOMAIN[ranked[0]['field']]

    def initial(self, report: dict, case_id: str) -> GateOutcome:
        valid = validate_final_report(report, case_id)
        score = self.sufficiency(valid)
        flags = tuple(valid['evidence_flags'])
        if score >= self.k:
            return GateOutcome('ISSUE', valid['proposed_risk_level'], score, None, flags)
        domain = self.choose_domain(valid)
        if domain is None:  # threshold up to 7, so this is impossible under valid inputs
            raise ContractError('Below threshold but no domain is eligible for verification')
        return GateOutcome('VERIFY', None, score, domain, flags)

    @staticmethod
    def merge_post_flags(previous: dict, revised: dict, verified_domain: str) -> list[dict]:
        """Never allow re-run agent to erase prior flags outside verified domain."""
        if verified_domain not in DOMAINS:
            raise ContractError('Cannot merge flags for unknown domain')
        # A single field may have multiple valid flag types, but the shared
        # sufficiency gate already counts unique fields. Canonicalize without
        # losing unverified-domain flags or selecting based on model outcomes.
        def choose(flags):
            selected = {}
            for flag in sorted(flags, key=lambda f: (
                    -FLAG_SEVERITY[f['type']], f['type'], f['reason'])):
                selected.setdefault(flag['field'], flag)
            return selected

        # Never allow a post-verification report to erase (or replace) an
        # existing flag outside the verified domain.
        protected = choose(f for f in previous['evidence_flags']
                           if FIELD_DOMAIN[f['field']] != verified_domain)
        revised_flags = choose(revised['evidence_flags'])
        selected = dict(protected)
        for field, flag in revised_flags.items():
            if FIELD_DOMAIN[field] == verified_domain or field not in selected:
                selected[field] = flag
        return [selected[field] for field in sorted(selected)]

    def post_verification(self, prior_report: dict, revised_report: dict,
                          case_id: str, verified_domain: str, *, verification_used: bool) -> GateOutcome:
        if not verification_used or verified_domain not in DOMAINS:
            raise ContractError('Post-verification decision requires one completed verification')
        before = validate_final_report(prior_report, case_id)
        after = validate_final_report(revised_report, case_id)
        merged_flags = self.merge_post_flags(before, after, verified_domain)
        merged = {**after, 'evidence_flags': merged_flags}
        valid = validate_final_report(merged, case_id)
        score = self.sufficiency(valid)
        return GateOutcome('ISSUE' if score >= self.k else 'ABSTAIN',
                           valid['proposed_risk_level'] if score >= self.k else None,
                           score, None, tuple(merged_flags))
