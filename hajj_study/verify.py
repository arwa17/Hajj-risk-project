"""G4 simulated single-domain verification, isolated from any agent prompt.

The oracle is PRIVATE. Build its clean values from original source-row records
and scorer metadata only in the execution environment. Never serialize the
oracle, gold labels or source row indices to an LLM message.
"""
from __future__ import annotations

from copy import deepcopy

from hajj_study.corrupt import DOMAINS
from hajj_study.schemas import ContractError, validate_observations


class VerificationOracle:
    """Only seven clean observations per case; no class label or proxy formula."""
    def __init__(self, clean_by_case: dict[str, dict]):
        if not isinstance(clean_by_case, dict):
            raise ValueError('Verification oracle requires a case-to-clean-observations map')
        self._clean = {}
        for case_id, clean in clean_by_case.items():
            valid = validate_observations(clean)
            if any(v is None or isinstance(v, list) for v in valid.values()):
                raise ContractError('Oracle input must have original clean values only')
            self._clean[case_id] = valid

    def new_session(self, case_id: str):
        if case_id not in self._clean:
            raise ContractError('Case missing from verification oracle')
        return VerificationSession(case_id, deepcopy(self._clean[case_id]))


class VerificationSession:
    def __init__(self, case_id: str, clean: dict):
        self.case_id = case_id
        self.__clean = clean
        self.used = False

    def verify_domain(self, observed: dict, domain: str, *, case_id: str) -> dict:
        if self.used:
            raise ContractError('Verification budget exceeded: only one domain is permitted')
        if case_id != self.case_id:
            raise ContractError('Verification case_id mismatch')
        if domain not in DOMAINS:
            raise ContractError('Unknown verification domain')
        updated = validate_observations(observed)
        self.used = True
        for field in DOMAINS[domain]:
            updated[field] = deepcopy(self.__clean[field])
        # Only selected domain is released. No labels/metadata included.
        return updated
