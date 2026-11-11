"""Fully offline M4/M6 execution demonstration using real decision logic.

IMPORTANT: All responses and observations below are SCRIPTED SYNTHETIC FIXTURES.
This is NOT a live Gemini evaluation, not the frozen 30-case test, and not a
measurement of agent quality. It demonstrates pipeline and failure-safe routing.

Run at the repository root: python -m examples.offline_demo
No credentials, network access, source CSV, or private experiment artifacts needed.
"""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

from hajj_study.gate import SharedDecisionGate
from hajj_study.generalist import execute_single_case
from hajj_study.multi_agent import execute_multi_case
from hajj_study.verify import VerificationOracle


# A hard network-denial guard helps ensure the demonstration cannot send any data.
def _deny_network(*_args, **_kwargs):
    raise RuntimeError("NETWORK_DENIED_IN_OFFLINE_DEMO")


def enforce_no_network():
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    socket.create_connection = _deny_network


CLEAN = {
    "Crowd_Density": "Low",
    "Queue_Time_minutes": 50.0,
    "Distance_Between_People_m": 1.2,
    "Fatigue_Level": "High",
    "Stress_Level": "Medium",
    "Emergency_Event": "No",
    "Temperature": 39.0,
}
PREDICTION = {
    "predicted_risk": "MEDIUM",
    "probabilities": {"LOW": 0.2, "MEDIUM": 0.6, "HIGH": 0.2},
}


def flag(field):
    return {"field": field, "type": "missing", "reason": "Synthetic reading is absent."}


def final_report(cid, flags=()):
    return {
        "case_id": cid,
        "proposed_risk_level": "MEDIUM",
        "evidence_flags": list(flags),
        "supporting_evidence": ["Synthetic observation review for an offline software demo."],
    }


def specialist_report(cid, domain, flags=()):
    return {
        "case_id": cid,
        "domain": domain,
        "domain_risk_signal": "MEDIUM",
        "evidence_flags": list(flags),
        "supporting_evidence": ["Synthetic domain-specific observation review."],
    }


class ScriptedProvider:
    """A deterministic, memory-only substitute for the provider transport."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_validated(self, *, system_prompt, user_prompt, response_schema, validator):
        # Exercise the actual prompt construction and schema-validation boundary.
        payload = json.loads(user_prompt)
        if "case_id" not in payload or not self.responses:
            raise AssertionError("Unexpected scripted provider call")
        response = validator(self.responses.pop(0))
        self.calls += 1
        return SimpleNamespace(
            report=response,
            model_version="OFFLINE_SCRIPTED_NOT_GEMINI",
            total_elapsed_ms=0.0,
            attempts=[{"attempt": 1, "status": "STOP", "finish_reason": "STOP",
                       "model_version": "OFFLINE_SCRIPTED_NOT_GEMINI",
                       "usage": {"prompt_tokens": 0, "candidates_tokens": 0},
                       "elapsed_ms": 0.0}],
        )


def scenario(name):
    cid = {"clean": "SYNTH_CLEAN", "one_missing": "SYNTH_ONE_MISSING",
           "two_missing": "SYNTH_TWO_MISSING"}[name]
    observations = dict(CLEAN)
    if name != "clean":
        observations["Temperature"] = None
    if name == "two_missing":
        observations["Crowd_Density"] = None
    case = {"case_id": cid, "observations": observations,
            "ml_prediction": {"predicted_risk": PREDICTION["predicted_risk"],
                              "probabilities": dict(PREDICTION["probabilities"])}}
    initial = [] if name == "clean" else [flag("Temperature")]
    if name == "two_missing":
        initial.insert(0, flag("Crowd_Density"))

    single_responses = [final_report(cid, initial)]
    if name != "clean":
        single_responses.append(final_report(cid))

    multi_responses = [
        specialist_report(cid, "crowd", [flag("Crowd_Density")] if name == "two_missing" else []),
        specialist_report(cid, "health"),
        specialist_report(cid, "environment", [flag("Temperature")] if name != "clean" else []),
        final_report(cid, initial),
    ]
    if name != "clean":
        verified_domain = "crowd" if name == "two_missing" else "environment"
        multi_responses.extend([specialist_report(cid, verified_domain), final_report(cid)])

    oracle = VerificationOracle({cid: CLEAN})
    single_provider, multi_provider = ScriptedProvider(single_responses), ScriptedProvider(multi_responses)
    single = execute_single_case(case, provider=single_provider,
                                 gate=SharedDecisionGate(7), oracle=oracle)
    multi = execute_multi_case(case, provider=multi_provider,
                               gate=SharedDecisionGate(7), oracle=oracle,
                               expected_model_version="OFFLINE_SCRIPTED_NOT_GEMINI")
    expected = {
        "clean": ("ISSUE", 0, 1, 4),
        "one_missing": ("ISSUE", 1, 2, 6),
        "two_missing": ("ABSTAIN", 1, 2, 6),
    }[name]
    status, verify, single_calls, multi_calls = expected
    assert single.status == multi.status == status, (name, single.status, multi.status)
    assert single.trace["verification_count"] == multi.trace["verification_count"] == verify
    assert single_provider.calls == single_calls and multi_provider.calls == multi_calls
    assert len(single_provider.responses) == len(multi_provider.responses) == 0
    assert single.trace["failure"] is None and multi.trace["failure"] is None
    assert single.risk_level == multi.risk_level == ("MEDIUM" if status == "ISSUE" else None)
    return {"scenario": name, "single_status": single.status,
            "multi_status": multi.status, "verification_count": verify,
            "single_scripted_calls": single_calls, "multi_scripted_calls": multi_calls}


def run_demo():
    return [scenario(name) for name in ("clean", "one_missing", "two_missing")]


def main():
    enforce_no_network()
    print("===== SYNTHETIC OFFLINE END-TO-END DEMONSTRATION =====")
    for outcome in run_demo():
        print(json.dumps(outcome, sort_keys=True))
    print("DEMO_ASSERTIONS: PASS")
    print("PROVIDER: SCRIPTED_FIXTURE_ONLY")
    print("GEMINI_CALLS: ZERO")
    print("NETWORK: DENIED")
    print("FROZEN_EVALUATION_REPRODUCED: NO")


if __name__ == "__main__":
    main()
