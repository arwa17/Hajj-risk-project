"""Deterministic, source-blind (to agents) corruption operators for G3.

IMPORTANT: all metadata returned by corrupt_case() is for private scorer records;
never serialize it into an agent message.  Golden labels stay based on the
unmodified clean source row.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product

from hajj_study.proxy import PROXY_FIELDS, risk_label

DOMAINS = {
    "crowd": ("Crowd_Density", "Queue_Time_minutes", "Distance_Between_People_m"),
    "health": ("Fatigue_Level", "Stress_Level", "Emergency_Event"),
    "environment": ("Temperature",),
}
FIELD_DOMAIN = {field: domain for domain, fields in DOMAINS.items() for field in fields}
CATEGORICAL = {"Low": "High", "Medium": "Low", "High": "Medium"}
CONFLICT_CATEGORICAL = {"Low": "Medium", "Medium": "High", "High": "Low"}
NUMERIC_PARAMS = {
    "Temperature": (30.0, 45.0, 37.5, 6.0),
    "Queue_Time_minutes": (0.0, 60.0, 30.0, 20.0),
    "Distance_Between_People_m": (0.5, 2.5, 1.5, 0.6),
}
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


def source_evidence(clean):
    """Extract only seven permitted observation fields, preserving source values."""
    return {field: float(clean[field]) if field in NUMERIC_PARAMS else clean[field]
            for field in PROXY_FIELDS}


def _numeric_alternative(field, value):
    minimum, maximum, midpoint, delta = NUMERIC_PARAMS[field]
    original = float(value)
    proposal = original + (delta if original <= midpoint else -delta)
    proposal = max(minimum, min(maximum, proposal))
    if proposal == original:
        raise AssertionError(f"No numeric change for {field}")
    return round(proposal, 10)


def alternative(field, value, conflicting=False):
    if field in NUMERIC_PARAMS:
        return _numeric_alternative(field, value)
    mapping = CONFLICT_CATEGORICAL if conflicting else CATEGORICAL
    if field not in ("Crowd_Density", "Fatigue_Level", "Stress_Level") or value not in mapping:
        raise ValueError(f"No fixed perturbation mapping for {field}={value!r}")
    return mapping[value]


def affected_fields(kind, label, multidomain=False):
    """Fixed patterns chosen before selecting rows, not from model errors."""
    patterns = {
        ("missing", "LOW"): ("Crowd_Density",),
        ("missing", "MEDIUM"): ("Fatigue_Level",),
        ("missing", "HIGH"): ("Temperature", "Queue_Time_minutes"),
        ("noisy", "LOW"): ("Temperature",),
        ("noisy", "MEDIUM"): ("Queue_Time_minutes", "Stress_Level"),
        ("noisy", "HIGH"): ("Fatigue_Level",),
        ("conflicting", "LOW"): ("Stress_Level", "Crowd_Density"),
        ("conflicting", "MEDIUM"): ("Temperature",),
        ("conflicting", "HIGH"): ("Distance_Between_People_m",),
    }
    fields = patterns[(kind, label)]
    if multidomain:
        if len({FIELD_DOMAIN[field] for field in fields}) != 2:
            raise ValueError(f"No predeclared two-domain pattern: {kind}/{label}")
        return fields
    return fields[:1]


def corrupt_case(clean, kind, label, multidomain=False, clean_first_by_field=None):
    """Return observed evidence and private metadata. Conflict order is controlled separately."""
    if risk_label(clean) != label:
        raise ValueError("Source label does not match clean reference")
    if kind not in ("missing", "noisy", "conflicting"):
        raise ValueError(f"Unexpected corruption kind: {kind}")
    fields = affected_fields(kind, label, multidomain)
    observed = deepcopy(source_evidence(clean))
    clean_first_by_field = clean_first_by_field or {}
    alternatives = {}
    for field in fields:
        if kind == "missing":
            observed[field] = None
        elif kind == "noisy":
            new_value = alternative(field, observed[field])
            observed[field] = new_value
            alternatives[field] = new_value
        else:
            original = observed[field]
            other = alternative(field, original, conflicting=True)
            alternatives[field] = other
            clean_first = bool(clean_first_by_field.get(field, True))
            pair = (original, other) if clean_first else (other, original)
            observed[field] = [
                {"source": "reading_a", "value": pair[0]},
                {"source": "reading_b", "value": pair[1]},
            ]
    metadata = {
        "kind": kind,
        "altered_fields": list(fields),
        "altered_domains": sorted({FIELD_DOMAIN[f] for f in fields}),
        "alternative_values_for_private_analysis": alternatives,
        "clean_first_by_field": {f: bool(clean_first_by_field.get(f, True)) for f in fields}
        if kind == "conflicting" else {},
    }
    assert observed != source_evidence(clean)
    return observed, metadata


def stress_influence(clean, kind, mutation_metadata):
    """Check whether the PREDECLARED stress operator can change the reconstructed
    proxy class, not the golden label, which ALWAYS remains from clean source.

    Missing: at least one plausible extreme filling would alter the class.
    Noisy/conflicting: the alternative reading changes the proxy class.
    This is a purposeful stress-test filter and not a prevalence estimate.
    """
    clean_proxy = risk_label(clean)
    fields = mutation_metadata["altered_fields"]
    if kind == "missing":
        ranges = []
        for field in fields:
            if field in NUMERIC_PARAMS:
                low, high, *_ = NUMERIC_PARAMS[field]
                ranges.append((low, high))
            elif field == "Emergency_Event":
                ranges.append(("No", "Yes"))
            else:
                ranges.append(("Low", "High"))
        for values in product(*ranges):
            candidate = dict(clean)
            candidate.update(dict(zip(fields, values)))
            if risk_label(candidate) != clean_proxy:
                return True
        return False
    changed = dict(clean)
    changed.update(mutation_metadata["alternative_values_for_private_analysis"])
    return risk_label(changed) != clean_proxy
