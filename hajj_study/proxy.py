"""Reconstructed seven-field reference proxy; this is not the author's original encoder.

Weights and risk thresholds come from the approved proposal/reference paper.
Categorical scores, numeric scaling, and No=0 are declared implementation choices.
Never pass this reference-label formula or its outputs to an inference agent.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

PROXY_FIELDS = (
    "Crowd_Density",
    "Temperature",
    "Fatigue_Level",
    "Stress_Level",
    "Queue_Time_minutes",
    "Emergency_Event",
    "Distance_Between_People_m",
)
WEIGHTS = {
    "Crowd_Density": 0.25,
    "Temperature": 0.20,
    "Fatigue_Level": 0.15,
    "Stress_Level": 0.15,
    "Queue_Time_minutes": 0.10,
    "Emergency_Event": 0.10,
    "Distance_Between_People_m": 0.05,
}
LOW_UPPER = 1.8
MEDIUM_UPPER = 2.4
CATEGORY_SCORES = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
EMERGENCY_SCORES = {"No": 0.0, "Yes": 3.0}


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid numeric value {value!r}") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name}: {parsed!r} outside [{minimum}, {maximum}]")
    return parsed


def encoded_fields(row: Mapping[str, object]) -> dict[str, float]:
    missing = [field for field in PROXY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing source fields: {missing}")
    for name in ("Crowd_Density", "Fatigue_Level", "Stress_Level"):
        if row[name] not in CATEGORY_SCORES:
            raise ValueError(f"{name}: unknown category {row[name]!r}")
    if row["Emergency_Event"] not in EMERGENCY_SCORES:
        raise ValueError(f"Emergency_Event: unknown category {row['Emergency_Event']!r}")
    temperature = _bounded_float(row["Temperature"], "Temperature", 30.0, 45.0)
    queue = _bounded_float(row["Queue_Time_minutes"], "Queue_Time_minutes", 0.0, 60.0)
    distance = _bounded_float(row["Distance_Between_People_m"], "Distance_Between_People_m", 0.5, 2.5)
    return {
        "Crowd_Density": CATEGORY_SCORES[row["Crowd_Density"]],
        "Temperature": 1.0 + 2.0 * (temperature - 30.0) / 15.0,
        "Fatigue_Level": CATEGORY_SCORES[row["Fatigue_Level"]],
        "Stress_Level": CATEGORY_SCORES[row["Stress_Level"]],
        "Queue_Time_minutes": 1.0 + 2.0 * queue / 60.0,
        "Emergency_Event": EMERGENCY_SCORES[row["Emergency_Event"]],
        "Distance_Between_People_m": 3.0 - (distance - 0.5),
    }


def risk_score(row: Mapping[str, object]) -> float:
    values = encoded_fields(row)
    return sum(WEIGHTS[name] * values[name] for name in PROXY_FIELDS)


def risk_label_for_score(score: float) -> str:
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Risk score must be finite")
    if score < LOW_UPPER:
        return "LOW"
    if score <= MEDIUM_UPPER:
        return "MEDIUM"
    return "HIGH"


def risk_label(row: Mapping[str, object]) -> str:
    return risk_label_for_score(risk_score(row))
