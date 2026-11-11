"""Frozen 17-feature allowlist for the shared ML model (not for agents)."""
from __future__ import annotations

ALLOWED_MODEL_FEATURES = (
    "Location_Lat",
    "Location_Long",
    "Movement_Speed",
    "Activity_Type",
    "Weather_Conditions",
    "Sound_Level_dB",
    "AR_System_Interaction",
    "Health_Condition",
    "Age_Group",
    "Transport_Mode",
    "Waiting_Time_for_Transport",
    "Security_Checkpoint_Wait_Time",
    "Incident_Type",
    "Crowd_Morale",
    "Interaction_Frequency",
    "Event_Type",
    "Time_Spent_at_Location_minutes",
)
CATEGORICAL_FEATURES = (
    "Activity_Type",
    "Weather_Conditions",
    "AR_System_Interaction",
    "Health_Condition",
    "Age_Group",
    "Transport_Mode",
    "Incident_Type",
    "Crowd_Morale",
    "Event_Type",
)
LABEL_CONSTRUCTION_FIELDS = (
    "Crowd_Density",
    "Temperature",
    "Fatigue_Level",
    "Stress_Level",
    "Queue_Time_minutes",
    "Emergency_Event",
    "Distance_Between_People_m",
)
POST_EVENT_EXCLUSIONS = (
    "Satisfaction_Rating",
    "Perceived_Safety_Rating",
    "Pilgrim_Experience",
    "AR_Navigation_Success",
)
OTHER_EXCLUSIONS = ("Timestamp", "Nationality")
EXCLUDED_FROM_MODEL = frozenset(LABEL_CONSTRUCTION_FIELDS + POST_EVENT_EXCLUSIONS + OTHER_EXCLUSIONS)


def validate_features(all_columns) -> None:
    columns = list(all_columns)
    selected = set(ALLOWED_MODEL_FEATURES)
    if len(ALLOWED_MODEL_FEATURES) != 17 or len(selected) != 17:
        raise ValueError("ML feature allowlist must contain exactly 17 unique names")
    if selected & EXCLUDED_FROM_MODEL:
        raise ValueError(f"Forbidden ML predictors: {sorted(selected & EXCLUDED_FROM_MODEL)}")
    if set(CATEGORICAL_FEATURES) - selected:
        raise ValueError("Categorical list is not a subset of the allowlist")
    if set(columns) != selected | EXCLUDED_FROM_MODEL:
        missing = (selected | EXCLUDED_FROM_MODEL) - set(columns)
        extra = set(columns) - (selected | EXCLUDED_FROM_MODEL)
        raise ValueError(f"Raw CSV features changed: missing={sorted(missing)}, extra={sorted(extra)}")
