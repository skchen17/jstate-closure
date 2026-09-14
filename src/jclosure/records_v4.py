"""Additive schema-v6 record types for protocol v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION_V4 = 6


@dataclass(frozen=True)
class TeacherCompetenceRecord:
    run_id: str
    domain: str
    example_id: str
    family: str
    variant: str
    horizon: int
    parseable: bool
    full_trajectory_correct: bool
    final_answer_correct: bool
    expected_actions: tuple[str, ...]
    generated_actions: tuple[str, ...]
    error: str | None = None
    schema_version: int = SCHEMA_VERSION_V4
    protocol_version: str = "jstate_predictive_protocol_v4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SingleArmTrialRecord:
    run_id: str
    base_trial_id: str
    condition: str
    prompt_id: str
    family: str
    horizon: int
    layer: int
    valid: bool
    metrics: dict[str, Any]
    exclusion_reason: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V4
    protocol_version: str = "single_arm_causal_v4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictiveStateRecord:
    representation: str
    state_dimension: int
    split: str
    semantic_retention: float
    causal_direction_retention: float
    causal_magnitude_retention: float
    future_prediction_cosine: float
    current_reconstruction_cosine: float
    schema_version: int = SCHEMA_VERSION_V4
    protocol_version: str = "predictive_state_v4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
