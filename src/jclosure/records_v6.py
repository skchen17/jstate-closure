"""Additive machine-record types for protocol v6.

The v6 records deliberately do not alter the v1--v5 loaders or schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION_V6 = 8
PROTOCOL_V6 = "peripheral_foundations_protocol_v6"


@dataclass(frozen=True)
class TeacherCausalEndpointRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    horizon: int
    split: str
    valid: bool
    delta_j_l2_f32: float
    delta_j_l2_f64_projection: float
    delta_j_f32_f64_cosine: float | None
    delta_hidden_l2: float
    delta_remainder_l2: float
    output_logit_delta_l2: float
    semantic_delta: bool
    output_sign: int
    metadata: dict[str, Any] = field(default_factory=dict)
    exclusion_reason: str | None = None
    schema_version: int = SCHEMA_VERSION_V6
    protocol_version: str = PROTOCOL_V6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestorationNullRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    horizon: int
    condition: str
    valid: bool
    metrics: dict[str, Any]
    restoration_events: list[dict[str, Any]] = field(default_factory=list)
    hook_execution_map: list[list[Any]] = field(default_factory=list)
    exclusion_reason: str | None = None
    schema_version: int = SCHEMA_VERSION_V6
    protocol_version: str = PROTOCOL_V6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeripheralCeilingRecord:
    run_id: str
    model: str
    split: str
    seed: int
    example_id: str
    family: str
    next_j_cosine: float
    action_correct: bool
    action_cross_entropy: float
    causal_projection_rmse: float
    top_causal_dimension_rmse: float
    parameter_count: int
    schema_version: int = SCHEMA_VERSION_V6
    protocol_version: str = PROTOCOL_V6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
