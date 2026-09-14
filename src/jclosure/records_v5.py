"""Additive machine-record types for the peripheral-state v5 protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION_V5 = 7


@dataclass(frozen=True)
class PeripheralReferenceRecord:
    run_id: str
    architecture: str
    input_state: str
    split: str
    seed: int
    next_j_cosine: float
    next_j_distance: float
    semantic_accuracy: float
    parameter_count: int
    family: str = "pooled"
    schema_version: int = SCHEMA_VERSION_V5
    protocol_version: str = "jstate_peripheral_protocol_v5"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeripheralStateRecord:
    run_id: str
    encoder_family: str
    state_dimension: int
    split: str
    next_j_cosine: float
    semantic_accuracy: float
    gap_closed: float | None
    conditional_residual_gain: float | None
    causal_direction_cosine: float | None
    causal_magnitude_ratio: float | None
    output_sign_agreement: float | None
    authorized: bool
    parameter_count: int
    family_metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V5
    protocol_version: str = "jstate_peripheral_protocol_v5"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class H2ReplicationRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    horizon: int
    condition: str
    valid: bool
    metrics: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)
    exclusion_reason: str | None = None
    schema_version: int = SCHEMA_VERSION_V5
    protocol_version: str = "h2_family_replication_v5"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
