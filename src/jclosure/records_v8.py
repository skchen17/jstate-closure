"""Additive machine-readable records for structured persistent-state protocol v8."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION_V8 = 11
PROTOCOL_V8 = "structured_persistent_state_protocol_v8"


@dataclass(frozen=True)
class PersistentPairRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    split: str
    donor_id: str
    valid: bool
    intervention_quality: dict[str, Any]
    artifact_shard: str | None = None
    artifact_row: int | None = None
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V8
    protocol_version: str = PROTOCOL_V8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawStateEffectRecord:
    run_id: str
    base_trial_id: str
    prompt_id: str
    family: str
    split: str
    condition: str
    atom_mask: int | None
    valid: bool
    metrics: dict[str, Any]
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V8
    protocol_version: str = PROTOCOL_V8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompressionSweepRecord:
    run_id: str
    method: str
    regime: str
    dimension: int
    family: str
    split: str
    count: int
    predictive_gap_closed: float
    conditional_residual_gain: float
    causal_gap_closed: float | None
    direction_cosine: float | None
    magnitude_ratio: float | None
    semantic_agreement: float
    output_sign_agreement: float | None
    authorized: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION_V8
    protocol_version: str = PROTOCOL_V8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
