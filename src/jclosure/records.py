"""Stable record schemas shared by experiment runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jclosure.provenance import SCHEMA_VERSION


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    prompt_id: str
    task_family: str
    layer: int
    position: int
    intervention: str
    valid: bool
    metrics: dict[str, float | int | bool | None]
    seed: int
    template_id: str | None = None
    source: str | None = None
    strength: float | None = None
    donor_id: str | None = None
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "phase0_protocol_v1"
    dictionary_size: int | None = None
    dictionary_hash: str | None = None
    position_scope: str = "explicit"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrialRecord:
        """Load v1 or v2 records, supplying absent v2 provenance fields."""

        accepted = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in accepted})


@dataclass(frozen=True)
class PassKSummary:
    family: str
    method: str
    pass1: float
    pass5: float
    pass10: float
    strict_all_layers_hit10: float
    best_layer: int | None
    item_count: int
    concept_count: int
    protocol_version: str = "phase0_protocol_v2"
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class LayerCalibrationRecord:
    layer: int
    multihop_hit10: float
    order_ops_hit10: float
    rank_advantage_ci_lower: float | None
    positive_control_ci_lower: float | None
    clamp_valid_rate: float
    numerical_checks_passed: bool
    eligible: bool
    reasons: tuple[str, ...] = ()
    protocol_version: str = "phase0_protocol_v2"
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class TokenMacroState:
    trajectory_id: str
    token_step: int
    macro_kind: str
    eligible_layers: tuple[int, ...]
    dense_scores: tuple[float, ...]
    layer_dispersion: float
    semantic_action: str | None = None
    answer: str | None = None
    memory: tuple[float, ...] = ()
    protocol_version: str = "phase0_protocol_v2"
    schema_version: int = SCHEMA_VERSION
