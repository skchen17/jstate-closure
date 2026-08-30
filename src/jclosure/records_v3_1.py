"""Additive schema-v4 records for corrective protocol v3.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InterventionEligibility:
    passed: bool
    reasons: tuple[str, ...]
    dense_cosine: float
    top10_overlap: float
    rms_drift: float
    displacement_fraction: float
    natural: bool
    finite: bool
    activation_explosion: bool


@dataclass(frozen=True)
class RestorationEligibility:
    passed: bool
    reasons: tuple[str, ...]
    dense_cosine: float
    top10_overlap: float
    rms_drift: float
    natural: bool
    finite: bool
    activation_explosion: bool
    correction_l2: float
    correction_rms: float
    correction_natural_fraction: float


@dataclass(frozen=True)
class RestorationEvent:
    layer: int
    position: int
    eligibility: RestorationEligibility
    construction_status: str
    construction_failure_reason: str | None = None


@dataclass(frozen=True)
class HookScheduleV31:
    mode: str
    initial_layer: int
    restoration_layers: tuple[int, ...]
    initial_positions: tuple[int, ...]
    final_position: int
    modified_layer_positions: tuple[tuple[int, int], ...]
    protocol_version: str = "corrective_exploratory_protocol_v3_1"
    schema_version: int = 4


@dataclass(frozen=True)
class CausalTrialRecordV31:
    run_id: str
    base_trial_id: str
    paired_trial_id: str
    prompt_id: str
    condition: str
    layer: int
    restoration_layers: tuple[int, ...]
    position_scope: str
    valid: bool
    hook_execution_map: tuple[tuple[int, int], ...]
    metrics: dict[str, Any]
    exclusion_reason: str | None = None
    restoration_events: tuple[RestorationEvent, ...] = ()
    protocol_version: str = "corrective_exploratory_protocol_v3_1"
    schema_version: int = 4


@dataclass(frozen=True)
class CompactMemoryRecord:
    run_id: str
    model_family: str
    state_representation: str
    state_dimension: int
    parameter_count: int
    split: str
    metrics: dict[str, Any]
    memory_dimension: int | None = None
    history_length: int | None = None
    seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "compact_memory_exploratory_v3_1"
    schema_version: int = 4
