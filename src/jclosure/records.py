"""Stable record schemas shared by experiment runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    # Keep the historical constructor default for v1/v2 callers. Exploratory-v3
    # runners pass schema_version=3 explicitly and use the dedicated JSON schema.
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrialRecord:
        """Load v1, v2, or v3 records, supplying absent provenance fields."""

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
    schema_version: int = 2


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
    schema_version: int = 2


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
    schema_version: int = 2


@dataclass(frozen=True)
class ClampSchedule:
    """Auditable declaration of intended and actually modified hook locations."""

    protocol_version: str
    mode: str
    initial_layer: int
    selected_layers: tuple[int, ...]
    position_scope: str
    resolved_positions: tuple[int, ...]
    modified_layer_positions: tuple[tuple[int, int], ...]
    state_definition: str
    dictionary_size: int
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.mode not in {"single", "persistent_final", "persistent_all"}:
            raise ValueError(f"unknown clamp mode: {self.mode}")
        if self.position_scope not in {
            "final",
            "explicit",
            "all_non_padding",
            "reasoning_span",
        }:
            raise ValueError(f"unknown position scope: {self.position_scope}")
        if self.state_definition not in {"V3-Dense", "V3-Sparse"}:
            raise ValueError(f"unknown v3 state definition: {self.state_definition}")

    @classmethod
    def build(
        cls,
        *,
        protocol_version: str,
        mode: str,
        initial_layer: int,
        future_layers: tuple[int, ...] | list[int],
        position_scope: str,
        initial_positions: tuple[int, ...] | list[int],
        final_position: int,
        state_definition: str,
        dictionary_size: int,
    ) -> ClampSchedule:
        initial_positions = tuple(int(value) for value in initial_positions)
        future_layers = tuple(sorted(set(int(value) for value in future_layers)))
        selected_layers: tuple[int, ...]
        modified: tuple[tuple[int, int], ...]
        if mode == "single":
            selected_layers = (int(initial_layer),)
            modified = tuple((int(initial_layer), value) for value in initial_positions)
        elif mode == "persistent_final":
            selected_layers = (int(initial_layer), *future_layers)
            modified = (
                *((int(initial_layer), value) for value in initial_positions),
                *((layer, int(final_position)) for layer in future_layers),
            )
        elif mode == "persistent_all":
            selected_layers = (int(initial_layer), *future_layers)
            modified = tuple(
                (layer, position)
                for layer in selected_layers
                for position in initial_positions
            )
        else:
            raise ValueError(f"unknown clamp mode: {mode}")
        return cls(
            protocol_version=protocol_version,
            mode=mode,
            initial_layer=int(initial_layer),
            selected_layers=selected_layers,
            position_scope=position_scope,
            resolved_positions=initial_positions,
            modified_layer_positions=modified,
            state_definition=state_definition,
            dictionary_size=int(dictionary_size),
        )
