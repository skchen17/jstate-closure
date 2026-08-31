"""Additive schema-v5 records for causal protocol v3.2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HookEventV32:
    layer: int
    position: int
    operation_type: str


@dataclass(frozen=True)
class HookScheduleV32:
    mode: str
    initial_layer: int
    restoration_layers: tuple[int, ...]
    initial_scope: str
    restoration_scope: str
    initial_positions: tuple[int, ...]
    restoration_positions: tuple[int, ...]
    events: tuple[HookEventV32, ...]
    protocol_version: str = "corrective_causal_protocol_v3_2"
    schema_version: int = 5

    @property
    def initial_events(self) -> tuple[HookEventV32, ...]:
        return tuple(event for event in self.events if event.operation_type == "initial")

    @property
    def restoration_events(self) -> tuple[HookEventV32, ...]:
        return tuple(
            event for event in self.events if event.operation_type == "restoration"
        )


@dataclass(frozen=True)
class ConditionalEligibilityV32:
    applicable: int
    successes: int
    rate: float
    ci_lower: float
    ci_upper: float
    eligible: bool

