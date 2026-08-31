"""Independent initial/restoration scope scheduling for protocol v3.2."""

from __future__ import annotations

from jclosure.records_v3_2 import HookEventV32, HookScheduleV32

VALID_INITIAL_SCOPES = {"final"}
VALID_RESTORATION_SCOPES = {"none", "final", "all_non_padding"}


def scope_positions(sequence_length: int, scope: str) -> tuple[int, ...]:
    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
    if scope == "final":
        return (sequence_length - 1,)
    if scope == "all_non_padding":
        return tuple(range(sequence_length))
    if scope == "none":
        return ()
    raise ValueError(f"unknown v3.2 scope: {scope}")


def build_v32_schedule(
    *,
    mode: str,
    initial_layer: int,
    restoration_layers: list[int] | tuple[int, ...],
    sequence_length: int,
    initial_scope: str,
    restoration_scope: str,
) -> HookScheduleV32:
    if mode not in {"single", "persistent_final", "persistent_all"}:
        raise ValueError(f"unknown v3.2 mode: {mode}")
    if initial_scope not in VALID_INITIAL_SCOPES:
        raise ValueError(f"unsupported initial scope: {initial_scope}")
    expected = {
        "single": "none",
        "persistent_final": "final",
        "persistent_all": "all_non_padding",
    }[mode]
    if restoration_scope != expected:
        raise ValueError(
            f"mode {mode} requires restoration_scope={expected}, got {restoration_scope}"
        )
    initial_positions = scope_positions(sequence_length, initial_scope)
    restoration_positions = scope_positions(sequence_length, restoration_scope)
    later = tuple(
        layer
        for layer in sorted({int(value) for value in restoration_layers})
        if layer > int(initial_layer)
    )
    events = tuple(
        HookEventV32(int(initial_layer), position, "initial")
        for position in initial_positions
    ) + tuple(
        HookEventV32(layer, position, "restoration")
        for layer in later
        for position in restoration_positions
    )
    return HookScheduleV32(
        mode=mode,
        initial_layer=int(initial_layer),
        restoration_layers=later,
        initial_scope=initial_scope,
        restoration_scope=restoration_scope,
        initial_positions=initial_positions,
        restoration_positions=restoration_positions,
        events=events,
    )


def schedules_share_initial_perturbation(
    schedules: list[HookScheduleV32] | tuple[HookScheduleV32, ...],
) -> bool:
    return bool(schedules) and len({value.initial_events for value in schedules}) == 1
