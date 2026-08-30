"""Eligibility and hook scheduling for corrective protocol v3.1.

Initial state construction and later restoration deliberately have different
eligibility contracts.  The latter never requires a large displacement.
"""

from __future__ import annotations

import math

import torch

from jclosure.clamp_v3 import V3ClampValidation
from jclosure.records_v3_1 import (
    HookScheduleV31,
    InterventionEligibility,
    RestorationEligibility,
)

DISPLACEMENT_REASONS = {
    "displacement_below_sensitivity",
    "displacement_below_formal",
}


def validate_intervention_eligibility(
    validation: V3ClampValidation,
    *,
    finite: bool,
    activation_explosion: bool,
    construction_status: str,
    require_position_displacement: bool = True,
) -> InterventionEligibility:
    reasons = set(validation.failure_reasons)
    if not require_position_displacement:
        reasons -= DISPLACEMENT_REASONS
    if construction_status == "FAILED":
        reasons.add("construction_failed")
    if not finite:
        reasons.add("nan_or_inf")
    if activation_explosion:
        reasons.add("activation_explosion")
    passed = not reasons
    return InterventionEligibility(
        passed=passed,
        reasons=tuple(sorted(reasons)),
        dense_cosine=validation.dense_cosine,
        top10_overlap=validation.top10_overlap,
        rms_drift=validation.rms_drift,
        displacement_fraction=validation.displacement_fraction,
        natural=validation.natural,
        finite=finite,
        activation_explosion=activation_explosion,
    )


def validate_restoration_eligibility(
    validation: V3ClampValidation,
    *,
    correction: torch.Tensor,
    natural_scale: float,
    finite: bool,
    activation_explosion: bool,
    construction_status: str,
) -> RestorationEligibility:
    reasons = set(validation.failure_reasons) - DISPLACEMENT_REASONS
    if construction_status == "FAILED":
        reasons.add("construction_failed")
    if not finite:
        reasons.add("nan_or_inf")
    if activation_explosion:
        reasons.add("activation_explosion")
    correction_float = correction.float()
    correction_l2 = float(torch.linalg.vector_norm(correction_float).item())
    correction_rms = float(torch.sqrt(torch.mean(correction_float**2)).item())
    correction_fraction = correction_l2 / max(float(natural_scale), 1e-20)
    return RestorationEligibility(
        passed=not reasons,
        reasons=tuple(sorted(reasons)),
        dense_cosine=validation.dense_cosine,
        top10_overlap=validation.top10_overlap,
        rms_drift=validation.rms_drift,
        natural=validation.natural,
        finite=finite,
        activation_explosion=activation_explosion,
        correction_l2=correction_l2,
        correction_rms=correction_rms,
        correction_natural_fraction=correction_fraction,
    )


def build_v31_schedule(
    *,
    mode: str,
    initial_layer: int,
    restoration_layers: list[int] | tuple[int, ...],
    initial_positions: list[int] | tuple[int, ...],
    final_position: int,
) -> HookScheduleV31:
    if mode not in {"single", "persistent_final", "persistent_all"}:
        raise ValueError(f"unknown v3.1 mode: {mode}")
    positions = tuple(sorted({int(value) for value in initial_positions}))
    if not positions:
        raise ValueError("initial positions must not be empty")
    later = tuple(
        value
        for value in sorted({int(value) for value in restoration_layers})
        if value > int(initial_layer)
    )
    if mode == "single":
        modified = tuple((int(initial_layer), position) for position in positions)
    elif mode == "persistent_final":
        modified = (
            *((int(initial_layer), position) for position in positions),
            *((layer, int(final_position)) for layer in later),
        )
    else:
        modified = (
            *((int(initial_layer), position) for position in positions),
            *((layer, position) for layer in later for position in positions),
        )
    if any(not math.isfinite(float(value)) for pair in modified for value in pair):
        raise ValueError("hook schedule contains non-finite coordinates")
    return HookScheduleV31(
        mode=mode,
        initial_layer=int(initial_layer),
        restoration_layers=later,
        initial_positions=positions,
        final_position=int(final_position),
        modified_layer_positions=modified,
    )
