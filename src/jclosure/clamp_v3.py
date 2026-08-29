"""Operational state construction and validation for exploratory protocol v3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from jclosure.clamp import one_shot_clamp
from jclosure.geometry import DenseJMap, DenseNullProjector, SparseStateEquality
from jclosure.interventions import resolve_position_scope
from jclosure.jstate import JStateEncoder
from jclosure.metrics import rms_drift, topk_overlap
from jclosure.records import ClampSchedule


@dataclass(frozen=True)
class V3ClampThresholds:
    dense_cosine: float = 0.995
    dense_top10_overlap: float = 0.8
    sparse_support_f1: float = 0.8
    sparse_weighted_jaccard: float = 0.95
    sparse_coefficient_cosine: float = 0.995
    sparse_reconstruction_cosine: float = 0.995
    rms_drift: float = 0.02
    formal_displacement: float = 0.20
    sensitivity_displacement: float = 0.05


@dataclass(frozen=True)
class V3ClampValidation:
    state_definition: str
    valid: bool
    formal_valid: bool
    small_perturbation_valid: bool
    failure_reasons: tuple[str, ...]
    dense_cosine: float
    dense_profile_l2: float
    top10_overlap: float
    rms_drift: float
    displacement_fraction: float
    natural: bool
    sparse_equality: SparseStateEquality | None = None


DEFAULT_V3_CLAMP_THRESHOLDS = V3ClampThresholds()


def validate_v3_clamp(
    clean: torch.Tensor,
    candidate: torch.Tensor,
    *,
    layer: int,
    state_definition: str,
    encoder: JStateEncoder,
    dense_map: DenseJMap,
    natural_scale: float,
    natural: bool,
    thresholds: V3ClampThresholds = DEFAULT_V3_CLAMP_THRESHOLDS,
) -> V3ClampValidation:
    clean_dense = dense_map.dense_state(clean.float(), layer)
    candidate_dense = dense_map.dense_state(candidate.float(), layer)
    clean_raw = dense_map.raw_scores(clean.float(), layer)
    candidate_raw = dense_map.raw_scores(candidate.float(), layer)
    dense_cosine = float(
        F.cosine_similarity(clean_dense[None], candidate_dense[None]).item()
    )
    dense_l2 = float(torch.linalg.vector_norm(clean_dense - candidate_dense).item())
    overlap = topk_overlap(clean_raw, candidate_raw, 10)
    drift = rms_drift(clean, candidate)
    displacement = float(
        torch.linalg.vector_norm((candidate - clean).float()).item()
        / max(float(natural_scale), 1e-20)
    )
    sparse: SparseStateEquality | None = None
    failures: list[str] = []
    if state_definition == "V3-Dense":
        if dense_cosine < thresholds.dense_cosine:
            failures.append("dense_cosine")
        if overlap < thresholds.dense_top10_overlap:
            failures.append("dense_top10_overlap")
    elif state_definition == "V3-Sparse":
        sparse = SparseStateEquality.compare(
            encoder.decompose(clean, layer),
            encoder.decompose(candidate, layer),
            support_f1_threshold=thresholds.sparse_support_f1,
            weighted_jaccard_threshold=thresholds.sparse_weighted_jaccard,
            coefficient_cosine_threshold=thresholds.sparse_coefficient_cosine,
            reconstruction_cosine_threshold=thresholds.sparse_reconstruction_cosine,
        )
        failures.extend(sparse.failure_reasons)
    else:
        raise ValueError(f"unknown v3 state definition: {state_definition}")
    if drift > thresholds.rms_drift:
        failures.append("rms_drift")
    if not natural:
        failures.append("naturality")
    equality_valid = not failures
    formal = equality_valid and displacement >= thresholds.formal_displacement
    sensitivity = (
        equality_valid
        and thresholds.sensitivity_displacement <= displacement
        < thresholds.formal_displacement
    )
    if displacement < thresholds.sensitivity_displacement:
        failures.append("displacement_below_sensitivity")
    elif displacement < thresholds.formal_displacement:
        failures.append("displacement_below_formal")
    return V3ClampValidation(
        state_definition=state_definition,
        valid=formal,
        formal_valid=formal,
        small_perturbation_valid=sensitivity,
        failure_reasons=tuple(failures),
        dense_cosine=dense_cosine,
        dense_profile_l2=dense_l2,
        top10_overlap=overlap,
        rms_drift=drift,
        displacement_fraction=displacement,
        natural=natural,
        sparse_equality=sparse,
    )


def construct_sparse_candidate(
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    layer: int,
    encoder: JStateEncoder,
) -> torch.Tensor:
    return one_shot_clamp(
        clean,
        perturbed,
        layer=layer,
        encoder=encoder,
    ).activation


def construct_dense_candidate(
    clean: torch.Tensor,
    donor_difference: torch.Tensor,
    *,
    layer: int,
    dense_map: DenseJMap,
    natural_scale: float,
    displacement_fraction: float,
    relative_tolerance: float,
    optimized: bool,
    naturality: Callable[[torch.Tensor], bool] | None = None,
    thresholds: V3ClampThresholds = DEFAULT_V3_CLAMP_THRESHOLDS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    projector = DenseNullProjector(dense_map, layer)
    direction, basis, values = projector.donor_projection(
        clean,
        donor_difference,
        relative_tolerance=relative_tolerance,
        sphere_tangent=True,
    )
    if basis.shape[1] == 0 or float(torch.linalg.vector_norm(direction)) <= 1e-20:
        return clean.clone(), {
            "status": "FAILED",
            "failure_reason": "zero_dimensional_intersection",
            "basis_dimension": int(basis.shape[1]),
            "singular_values": values.detach().cpu().tolist(),
        }
    target = float(natural_scale) * float(displacement_fraction)
    if optimized:
        result = projector.optimize_hard_constraints(
            clean,
            donor_difference,
            basis,
            target_displacement=target,
            dense_cosine_threshold=thresholds.dense_cosine,
            top10_overlap_threshold=thresholds.dense_top10_overlap,
            rms_drift_threshold=thresholds.rms_drift,
            naturality=naturality,
        )
        return result.activation, {
            "status": result.status,
            "failure_reason": result.failure_reason,
            "iterations": result.iterations,
            "basis_dimension": int(basis.shape[1]),
            "singular_values": values.detach().cpu().tolist(),
        }
    try:
        tangent_step = projector.tangent_step_for_chord(clean, target)
    except ValueError:
        return clean.clone(), {
            "status": "FAILED",
            "failure_reason": "target_outside_sphere_retraction",
            "basis_dimension": int(basis.shape[1]),
            "singular_values": values.detach().cpu().tolist(),
        }
    scaled = direction * (
        tangent_step / torch.linalg.vector_norm(direction.float()).clamp_min(1e-20)
    )
    delta = projector.retract_to_sphere(clean, scaled)
    return clean + delta, {
        "status": "CONSTRUCTED",
        "failure_reason": None,
        "iterations": 0,
        "basis_dimension": int(basis.shape[1]),
        "singular_values": values.detach().cpu().tolist(),
    }


def project_dense_candidate(
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    layer: int,
    dense_map: DenseJMap,
    relative_tolerance: float,
    optimized: bool,
    naturality: Callable[[torch.Tensor], bool] | None = None,
    thresholds: V3ClampThresholds = DEFAULT_V3_CLAMP_THRESHOLDS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project the observed displacement into the local tangent-null subspace."""

    projector = DenseNullProjector(dense_map, layer)
    observed = perturbed.float() - clean.float()
    direction, basis, values = projector.donor_projection(
        clean,
        observed,
        relative_tolerance=relative_tolerance,
        sphere_tangent=True,
    )
    if basis.shape[1] == 0:
        return clean.clone(), {
            "status": "FAILED",
            "failure_reason": "zero_dimensional_intersection",
            "basis_dimension": 0,
            "singular_values": values.detach().cpu().tolist(),
        }
    if float(torch.linalg.vector_norm(direction.float())) <= 1e-20:
        return clean.clone(), {
            "status": "NO_CHANGE",
            "failure_reason": None,
            "basis_dimension": int(basis.shape[1]),
            "singular_values": values.detach().cpu().tolist(),
        }
    local_delta = projector.retract_to_sphere(clean, direction)
    if optimized:
        target = float(torch.linalg.vector_norm(local_delta.float()).item())
        result = projector.optimize_hard_constraints(
            clean,
            observed,
            basis,
            target_displacement=target,
            dense_cosine_threshold=thresholds.dense_cosine,
            top10_overlap_threshold=thresholds.dense_top10_overlap,
            rms_drift_threshold=thresholds.rms_drift,
            naturality=naturality,
        )
        return result.activation, {
            "status": result.status,
            "failure_reason": result.failure_reason,
            "iterations": result.iterations,
            "basis_dimension": int(basis.shape[1]),
            "singular_values": values.detach().cpu().tolist(),
        }
    return clean + local_delta, {
        "status": "PROJECTED",
        "failure_reason": None,
        "iterations": 0,
        "basis_dimension": int(basis.shape[1]),
        "singular_values": values.detach().cpu().tolist(),
    }


def build_clamp_schedule(
    *,
    mode: str,
    initial_layer: int,
    future_layers: list[int] | tuple[int, ...],
    position_scope: str,
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    explicit_positions: tuple[int, ...] | None,
    reasoning_span: tuple[int, int] | None,
    state_definition: str,
    dictionary_size: int,
) -> ClampSchedule:
    initial_positions = resolve_position_scope(
        sequence_length,
        scope=position_scope,  # type: ignore[arg-type]
        positions=explicit_positions,
        attention_mask=attention_mask,
        reasoning_span=reasoning_span,
    )
    return ClampSchedule.build(
        protocol_version="exploratory_protocol_v3",
        mode=mode,
        initial_layer=initial_layer,
        future_layers=future_layers,
        position_scope=position_scope,
        initial_positions=initial_positions,
        final_position=sequence_length - 1,
        state_definition=state_definition,
        dictionary_size=dictionary_size,
    )


def scheduled_sparse_clamp_transforms(
    schedule: ClampSchedule,
    clean_by_layer: dict[int, torch.Tensor],
    *,
    encoder: JStateEncoder,
    capture: dict[tuple[int, int], V3ClampValidation] | None = None,
) -> dict[int, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Build sparse clamps from the exact recorded schedule."""

    positions_by_layer: dict[int, list[int]] = {}
    for layer, position in schedule.modified_layer_positions:
        positions_by_layer.setdefault(layer, []).append(position)
    transforms: dict[int, Callable[[torch.Tensor, int], torch.Tensor]] = {}
    for layer, positions in positions_by_layer.items():
        clean_sequence = clean_by_layer[layer].detach().clone()

        def transform(
            activation: torch.Tensor,
            current_layer: int,
            *,
            clean_reference: torch.Tensor = clean_sequence,
            selected_positions: tuple[int, ...] = tuple(positions),
        ) -> torch.Tensor:
            output = activation.clone()
            for position in selected_positions:
                current = activation[0, position] if activation.ndim == 3 else activation[position]
                clean = clean_reference[position].to(current.device, current.dtype)
                candidate = construct_sparse_candidate(
                    clean, current, layer=current_layer, encoder=encoder
                )
                if activation.ndim == 3:
                    output[:, position, :] = candidate
                else:
                    output[position, :] = candidate
            return output

        transforms[layer] = transform
    return transforms
