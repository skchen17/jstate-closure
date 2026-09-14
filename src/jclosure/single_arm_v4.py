"""State construction and paired metrics for the protocol-v4 single-arm test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jclosure.clamp_v3 import (
    DenseCandidateGeometry,
    V3ClampThresholds,
    construct_dense_candidate,
    validate_v3_clamp,
)
from jclosure.geometry import DenseJMap, DenseNullProjector
from jclosure.metrics import jensen_shannon_from_logits


@dataclass(frozen=True)
class SharedLowSingularBasis:
    layer: int
    relative_tolerance: float
    singular_values: tuple[float, ...]
    rank: int
    basis: torch.Tensor

    @property
    def null_dimension(self) -> int:
        return int(self.basis.shape[1])


def shared_low_singular_basis(
    dense_map: DenseJMap,
    layer: int,
    *,
    relative_tolerance: float,
    device: torch.device | str,
) -> SharedLowSingularBasis:
    """Compute the fixed low-singular subspace of the centered dense map once."""

    matrix = dense_map.centered_map(layer, device=device, dtype=torch.float32)
    _, values, vh = torch.linalg.svd(matrix, full_matrices=False)
    maximum = values[0].clamp_min(1e-30)
    rank = int(torch.count_nonzero(values > relative_tolerance * maximum).item())
    basis = vh[rank:].T.contiguous()
    return SharedLowSingularBasis(
        layer=int(layer),
        relative_tolerance=float(relative_tolerance),
        singular_values=tuple(float(value) for value in values.detach().cpu()),
        rank=rank,
        basis=basis,
    )


def construct_from_shared_basis(
    clean: torch.Tensor,
    donor: torch.Tensor,
    *,
    shared: SharedLowSingularBasis,
    dense_map: DenseJMap,
    encoder: Any,
    natural_scale: float,
    displacement_fraction: float,
    thresholds: V3ClampThresholds,
    naturality: Any | None,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    """Construct and freshly validate a non-radial measured-J-preserving state."""

    projector = DenseNullProjector(dense_map, shared.layer)
    tangent = projector.tangent_intersection(shared.basis, clean.float())
    difference = donor.float() - clean.float()
    direction = projector.project(difference, tangent)
    prepared = DenseCandidateGeometry(
        direction=direction,
        basis=tangent,
        singular_values=torch.as_tensor(
            shared.singular_values, device=clean.device, dtype=torch.float32
        ),
    )
    naturality_callback = None
    if naturality is not None:
        naturality_callback = lambda value: naturality.score(  # noqa: E731
            value.detach().float().cpu().numpy()
        ).natural
    candidate, construction = construct_dense_candidate(
        clean.float(),
        difference,
        layer=shared.layer,
        dense_map=dense_map,
        natural_scale=float(natural_scale),
        displacement_fraction=float(displacement_fraction),
        relative_tolerance=shared.relative_tolerance,
        optimized=True,
        naturality=naturality_callback,
        thresholds=thresholds,
        prepared=prepared,
    )
    natural = (
        True
        if naturality is None
        else bool(naturality.score(candidate.detach().cpu().numpy()).natural)
    )
    validation = validate_v3_clamp(
        clean.float(),
        candidate.float(),
        layer=shared.layer,
        state_definition="V3-Dense",
        encoder=encoder,
        dense_map=dense_map,
        natural_scale=float(natural_scale),
        natural=natural,
        thresholds=thresholds,
    )
    quality = {
        "construction": construction,
        "validation": asdict(validation),
        "shared_rank": shared.rank,
        "shared_null_dimension": shared.null_dimension,
        "tangent_dimension": int(tangent.shape[1]),
        "radial_cosine": float(
            F.cosine_similarity(
                (candidate - clean).float()[None], clean.float()[None]
            ).item()
        ),
    }
    return candidate.to(clean.dtype), quality, bool(validation.formal_valid)


def matched_controls(
    clean: torch.Tensor,
    donor: torch.Tensor,
    preserving: torch.Tensor,
    j_direction: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return controls with displacement norm matched to the preserving arm."""

    displacement = preserving.float() - clean.float()
    norm = torch.linalg.vector_norm(displacement).clamp_min(1e-20)

    def scale(value: torch.Tensor) -> torch.Tensor:
        return value.float() * (
            norm / torch.linalg.vector_norm(value.float()).clamp_min(1e-20)
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(
        clean.device
    )
    return {
        "clean": clean.clone(),
        "identity": clean.clone(),
        "matched_random": clean.float() + scale(random),
        "j_positive": clean.float() + scale(j_direction),
        "full_perturbation": clean.float() + scale(donor.float() - clean.float()),
        "j_preserving": preserving.float(),
    }


def multiple_token_log_odds(logits: torch.Tensor, token_ids: list[int]) -> float:
    ids = sorted(set(int(value) for value in token_ids))
    if not ids:
        return float("nan")
    probabilities = torch.softmax(logits.double().reshape(-1), dim=0)
    probability = probabilities[ids].sum().clamp(1e-15, 1 - 1e-15)
    return float(torch.log(probability / (1 - probability)).item())


def aligned_rollout_metrics(
    clean: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare autonomous semantic-step trajectories without teacher forcing."""

    length = min(len(clean["logits"]), len(candidate["logits"]))
    js_curve = [
        jensen_shannon_from_logits(
            torch.as_tensor(clean["logits"][index]),
            torch.as_tensor(candidate["logits"][index]),
        )
        for index in range(length)
    ]
    state_curve = []
    for index in range(min(len(clean["j_states"]), len(candidate["j_states"]))):
        left = torch.as_tensor(clean["j_states"][index]).float()
        right = torch.as_tensor(candidate["j_states"][index]).float()
        state_curve.append(
            float(1 - F.cosine_similarity(left[None], right[None]).item())
        )
    layer_curve = {}
    for layer in sorted(
        set(clean.get("within_forward_j_states", {}))
        & set(candidate.get("within_forward_j_states", {})),
        key=int,
    ):
        left = torch.as_tensor(clean["within_forward_j_states"][layer]).float()
        right = torch.as_tensor(candidate["within_forward_j_states"][layer]).float()
        layer_curve[layer] = float(
            1 - F.cosine_similarity(left[None], right[None]).item()
        )
    clean_actions = tuple(clean["actions"])
    candidate_actions = tuple(candidate["actions"])
    expected = tuple(clean["expected_actions"])
    clean_final = clean_actions[-1] if clean_actions else None
    candidate_final = candidate_actions[-1] if candidate_actions else None
    return {
        "output_js_curve": js_curve,
        "output_js_divergence": float(np.mean(js_curve)) if js_curve else None,
        "future_j_divergence_curve": state_curve,
        "future_j_trajectory_divergence": (
            float(np.mean(state_curve)) if state_curve else None
        ),
        "within_forward_layer_j_divergence": layer_curve,
        "target_log_odds_curve": list(candidate["target_log_odds"]),
        "target_log_odds_clean_curve": list(clean["target_log_odds"]),
        "target_log_odds_change": (
            float(
                np.mean(candidate["target_log_odds"][:length])
                - np.mean(clean["target_log_odds"][:length])
            )
            if length
            else None
        ),
        "answer_flip": bool(candidate_final != clean_final),
        "task_accuracy": float(candidate_actions == expected),
        "clean_task_accuracy": float(clean_actions == expected),
        "task_accuracy_change": float(candidate_actions == expected)
        - float(clean_actions == expected),
        "final_answer_accuracy": float(candidate_final == expected[-1]),
        "full_trajectory_accuracy": float(candidate_actions == expected),
        "generated_actions": list(candidate_actions),
        "parseable": bool(candidate["parseable"]),
    }
