"""Deterministic residual-stream interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from jclosure.decomposition import DecompositionResult, strip_j_component

InterventionKind = Literal[
    "steer",
    "coordinate_swap",
    "replace",
    "random",
    "non_j",
    "full_patch",
    "clamp",
]
PositionScope = Literal["final", "explicit", "all_non_padding", "reasoning_span"]


@dataclass(frozen=True)
class InterventionSpec:
    kind: InterventionKind
    layers: tuple[int, ...]
    positions: tuple[int, ...] | None
    strength: float = 1.0
    source_token_id: int | None = None
    target_token_id: int | None = None
    seed: int | None = None
    source: str | None = None
    position_scope: PositionScope = "explicit"
    protocol_version: str = "phase0_protocol_v1"
    dictionary_size: int | None = None


def _position_indices(length: int, positions: tuple[int, ...] | None) -> list[int]:
    if positions is None:
        return list(range(length))
    resolved: list[int] = []
    for position in positions:
        index = position if position >= 0 else length + position
        if not 0 <= index < length:
            raise IndexError(f"position {position} out of bounds for sequence length {length}")
        resolved.append(index)
    return resolved


def apply_at_positions(
    activation: torch.Tensor,
    transform,
    positions: tuple[int, ...] | None,
) -> torch.Tensor:
    """Clone and edit sequence positions in a [B,S,D] or [S,D] activation."""

    if activation.ndim not in (2, 3):
        raise ValueError("activation must have shape [S,D] or [B,S,D]")
    result = activation.clone()
    sequence_axis = 1 if activation.ndim == 3 else 0
    indices = _position_indices(activation.shape[sequence_axis], positions)
    if activation.ndim == 3:
        selected = result[:, indices, :]
        result[:, indices, :] = transform(selected)
    else:
        selected = result[indices, :]
        result[indices, :] = transform(selected)
    return result


def steer_activation(
    activation: torch.Tensor,
    direction: torch.Tensor,
    *,
    strength: float,
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    direction = direction.to(device=activation.device, dtype=activation.dtype)
    if direction.ndim != 1 or direction.shape[0] != activation.shape[-1]:
        raise ValueError("steering direction must have shape [d_model]")
    return apply_at_positions(
        activation, lambda selected: selected + strength * direction, positions
    )


def coordinate_swap_activation(
    activation: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    strength: float = 1.0,
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Swap coordinates in the two-direction span using a pseudoinverse."""

    source = source_direction.to(device=activation.device, dtype=torch.float32)
    target = target_direction.to(device=activation.device, dtype=torch.float32)
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("source and target directions must be equal 1D shapes")
    basis = torch.stack((source, target), dim=1)  # [D, 2]
    pseudoinverse = torch.linalg.pinv(basis)  # [2, D]

    def swap(selected: torch.Tensor) -> torch.Tensor:
        original_shape = selected.shape
        flat = selected.float().reshape(-1, selected.shape[-1])
        coordinates = flat @ pseudoinverse.T
        swapped = coordinates.flip(-1)
        delta = (swapped - coordinates) @ basis.T
        return (flat + strength * delta).reshape(original_shape).to(selected.dtype)

    return apply_at_positions(activation, swap, positions)


def replace_activation(
    activation: torch.Tensor,
    replacement: torch.Tensor,
    *,
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    replacement = replacement.to(device=activation.device, dtype=activation.dtype)

    def replace(selected: torch.Tensor) -> torch.Tensor:
        if replacement.shape == selected.shape:
            return replacement
        if replacement.ndim == 1 and replacement.shape[0] == selected.shape[-1]:
            return replacement.expand_as(selected)
        raise ValueError(
            f"replacement shape {replacement.shape} cannot fill {selected.shape}"
        )

    return apply_at_positions(activation, replace, positions)


def matched_random_direction(
    reference: torch.Tensor,
    *,
    seed: int,
    norm: float | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    target_norm = float(torch.linalg.vector_norm(reference.float())) if norm is None else norm
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return (direction * target_norm).to(device=reference.device, dtype=reference.dtype)


def non_j_direction(
    vector: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25
) -> tuple[torch.Tensor, DecompositionResult]:
    stripped, decomposition = strip_j_component(vector, dictionary, k=k)
    return stripped.to(vector.dtype), decomposition
