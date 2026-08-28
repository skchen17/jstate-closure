"""One-shot and persistent J-component clamps with strict quality checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from jclosure.jstate import JState, JStateEncoder, jstate_similarity
from jclosure.metrics import rms_drift


@dataclass(frozen=True)
class ClampThresholds:
    dense_cosine: float = 0.995
    top10_overlap: float = 0.8
    rms_drift: float = 0.02
    min_remainder_fraction: float = 0.20


DEFAULT_CLAMP_THRESHOLDS = ClampThresholds()


@dataclass(frozen=True)
class ClampResult:
    activation: torch.Tensor
    target_state: JState
    observed_state: JState
    dense_cosine: float
    top10_overlap: float
    activation_rms_drift: float
    remainder_distance: float
    remainder_fraction: float
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MultiPositionClampResult:
    activation: torch.Tensor
    positions: tuple[int, ...]
    results: tuple[ClampResult, ...]
    passed: bool
    valid_fraction: float
    failure_reasons: tuple[str, ...]


def validate_clamp(
    clean: torch.Tensor,
    experimental: torch.Tensor,
    *,
    layer: int,
    encoder: JStateEncoder,
    thresholds: ClampThresholds = DEFAULT_CLAMP_THRESHOLDS,
    natural_scale: float | None = None,
    position: int = -1,
) -> ClampResult:
    target_state = encoder.encode(clean, layer, position=position)
    observed_state = encoder.encode(experimental, layer, position=position)
    dense = jstate_similarity(target_state, observed_state, "dense_cosine")
    overlap = jstate_similarity(target_state, observed_state, "top10_overlap")
    drift = rms_drift(clean, experimental)
    clean_decomposition = encoder.decompose(clean, layer)
    experimental_decomposition = encoder.decompose(experimental, layer)
    remainder_distance = float(
        torch.linalg.vector_norm(
            experimental_decomposition.remainder.float()
            - clean_decomposition.remainder.float()
        ).item()
    )
    denominator = (
        float(natural_scale)
        if natural_scale is not None
        else float(torch.linalg.vector_norm(clean.float()).item())
    )
    remainder_fraction = remainder_distance / max(denominator, 1e-12)
    failures: list[str] = []
    if dense < thresholds.dense_cosine:
        failures.append("dense_cosine")
    if overlap < thresholds.top10_overlap:
        failures.append("top10_overlap")
    if drift > thresholds.rms_drift:
        failures.append("rms_drift")
    if remainder_fraction < thresholds.min_remainder_fraction:
        failures.append("remainder_distance")
    return ClampResult(
        activation=experimental,
        target_state=target_state,
        observed_state=observed_state,
        dense_cosine=dense,
        top10_overlap=overlap,
        activation_rms_drift=drift,
        remainder_distance=remainder_distance,
        remainder_fraction=remainder_fraction,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def one_shot_clamp(
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    layer: int,
    encoder: JStateEncoder,
    thresholds: ClampThresholds = DEFAULT_CLAMP_THRESHOLDS,
    natural_scale: float | None = None,
    position: int = -1,
) -> ClampResult:
    """Combine clean sparse-J reconstruction with perturbed remainder."""

    if clean.shape != perturbed.shape or clean.ndim != 1:
        raise ValueError("clean and perturbed must be equal [d_model] vectors")
    clean_decomposition = encoder.decompose(clean, layer)
    perturbed_decomposition = encoder.decompose(perturbed, layer)
    experimental = (
        clean_decomposition.reconstruction + perturbed_decomposition.remainder
    ).to(clean.dtype)
    return validate_clamp(
        clean,
        experimental,
        layer=layer,
        encoder=encoder,
        thresholds=thresholds,
        natural_scale=natural_scale,
        position=position,
    )


def _resolve_positions(length: int, positions: Iterable[int] | None) -> tuple[int, ...]:
    requested = range(length) if positions is None else positions
    resolved: list[int] = []
    for position in requested:
        index = int(position) if int(position) >= 0 else length + int(position)
        if not 0 <= index < length:
            raise IndexError(f"position {position} out of bounds for length {length}")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


def one_shot_clamp_positions(
    clean: torch.Tensor,
    perturbed: torch.Tensor,
    *,
    layer: int,
    encoder: JStateEncoder,
    positions: Iterable[int] | None = None,
    thresholds: ClampThresholds = DEFAULT_CLAMP_THRESHOLDS,
    natural_scales: dict[int, float] | None = None,
) -> MultiPositionClampResult:
    """Restore measured-J independently at selected sequence positions."""

    if clean.shape != perturbed.shape or clean.ndim != 2:
        raise ValueError("clean and perturbed must be equal [sequence,d_model] tensors")
    indices = _resolve_positions(clean.shape[0], positions)
    output = perturbed.clone()
    results: list[ClampResult] = []
    for index in indices:
        result = one_shot_clamp(
            clean[index],
            perturbed[index],
            layer=layer,
            encoder=encoder,
            thresholds=thresholds,
            natural_scale=(natural_scales or {}).get(index),
            position=index,
        )
        output[index] = result.activation
        results.append(result)
    passed_count = sum(result.passed for result in results)
    reasons = tuple(
        sorted({reason for result in results for reason in result.failure_reasons})
    )
    return MultiPositionClampResult(
        activation=output,
        positions=indices,
        results=tuple(results),
        passed=bool(results) and passed_count == len(results),
        valid_fraction=passed_count / max(len(results), 1),
        failure_reasons=reasons,
    )


def persistent_clamp_transforms(
    clean_by_layer: dict[int, torch.Tensor],
    encoder: JStateEncoder,
    *,
    position: int = -1,
    positions: Iterable[int] | None = (),
) -> dict[int, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Return layer transforms suitable for ``ResidualEditor``."""

    transforms: dict[int, Callable[[torch.Tensor, int], torch.Tensor]] = {}
    selected_positions = None if positions is None else tuple(positions)
    for layer, clean in clean_by_layer.items():
        clean_vector = clean.detach().clone()

        def transform(
            activation: torch.Tensor, current_layer: int, clean_ref=clean_vector
        ):
            output = activation.clone()
            length = activation.shape[-2]
            requested = (position,) if selected_positions == () else selected_positions
            indices = _resolve_positions(length, requested)
            for index in indices:
                perturbed_vector = (
                    activation[0, index] if activation.ndim == 3 else activation[index]
                )
                clean_at = clean_ref[index] if clean_ref.ndim == 2 else clean_ref
                clamp = one_shot_clamp(
                    clean_at.to(perturbed_vector.device),
                    perturbed_vector,
                    layer=current_layer,
                    encoder=encoder,
                    thresholds=ClampThresholds(min_remainder_fraction=0.0),
                    position=index,
                )
                if activation.ndim == 3:
                    output[:, index, :] = clamp.activation
                else:
                    output[index, :] = clamp.activation
            return output

        transforms[int(layer)] = transform
    return transforms
