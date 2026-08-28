"""Sparse non-negative decomposition in an overcomplete J-direction frame."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecompositionResult:
    atom_indices: torch.Tensor
    coefficients: torch.Tensor
    reconstruction: torch.Tensor
    remainder: torch.Tensor
    reconstruction_error: float
    variance_explained: float


def normalize_dictionary(dictionary: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if dictionary.ndim != 2:
        raise ValueError("dictionary must have shape [n_atoms, d_model]")
    return F.normalize(dictionary.float(), dim=-1, eps=eps)


def _projected_nnls(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    max_iter: int = 256,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Solve a small NNLS problem with deterministic projected gradients.

    ``design`` is ``[d_model, n_active]`` and active sets are normally at most
    25 columns, so an exact spectral step is inexpensive and stable.
    """

    if design.shape[1] == 0:
        return target.new_zeros((0,), dtype=torch.float32)
    design = design.float()
    target = target.float()
    gram = design.T @ design
    rhs = design.T @ target
    try:
        initial = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution[:, 0]
    except RuntimeError:
        initial = torch.zeros_like(rhs)
    coefficients = initial.clamp_min(0)
    lipschitz = torch.linalg.eigvalsh(gram).amax().clamp_min(1e-8)
    step = 1.0 / lipschitz
    for _ in range(max_iter):
        updated = (coefficients - step * (gram @ coefficients - rhs)).clamp_min(0)
        if torch.max(torch.abs(updated - coefficients)) <= tolerance:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def gradient_pursuit(
    vector: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 25,
    correlation_tolerance: float = 1e-8,
    nnls_max_iter: int = 256,
) -> DecompositionResult:
    """Approximate ``vector`` with at most ``k`` non-negative atoms.

    Dictionary rows are normalized internally. At every step the atom with the
    largest positive correlation with the residual is selected; ties resolve
    to the lowest row index through ``torch.argmax``. Coefficients are refit by
    NNLS on the full active support after each selection.
    """

    if vector.ndim != 1:
        raise ValueError("vector must have shape [d_model]")
    if dictionary.ndim != 2 or dictionary.shape[1] != vector.shape[0]:
        raise ValueError("dictionary shape is incompatible with vector")
    if k <= 0:
        raise ValueError("k must be positive")

    atoms = normalize_dictionary(dictionary).to(vector.device)
    target = vector.float()
    reconstruction = torch.zeros_like(target)
    residual = target.clone()
    selected: list[int] = []
    coefficients = target.new_zeros((0,))
    max_atoms = min(k, atoms.shape[0])

    for _ in range(max_atoms):
        correlations = atoms @ residual
        if selected:
            correlations[torch.tensor(selected, device=atoms.device)] = -torch.inf
        index = int(torch.argmax(correlations).item())
        if not torch.isfinite(correlations[index]) or float(correlations[index]) <= correlation_tolerance:
            break
        selected.append(index)
        active = atoms[selected].T
        coefficients = _projected_nnls(
            active, target, max_iter=nnls_max_iter
        )
        reconstruction = active @ coefficients
        residual = target - reconstruction

    atom_indices = torch.tensor(selected, dtype=torch.long, device=vector.device)
    error = float(torch.linalg.vector_norm(residual).item())
    total_sq = float(torch.dot(target, target).item())
    residual_sq = float(torch.dot(residual, residual).item())
    variance_explained = 0.0 if total_sq <= 1e-20 else 1.0 - residual_sq / total_sq
    return DecompositionResult(
        atom_indices=atom_indices,
        coefficients=coefficients,
        reconstruction=reconstruction.to(vector.dtype),
        remainder=residual.to(vector.dtype),
        reconstruction_error=error,
        variance_explained=variance_explained,
    )


def strip_j_component(
    vector: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25
) -> tuple[torch.Tensor, DecompositionResult]:
    result = gradient_pursuit(vector, dictionary, k=k)
    return result.remainder, result

