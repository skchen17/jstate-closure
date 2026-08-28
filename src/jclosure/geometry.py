"""Geometry of normalized dense measured-J states for exploratory protocol v3.

Dense maps in this module are intentionally *not* row-normalized.  If ``A`` is
the selected-token map ``W_U J_l``, the normalized dense state is

``s(h) = C A h / ||C A h||``

where ``C`` centers across concept rows.  Sparse pursuit continues to use the
independent row-normalized dictionary in :mod:`jclosure.decomposition`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from jclosure.decomposition import DecompositionResult
from jclosure.metrics import sparse_support_f1

RankStatus = Literal["RESOLVED", "NUMERICALLY_UNRESOLVED"]


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left.float())
    right_norm = torch.linalg.vector_norm(right.float())
    if float(left_norm) <= 1e-20 and float(right_norm) <= 1e-20:
        return 1.0
    if float(left_norm) <= 1e-20 or float(right_norm) <= 1e-20:
        return 0.0
    return float(F.cosine_similarity(left.float()[None], right.float()[None]).item())


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-20)
    return float(
        torch.linalg.vector_norm(candidate.float() - reference.float()).item()
        / denominator.item()
    )


@dataclass(frozen=True)
class SpectrumSummary:
    """Serializable spectrum diagnostics using explicit, declared tolerances."""

    singular_values: tuple[float, ...]
    tolerance_ranks: dict[str, int]
    tolerance_values: dict[str, float]
    stable_rank: float
    participation_rank: float
    entropy_effective_rank: float
    truncated_condition: dict[str, float | None]
    cumulative_variance: tuple[float, ...]
    rank_status: RankStatus = "RESOLVED"
    dtype: str = "float32"

    @property
    def rank_fraction(self) -> dict[str, float]:
        dimension = max(len(self.singular_values), 1)
        return {
            key: value / dimension for key, value in self.tolerance_ranks.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "singular_values": list(self.singular_values),
            "tolerance_ranks": dict(self.tolerance_ranks),
            "tolerance_values": dict(self.tolerance_values),
            "rank_fraction": self.rank_fraction,
            "stable_rank": self.stable_rank,
            "participation_rank": self.participation_rank,
            "entropy_effective_rank": self.entropy_effective_rank,
            "truncated_condition": dict(self.truncated_condition),
            "cumulative_variance": list(self.cumulative_variance),
            "rank_status": self.rank_status,
            "dtype": self.dtype,
        }

    @classmethod
    def from_singular_values(
        cls,
        singular_values: torch.Tensor | np.ndarray | Iterable[float],
        *,
        rows: int,
        cols: int,
        dtype: torch.dtype = torch.float32,
        relative_tolerances: Iterable[float] = (1e-2, 1e-3, 1e-4, 1e-5),
        unresolved_band: float = 4.0,
    ) -> SpectrumSummary:
        values = torch.as_tensor(singular_values, dtype=torch.float64).reshape(-1)
        values = torch.sort(values.clamp_min(0), descending=True).values
        sigma_max = float(values[0].item()) if values.numel() else 0.0
        eps = torch.finfo(dtype).eps
        tolerance_values: dict[str, float] = {}
        for tolerance in relative_tolerances:
            tolerance_values[f"relative_{tolerance:.0e}"] = sigma_max * float(tolerance)
        tolerance_values["machine"] = max(rows, cols) * eps * sigma_max
        tolerance_ranks = {
            key: int(torch.count_nonzero(values > threshold).item())
            for key, threshold in tolerance_values.items()
        }
        squares = values.square()
        square_sum = float(squares.sum().item())
        stable = 0.0 if sigma_max <= 0 else square_sum / (sigma_max * sigma_max)
        fourth_sum = float(squares.square().sum().item())
        participation = 0.0 if fourth_sum <= 0 else square_sum * square_sum / fourth_sum
        if square_sum <= 0:
            entropy_rank = 0.0
            cumulative: tuple[float, ...] = tuple(0.0 for _ in values)
        else:
            probabilities = squares / square_sum
            positive = probabilities[probabilities > 0]
            entropy_rank = float(torch.exp(-(positive * positive.log()).sum()).item())
            cumulative = tuple(float(value) for value in probabilities.cumsum(0))
        conditions: dict[str, float | None] = {}
        for key, rank in tolerance_ranks.items():
            conditions[key] = (
                None
                if rank <= 0 or sigma_max <= 0
                else sigma_max / float(values[rank - 1].item())
            )
        machine_threshold = tolerance_values["machine"]
        unresolved = bool(
            machine_threshold > 0
            and torch.any(
                (values >= machine_threshold / unresolved_band)
                & (values <= machine_threshold * unresolved_band)
            )
        )
        return cls(
            singular_values=tuple(float(value) for value in values),
            tolerance_ranks=tolerance_ranks,
            tolerance_values=tolerance_values,
            stable_rank=float(stable),
            participation_rank=float(participation),
            entropy_effective_rank=float(entropy_rank),
            truncated_condition=conditions,
            cumulative_variance=cumulative,
            rank_status="NUMERICALLY_UNRESOLVED" if unresolved else "RESOLVED",
            dtype=str(dtype).removeprefix("torch."),
        )


class DenseJMap:
    """Layer-indexed raw and centered dense measured-J maps."""

    def __init__(
        self,
        maps: dict[int, torch.Tensor] | None = None,
        *,
        builder: Callable[[int], torch.Tensor] | None = None,
        layers: Iterable[int] = (),
        eps: float = 1e-12,
    ) -> None:
        if not maps and builder is None:
            raise ValueError("DenseJMap requires maps or a builder")
        self._raw = {int(layer): value.detach() for layer, value in (maps or {}).items()}
        self._builder = builder
        self.layers = tuple(sorted(set(int(value) for value in (*layers, *self._raw))))
        self.eps = float(eps)
        self._device_cache: dict[tuple[int, str, str, bool], torch.Tensor] = {}

    @classmethod
    def from_encoder(cls, encoder: Any) -> DenseJMap:
        return cls(
            builder=lambda layer: encoder.raw_dictionary(layer).detach().cpu(),
            layers=encoder.available_layers,
        )

    def _ensure(self, layer: int) -> None:
        layer = int(layer)
        if layer in self._raw:
            return
        if self._builder is None or (self.layers and layer not in self.layers):
            raise KeyError(f"no dense measured-J map for layer {layer}")
        value = self._builder(layer)
        if value.ndim != 2:
            raise ValueError("dense measured-J maps must have shape [concepts,d_model]")
        self._raw[layer] = value.detach()

    def raw_map(
        self,
        layer: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        self._ensure(layer)
        value = self._raw[int(layer)]
        if device is None and value.dtype == dtype:
            return value
        target_device = value.device if device is None else torch.device(device)
        key = (int(layer), str(target_device), str(dtype), False)
        if key not in self._device_cache:
            self._device_cache[key] = value.to(device=target_device, dtype=dtype)
        return self._device_cache[key]

    def centered_map(
        self,
        layer: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        raw = self.raw_map(layer, device=device, dtype=dtype)
        key = (int(layer), str(raw.device), str(dtype), True)
        if key not in self._device_cache:
            self._device_cache[key] = raw - raw.mean(dim=0, keepdim=True)
        return self._device_cache[key]

    def raw_scores(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return self.raw_map(layer, device=h.device, dtype=h.dtype) @ h

    def dense_state(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if h.ndim != 1:
            raise ValueError("dense_state expects a [d_model] vector")
        centered = self.centered_map(layer, device=h.device, dtype=h.dtype) @ h
        norm = torch.linalg.vector_norm(centered)
        if float(norm.detach()) <= self.eps:
            raise ValueError("centered dense scores have zero norm")
        return centered / norm

    def dense_state_jvp(
        self, h: torch.Tensor, v: torch.Tensor, layer: int
    ) -> torch.Tensor:
        if h.shape != v.shape or h.ndim != 1:
            raise ValueError("h and v must be equal [d_model] vectors")
        centered = self.centered_map(layer, device=h.device, dtype=h.dtype)
        scores = centered @ h
        norm = torch.linalg.vector_norm(scores).clamp_min(self.eps)
        state = scores / norm
        mapped = centered @ v
        return (mapped - state * torch.dot(state, mapped)) / norm

    def dense_state_vjp(
        self, h: torch.Tensor, u: torch.Tensor, layer: int
    ) -> torch.Tensor:
        centered = self.centered_map(layer, device=h.device, dtype=h.dtype)
        if h.ndim != 1 or u.shape != (centered.shape[0],):
            raise ValueError("h or cotangent has incompatible shape")
        scores = centered @ h
        norm = torch.linalg.vector_norm(scores).clamp_min(self.eps)
        state = scores / norm
        projected = u - state * torch.dot(state, u)
        return centered.T @ projected / norm

    def local_jacobian(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Materialize ``J_s(h)``; use only when the tall matrix fits memory."""

        centered = self.centered_map(layer, device=h.device, dtype=h.dtype)
        scores = centered @ h
        norm = torch.linalg.vector_norm(scores).clamp_min(self.eps)
        state = scores / norm
        return (centered - state[:, None] * (state @ centered)[None, :]) / norm

    def local_jacobian_gram(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Return the analytic d_model×d_model Gram matrix ``J_s.T @ J_s``."""

        centered = self.centered_map(layer, device=h.device, dtype=h.dtype)
        scores = centered @ h
        norm = torch.linalg.vector_norm(scores).clamp_min(self.eps)
        state = scores / norm
        projected_row = centered.T @ state
        return (
            centered.T @ centered - projected_row[:, None] @ projected_row[None, :]
        ) / norm.square()

    def radial_residual(self, h: torch.Tensor, layer: int) -> float:
        jvp = self.dense_state_jvp(h, h, layer)
        gram = self.local_jacobian_gram(h, layer)
        operator_norm = torch.linalg.eigvalsh(gram.double()).amax().clamp_min(0).sqrt()
        denominator = operator_norm * torch.linalg.vector_norm(h.double())
        if float(denominator) <= self.eps:
            return 0.0
        return float(torch.linalg.vector_norm(jvp.double()).item() / denominator.item())


@dataclass(frozen=True)
class SparseStateEquality:
    support_f1: float
    weighted_jaccard: float
    coefficient_cosine: float
    coefficient_relative_l2: float
    reconstruction_cosine: float
    reconstruction_relative_l2: float
    passed: bool
    failure_reasons: tuple[str, ...]

    @classmethod
    def compare(
        cls,
        clean: DecompositionResult,
        candidate: DecompositionResult,
        *,
        support_f1_threshold: float = 0.8,
        weighted_jaccard_threshold: float = 0.95,
        coefficient_cosine_threshold: float = 0.995,
        reconstruction_cosine_threshold: float = 0.995,
    ) -> SparseStateEquality:
        clean_map = {
            int(index): float(value)
            for index, value in zip(
                clean.atom_indices, clean.coefficients, strict=True
            )
        }
        candidate_map = {
            int(index): float(value)
            for index, value in zip(
                candidate.atom_indices, candidate.coefficients, strict=True
            )
        }
        union = sorted(clean_map.keys() | candidate_map.keys())
        clean_coefficients = torch.tensor(
            [clean_map.get(index, 0.0) for index in union], dtype=torch.float64
        )
        candidate_coefficients = torch.tensor(
            [candidate_map.get(index, 0.0) for index in union], dtype=torch.float64
        )
        denominator = sum(
            max(clean_map.get(index, 0.0), candidate_map.get(index, 0.0))
            for index in union
        )
        numerator = sum(
            min(clean_map.get(index, 0.0), candidate_map.get(index, 0.0))
            for index in union
        )
        weighted = 1.0 if denominator <= 1e-20 else numerator / denominator
        coefficient_cosine = _safe_cosine(clean_coefficients, candidate_coefficients)
        coefficient_relative_l2 = _relative_l2(
            clean_coefficients, candidate_coefficients
        )
        reconstruction_cosine = _safe_cosine(
            clean.reconstruction, candidate.reconstruction
        )
        reconstruction_relative_l2 = _relative_l2(
            clean.reconstruction, candidate.reconstruction
        )
        support = sparse_support_f1(list(clean_map), list(candidate_map))
        failures: list[str] = []
        if support < support_f1_threshold:
            failures.append("sparse_support_f1")
        if weighted < weighted_jaccard_threshold:
            failures.append("sparse_weighted_jaccard")
        if coefficient_cosine < coefficient_cosine_threshold:
            failures.append("sparse_coefficient_cosine")
        if reconstruction_cosine < reconstruction_cosine_threshold:
            failures.append("sparse_reconstruction_cosine")
        return cls(
            support_f1=support,
            weighted_jaccard=weighted,
            coefficient_cosine=coefficient_cosine,
            coefficient_relative_l2=coefficient_relative_l2,
            reconstruction_cosine=reconstruction_cosine,
            reconstruction_relative_l2=reconstruction_relative_l2,
            passed=not failures,
            failure_reasons=tuple(failures),
        )


@dataclass(frozen=True)
class DenseOptimizationResult:
    activation: torch.Tensor
    delta: torch.Tensor
    status: str
    iterations: int
    dense_cosine: float
    top10_overlap: float
    rms_drift: float
    displacement: float
    donor_alignment: float
    failure_reason: str | None = None


class DenseNullProjector:
    """Construct local dense-null and sphere-tangent perturbations."""

    def __init__(self, dense_map: DenseJMap, layer: int) -> None:
        self.dense_map = dense_map
        self.layer = int(layer)

    def low_singular_basis(
        self, h: torch.Tensor, *, relative_tolerance: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        jacobian = self.dense_map.local_jacobian(h, self.layer)
        _, values, vh = torch.linalg.svd(jacobian, full_matrices=True)
        maximum = values[0].clamp_min(1e-30) if values.numel() else h.new_tensor(1.0)
        rank = int(torch.count_nonzero(values > relative_tolerance * maximum).item())
        basis = vh[rank:].T.contiguous()
        return basis, values

    @staticmethod
    def tangent_intersection(
        basis: torch.Tensor, h: torch.Tensor, *, tolerance: float = 1e-10
    ) -> torch.Tensor:
        if basis.ndim != 2 or basis.shape[0] != h.numel():
            raise ValueError("basis must have shape [d_model,q]")
        if basis.shape[1] == 0:
            return basis
        radial_coordinates = basis.T @ h
        radial_norm = torch.linalg.vector_norm(radial_coordinates)
        if float(radial_norm) <= tolerance:
            return basis
        _, _, vh = torch.linalg.svd(radial_coordinates[None, :], full_matrices=True)
        coefficient_null = vh[1:].T
        return basis @ coefficient_null

    @staticmethod
    def project(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        if basis.shape[1] == 0:
            return torch.zeros_like(vector)
        return basis @ (basis.T @ vector)

    def donor_projection(
        self,
        h: torch.Tensor,
        donor_difference: torch.Tensor,
        *,
        relative_tolerance: float,
        sphere_tangent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis, singular_values = self.low_singular_basis(
            h, relative_tolerance=relative_tolerance
        )
        if sphere_tangent:
            basis = self.tangent_intersection(basis, h)
        return self.project(donor_difference, basis), basis, singular_values

    @staticmethod
    def retract_to_sphere(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        target_norm = torch.linalg.vector_norm(h.float())
        candidate = h.float() + delta.float()
        candidate_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-20)
        return (candidate * (target_norm / candidate_norm) - h.float()).to(h.dtype)

    def optimize_hard_constraints(
        self,
        h: torch.Tensor,
        donor_difference: torch.Tensor,
        basis: torch.Tensor,
        *,
        target_displacement: float,
        dense_cosine_threshold: float = 0.995,
        top10_overlap_threshold: float = 0.8,
        rms_drift_threshold: float = 0.02,
        naturality: Callable[[torch.Tensor], bool] | None = None,
        max_iter: int = 128,
        min_step: float = 1e-6,
    ) -> DenseOptimizationResult:
        if basis.shape[1] == 0:
            return self._failure(h, "zero_dimensional_intersection")
        clean_state = self.dense_map.dense_state(h.float(), self.layer)
        raw_clean = self.dense_map.raw_scores(h.float(), self.layer)
        projected = self.project(donor_difference.float(), basis.float())
        projected_norm = torch.linalg.vector_norm(projected)
        if not torch.isfinite(projected_norm) or float(projected_norm) <= 1e-20:
            return self._failure(h, "degenerate_donor_projection")
        direction = projected / projected_norm
        displacement = min(float(target_displacement), float(projected_norm))
        step = displacement
        iterations = 0
        last_reason = "line_search_exhausted"
        while iterations < max_iter and step >= min_step:
            iterations += 1
            delta = self.retract_to_sphere(h, direction * step)
            candidate = h + delta
            if not torch.isfinite(candidate).all():
                return self._failure(h, "nan_or_inf", iterations=iterations)
            state = self.dense_map.dense_state(candidate.float(), self.layer)
            raw_candidate = self.dense_map.raw_scores(candidate.float(), self.layer)
            cosine = _safe_cosine(clean_state, state)
            top10 = _topk_overlap(raw_clean, raw_candidate, 10)
            drift = _rms_norm_drift(h, candidate)
            natural = True if naturality is None else bool(naturality(candidate))
            if (
                cosine >= dense_cosine_threshold
                and top10 >= top10_overlap_threshold
                and drift <= rms_drift_threshold
                and natural
            ):
                donor_alignment = _safe_cosine(delta, donor_difference)
                return DenseOptimizationResult(
                    activation=candidate.detach(),
                    delta=delta.detach(),
                    status="CONVERGED",
                    iterations=iterations,
                    dense_cosine=cosine,
                    top10_overlap=top10,
                    rms_drift=drift,
                    displacement=float(torch.linalg.vector_norm(delta.float()).item()),
                    donor_alignment=donor_alignment,
                )
            if not natural:
                last_reason = "naturality_constraint"
            elif cosine < dense_cosine_threshold:
                last_reason = "dense_equality_constraint"
            elif top10 < top10_overlap_threshold:
                last_reason = "top10_constraint"
            elif drift > rms_drift_threshold:
                last_reason = "rms_constraint"
            step *= 0.5
        return self._failure(h, last_reason, iterations=iterations)

    def _failure(
        self, h: torch.Tensor, reason: str, *, iterations: int = 0
    ) -> DenseOptimizationResult:
        return DenseOptimizationResult(
            activation=h.detach().clone(),
            delta=torch.zeros_like(h),
            status="FAILED",
            iterations=iterations,
            dense_cosine=1.0,
            top10_overlap=1.0,
            rms_drift=0.0,
            displacement=0.0,
            donor_alignment=0.0,
            failure_reason=reason,
        )


def _topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    k = min(k, left.numel(), right.numel())
    if k <= 0:
        return 1.0
    left_ids = set(torch.topk(left.float(), k).indices.tolist())
    right_ids = set(torch.topk(right.float(), k).indices.tolist())
    return len(left_ids & right_ids) / k


def _rms_norm_drift(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    candidate_rms = torch.sqrt(torch.mean(candidate.float().square()))
    return float(
        torch.abs(candidate_rms - reference_rms).item()
        / reference_rms.clamp_min(1e-20).item()
    )


def pareto_nondominated(
    records: Iterable[dict[str, float]],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> list[dict[str, float]]:
    """Return deterministic non-dominated records without inventing scalar weights."""

    rows = list(records)
    output: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            weak = all(other[key] >= row[key] for key in maximize) and all(
                other[key] <= row[key] for key in minimize
            )
            strict = any(other[key] > row[key] for key in maximize) or any(
                other[key] < row[key] for key in minimize
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            output.append(row)
    return output


def maximum_feasible_displacement(
    records: Iterable[dict[str, Any]],
    *,
    dense_cosine: float = 0.995,
    top10_overlap: float = 0.8,
    rms_drift: float = 0.02,
    require_natural: bool = True,
) -> float | None:
    values = [
        float(record["displacement_fraction"])
        for record in records
        if float(record["dense_cosine"]) >= dense_cosine
        and float(record["top10_overlap"]) >= top10_overlap
        and float(record["rms_drift"]) <= rms_drift
        and (not require_natural or bool(record.get("natural", False)))
    ]
    return max(values) if values else None
