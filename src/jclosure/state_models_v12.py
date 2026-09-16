"""Global and state-dependent linear causal-coordinate models for v12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jclosure.experiments.sufficiency_v9 import (
    _apply_ridge_weights,
    _cosine_rows,
    _ridge_weights,
)

CHANNELS = ("recurrent", "conv", "kv")


@dataclass
class LinearBasis:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    scores: np.ndarray
    singular: np.ndarray
    rank: int
    definition: str


def _standardize(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = values[train].mean(axis=0, keepdims=True)
    scale = values[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return mean, scale, (values - mean) / scale


def fit_pca_basis(values: np.ndarray, train: np.ndarray) -> LinearBasis:
    mean, scale, standardized = _standardize(values, train)
    _, singular, vh = np.linalg.svd(standardized[train], full_matrices=False)
    keep = singular > max(float(singular[0]), 1e-20) * 1e-8
    basis = vh[keep].astype(np.float32)
    return LinearBasis(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        basis=basis,
        scores=(standardized @ basis.T).astype(np.float32),
        singular=singular[keep].astype(np.float32),
        rank=int(np.sum(keep)),
        definition="global variance PCA",
    )


def _target_scores(target: np.ndarray, train: np.ndarray) -> np.ndarray:
    fit = target[train].astype(np.float64)
    fit -= fit.mean(axis=0, keepdims=True)
    gram = fit @ fit.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    vectors = vectors[:, order]
    keep = values > max(float(values[0]), 1e-20) * 1e-10
    return (vectors[:, keep] * np.sqrt(values[keep])[None]).astype(np.float32)


def fit_causal_basis(
    values: np.ndarray, target: np.ndarray, train: np.ndarray
) -> LinearBasis:
    mean, scale, standardized = _standardize(values, train)
    _, singular, vh = np.linalg.svd(standardized[train], full_matrices=False)
    keep = singular > max(float(singular[0]), 1e-20) * 1e-8
    vh = vh[keep]
    pca_scores = standardized @ vh.T
    target_scores = _target_scores(target, train)
    cross = pca_scores[train].T @ target_scores
    left, causal_singular, _ = np.linalg.svd(cross, full_matrices=False)
    causal_keep = causal_singular > max(float(causal_singular[0]), 1e-20) * 1e-8
    left = left[:, causal_keep]
    causal_singular = causal_singular[causal_keep]
    basis = (left.T @ vh).astype(np.float32)
    return LinearBasis(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        basis=basis,
        scores=(standardized @ basis.T).astype(np.float32),
        singular=causal_singular.astype(np.float32),
        rank=int(len(causal_singular)),
        definition=(
            "reduced-rank causal covariance basis over h1/h2/h4 J effects and output"
        ),
    )


def project_basis(
    model: LinearBasis, values: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    if dimension > model.rank:
        raise ValueError(f"dimension {dimension} exceeds fitted rank {model.rank}")
    standardized = (values - model.mean) / model.scale
    latent = standardized @ model.basis[:dimension].T
    reconstructed = (latent @ model.basis[:dimension]) * model.scale + model.mean
    return reconstructed.astype(np.float32), latent.astype(np.float32)


def target_bundle(data: dict[str, Any]) -> np.ndarray:
    targets = data["targets"]
    return np.concatenate(
        [
            targets["next_delta"],
            targets["future_delta_h2"],
            targets["future_delta_h4"],
            targets["output_delta"],
        ],
        axis=1,
    ).astype(np.float32)


def nested_indices(data: dict[str, Any], base: dict[str, Any]) -> dict[int, np.ndarray]:
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    return {
        int(size): np.asarray([lookup[value] for value in ids], dtype=int)
        for size, ids in base["train_scaling"]["nested_base_trial_ids"].items()
    }


def _prediction_metrics(
    latent: np.ndarray,
    data: dict[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
) -> dict[str, float]:
    x = np.concatenate((data["targets"]["current_j"], latent), axis=1)
    weights = _ridge_weights(x[train], x[validation], 1.0)
    output = {}
    for name, key in (
        ("h1_direction", "next_delta"),
        ("h2_direction", "future_delta_h2"),
        ("h4_direction", "future_delta_h4"),
    ):
        predicted = _apply_ridge_weights(weights, data["targets"][key][train])
        output[name] = float(
            _cosine_rows(predicted, data["targets"][key][validation]).mean()
        )
    predicted_output = _apply_ridge_weights(
        weights, data["targets"]["output_delta"][train]
    )[:, 0]
    target_output = data["targets"]["output_delta"][validation, 0]
    output["output_correlation"] = float(
        np.corrcoef(predicted_output, target_output)[0, 1]
    )
    return output


def scaling_records(
    data: dict[str, Any], base: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    train_sets = nested_indices(data, base)
    validation = np.flatnonzero(data["splits"] == "validation")
    dimensions = [int(value) for value in config["causal_geometry_v12"]["dimensions"]]
    requested_sizes = [
        int(value) for value in config["causal_geometry_v12"]["train_sizes"]
    ]
    target = target_bundle(data)
    rows = []
    for size in requested_sizes:
        if size not in train_sets:
            for method in ("combined_pca", "architecture_joint_pca", "causal_weighted"):
                for dimension in dimensions:
                    rows.append(
                        {
                            "train_size": size,
                            "method": method,
                            "dimension": dimension,
                            "status": "NOT_IDENTIFIED_BANK_SIZE",
                        }
                    )
            continue
        train = train_sets[size]
        models = {
            "combined_pca": (fit_pca_basis(data["features"], train), data["features"]),
            "architecture_joint_pca": (
                fit_pca_basis(data["layerwise"], train),
                data["layerwise"],
            ),
            "causal_weighted": (
                fit_causal_basis(data["layerwise"], target, train),
                data["layerwise"],
            ),
        }
        for method, (model, values) in models.items():
            for dimension in dimensions:
                if dimension > model.rank:
                    rows.append(
                        {
                            "train_size": size,
                            "method": method,
                            "dimension": dimension,
                            "empirical_rank": model.rank,
                            "status": "NOT_IDENTIFIED_RANK_LIMIT",
                        }
                    )
                    continue
                reconstructed, latent = project_basis(model, values, dimension)
                residual = values[validation] - reconstructed[validation]
                centered = values[validation] - values[train].mean(
                    axis=0, keepdims=True
                )
                fraction = 1.0 - float(np.sum(residual**2)) / max(
                    float(np.sum(centered**2)), 1e-20
                )
                rows.append(
                    {
                        "train_size": size,
                        "method": method,
                        "dimension": dimension,
                        "empirical_rank": model.rank,
                        "status": "IDENTIFIED",
                        "reconstruction_fraction": fraction,
                        "raw_teacher_effect_coverage": fraction,
                        **_prediction_metrics(latent, data, train, validation),
                    }
                )
    return rows


class V12StateBank:
    def __init__(self, data: dict[str, Any], train: np.ndarray) -> None:
        self.data = data
        self.train = train
        self.rank = int(data["features"].shape[1])
        self.blocks = {
            "recurrent": data["layerwise"][:, : self.rank],
            "conv": data["layerwise"][:, self.rank : 2 * self.rank],
            "kv": data["layerwise"][:, 2 * self.rank :],
        }
        self.target = target_bundle(data)
        self.arch_pca = fit_pca_basis(data["layerwise"], train)
        self.arch_causal = fit_causal_basis(data["layerwise"], self.target, train)
        self.combined_pca = fit_pca_basis(data["features"], train)
        context = data["targets"]["current_j"]
        self.context_mean = context[train].mean(axis=0, keepdims=True)
        self.context_scale = context[train].std(axis=0, keepdims=True)
        self.context_scale[self.context_scale < 1e-6] = 1.0
        self.context = (context - self.context_mean) / self.context_scale
        self._local_cache: dict[tuple[int, int, bool], LinearBasis] = {}

    def specs(self, dimensions: list[int]) -> list[dict[str, Any]]:
        output = []
        for dimension in dimensions:
            for kind, label in (
                ("combined_pca", "global_combined_pca"),
                ("architecture_pca", "global_architecture_pca"),
                ("global_causal", "global_causal_basis"),
                ("local_pca", "local_pca"),
                ("local_causal", "local_causal_basis"),
            ):
                if kind == "combined_pca":
                    rank = self.combined_pca.rank
                elif kind == "architecture_pca":
                    rank = self.arch_pca.rank
                elif kind == "global_causal":
                    rank = self.arch_causal.rank
                else:
                    # A centered neighborhood of n states has rank at most n - 1.
                    rank = min(511, self.arch_pca.rank)
                status = (
                    "IDENTIFIED"
                    if int(dimension) <= rank
                    else "NOT_IDENTIFIED_RANK_LIMIT"
                )
                output.append(
                    {
                        "method": f"{label}_d{dimension}",
                        "kind": kind,
                        "class": label,
                        "dimension": int(dimension),
                        "empirical_rank_bound": int(rank),
                        "status": status,
                    }
                )
        return output

    def _local_model(
        self, source_index: int, neighbors: int, causal: bool
    ) -> LinearBasis:
        key = (int(source_index), int(neighbors), bool(causal))
        if key in self._local_cache:
            return self._local_cache[key]
        distance = np.linalg.norm(
            self.context[self.train] - self.context[source_index], axis=1
        )
        local = self.train[np.argsort(distance)[:neighbors]]
        if causal:
            model = fit_causal_basis(self.data["layerwise"], self.target, local)
        else:
            model = fit_pca_basis(self.data["layerwise"], local)
        self._local_cache[key] = model
        return model

    def reconstruction_inputs(
        self,
        spec: dict[str, Any],
        selected: np.ndarray,
        *,
        neighbors: int,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
        kind = str(spec["kind"])
        dimension = int(spec["dimension"])
        if spec.get("status", "IDENTIFIED") != "IDENTIFIED":
            raise ValueError(
                f"{spec['method']} is {spec['status']} at dimension {dimension}"
            )
        if kind == "combined_pca":
            reconstructed, latent = project_basis(
                self.combined_pca, self.data["features"][selected], dimension
            )
            inputs = {name: (self.data["features"], reconstructed) for name in CHANNELS}
            return inputs, latent, {"empirical_rank": self.combined_pca.rank}
        if kind in {"architecture_pca", "global_causal"}:
            model = self.arch_pca if kind == "architecture_pca" else self.arch_causal
            reconstructed, latent = project_basis(
                model, self.data["layerwise"][selected], dimension
            )
            ranks = [self.rank * index for index in range(4)]
            inputs = {
                name: (
                    self.blocks[name],
                    reconstructed[:, ranks[index] : ranks[index + 1]],
                )
                for index, name in enumerate(CHANNELS)
            }
            return inputs, latent, {"empirical_rank": model.rank}
        reconstructed_rows, latent_rows, ranks = [], [], []
        causal = kind == "local_causal"
        for source_index in selected:
            model = self._local_model(int(source_index), neighbors, causal)
            reconstructed, latent = project_basis(
                model, self.data["layerwise"][[source_index]], dimension
            )
            reconstructed_rows.append(reconstructed[0])
            latent_rows.append(latent[0])
            ranks.append(model.rank)
        reconstructed = np.stack(reconstructed_rows)
        latent = np.stack(latent_rows)
        inputs = {
            name: (
                self.blocks[name],
                reconstructed[:, index * self.rank : (index + 1) * self.rank],
            )
            for index, name in enumerate(CHANNELS)
        }
        return (
            inputs,
            latent,
            {
                "minimum_local_rank": int(min(ranks)),
                "maximum_local_rank": int(max(ranks)),
                "neighbors": neighbors,
            },
        )
