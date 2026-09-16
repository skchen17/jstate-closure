"""Train-only architecture-resolved score models for protocol v11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jclosure.experiments.decoded_causal_v10 import (
    _decoded_feature_scores,
    _latent_decoder,
)
from jclosure.experiments.sufficiency_v9 import (
    _apply_ridge_weights,
    _cosine_rows,
    _ridge_weights,
)

CHANNELS = ("recurrent", "conv", "kv")


@dataclass
class PCAModel:
    mean: np.ndarray
    scale: np.ndarray
    vh: np.ndarray
    scores: np.ndarray
    singular: np.ndarray


def fit_pca(values: np.ndarray, train: np.ndarray) -> PCAModel:
    mean = values[train].mean(axis=0, keepdims=True)
    scale = values[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    standardized = (values - mean) / scale
    _, singular, vh = np.linalg.svd(standardized[train], full_matrices=False)
    keep = singular > max(float(singular[0]), 1e-20) * 1e-8
    vh = vh[keep]
    return PCAModel(
        mean=mean,
        scale=scale,
        vh=vh,
        scores=standardized @ vh.T,
        singular=singular[keep],
    )


def _energy(scores: np.ndarray, target: np.ndarray, train: np.ndarray) -> np.ndarray:
    centered = target[train] - target[train].mean(axis=0, keepdims=True)
    covariance = scores[train].T @ centered
    numerator = np.sum(covariance * covariance, axis=1)
    denominator = np.maximum(np.sum(scores[train] * scores[train], axis=0), 1e-12)
    return numerator / denominator


def semantic_effect_target(delta: np.ndarray, top_k: int = 10) -> np.ndarray:
    output = np.zeros_like(delta, dtype=np.float32)
    k = min(top_k, delta.shape[1])
    indices = np.argpartition(np.abs(delta), -k, axis=1)[:, -k:]
    rows = np.arange(len(delta))[:, None]
    output[rows, indices] = np.sign(delta[rows, indices])
    return output


def objective_order(
    model: PCAModel,
    objective: str,
    targets: dict[str, np.ndarray],
    train: np.ndarray,
    effect_weights: np.ndarray,
) -> np.ndarray:
    rank = model.scores.shape[1]
    if objective == "cache_pca":
        return np.arange(rank)
    h1 = _energy(model.scores, targets["next_delta"], train)
    semantic = _energy(
        model.scores, semantic_effect_target(targets["next_delta"]), train
    )
    output = _energy(model.scores, targets["output_delta"], train)
    h4 = _energy(model.scores, targets["future_delta_h4"], train)
    if objective == "h1_direction":
        value = h1
    elif objective == "semantic_h1":
        value = semantic
    elif objective == "output_h1":
        value = output
    else:
        value = 0.6 * h1 + 0.4 * h4
        if objective == "effect_weighted_multistep":
            weighted_h1 = targets["next_delta"] * effect_weights[:, None]
            weighted_h4 = targets["future_delta_h4"] * effect_weights[:, None]
            value = (
                0.3 * value
                + 0.42 * _energy(model.scores, weighted_h1, train)
                + 0.28 * _energy(model.scores, weighted_h4, train)
            )
        elif objective == "causal_composite":
            value = 0.35 * h1 + 0.20 * h4 + 0.25 * semantic + 0.20 * output
        elif objective == "manifold_regularized":
            variance = model.singular.astype(np.float64) ** 2
            variance = variance / max(float(variance.max()), 1e-20)
            value = (
                0.30 * h1
                + 0.15 * h4
                + 0.20 * semantic
                + 0.15 * output
                + 0.20 * variance
            )
        elif objective != "multistep_direction":
            raise ValueError(f"unknown v11 objective: {objective}")
    return np.argsort(-value)


def project(
    model: PCAModel, order: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.zeros_like(model.scores, dtype=np.float32)
    selected = order[:dimension]
    retained[:, selected] = model.scores[:, selected]
    standardized = retained @ model.vh
    reconstructed = standardized * model.scale + model.mean
    return reconstructed.astype(np.float32), model.scores[:, selected].astype(
        np.float32
    )


def proportional_allocations(total: int) -> list[list[int]]:
    candidates = []
    for weights in ((0.45, 0.35, 0.20), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)):
        rec = int(round(total * weights[0]))
        conv = int(round(total * weights[1]))
        kv = total - rec - conv
        candidates.append([rec, conv, kv, 0])
    return candidates


class ScoreBank:
    def __init__(
        self,
        data: dict[str, Any],
        train: np.ndarray,
        effect_weights: np.ndarray,
        semantic_weights: dict[str, float],
    ) -> None:
        self.data = data
        self.train = train
        self.effect_weights = effect_weights
        self.rank = int(data["features"].shape[1])
        self.blocks = {
            "recurrent": data["layerwise"][:, : self.rank],
            "conv": data["layerwise"][:, self.rank : 2 * self.rank],
            "kv": data["layerwise"][:, 2 * self.rank :],
        }
        self.models = {
            name: fit_pca(values, train) for name, values in self.blocks.items()
        }
        self.joint_values = np.concatenate(
            [self.blocks[name] for name in CHANNELS], axis=1
        )
        self.joint_model = fit_pca(self.joint_values, train)
        self.unified_model = _latent_decoder(
            data["features"], train, data["targets"], semantic_weights
        )

    def _order(self, model: PCAModel, objective: str) -> np.ndarray:
        return objective_order(
            model,
            objective,
            self.data["targets"],
            self.train,
            self.effect_weights,
        )

    def factorized(
        self,
        allocation: list[int],
        objective: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
        decoded: dict[str, np.ndarray] = {}
        latents: list[np.ndarray] = []
        for name, dimension in zip(CHANNELS, allocation[:3], strict=True):
            order = self._order(self.models[name], objective)
            values, latent = project(self.models[name], order, int(dimension))
            decoded[name] = values
            latents.append(latent)
        interaction_dimension = int(allocation[3])
        interaction_meta: dict[str, Any] = {"dimension": interaction_dimension}
        if interaction_dimension:
            residual = np.concatenate(
                [self.blocks[name] - decoded[name] for name in CHANNELS], axis=1
            )
            interaction_model = fit_pca(residual, self.train)
            interaction_order = objective_order(
                interaction_model,
                objective,
                self.data["targets"],
                self.train,
                self.effect_weights,
            )
            correction, interaction_latent = project(
                interaction_model, interaction_order, interaction_dimension
            )
            for index, name in enumerate(CHANNELS):
                start = index * self.rank
                decoded[name] = decoded[name] + correction[:, start : start + self.rank]
            latents.append(interaction_latent)
            interaction_meta["rank"] = int(interaction_model.scores.shape[1])
        return decoded, np.concatenate(latents, axis=1), interaction_meta

    def oracle_joint(
        self, dimension: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
        reconstructed, latent = project(
            self.joint_model, np.arange(self.joint_model.scores.shape[1]), dimension
        )
        decoded = {
            name: reconstructed[:, index * self.rank : (index + 1) * self.rank]
            for index, name in enumerate(CHANNELS)
        }
        return decoded, latent, {"joint_rank": int(self.joint_model.scores.shape[1])}

    def unified(
        self, dimension: int
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
        values = _decoded_feature_scores(
            self.unified_model,
            "causal_bottleneck",
            dimension,
            np.arange(len(self.data["ids"])),
        )
        order = self.unified_model["orders"]["causal_bottleneck"][:dimension]
        latent = self.unified_model["scores"][:, order].astype(np.float32)
        inputs = {name: (self.data["features"], values) for name in CHANNELS}
        return inputs, latent, {"unified_rank": int(self.data["features"].shape[1])}

    def decode_inputs(
        self, spec: dict[str, Any]
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
        kind = str(spec["kind"])
        if kind == "unified":
            return self.unified(int(spec["dimension"]))
        if kind == "oracle_joint":
            decoded, latent, metadata = self.oracle_joint(int(spec["dimension"]))
        else:
            decoded, latent, metadata = self.factorized(
                [int(value) for value in spec["allocation"]],
                str(spec.get("objective", "cache_pca")),
            )
        inputs = {name: (self.blocks[name], decoded[name]) for name in CHANNELS}
        return inputs, latent, metadata


def _prediction_proxy(
    latent: np.ndarray,
    data: dict[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    alpha: float = 1.0,
) -> tuple[float, float]:
    current = data["targets"]["current_j"]
    x = np.concatenate((current, latent), axis=1)
    weights = _ridge_weights(x[train], x[validation], alpha)
    h1 = _apply_ridge_weights(weights, data["targets"]["next_delta"][train])
    h4 = _apply_ridge_weights(weights, data["targets"]["future_delta_h4"][train])
    return (
        float(_cosine_rows(h1, data["targets"]["next_delta"][validation]).mean()),
        float(_cosine_rows(h4, data["targets"]["future_delta_h4"][validation]).mean()),
    )


def _score_cosine(
    decoded: dict[str, np.ndarray],
    bank: ScoreBank,
    validation: np.ndarray,
) -> float:
    values = []
    for name in CHANNELS:
        left = decoded[name][validation]
        right = bank.blocks[name][validation]
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        values.append(float(np.mean(numerator / np.maximum(denominator, 1e-20))))
    return float(np.mean(values))


def build_method_specs(
    bank: ScoreBank,
    config: dict[str, Any],
    validation: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    section = config["causal_geometry_v11"]
    specs: list[dict[str, Any]] = [
        {
            "method": "unified_causal_d512",
            "kind": "unified",
            "dimension": 512,
            "class": "unified",
            "objective": "h1_direction",
        }
    ]
    search_rows: list[dict[str, Any]] = []
    for dimension in section["oracle"]["dimensions"]:
        dimension = int(dimension)
        specs.append(
            {
                "method": f"oracle_joint_pca_d{dimension}",
                "kind": "oracle_joint",
                "dimension": dimension,
                "class": "oracle_joint",
                "objective": "cache_pca",
            }
        )
        configured = section["factorized"]["allocations"].get(str(dimension))
        allocations = (
            configured["no_interaction"]
            if configured is not None
            else proportional_allocations(dimension)
        )
        candidates = []
        for allocation in allocations:
            decoded, latent, _ = bank.factorized(
                [int(value) for value in allocation], "cache_pca"
            )
            h1, h4 = _prediction_proxy(latent, bank.data, bank.train, validation)
            reconstruction = _score_cosine(decoded, bank, validation)
            row = {
                "record_type": "oracle_factorized_allocation_search",
                "dimension": dimension,
                "allocation": [int(value) for value in allocation],
                "objective": "cache_pca",
                "h1_proxy": h1,
                "h4_proxy": h4,
                "score_reconstruction_cosine": reconstruction,
                "selection_score": reconstruction,
            }
            search_rows.append(row)
            candidates.append(row)
        best = max(
            candidates,
            key=lambda value: (value["selection_score"], value["allocation"]),
        )
        specs.append(
            {
                "method": f"oracle_factorized_pca_d{dimension}",
                "kind": "factorized",
                "dimension": dimension,
                "allocation": best["allocation"],
                "class": "oracle_factorized",
                "objective": "cache_pca",
            }
        )
    factorized = section["factorized"]
    for budget in factorized["total_budgets"]:
        budget = int(budget)
        for objective in factorized["objectives"]:
            for interaction_class, allocations in factorized["allocations"][
                str(budget)
            ].items():
                candidates = []
                for allocation in allocations:
                    decoded, latent, _ = bank.factorized(
                        [int(value) for value in allocation], str(objective)
                    )
                    h1, h4 = _prediction_proxy(
                        latent, bank.data, bank.train, validation
                    )
                    reconstruction = _score_cosine(decoded, bank, validation)
                    selection_score = 0.45 * h1 + 0.35 * h4 + 0.20 * reconstruction
                    row = {
                        "record_type": "factorized_allocation_search",
                        "dimension": budget,
                        "allocation": [int(value) for value in allocation],
                        "objective": str(objective),
                        "interaction_class": str(interaction_class),
                        "h1_proxy": h1,
                        "h4_proxy": h4,
                        "score_reconstruction_cosine": reconstruction,
                        "selection_score": selection_score,
                    }
                    search_rows.append(row)
                    candidates.append(row)
                best = max(
                    candidates,
                    key=lambda value: (value["selection_score"], value["allocation"]),
                )
                tag = "int" if interaction_class == "interaction" else "no_int"
                specs.append(
                    {
                        "method": f"factorized_{objective}_{tag}_d{budget}",
                        "kind": "factorized",
                        "dimension": budget,
                        "allocation": best["allocation"],
                        "class": f"factorized_{tag}",
                        "objective": str(objective),
                    }
                )
    return specs, search_rows
