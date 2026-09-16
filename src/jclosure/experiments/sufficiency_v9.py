"""Rank-aware conditional-sufficiency and residual-localization protocol v9."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.arch_compression_v7 import _cosine_rows
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v9 import (
    PROTOCOL_V9,
    SCHEMA_VERSION_V9,
    build_freeze,
    verify_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

FEATURES = Path("artifacts/persistent/v8/structured_features_v8.npz")
V8_SWEEP = Path("results/v8/processed/persistent_state_compression_v8.parquet")
V8_SCREEN = Path("results/v8/processed/structured_component_screen_v8.parquet")
SWEEP = Path("results/v9/processed/sufficiency_dimension_sweep_v9.parquet")
SUMMARY = Path("results/v9/processed/sufficiency_dimension_sweep_v9.json")
RESIDUAL = Path("results/v9/processed/residual_information_localization_v9.parquet")
RESIDUAL_SUMMARY = Path(
    "results/v9/processed/residual_information_localization_v9.json"
)
MULTIHORIZON = Path("results/v9/processed/multihorizon_sufficiency_v9.parquet")
MULTIHORIZON_SUMMARY = Path(
    "results/v9/processed/multihorizon_sufficiency_v9.json"
)
ENRICHED = Path("results/v9/processed/behavior_enriched_causal_v9.parquet")
ENRICHED_SUMMARY = Path("results/v9/processed/behavior_enriched_causal_v9.json")

FULL_CONDITION = "rec_matrix_all+conv_all+kv_l27_h3+kv_other"


def _bootstrap_ci(
    values: np.ndarray, seeds: list[int], resamples: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    per_seed: dict[str, dict[str, float]] = {}
    for seed in seeds:
        generator = np.random.default_rng(seed)
        means = np.empty(resamples, dtype=np.float64)
        batch_size = min(1000, resamples)
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            draws = generator.integers(
                0, len(values), size=(stop - start, len(values))
            )
            means[start:stop] = values[draws].mean(axis=1)
        per_seed[str(seed)] = {
            "lower": float(np.quantile(means, 0.025)),
            "upper": float(np.quantile(means, 0.975)),
        }
    return {
        "estimate": float(values.mean()),
        "lower": min(item["lower"] for item in per_seed.values()),
        "upper": max(item["upper"] for item in per_seed.values()),
        "n_clusters": int(len(values)),
        "n_resamples_per_seed": int(resamples),
        "seeds": per_seed,
    }


def safe_gap_ratio(
    baseline: float, candidate: float, ceiling: float, epsilon: float
) -> dict[str, Any]:
    candidate_delta = float(candidate - baseline)
    ceiling_delta = float(ceiling - baseline)
    identified = abs(ceiling_delta) >= float(epsilon)
    return {
        "baseline_score": float(baseline),
        "candidate_score": float(candidate),
        "ceiling_score": float(ceiling),
        "candidate_baseline_delta": candidate_delta,
        "ceiling_baseline_delta": ceiling_delta,
        "gap_closed": float(candidate_delta / ceiling_delta) if identified else None,
        "identified": identified,
        "epsilon": float(epsilon),
    }


def _top10_agreement(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    k = min(10, predicted.shape[1])
    left = np.argpartition(predicted, -k, axis=1)[:, -k:]
    right = np.argpartition(target, -k, axis=1)[:, -k:]
    return np.asarray(
        [len(set(a) & set(b)) / k for a, b in zip(left, right, strict=True)],
        dtype=np.float64,
    )


def _standardize(
    train_values: np.ndarray, all_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0, keepdims=True)
    scale = train_values.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train_values - mean) / scale, (all_values - mean) / scale


def _energy(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = target - target.mean(axis=0, keepdims=True)
    covariance = scores.T @ centered
    values = np.sum(covariance * covariance, axis=1) / np.maximum(
        np.sum(scores * scores, axis=0), 1e-12
    )
    maximum = float(np.max(values))
    return values / maximum if maximum > 0 else values


def build_ordered_latents(
    features: np.ndarray,
    train: np.ndarray,
    targets: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fit, standardized = _standardize(features[train], features)
    _, singular, vh = np.linalg.svd(fit, full_matrices=False)
    keep = singular > max(float(singular[0]), 1e-20) * 1e-8
    singular = singular[keep]
    scores = standardized @ vh[keep].T
    fit_scores = scores[train]
    orders: dict[str, np.ndarray] = {
        "pca": np.arange(scores.shape[1]),
        "predictive_bottleneck": np.argsort(
            -_energy(fit_scores, targets["next_absolute"][train])
        ),
        "causal_bottleneck": np.argsort(
            -_energy(fit_scores, targets["next_delta"][train])
        ),
    }
    semantic_energy = (
        float(weights["next_absolute"])
        * _energy(fit_scores, targets["next_absolute"][train])
        + float(weights["next_causal_delta"])
        * _energy(fit_scores, targets["next_delta"][train])
        + float(weights["future_causal_delta"])
        * _energy(fit_scores, targets["future_delta_h4"][train])
    )
    orders["semantic_causal_bottleneck"] = np.argsort(-semantic_energy)
    return (
        {method: scores[:, order] for method, order in orders.items()},
        {
            "rank": int(scores.shape[1]),
            "largest_singular_value": float(singular[0]),
            "smallest_retained_singular_value": float(singular[-1]),
        },
    )


def _load_data(root: Path) -> dict[str, Any]:
    with np.load(root / FEATURES, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        layerwise = payload["layerwise_features"].astype(np.float32)
        full_features = payload["full_features"].astype(np.float32)
        ids = payload["base_trial_id"].astype(str)
        families = payload["family"].astype(str)
        splits = payload["split"].astype(str)
    combined: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "validation", "final_test"):
        capture = json.loads(
            (
                root / f"results/v8/processed/persistent_capture_{split}_v8.json"
            ).read_text(encoding="utf-8")
        )
        with np.load(root / capture["endpoint_artifact"], allow_pickle=False) as data:
            combined["base_trial_id"].append(data["base_trial_id"].astype(str))
            combined["current_j"].append(data["current_j_perturbed"].astype(np.float32))
            none_next = data["next_j__none"].astype(np.float32)
            full_next = data[f"next_j__{FULL_CONDITION}"].astype(np.float32)
            combined["next_delta"].append(full_next - none_next)
            combined["next_absolute"].append(full_next)
            none_future = data["future_j__none"].astype(np.float32)
            full_future = data[f"future_j__{FULL_CONDITION}"].astype(np.float32)
            for horizon, index in ((1, 0), (2, 1), (4, 3)):
                combined[f"future_delta_h{horizon}"].append(
                    full_future[:, index] - none_future[:, index]
                )
                combined[f"future_absolute_h{horizon}"].append(full_future[:, index])
            combined["output_delta"].append(
                (
                    data[f"output_log_odds__{FULL_CONDITION}"].astype(np.float32)
                    - data["output_log_odds__none"].astype(np.float32)
                )[:, None]
            )
    endpoint = {key: np.concatenate(values) for key, values in combined.items()}
    lookup = {
        value: index for index, value in enumerate(endpoint["base_trial_id"].astype(str))
    }
    order = np.asarray([lookup[value] for value in ids])
    targets = {
        key: value[order]
        for key, value in endpoint.items()
        if key != "base_trial_id"
    }
    return {
        "features": features,
        "layerwise": layerwise,
        "full_features": full_features,
        "ids": ids,
        "families": families,
        "splits": splits,
        "targets": targets,
    }


def _ridge_weights(
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean_x = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    fit = (train - mean_x) / scale
    held = (test - mean_x) / scale
    kernel = fit @ fit.T / max(1, fit.shape[1])
    cross = held @ fit.T / max(1, fit.shape[1])
    regularized = kernel + alpha * np.eye(len(fit))
    return np.linalg.solve(regularized, cross.T).T


def _apply_ridge_weights(
    weights: np.ndarray, train_target: np.ndarray
) -> np.ndarray:
    mean_y = train_target.mean(axis=0, keepdims=True)
    centered_y = train_target - mean_y
    return (mean_y + weights @ centered_y).astype(np.float32)


def _predict(
    x: np.ndarray,
    targets: dict[str, np.ndarray],
    train: np.ndarray,
    evaluate: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    weights = _ridge_weights(x[train], x[evaluate], alpha)
    return {
        name: _apply_ridge_weights(weights, value[train])
        for name, value in targets.items()
    }


def _vector_record(
    baseline: np.ndarray,
    candidate: np.ndarray,
    ceiling: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    local: np.ndarray,
    *,
    epsilon: float,
    seeds: list[int],
    resamples: int,
) -> dict[str, Any]:
    base = _cosine_rows(baseline[local], truth[local])
    compact = _cosine_rows(candidate[local], truth[local])
    full = _cosine_rows(ceiling[local], truth[local])
    raw_residual = _cosine_rows(residual[local], truth[local])
    gap = safe_gap_ratio(base.mean(), compact.mean(), full.mean(), epsilon)
    conditional = _bootstrap_ci(raw_residual - compact, seeds, resamples)
    return {
        **gap,
        "conditional_residual_gain": conditional,
        "compact_plus_residual_score": float(raw_residual.mean()),
    }


def _semantic_record(
    baseline: np.ndarray,
    candidate: np.ndarray,
    ceiling: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    local: np.ndarray,
    *,
    epsilon: float,
    seeds: list[int],
    resamples: int,
) -> dict[str, Any]:
    base = _top10_agreement(baseline[local], truth[local])
    compact = _top10_agreement(candidate[local], truth[local])
    full = _top10_agreement(ceiling[local], truth[local])
    raw_residual = _top10_agreement(residual[local], truth[local])
    return {
        **safe_gap_ratio(base.mean(), compact.mean(), full.mean(), epsilon),
        "conditional_residual_gain": _bootstrap_ci(
            raw_residual - compact, seeds, resamples
        ),
        "compact_plus_residual_score": float(raw_residual.mean()),
        "semantic_agreement": _bootstrap_ci(compact, seeds, resamples),
    }


def _scalar_record(
    baseline: np.ndarray,
    candidate: np.ndarray,
    ceiling: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    local: np.ndarray,
    *,
    epsilon: float,
    seeds: list[int],
    resamples: int,
) -> dict[str, Any]:
    base_error = np.abs(baseline[local].reshape(-1) - truth[local].reshape(-1))
    candidate_error = np.abs(candidate[local].reshape(-1) - truth[local].reshape(-1))
    ceiling_error = np.abs(ceiling[local].reshape(-1) - truth[local].reshape(-1))
    residual_error = np.abs(residual[local].reshape(-1) - truth[local].reshape(-1))
    sign = (
        np.sign(candidate[local].reshape(-1)) == np.sign(truth[local].reshape(-1))
    ).astype(float)
    return {
        **safe_gap_ratio(
            -base_error.mean(), -candidate_error.mean(), -ceiling_error.mean(), epsilon
        ),
        "conditional_residual_gain": _bootstrap_ci(
            candidate_error - residual_error, seeds, resamples
        ),
        "compact_plus_residual_score": float(-residual_error.mean()),
        "task_decision_sign_agreement": _bootstrap_ci(sign, seeds, resamples),
    }


def _flat_row(
    *,
    run_id: str,
    method: str,
    dimension: int,
    effective_dimension: int | None,
    family: str,
    endpoint: str,
    status: str,
    metrics: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V9,
        "protocol_version": PROTOCOL_V9,
        "record_type": "dimension_sweep",
        "run_id": run_id,
        "method": method,
        "dimension": int(dimension),
        "effective_dimension": effective_dimension,
        "family": family,
        "endpoint": endpoint,
        "split": "final_test",
        "status": status,
        "source": source,
        "causal_gap_closed": None,
        "causal_direction_cosine": None,
        "causal_magnitude_ratio": None,
        "causal_output_sign_agreement": None,
        "causal_status": "GATED_PENDING_OBSERVATIONAL_SUFFICIENCY",
    }
    if metrics is None:
        value.update(
            {
                "baseline_score": None,
                "candidate_score": None,
                "ceiling_score": None,
                "candidate_baseline_delta": None,
                "ceiling_baseline_delta": None,
                "predictive_gap_closed": None,
                "gap_identified": False,
                "conditional_residual_gain": None,
                "conditional_lower": None,
                "conditional_upper": None,
                "semantic_agreement": None,
                "metadata": json.dumps({}, sort_keys=True),
            }
        )
        return value
    conditional = metrics["conditional_residual_gain"]
    semantic = metrics.get("semantic_agreement")
    value.update(
        {
            "baseline_score": metrics["baseline_score"],
            "candidate_score": metrics["candidate_score"],
            "ceiling_score": metrics["ceiling_score"],
            "candidate_baseline_delta": metrics["candidate_baseline_delta"],
            "ceiling_baseline_delta": metrics["ceiling_baseline_delta"],
            "predictive_gap_closed": metrics["gap_closed"],
            "gap_identified": metrics["identified"],
            "conditional_residual_gain": conditional["estimate"],
            "conditional_lower": conditional["lower"],
            "conditional_upper": conditional["upper"],
            "semantic_agreement": semantic["estimate"] if semantic else None,
            "metadata": json.dumps(metrics, sort_keys=True),
        }
    )
    return value


def _legacy_rows(
    root: Path, run_id: str, epsilon: float
) -> list[dict[str, Any]]:
    frame = pd.read_parquet(root / V8_SWEEP)
    selected = frame[
        (frame["regime"] == "universal")
        & (frame["family"] == "pooled")
        & (frame["dimension"] < 512)
        & frame["method"].isin(
            ["pca", "predictive_bottleneck", "causal_bottleneck"]
        )
    ]
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        metadata = json.loads(record["metadata"])
        final = metadata["final"]
        metrics = {
            **safe_gap_ratio(
                final["j_only_cosine"]["estimate"],
                final["candidate_cosine"]["estimate"],
                final["full_state_cosine"]["estimate"],
                epsilon,
            ),
            "conditional_residual_gain": final["conditional_residual_gain"],
            "compact_plus_residual_score": final["candidate_cosine"]["estimate"]
            + final["conditional_residual_gain"]["estimate"],
        }
        rows.append(
            _flat_row(
                run_id=run_id,
                method=str(record["method"]),
                dimension=int(record["dimension"]),
                effective_dimension=int(record["dimension"]),
                family="pooled",
                endpoint="next_j",
                status="V8_FROZEN_REANALYSIS",
                metrics=metrics,
                source=str(V8_SWEEP),
            )
        )
    return rows


def _effect_enriched_indices(
    root: Path,
    ids: np.ndarray,
    families: np.ndarray,
    final: np.ndarray,
    section: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_parquet(root / V8_SCREEN)
    frame = frame[(frame["split"] == "final_test") & (frame["condition"] == "R7")]
    columns = [str(value) for value in section["score_columns"]]
    chosen: list[str] = []
    details: dict[str, Any] = {}
    for family, values in frame.groupby("family", sort=True):
        normalized = []
        for column in columns:
            raw = values[column].to_numpy(dtype=float)
            scale = raw.std()
            normalized.append((raw - raw.mean()) / (scale if scale > 1e-12 else 1.0))
        score = np.mean(np.stack(normalized, axis=1), axis=1)
        ranked = values.assign(_score=score).sort_values(
            ["_score", "base_trial_id"], ascending=[False, True]
        )
        count = max(
            int(section["minimum_per_family"]),
            int(math.ceil(len(ranked) * float(section["fraction_per_family"]))),
        )
        selected = ranked.head(count)
        chosen.extend(selected["base_trial_id"].astype(str).tolist())
        details[str(family)] = {
            "available": int(len(ranked)),
            "selected": int(len(selected)),
            "minimum_composite_score": float(selected["_score"].min()),
        }
    selected_ids = set(chosen)
    indices = np.asarray([index for index in final if ids[index] in selected_ids])
    payload = "\n".join(sorted(chosen)).encode()
    return indices, {
        "selection_rule": "top within-family standardized mean causal-effect score",
        "score_columns": columns,
        "count": int(len(indices)),
        "family_counts": details,
        "base_trial_id_sha256": hashlib.sha256(payload).hexdigest(),
        "model_fit_overlap": False,
        "held_out_split": "final_test",
    }


def _candidate_metrics(
    predictions: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    evaluation: np.ndarray,
    group: np.ndarray,
    *,
    epsilon: float,
    seeds: list[int],
    resamples: int,
) -> dict[str, dict[str, Any]]:
    local_lookup = {value: index for index, value in enumerate(evaluation.tolist())}
    local = np.asarray([local_lookup[value] for value in group])
    output: dict[str, dict[str, Any]] = {}
    for name in ("next_delta", "future_delta_h1", "future_delta_h2", "future_delta_h4"):
        output[name] = _vector_record(
            predictions["baseline"][name],
            predictions["candidate"][name],
            predictions["ceiling"][name],
            predictions["residual"][name],
            targets[name][evaluation],
            local,
            epsilon=epsilon,
            seeds=seeds,
            resamples=resamples,
        )
    output["output_delta"] = _scalar_record(
        predictions["baseline"]["output_delta"],
        predictions["candidate"]["output_delta"],
        predictions["ceiling"]["output_delta"],
        predictions["residual"]["output_delta"],
        targets["output_delta"][evaluation],
        local,
        epsilon=epsilon,
        seeds=seeds,
        resamples=resamples,
    )
    for horizon in (1, 2, 4):
        name = f"future_absolute_h{horizon}"
        if name not in predictions["candidate"]:
            continue
        output[name] = _semantic_record(
            predictions["baseline"][name],
            predictions["candidate"][name],
            predictions["ceiling"][name],
            predictions["residual"][name],
            targets[name][evaluation],
            local,
            epsilon=epsilon,
            seeds=seeds,
            resamples=resamples,
        )
    output["next_absolute"] = _semantic_record(
        predictions["baseline"]["next_absolute"],
        predictions["candidate"]["next_absolute"],
        predictions["ceiling"]["next_absolute"],
        predictions["residual"]["next_absolute"],
        targets["next_absolute"][evaluation],
        local,
        epsilon=epsilon,
        seeds=seeds,
        resamples=resamples,
    )
    return output


def _residual_localization(
    *,
    context: Any,
    data: dict[str, Any],
    latent_bank: dict[str, np.ndarray],
    train: np.ndarray,
    evaluation: np.ndarray,
    final: np.ndarray,
    dimensions: list[int],
    alpha: float,
    seeds: list[int],
    resamples: int,
    primary_method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = data["targets"]["current_j"]
    target = data["targets"]["next_delta"]
    layerwise = data["layerwise"]
    rank = data["features"].shape[1]
    if layerwise.shape[1] != 3 * rank:
        raise RuntimeError("v9 expected recurrent/conv/KV block scores with equal rank")
    blocks = {
        "recurrent": layerwise[:, :rank],
        "conv": layerwise[:, rank : 2 * rank],
        "kv": layerwise[:, 2 * rank :],
    }
    local_lookup = {value: index for index, value in enumerate(evaluation.tolist())}
    local = np.asarray([local_lookup[value] for value in final])
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for dimension in dimensions:
        if dimension > rank:
            summaries[str(dimension)] = {"status": "NOT_IDENTIFIED_RANK_LIMIT"}
            continue
        latent = latent_bank[primary_method][:, :dimension]
        base_x = np.concatenate((current, latent), axis=1)
        predictions: dict[int, np.ndarray] = {
            0: _apply_ridge_weights(
                _ridge_weights(base_x[train], base_x[evaluation], alpha),
                target[train],
            )
        }
        for mask in range(1, 8):
            selected = [
                blocks[name]
                for index, name in enumerate(("recurrent", "conv", "kv"))
                if mask & (1 << index)
            ]
            x = np.concatenate((base_x, *selected), axis=1)
            predictions[mask] = _apply_ridge_weights(
                _ridge_weights(x[train], x[evaluation], alpha), target[train]
            )
        truth = target[final]
        cosine = {
            mask: _cosine_rows(value[local], truth) for mask, value in predictions.items()
        }
        gain = {mask: cosine[mask] - cosine[0] for mask in range(8)}
        names = ("recurrent", "conv", "kv")
        standalone = {
            name: gain[1 << index] for index, name in enumerate(names)
        }
        interaction = gain[7] - gain[1] - gain[2] - gain[4]
        dimension_rows = []
        source_values = {**standalone, "interaction": interaction}
        for name, values in source_values.items():
            ci = _bootstrap_ci(values, seeds, resamples)
            row = {
                "schema_version": SCHEMA_VERSION_V9,
                "protocol_version": PROTOCOL_V9,
                "record_type": "residual_localization",
                "run_id": context.run_id,
                "method": primary_method,
                "dimension": int(dimension),
                "source": name,
                "conditional_gain": ci["estimate"],
                "lower": ci["lower"],
                "upper": ci["upper"],
                "metadata": json.dumps(ci, sort_keys=True),
            }
            rows.append(row)
            dimension_rows.append(row)
        joint = _bootstrap_ci(gain[7], seeds, resamples)
        summaries[str(dimension)] = {
            "status": "COMPLETED",
            "joint_conditional_gain": joint,
            "largest_residual_source": max(
                source_values,
                key=lambda name: float(source_values[name].mean()),
            ),
            "rows": dimension_rows,
        }
    return rows, summaries


def _analysis(context: Any, freeze: dict[str, Any]) -> None:
    section = context.config["sufficiency_v9"]
    data = _load_data(context.root)
    targets = data["targets"]
    splits = data["splits"]
    families = data["families"]
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    final = np.flatnonzero(splits == "final_test")
    evaluation = np.concatenate((validation, final))
    rank = int(data["features"].shape[1])
    if rank != int(freeze["effective_feature_rank"]):
        raise RuntimeError("v9 effective rank differs from frozen rank")
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    epsilon = float(section["gap_denominator_epsilon"])
    alpha = float(section["ridge_alpha"])
    requested = [int(value) for value in section["requested_dimensions"]]
    focus = [int(value) for value in section["newly_evaluated_dimensions"]]
    methods = [str(value) for value in section["methods"]]
    primary_method = str(section["primary_method"])
    latent_bank, latent_meta = build_ordered_latents(
        data["features"],
        train,
        targets,
        section["semantic_objective_weights"],
    )
    if latent_meta["rank"] != rank:
        raise RuntimeError("v9 latent rank differs from frozen feature rank")
    common_targets = {
        name: targets[name]
        for name in (
            "next_delta",
            "future_delta_h1",
            "future_delta_h2",
            "future_delta_h4",
            "output_delta",
            "next_absolute",
            "future_absolute_h1",
            "future_absolute_h2",
            "future_absolute_h4",
        )
    }
    baseline = _predict(targets["current_j"], common_targets, train, evaluation, alpha)
    full_x = np.concatenate((targets["current_j"], data["full_features"]), axis=1)
    ceiling = _predict(full_x, common_targets, train, evaluation, alpha)
    groups = [("pooled", final)] + [
        (str(family), final[families[final] == family])
        for family in sorted(set(families))
    ]
    enriched_indices, enriched_selection = _effect_enriched_indices(
        context.root,
        data["ids"],
        families,
        final,
        section["behavior_enriched"],
    )
    rows = _legacy_rows(context.root, context.run_id, epsilon)
    enriched_rows: list[dict[str, Any]] = []
    multihorizon_rows: list[dict[str, Any]] = []
    candidate_status: dict[str, Any] = {}
    progress_path = context.raw_dir / context.run_id / "analysis_progress.json"
    jobs = [
        (method, dimension)
        for method in methods
        for dimension in focus
        if dimension <= rank
    ]
    for job_index, (method, dimension) in enumerate(jobs):
        latent = latent_bank[method][:, :dimension]
        candidate_x = np.concatenate((targets["current_j"], latent), axis=1)
        candidate = _predict(candidate_x, common_targets, train, evaluation, alpha)
        predictions = {
            "baseline": baseline,
            "candidate": candidate,
            "ceiling": ceiling,
            "residual": ceiling,
        }
        group_passes: list[bool] = []
        group_details: dict[str, Any] = {}
        pooled_metrics: dict[str, dict[str, Any]] | None = None
        for family, indices in groups:
            metrics = _candidate_metrics(
                predictions,
                targets,
                evaluation,
                indices,
                epsilon=epsilon,
                seeds=seeds,
                resamples=resamples,
            )
            if family == "pooled":
                pooled_metrics = metrics
            for endpoint, endpoint_metrics in metrics.items():
                label = endpoint.replace("_delta", "").replace("_absolute", "_semantic")
                rows.append(
                    _flat_row(
                        run_id=context.run_id,
                        method=method,
                        dimension=dimension,
                        effective_dimension=dimension,
                        family=family,
                        endpoint=label,
                        status="COMPLETED_HELD_OUT",
                        metrics=endpoint_metrics,
                        source="v9_rank_aware_reanalysis",
                    )
                )
            predictive_endpoints = (
                "next_delta",
                "future_delta_h1",
                "future_delta_h2",
                "future_delta_h4",
                "output_delta",
            )
            predictive_pass = True
            for endpoint in predictive_endpoints:
                value = metrics[endpoint]
                if value["identified"]:
                    predictive_pass &= bool(
                        value["gap_closed"]
                        >= float(section["predictive_gap_closed_minimum"])
                    )
                else:
                    predictive_pass &= bool(
                        abs(value["candidate_score"] - value["ceiling_score"])
                        <= epsilon
                    )
            conditional_pass = all(
                metrics[endpoint]["conditional_residual_gain"]["upper"]
                <= float(section["conditional_residual_gain_maximum"])
                for endpoint in predictive_endpoints
            )
            semantic_pass = all(
                metrics[endpoint]["semantic_agreement"]["estimate"]
                >= float(section["semantic_agreement_minimum"])
                for endpoint in (
                    "next_absolute",
                    "future_absolute_h1",
                    "future_absolute_h2",
                    "future_absolute_h4",
                )
            )
            passed = predictive_pass and conditional_pass and semantic_pass
            group_passes.append(passed)
            group_details[family] = {
                "predictive_pass": predictive_pass,
                "conditional_pass": conditional_pass,
                "semantic_pass": semantic_pass,
                "observational_gate_pass": passed,
            }
        observational_pass = bool(group_passes and all(group_passes))
        key = f"{method}/{dimension}"
        candidate_status[key] = {
            "method": method,
            "dimension": dimension,
            "effective_dimension": dimension,
            "status": (
                "PENDING_CAUSAL_DECODING"
                if observational_pass
                else "GATED_BY_PREDICTIVE_CONDITIONAL_OR_SEMANTIC"
            ),
            "group_gates": group_details,
            "causal_gap_closed": None,
            "causal_direction_cosine": None,
            "causal_magnitude_ratio": None,
            "causal_output_sign_agreement": None,
            "authorized": False,
        }
        if method == primary_method:
            if pooled_metrics is None:
                raise RuntimeError("v9 pooled final-test metrics missing")
            enriched_metrics = _candidate_metrics(
                predictions,
                targets,
                evaluation,
                enriched_indices,
                epsilon=epsilon,
                seeds=seeds,
                resamples=resamples,
            )
            for endpoint, endpoint_metrics in enriched_metrics.items():
                label = endpoint.replace("_delta", "").replace("_absolute", "_semantic")
                enriched_rows.append(
                    _flat_row(
                        run_id=context.run_id,
                        method=method,
                        dimension=dimension,
                        effective_dimension=dimension,
                        family="effect_enriched",
                        endpoint=label,
                        status="COMPLETED_HELD_OUT_EFFECT_ENRICHED",
                        metrics=endpoint_metrics,
                        source=str(V8_SCREEN),
                    )
            )
            for horizon in (1, 2, 4):
                causal = pooled_metrics[f"future_delta_h{horizon}"]
                semantic = pooled_metrics[f"future_absolute_h{horizon}"]
                multihorizon_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION_V9,
                        "protocol_version": PROTOCOL_V9,
                        "record_type": "multihorizon_sufficiency",
                        "run_id": context.run_id,
                        "method": method,
                        "dimension": dimension,
                        "horizon": horizon,
                        "status": "COMPLETED_FROZEN_V8_CAPTURE",
                        "predictive_gap_closed": causal["gap_closed"],
                        "gap_identified": causal["identified"],
                        "baseline_score": causal["baseline_score"],
                        "candidate_score": causal["candidate_score"],
                        "ceiling_score": causal["ceiling_score"],
                        "compact_plus_residual_score": causal[
                            "compact_plus_residual_score"
                        ],
                        "conditional_residual_gain": causal[
                            "conditional_residual_gain"
                        ]["estimate"],
                        "conditional_lower": causal["conditional_residual_gain"][
                            "lower"
                        ],
                        "conditional_upper": causal["conditional_residual_gain"][
                            "upper"
                        ],
                        "semantic_agreement": semantic["semantic_agreement"][
                            "estimate"
                        ],
                        "output_predictive_gap_closed": (
                            pooled_metrics["output_delta"]["gap_closed"]
                            if horizon == 4
                            else None
                        ),
                        "output_conditional_residual_gain": (
                            pooled_metrics["output_delta"][
                                "conditional_residual_gain"
                            ]["estimate"]
                            if horizon == 4
                            else None
                        ),
                        "task_decision_sign_agreement": (
                            pooled_metrics["output_delta"][
                                "task_decision_sign_agreement"
                            ]["estimate"]
                            if horizon == 4
                            else None
                        ),
                        "output_logits_status": (
                            "AVAILABLE_AT_CAPTURED_TRAJECTORY_ENDPOINT"
                            if horizon == 4
                            else "NOT_SEPARATELY_CAPTURED"
                        ),
                        "task_decision_status": (
                            "AVAILABLE_AT_CAPTURED_TRAJECTORY_ENDPOINT"
                            if horizon == 4
                            else "NOT_SEPARATELY_CAPTURED"
                        ),
                    }
                )
            multihorizon_rows.append(
                {
                    "schema_version": SCHEMA_VERSION_V9,
                    "protocol_version": PROTOCOL_V9,
                    "record_type": "multihorizon_sufficiency",
                    "run_id": context.run_id,
                    "method": method,
                    "dimension": dimension,
                    "horizon": 8,
                    "status": "NOT_MEASURED_FROZEN_CAPTURE_LIMIT",
                    "predictive_gap_closed": None,
                    "gap_identified": False,
                    "baseline_score": None,
                    "candidate_score": None,
                    "ceiling_score": None,
                    "compact_plus_residual_score": None,
                    "conditional_residual_gain": None,
                    "conditional_lower": None,
                    "conditional_upper": None,
                    "semantic_agreement": None,
                    "output_predictive_gap_closed": None,
                    "output_conditional_residual_gain": None,
                    "task_decision_sign_agreement": None,
                    "output_logits_status": "NOT_CAPTURED",
                    "task_decision_status": "NOT_CAPTURED",
                }
            )
        write_json_atomic(
            progress_path,
            {
                "status": "RUNNING",
                "completed": job_index + 1,
                "total": len(jobs),
                "last_candidate": key,
            },
        )
    for method in methods:
        for dimension in requested:
            if dimension <= rank or dimension not in focus:
                continue
            key = f"{method}/{dimension}"
            candidate_status[key] = {
                "method": method,
                "dimension": dimension,
                "effective_dimension": None,
                "status": "NOT_IDENTIFIED_RANK_LIMIT",
                "rank": rank,
                "authorized": False,
            }
            rows.append(
                _flat_row(
                    run_id=context.run_id,
                    method=method,
                    dimension=dimension,
                    effective_dimension=None,
                    family="pooled",
                    endpoint="next_j",
                    status="NOT_IDENTIFIED_RANK_LIMIT",
                    metrics=None,
                    source="v9_rank_audit",
                )
            )
    residual_dimensions = [
        int(value) for value in section["residual_localization_dimensions"]
    ]
    residual_rows, residual_summary = _residual_localization(
        context=context,
        data=data,
        latent_bank=latent_bank,
        train=train,
        evaluation=evaluation,
        final=final,
        dimensions=residual_dimensions,
        alpha=alpha,
        seeds=seeds,
        resamples=resamples,
        primary_method=primary_method,
    )
    context.root.joinpath(SWEEP).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(context.root / SWEEP, index=False, compression="zstd")
    pd.DataFrame(residual_rows).to_parquet(
        context.root / RESIDUAL, index=False, compression="zstd"
    )
    pd.DataFrame(multihorizon_rows).to_parquet(
        context.root / MULTIHORIZON, index=False, compression="zstd"
    )
    pd.DataFrame(enriched_rows).to_parquet(
        context.root / ENRICHED, index=False, compression="zstd"
    )
    observational_candidates = [
        value
        for value in candidate_status.values()
        if value.get("status") == "PENDING_CAUSAL_DECODING"
    ]
    summary = {
        "schema_version": SCHEMA_VERSION_V9,
        "protocol_version": PROTOCOL_V9,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "source_v8_freeze_digest": freeze["v8_source_freeze_digest"],
        "status": "COMPLETED_OBSERVATIONAL_SCREEN_CAUSAL_GATED",
        "rank_audit": {
            "train_count": int(len(train)),
            "effective_rank": rank,
            "requested_dimensions": requested,
            "identified_dimensions": [value for value in requested if value <= rank],
            "rank_limited_dimensions": [value for value in requested if value > rank],
            "conclusion": "dimensions above 599 are not identifiable from the frozen 600-pair training split",
        },
        "latent_metadata": latent_meta,
        "candidate_status": candidate_status,
        "observational_gate_pass_count": len(observational_candidates),
        "smallest_observational_candidate": min(
            (value["dimension"] for value in observational_candidates), default=None
        ),
        "smallest_authorized_dimension": None,
        "causal_decoding_executed": False,
        "controller_authorized": False,
        "records": str(SWEEP),
        "records_sha256": sha256_file(context.root / SWEEP),
    }
    write_json_atomic(context.root / SUMMARY, summary)
    write_json_atomic(
        context.root / RESIDUAL_SUMMARY,
        {
            "schema_version": SCHEMA_VERSION_V9,
            "protocol_version": PROTOCOL_V9,
            "run_id": context.run_id,
            "source_freeze_digest": freeze["freeze_digest"],
            "method": primary_method,
            "dimensions": residual_summary,
            "records": str(RESIDUAL),
            "records_sha256": sha256_file(context.root / RESIDUAL),
        },
    )
    write_json_atomic(
        context.root / MULTIHORIZON_SUMMARY,
        {
            "schema_version": SCHEMA_VERSION_V9,
            "protocol_version": PROTOCOL_V9,
            "run_id": context.run_id,
            "source_freeze_digest": freeze["freeze_digest"],
            "captured_horizons": [1, 2, 4],
            "uncaptured_horizons": [8],
            "records": str(MULTIHORIZON),
            "records_sha256": sha256_file(context.root / MULTIHORIZON),
        },
    )
    write_json_atomic(
        context.root / ENRICHED_SUMMARY,
        {
            "schema_version": SCHEMA_VERSION_V9,
            "protocol_version": PROTOCOL_V9,
            "run_id": context.run_id,
            "source_freeze_digest": freeze["freeze_digest"],
            "selection": enriched_selection,
            "records": str(ENRICHED),
            "records_sha256": sha256_file(context.root / ENRICHED),
            "causal_fidelity_status": "GATED_PENDING_OBSERVATIONAL_SUFFICIENCY",
        },
    )
    write_json_atomic(
        progress_path,
        {"status": "COMPLETED", "completed": len(jobs), "total": len(jobs)},
    )
    context.finish("COMPLETED_V9_ANALYSIS", summary=summary)


def main() -> None:
    parser = standard_parser(
        "conditional sufficiency dimension protocol v9",
        "configs/sufficiency_v9.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "analyze"))
    args = parser.parse_args()
    context = initialize_context("sufficiency-v9", args)
    try:
        if args.stage == "freeze":
            context.finish(
                "COMPLETED_V9_FREEZE",
                freeze=build_freeze(context.root, context.config),
            )
            return
        freeze = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        _analysis(context, freeze)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
