"""V13 data scaling, exact-JVP probe scaling, tangent atlas, and path oracles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments.bank_v13 import CAPTURE_FREEZE_PATH
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.causal_v12 import _continuous_semantic, _effect_metrics
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import _dual_pca_scores
from jclosure.experiments.decoded_causal_v10 import apply_decoded_state
from jclosure.experiments.jvp_v12 import _apply_direction, _spectrum
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import _semantic_ids
from jclosure.model import load_model_bundle
from jclosure.protocol_v13 import (
    PROTOCOL_V13,
    SCHEMA_VERSION_V13,
    SELECTION_PATH,
    build_stage_freeze,
    verify_base_freeze,
    verify_stage_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

FEATURES = Path("artifacts/causal/v13/causal_features_v13.npz")
FEATURE_SUMMARY = Path("results/v13/processed/causal_features_v13.json")
SCALING_RECORDS = Path("results/v13/processed/data_dimension_scaling_v13.parquet")
SCALING_SUMMARY = Path("results/v13/processed/data_dimension_scaling_v13.json")
DIRECTIONS = Path("artifacts/causal/v13/probe_directions_v13.pt")
DIRECTION_SUMMARY = Path("results/v13/processed/probe_directions_v13.json")
GEOMETRY_FREEZE = Path("artifacts/causal_geometry_v13_jvp.freeze.json")
JVP_RECORDS = Path("results/v13/processed/causal_probe_scaling_v13.parquet")
JVP_SUMMARY = Path("results/v13/processed/causal_probe_scaling_v13.json")
JVP_MATRIX_ROOT = Path("results/v13/processed/jvp_matrices")
ORACLE_RECORDS = Path("results/v13/processed/moving_tangent_oracle_v13.parquet")
ORACLE_SUMMARY = Path("results/v13/processed/moving_tangent_oracle_v13.json")
ANALYSIS_SUMMARY = Path("results/v13/processed/geometry_analysis_v13.json")
LINEARITY_RECORDS = Path("results/v13/processed/local_linearity_radius_v13.parquet")
LINEARITY_SUMMARY = Path("results/v13/processed/local_linearity_radius_v13.json")


def _capture_manifests(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(
            (
                root / f"results/v13/processed/causal_capture_{split}_v13.json"
            ).read_text()
        )
        for split in ("train", "validation", "final_test")
    ]


def _load_raw_deltas(
    root: Path,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    recurrent_rows: list[torch.Tensor] = []
    conv_rows: list[torch.Tensor] = []
    temporary_kv: list[torch.Tensor] = []
    ids: list[str] = []
    families: list[str] = []
    splits: list[str] = []
    maximum_tokens = 256
    for capture in _capture_manifests(root):
        for declaration in capture["state_shards"]:
            path = root / declaration["path"]
            if sha256_file(path) != declaration["sha256"]:
                raise RuntimeError(f"V13 state shard hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                clean, perturbed = row["clean"], row["perturbed"]
                recurrent_rows.append(perturbed["recurrent"] - clean["recurrent"])
                conv_rows.append(perturbed["conv"] - clean["conv"])
                current = torch.stack(
                    [
                        torch.stack(
                            [
                                perturbed["kv"][layer][name] - clean["kv"][layer][name]
                                for name in ("keys", "values")
                            ]
                        )
                        for layer in ("27", "31")
                    ]
                )
                if int(current.shape[-2]) > maximum_tokens:
                    raise RuntimeError(
                        f"V13 prompt cache length {current.shape[-2]} exceeds frozen "
                        f"KV analysis width {maximum_tokens}"
                    )
                temporary_kv.append(current)
                ids.append(str(row["base_trial_id"]))
                families.append(str(row["family"]))
                splits.append(str(row["split"]))
    kv_rows = []
    for current in temporary_kv:
        padded = torch.zeros(
            (*current.shape[:-2], maximum_tokens, current.shape[-1]),
            dtype=torch.bfloat16,
        )
        padded[..., : current.shape[-2], :] = current
        kv_rows.append(padded)
    return (
        {
            "recurrent": torch.stack(recurrent_rows),
            "conv": torch.stack(conv_rows),
            "kv": torch.stack(kv_rows),
        },
        ids,
        families,
        splits,
    )


def _endpoints(root: Path, ids: list[str]) -> dict[str, np.ndarray]:
    by_id: dict[str, dict[str, np.ndarray]] = {}
    for capture in _capture_manifests(root):
        with np.load(root / capture["endpoint_artifact"], allow_pickle=False) as data:
            names = list(data.files)
            for index, base_id in enumerate(data["base_trial_id"].astype(str)):
                by_id[str(base_id)] = {
                    name: np.asarray(data[name][index])
                    for name in names
                    if name not in {"base_trial_id", "prompt_id", "family"}
                }
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for base_id in ids:
        for name, value in by_id[base_id].items():
            output[name].append(value)
    return {
        name: np.stack(values)
        for name, values in output.items()
        if len(values) == len(ids)
    }


def _build_features(context: Any) -> dict[str, Any]:
    freeze = verify_stage_freeze(context.root, context.config, CAPTURE_FREEZE_PATH)
    raw, ids, families, splits = _load_raw_deltas(context.root)
    train = np.flatnonzero(np.asarray(splits) == "train")
    device = (
        f"cuda:{int(context.config['model'].get('device', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    features, layerwise, metadata = _dual_pca_scores(
        [raw["recurrent"], raw["conv"], raw["kv"]], train, device=device
    )
    endpoints = _endpoints(context.root, ids)
    path = context.root / FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features,
        "layerwise": layerwise,
        "base_trial_id": np.asarray(ids),
        "family": np.asarray(families),
        "split": np.asarray(splits),
    }
    payload.update({f"endpoint__{name}": value for name, value in endpoints.items()})
    np.savez_compressed(path, **payload)
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "source_capture_freeze_digest": freeze["freeze_digest"],
        "count": len(ids),
        "split_counts": {split: splits.count(split) for split in sorted(set(splits))},
        "feature_dimension": int(features.shape[1]),
        "layerwise_dimension": int(layerwise.shape[1]),
        "architecture_blocks": metadata,
        "artifact": str(FEATURES),
        "artifact_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / FEATURE_SUMMARY, summary)
    return summary


def _load_features(root: Path) -> dict[str, Any]:
    with np.load(root / FEATURES, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _target_bundle(data: dict[str, Any]) -> np.ndarray:
    values = []
    train = np.flatnonzero(data["split"].astype(str) == "train")
    for horizon in (1, 2, 4):
        j_delta = data[f"endpoint__perturbed_j_h{horizon}"].astype(np.float32) - data[
            f"endpoint__clean_j_h{horizon}"
        ].astype(np.float32)
        selected_j = np.argsort(-j_delta[train].var(axis=0))[:256]
        values.append(j_delta[:, selected_j])
        values.append(
            data[f"endpoint__perturbed_logits_h{horizon}"].astype(np.float32)
            - data[f"endpoint__clean_logits_h{horizon}"].astype(np.float32)
        )
        values.append(
            data[f"endpoint__semantic_delta_h{horizon}"].astype(np.float32)[:, None]
        )
    return np.concatenate(values, axis=1)


def _standardize(
    values: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train].mean(axis=0, keepdims=True)
    scale = values[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return mean, scale, ((values - mean) / scale).astype(np.float32)


def _fit_basis(
    values: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    kind: str,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    _, _, x = _standardize(values, train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = torch.from_numpy(x[train]).float().to(device)
    q = min(maximum, len(train) - 1, fit.shape[1])
    if kind == "pca":
        _, singular, vectors = torch.pca_lowrank(fit, q=q, center=False, niter=3)
        basis = vectors.T.cpu().numpy().astype(np.float32)
        scores = x @ basis.T
        rank = int(np.sum(singular.cpu().numpy() > float(singular[0]) * 1e-8))
        return basis, scores.astype(np.float32), rank
    _, _, y = _standardize(target, train)
    q = min(q, y.shape[1])
    cross = (fit.T @ torch.from_numpy(y[train]).float().to(device)) / max(len(train), 1)
    left, singular, _ = torch.svd_lowrank(cross, q=q, niter=3)
    basis = left[:, :q].T.cpu().numpy().astype(np.float32)
    scores = x @ basis.T
    rank = int(np.sum(singular.cpu().numpy() > float(singular[0]) * 1e-8))
    return basis, scores.astype(np.float32), min(rank, q)


def _balanced_nested(data: dict[str, Any], sizes: list[int]) -> dict[int, np.ndarray]:
    families = data["family"].astype(str)
    ids = data["base_trial_id"].astype(str)
    train = np.flatnonzero(data["split"].astype(str) == "train")
    output = {}
    for size in sizes:
        per_family = size // 5
        chosen = []
        for family in sorted(set(families[train])):
            current = [int(index) for index in train if families[index] == family]
            current.sort(
                key=lambda index: hashlib.sha256(
                    f"nested:{ids[index]}".encode()
                ).hexdigest()
            )
            chosen.extend(current[:per_family])
        output[size] = np.asarray(sorted(chosen), dtype=int)
    return output


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-20
    )


def _knn_predict(
    latent: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    neighbors: int = 16,
) -> np.ndarray:
    mean = latent[train].mean(axis=0, keepdims=True)
    scale = latent[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    fit = (latent[train] - mean) / scale
    held = (latent[validation] - mean) / scale
    distance = (
        np.sum(held**2, axis=1, keepdims=True)
        + np.sum(fit**2, axis=1)[None]
        - 2 * held @ fit.T
    )
    count = min(neighbors, len(train))
    nearest = np.argpartition(distance, count - 1, axis=1)[:, :count]
    selected = np.take_along_axis(distance, nearest, axis=1)
    weights = 1.0 / np.maximum(selected, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkt->nt", weights, target[train][nearest]).astype(np.float32)


def _scaling(context: Any) -> dict[str, Any]:
    verify_stage_freeze(context.root, context.config, CAPTURE_FREEZE_PATH)
    data = _load_features(context.root)
    section = context.config["causal_geometry_v13"]
    sizes = [int(value) for value in section["train_sizes"]]
    dimensions = [int(value) for value in section["dimensions"]]
    nested = _balanced_nested(data, sizes)
    validation_all = np.flatnonzero(data["split"].astype(str) == "validation")
    families = data["family"].astype(str)
    panel_per_family = int(section["evaluation_panel_per_family"])
    validation = np.concatenate(
        [
            np.asarray(
                [index for index in validation_all if families[index] == family][
                    :panel_per_family
                ]
            )
            for family in sorted(set(families[validation_all]))
        ]
    )
    target = _target_bundle(data)
    current_j = data["endpoint__current_j_perturbed"].astype(np.float32)
    rows = []
    for size in sizes:
        train = nested[size]
        methods = {
            "global_joint_pca": (data["features"].astype(np.float32), "pca"),
            "architecture_resolved_pca": (data["layerwise"].astype(np.float32), "pca"),
            "global_causal_weighted": (data["layerwise"].astype(np.float32), "causal"),
        }
        for method, (values, kind) in methods.items():
            maximum = min(max(dimensions), len(train) - 1, values.shape[1])
            basis, scores, rank = _fit_basis(values, target, train, kind, maximum)
            for dimension in dimensions:
                if dimension > rank:
                    rows.append(
                        {
                            "train_size": size,
                            "method": method,
                            "dimension": dimension,
                            "status": "NOT_IDENTIFIED_RANK_LIMIT",
                            "empirical_rank": rank,
                        }
                    )
                    continue
                latent = scores[:, :dimension]
                predicted = _knn_predict(
                    latent, target, train, validation, neighbors=16
                )
                cursor = 0
                metrics = {}
                for horizon in (1, 2, 4):
                    width = 256
                    case_direction = _cosine_rows(
                        predicted[:, cursor : cursor + width],
                        target[validation, cursor : cursor + width],
                    )
                    predicted_j = predicted[:, cursor : cursor + width]
                    teacher_j = target[validation, cursor : cursor + width]
                    metrics[f"direction_h{horizon}"] = float(case_direction.mean())
                    metrics[f"magnitude_h{horizon}"] = float(
                        np.linalg.norm(predicted_j)
                        / max(float(np.linalg.norm(teacher_j)), 1e-20)
                    )
                    metrics[f"direction_h{horizon}_by_case"] = case_direction.astype(
                        np.float32
                    ).tolist()
                    cursor += width
                    logit_width = 64
                    metrics[f"output_direction_h{horizon}"] = float(
                        _cosine_rows(
                            predicted[:, cursor : cursor + logit_width],
                            target[validation, cursor : cursor + logit_width],
                        ).mean()
                    )
                    metrics[f"output_magnitude_h{horizon}"] = float(
                        np.linalg.norm(predicted[:, cursor : cursor + logit_width])
                        / max(
                            float(
                                np.linalg.norm(
                                    target[
                                        validation,
                                        cursor : cursor + logit_width,
                                    ]
                                )
                            ),
                            1e-20,
                        )
                    )
                    cursor += logit_width
                    semantic_pred = predicted[:, cursor]
                    semantic_true = target[validation, cursor]
                    metrics[f"semantic_sign_h{horizon}"] = float(
                        np.mean(np.sign(semantic_pred) == np.sign(semantic_true))
                    )
                    metrics[f"semantic_cosine_h{horizon}"] = _safe_cosine(
                        semantic_pred, semantic_true
                    )
                    cursor += 1
                rows.append(
                    {
                        "train_size": size,
                        "method": method,
                        "dimension": dimension,
                        "status": "IDENTIFIED",
                        "empirical_rank": rank,
                        **metrics,
                    }
                )
        # State-specific causal basis is fit separately per validation state on its nearest J neighbors.
        _, _, standardized_j = _standardize(current_j, train)
        local_models: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for family in sorted(set(families[validation])):
            source = next(index for index in validation if families[index] == family)
            distance = np.linalg.norm(
                standardized_j[train] - standardized_j[source], axis=1
            )
            local = train[np.argsort(distance)[: min(size, 2400)]]
            maximum = min(max(dimensions), len(local) - 1)
            _, scores, rank = _fit_basis(
                data["layerwise"].astype(np.float32),
                target,
                local,
                "causal",
                maximum,
            )
            local_models[family] = (local, scores, rank)
        for dimension in dimensions:
            local_rank = int(min(value[2] for value in local_models.values()))
            if dimension >= size or dimension > local_rank:
                rows.append(
                    {
                        "train_size": size,
                        "method": "local_causal_basis",
                        "dimension": dimension,
                        "status": "NOT_IDENTIFIED_RANK_LIMIT",
                        "empirical_rank": local_rank,
                    }
                )
                continue
            predictions = []
            ranks = []
            for source in validation:
                local, scores, rank = local_models[families[source]]
                ranks.append(rank)
                predictions.append(
                    _knn_predict(
                        scores[:, :dimension],
                        target,
                        local,
                        np.asarray([source]),
                        neighbors=16,
                    )[0]
                )
            predicted = np.stack(predictions)
            cursor, metrics = 0, {}
            for horizon in (1, 2, 4):
                width = 256
                case_direction = _cosine_rows(
                    predicted[:, cursor : cursor + width],
                    target[validation, cursor : cursor + width],
                )
                predicted_j = predicted[:, cursor : cursor + width]
                teacher_j = target[validation, cursor : cursor + width]
                metrics[f"direction_h{horizon}"] = float(case_direction.mean())
                metrics[f"magnitude_h{horizon}"] = float(
                    np.linalg.norm(predicted_j)
                    / max(float(np.linalg.norm(teacher_j)), 1e-20)
                )
                metrics[f"direction_h{horizon}_by_case"] = case_direction.astype(
                    np.float32
                ).tolist()
                cursor += width
                metrics[f"output_direction_h{horizon}"] = float(
                    _cosine_rows(
                        predicted[:, cursor : cursor + 64],
                        target[validation, cursor : cursor + 64],
                    ).mean()
                )
                metrics[f"output_magnitude_h{horizon}"] = float(
                    np.linalg.norm(predicted[:, cursor : cursor + 64])
                    / max(
                        float(np.linalg.norm(target[validation, cursor : cursor + 64])),
                        1e-20,
                    )
                )
                cursor += 64
                metrics[f"semantic_sign_h{horizon}"] = float(
                    np.mean(
                        np.sign(predicted[:, cursor])
                        == np.sign(target[validation, cursor])
                    )
                )
                metrics[f"semantic_cosine_h{horizon}"] = _safe_cosine(
                    predicted[:, cursor], target[validation, cursor]
                )
                cursor += 1
            rows.append(
                {
                    "train_size": size,
                    "method": "local_causal_basis",
                    "dimension": dimension,
                    "status": "IDENTIFIED",
                    "empirical_rank": int(min(ranks)),
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    path = context.root / SCALING_RECORDS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    epsilon = float(section["saturation"]["epsilon"])
    curves = {}
    for method in sorted(frame["method"].unique()):
        for dimension in (128, 256, 512, 1024):
            subset = frame[
                (frame.method == method)
                & (frame.dimension == dimension)
                & (frame.status == "IDENTIFIED")
            ].sort_values("train_size")
            values = subset["direction_h1"].astype(float).tolist()
            sizes_found = subset["train_size"].astype(int).tolist()
            deltas = np.diff(values).tolist() if len(values) > 1 else []
            arrays = [
                np.asarray(value, dtype=float)
                for value in subset["direction_h1_by_case"].tolist()
            ]
            rng = np.random.default_rng(20270922 + dimension)
            doubling_ci = []
            for left, right in zip(arrays[:-1], arrays[1:], strict=True):
                paired = right - left
                indices = rng.integers(
                    0,
                    len(paired),
                    size=(int(section["bootstrap_resamples"]), len(paired)),
                )
                sampled = paired[indices].mean(axis=1)
                doubling_ci.append(
                    {
                        "lower": float(np.quantile(sampled, 0.025)),
                        "upper": float(np.quantile(sampled, 0.975)),
                    }
                )
            saturated = (
                len(deltas) >= 2
                and all(abs(value) < epsilon for value in deltas[-2:])
                and all(
                    value["lower"] <= epsilon and value["upper"] >= -epsilon
                    for value in doubling_ci[-2:]
                )
            )
            curves[f"{method}/d{dimension}"] = {
                "train_sizes": sizes_found,
                "direction_h1": values,
                "doubling_deltas": deltas,
                "doubling_bootstrap_ci": doubling_ci,
                "status": "SATURATION_OBSERVED" if saturated else "DATA_LIMITED",
            }
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "records": str(SCALING_RECORDS),
        "records_sha256": sha256_file(path),
        "saturation_epsilon": epsilon,
        "local_basis_definition": "family-stratified nearest-J neighborhood centered on one frozen validation representative per family; shared within family",
        "curves": curves,
    }
    write_json_atomic(context.root / SCALING_SUMMARY, summary)
    return summary


def _direction_block(
    block: torch.Tensor,
    train: np.ndarray,
    scores: np.ndarray,
    requested: np.ndarray,
    device: torch.device,
    chunk: int,
) -> torch.Tensor:
    fit = scores[train].astype(np.float64)
    fit -= fit.mean(axis=0, keepdims=True)
    alpha = requested.astype(np.float64) @ np.linalg.pinv(fit, rcond=1e-10)
    flat = block.reshape(block.shape[0], -1)
    output = torch.empty((len(alpha), flat.shape[1]), dtype=torch.bfloat16)
    alpha_device = torch.from_numpy(alpha.astype(np.float32)).to(device)
    for start in range(0, flat.shape[1], chunk):
        stop = min(start + chunk, flat.shape[1])
        values = flat[train, start:stop].float()
        mean = values.double().mean(0).float()
        output[:, start:stop] = (
            (alpha_device @ (values - mean).to(device)).cpu().to(torch.bfloat16)
        )
    return output.reshape((len(alpha), *block.shape[1:]))


def _prepare_directions(context: Any) -> dict[str, Any]:
    verify_stage_freeze(context.root, context.config, CAPTURE_FREEZE_PATH)
    data = _load_features(context.root)
    raw, raw_ids, _, _ = _load_raw_deltas(context.root)
    lookup = {value: index for index, value in enumerate(raw_ids)}
    order = np.asarray([lookup[value] for value in data["base_trial_id"].astype(str)])
    raw = {name: value[order] for name, value in raw.items()}
    train = np.flatnonzero(data["split"].astype(str) == "train")
    x = data["layerwise"].astype(np.float32)
    target = _target_bundle(data)
    _, score_scale, standardized = _standardize(x, train)
    maximum = int(
        context.config["causal_geometry_v13"]["jvp"]["maximum_probe_directions"]
    )
    per_family = int(np.ceil(maximum / 5))
    pca_basis, _, _ = _fit_basis(x, target, train, "pca", per_family)
    causal_basis, _, _ = _fit_basis(x, target, train, "causal", per_family)
    variance = standardized[train].var(axis=0)
    low = np.argsort(variance)[:per_family]
    low_basis = np.eye(x.shape[1], dtype=np.float32)[low]
    rng = np.random.default_rng(20270921)
    random_basis = rng.normal(size=(per_family, x.shape[1])).astype(np.float32)
    random_basis /= np.maximum(
        np.linalg.norm(random_basis, axis=1, keepdims=True), 1e-12
    )
    rank = data["features"].shape[1]
    balanced_indices = []
    for offset in range(per_family):
        channel = offset % 3
        balanced_indices.append(channel * rank + (offset // 3) % rank)
    balanced_basis = np.eye(x.shape[1], dtype=np.float32)[balanced_indices]
    pools = {
        "high_variance_pca": pca_basis,
        "causal_weighted": causal_basis,
        "random_raw": random_basis,
        "low_variance": low_basis,
        "architecture_balanced": balanced_basis,
    }
    rows, labels = [], []
    for offset in range(per_family):
        for name in pools:
            if len(rows) >= maximum:
                break
            rows.append(pools[name][offset])
            labels.append(name)
    score_directions = np.stack(rows).astype(np.float32)
    typical = float(np.sqrt(np.mean(np.sum(standardized[train] ** 2, axis=1))))
    score_directions *= typical / np.maximum(
        np.linalg.norm(score_directions, axis=1, keepdims=True), 1e-12
    )
    score_directions *= score_scale
    device = torch.device(
        f"cuda:{int(context.config['model'].get('device', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    chunk = int(
        context.config["causal_geometry_v13"]["decoder"]["reconstruction_chunk"]
    )
    directions = {}
    channel_energy = {}
    for index, name in enumerate(("recurrent", "conv", "kv")):
        current = score_directions[:, index * rank : (index + 1) * rank]
        directions[name] = _direction_block(
            raw[name],
            train,
            x[:, index * rank : (index + 1) * rank],
            current,
            device,
            chunk,
        )
        channel_energy[name] = np.sum(current.astype(np.float64) ** 2, axis=1).tolist()
    path = context.root / DIRECTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "v13_mixed_empirical_raw_directions_bf16",
            "labels": labels,
            "score_directions": score_directions,
            **directions,
        },
        path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "operator_domain": "512 frozen mixed empirical raw-state directions",
        "probe_direction_count": len(labels),
        "probe_families": {
            name: [index for index, label in enumerate(labels) if label == name]
            for name in pools
        },
        "channel_score_energy": channel_energy,
        "artifact": str(DIRECTIONS),
        "artifact_sha256": sha256_file(path),
        "full_raw_jacobian_claimed": False,
    }
    write_json_atomic(context.root / DIRECTION_SUMMARY, summary)
    return summary


def _freeze_jvp(context: Any) -> dict[str, Any]:
    return build_stage_freeze(
        context.root,
        context.config,
        path=GEOMETRY_FREEZE,
        purpose="freeze V13 mixed probe operator, scaling records, target bundles, and atlas anchors",
        inputs=[
            FEATURES,
            FEATURE_SUMMARY,
            SCALING_RECORDS,
            SCALING_SUMMARY,
            DIRECTIONS,
            DIRECTION_SUMMARY,
        ],
        payload={
            "probe_sizes": context.config["causal_geometry_v13"]["jvp"]["probe_sizes"],
            "target_bundles": context.config["causal_geometry_v13"]["target_bundles"],
            "confirmatory_used_for_selection": False,
        },
    )


def _pair_metadata(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for capture in _capture_manifests(root):
        for line in (root / capture["pair_records"]).read_text().splitlines():
            row = json.loads(line)
            output[str(row["base_trial_id"])] = row
    return output


def _advance_cache(bundle: Any, cache: Any, token: int, length: int) -> Any:
    device = next(bundle.hf_model.parameters()).device
    with torch.no_grad():
        output = bundle.hf_model(
            input_ids=torch.tensor([[token]], device=device),
            attention_mask=torch.ones((1, length + 1), dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
    return output.past_key_values


def _anchor_ids(data: dict[str, Any], per_family: int) -> list[str]:
    ids = data["base_trial_id"].astype(str)
    families = data["family"].astype(str)
    train = np.flatnonzero(data["split"].astype(str) == "train")
    output = []
    for family in sorted(set(families[train])):
        current = [ids[index] for index in train if families[index] == family]
        current.sort(
            key=lambda value: hashlib.sha256(f"jvp:{value}".encode()).hexdigest()
        )
        output.extend(current[:per_family])
    return output


def _run_jvp(context: Any) -> dict[str, Any]:
    freeze = verify_stage_freeze(context.root, context.config, GEOMETRY_FREEZE)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    directions = torch.load(
        context.root / DIRECTIONS, map_location="cpu", weights_only=False
    )
    data = _load_features(context.root)
    metadata = _pair_metadata(context.root)
    tasks = {
        task.example_id: task for _, task in load_tasks(context.root / SELECTION_PATH)
    }
    bundle = load_model_bundle(context.config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    section = context.config["causal_geometry_v13"]["jvp"]
    anchors = _anchor_ids(data, int(section["anchors_per_family"]))
    positions = [int(value) for value in section["token_positions"]]
    horizons = [int(value) for value in section["target_horizons"]]
    max_count = int(section["maximum_probe_directions"])
    probe_sizes = [int(value) for value in section["probe_sizes"]]
    j_variance = (
        data["endpoint__perturbed_j_h1"].astype(np.float32)
        - data["endpoint__clean_j_h1"].astype(np.float32)
    ).var(0)
    selected_j = np.argsort(-j_variance)[: int(section["selected_j_count"])]
    device = next(bundle.hf_model.parameters()).device
    selected_j_tensor = torch.as_tensor(selected_j, device=device)
    matrix_root = context.root / JVP_MATRIX_ROOT
    matrix_root.mkdir(parents=True, exist_ok=True)
    records = []
    labels = list(directions["labels"])
    sensitivity: dict[str, list[float]] = defaultdict(list)
    for anchor_index, base_id in enumerate(anchors):
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=max(positions) + max(horizons),
            measured_layers=measured,
            dense_map=dense_map,
        )
        for position in positions:
            base_cache = clone_hybrid_cache(clean["cache"])
            current_length = int(clean["prompt_length"])
            for offset in range(position):
                base_cache = _advance_cache(
                    bundle, base_cache, tokens[offset], current_length
                )
                current_length += 1
            local_tokens = tokens[position : position + max(horizons)]
            with torch.no_grad():
                probe = bundle.hf_model(
                    input_ids=torch.tensor([[local_tokens[0]]], device=device),
                    attention_mask=torch.ones(
                        (1, current_length + 1), dtype=torch.long, device=device
                    ),
                    past_key_values=clone_hybrid_cache(base_cache),
                    use_cache=True,
                )
            top_logits = torch.topk(
                probe.logits[0, -1], int(section["selected_logit_count"])
            ).indices.tolist()
            semantic_ids = _semantic_ids(
                bundle.tokenizer,
                task.semantic_actions[min(position, len(task.semantic_actions) - 1)],
            )
            selected_logits = list(dict.fromkeys([*semantic_ids, *top_logits]))[
                : int(section["selected_logit_count"])
            ]
            columns = []
            slices: dict[str, list[int]] = defaultdict(list)
            for direction_index in range(max_count):
                row = {
                    name: directions[name][direction_index]
                    for name in ("recurrent", "conv", "kv")
                }

                def target(
                    epsilon: torch.Tensor,
                    base_cache: Any = base_cache,
                    row: dict[str, torch.Tensor] = row,
                    local_tokens: list[int] = local_tokens,
                    current_length: int = current_length,
                    selected_logits: list[int] = selected_logits,
                    slices_ref: dict[str, list[int]] = slices,
                    record_slices: bool = direction_index == 0,
                ) -> torch.Tensor:
                    cache = _apply_direction(
                        base_cache,
                        epsilon,
                        row,
                        recurrent_layers,
                        attention_layers,
                    )
                    parts = []
                    for step, token in enumerate(local_tokens, start=1):
                        with ActivationRecorder(
                            bundle.layers,
                            at=[int(value) for value in section["workspace_layers"]],
                            clone=False,
                            detach=False,
                        ) as recorder:
                            output = bundle.hf_model(
                                input_ids=torch.tensor([[token]], device=device),
                                attention_mask=torch.ones(
                                    (1, current_length + step),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                past_key_values=cache,
                                use_cache=True,
                            )
                        cache = output.past_key_values
                        if step in horizons:
                            start = sum(part.numel() for part in parts)
                            hidden = recorder.activations[main_layer][0, -1].float()
                            j_state = dense_map.dense_state(hidden, main_layer)
                            parts.append(j_state[selected_j_tensor])
                            if record_slices:
                                slices_ref[f"j_h{step}"].extend(
                                    range(start, start + len(selected_j))
                                )
                            start += len(selected_j)
                            parts.append(output.logits[0, -1, selected_logits].float())
                            if record_slices:
                                slices_ref[f"logits_h{step}"].extend(
                                    range(start, start + len(selected_logits))
                                )
                            start += len(selected_logits)
                            semantic = torch.log_softmax(
                                output.logits[0, -1].float(), dim=-1
                            )[selected_logits]
                            parts.append(semantic)
                            if record_slices:
                                slices_ref[f"semantic_h{step}"].extend(
                                    range(start, start + len(selected_logits))
                                )
                            for workspace_layer in section["workspace_layers"]:
                                workspace = recorder.activations[int(workspace_layer)][
                                    0, -1
                                ].float()[: int(section["selected_workspace_count"])]
                                start = sum(part.numel() for part in parts)
                                parts.append(workspace)
                                if record_slices:
                                    slices_ref[
                                        f"workspace_l{workspace_layer}_h{step}"
                                    ].extend(range(start, start + len(workspace)))
                    return torch.cat(parts)

                epsilon = torch.zeros((), device=device, dtype=torch.float32)
                _, derivative = torch.autograd.functional.jvp(
                    target,
                    epsilon,
                    torch.ones_like(epsilon),
                    create_graph=False,
                    strict=True,
                )
                columns.append(derivative.detach().cpu().numpy().astype(np.float32))
                if direction_index == 0:
                    slices = {
                        name: sorted(set(indices)) for name, indices in slices.items()
                    }
            matrix = np.stack(columns, axis=1)
            column_norms = np.linalg.norm(matrix.astype(np.float64), axis=0)
            for probe_family in sorted(set(labels)):
                mask = np.asarray(
                    [label == probe_family for label in labels], dtype=bool
                )
                sensitivity[probe_family].extend(column_norms[mask].tolist())
            matrix_path = (
                matrix_root / f"anchor_{anchor_index:02d}_position_{position}.npz"
            )
            np.savez_compressed(
                matrix_path,
                matrix=matrix,
                selected_j=selected_j,
                selected_logits=np.asarray(selected_logits),
                slices_json=np.asarray(json.dumps(slices)),
            )
            for probe_size in probe_sizes:
                current = matrix[:, :probe_size]
                spectrum, _ = _spectrum(
                    current, [float(value) for value in section["energy_thresholds"]]
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V13,
                        "protocol_version": PROTOCOL_V13,
                        "base_trial_id": base_id,
                        "prompt_id": pair["prompt_id"],
                        "family": pair["family"],
                        "token_position": position,
                        "probe_family": "mixed",
                        "probe_direction_count": probe_size,
                        "matrix_path": str(matrix_path.relative_to(context.root)),
                        "matrix_sha256": sha256_file(matrix_path),
                        **spectrum,
                    }
                )
            for probe_family in sorted(set(labels)):
                indices = [
                    index for index, label in enumerate(labels) if label == probe_family
                ]
                spectrum, _ = _spectrum(matrix[:, indices], [0.90, 0.95, 0.99])
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V13,
                        "protocol_version": PROTOCOL_V13,
                        "base_trial_id": base_id,
                        "prompt_id": pair["prompt_id"],
                        "family": pair["family"],
                        "token_position": position,
                        "probe_family": probe_family,
                        "probe_direction_count": len(indices),
                        "matrix_path": str(matrix_path.relative_to(context.root)),
                        "matrix_sha256": sha256_file(matrix_path),
                        **spectrum,
                    }
                )
            write_json_atomic(
                context.raw_dir / context.run_id / "jvp_progress.json",
                {
                    "status": "RUNNING",
                    "completed_local_states": anchor_index * len(positions)
                    + position
                    + 1,
                    "total_local_states": len(anchors) * len(positions),
                    "completed_directions_current_state": max_count,
                },
            )
    frame = pd.DataFrame(records)
    path = context.root / JVP_RECORDS
    frame.to_parquet(path, index=False, compression="zstd")
    mixed = frame[frame.probe_family == "mixed"]
    curves = {
        str(size): {
            "median_r90": float(group.rank_90.median()),
            "median_r95": float(group.rank_95.median()),
            "median_r99": float(group.rank_99.median()),
            "mean_stable_rank": float(group.stable_rank.mean()),
            "mean_effective_rank": float(group.effective_rank.mean()),
        }
        for size, group in mixed.groupby("probe_direction_count")
    }
    robustness = {
        str(name): {
            "median_r95": float(group.rank_95.median()),
            "mean_r95": float(group.rank_95.mean()),
        }
        for name, group in frame[frame.probe_family != "mixed"].groupby("probe_family")
    }
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "exact_autograd_jvp": True,
        "full_raw_jacobian_claimed": False,
        "local_state_count": len(anchors) * len(positions),
        "probe_scaling": curves,
        "probe_family_robustness": robustness,
        "probe_family_mean_column_sensitivity": {
            name: float(np.mean(values)) for name, values in sensitivity.items()
        },
        "records": str(JVP_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / JVP_SUMMARY, summary)
    return summary


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left.reshape(-1), right.reshape(-1))
        / max(
            float(np.linalg.norm(left.reshape(-1)) * np.linalg.norm(right.reshape(-1))),
            1e-20,
        )
    )


def _atlas(root: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(root / JVP_RECORDS)
    rows = frame[
        (frame["probe_family"] == "mixed")
        & (frame["probe_direction_count"] == frame["probe_direction_count"].max())
    ]
    output = []
    for _, row in rows.iterrows():
        with np.load(root / row["matrix_path"], allow_pickle=False) as data:
            matrix = data["matrix"].astype(np.float32)
        _, singular, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
        energy = singular**2
        cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-20)
        r95 = int(np.searchsorted(cumulative, 0.95) + 1)
        output.append(
            {
                "base_trial_id": str(row["base_trial_id"]),
                "family": str(row["family"]),
                "token_position": int(row["token_position"]),
                "matrix_path": str(row["matrix_path"]),
                "vh": vh.astype(np.float32),
                "singular": singular.astype(np.float32),
                "r95": r95,
            }
        )
    return output


def _coefficients(
    data: dict[str, Any], directions: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    train = np.flatnonzero(data["split"].astype(str) == "train")
    scores = data["layerwise"].astype(np.float64)
    mean = scores[train].mean(axis=0, keepdims=True)
    operator = np.asarray(directions["score_directions"], dtype=np.float64)
    gram_inverse = np.linalg.pinv(operator @ operator.T, rcond=1e-8)
    coefficients = (scores - mean) @ operator.T @ gram_inverse
    return coefficients.astype(np.float32), mean.astype(np.float32)


def _decoded_delta(
    directions: dict[str, Any], coefficients: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.from_numpy(coefficients.astype(np.float32))
    return tuple(
        torch.tensordot(weight, directions[name].float(), dims=([0], [0])).to(
            torch.bfloat16
        )
        for name in ("recurrent", "conv", "kv")
    )  # type: ignore[return-value]


def _project_static(coefficients: np.ndarray, atlas_row: dict[str, Any]) -> np.ndarray:
    rank = min(32, int(atlas_row["r95"]), atlas_row["vh"].shape[0])
    basis = atlas_row["vh"][:rank]
    return (coefficients @ basis.T @ basis).astype(np.float32)


def _moving_coefficients(
    target: np.ndarray,
    atlas: list[dict[str, Any]],
    anchor_scores: np.ndarray,
    *,
    alpha: float,
    steps: int,
    neighbors: int = 1,
) -> tuple[np.ndarray, list[int]]:
    current = np.zeros_like(target)
    visited = []
    for _ in range(steps):
        distance = np.linalg.norm(anchor_scores - current[None], axis=1)
        nearest = np.argsort(distance)[:neighbors]
        visited.extend(int(index) for index in nearest)
        residual = target - current
        if neighbors == 1:
            projected = _project_static(residual, atlas[int(nearest[0])])
        else:
            weights = 1.0 / np.maximum(distance[nearest], 1e-6)
            weights /= weights.sum()
            projection = np.zeros((len(target), len(target)), dtype=np.float32)
            for weight, index in zip(weights, nearest, strict=True):
                rank = min(32, int(atlas[int(index)]["r95"]))
                basis = atlas[int(index)]["vh"][:rank]
                projection += float(weight) * (basis.T @ basis)
            projected = residual @ projection
        current = current + alpha * projected
        if np.linalg.norm(residual) <= 0.01 * max(np.linalg.norm(target), 1e-20):
            break
    return current.astype(np.float32), visited


def _panel_ids(data: dict[str, Any], split: str, per_family: int) -> list[str]:
    ids = data["base_trial_id"].astype(str)
    families = data["family"].astype(str)
    selected = np.flatnonzero(data["split"].astype(str) == split)
    output = []
    for family in sorted(set(families[selected])):
        current = [ids[index] for index in selected if families[index] == family]
        current.sort(
            key=lambda value: hashlib.sha256(
                f"oracle:{split}:{value}".encode()
            ).hexdigest()
        )
        output.extend(current[:per_family])
    return output


def _run_oracle(context: Any, split: str) -> dict[str, Any]:
    freeze = verify_stage_freeze(context.root, context.config, GEOMETRY_FREEZE)
    data = _load_features(context.root)
    directions = torch.load(
        context.root / DIRECTIONS, map_location="cpu", weights_only=False
    )
    coefficients, _ = _coefficients(data, directions)
    ids = data["base_trial_id"].astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    atlas = _atlas(context.root)
    anchor_indices = np.asarray([lookup[row["base_trial_id"]] for row in atlas])
    anchor_scores = coefficients[anchor_indices]
    metadata = _pair_metadata(context.root)
    tasks = {
        task.example_id: task for _, task in load_tasks(context.root / SELECTION_PATH)
    }
    raw, raw_ids, _, _ = _load_raw_deltas(context.root)
    raw_lookup = {value: index for index, value in enumerate(raw_ids)}
    section = context.config["causal_geometry_v13"]
    per_family = int(
        section[
            "evaluation_panel_per_family"
            if split == "validation"
            else "confirmatory_panel_per_family"
        ]
    )
    panel = _panel_ids(data, split, per_family)
    if split == "final_test":
        finalist_path = (
            context.root / "artifacts/causal_geometry_v13_finalists.freeze.json"
        )
        finalist = verify_stage_freeze(context.root, context.config, finalist_path)
        alphas = [float(finalist["moving_alpha"])]
        moving_methods = [str(finalist["moving_method"])]
    else:
        alphas = [float(value) for value in section["moving_tangent"]["alphas"]]
        moving_methods = [
            "moving_tangent_oracle",
            "moving_tangent_interpolated_oracle",
        ]
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    labels = list(directions["labels"])
    pca_mask = np.asarray([label == "high_variance_pca" for label in labels])
    causal_mask = np.asarray([label == "causal_weighted" for label in labels])
    rows = []
    for case_index, base_id in enumerate(panel):
        index = lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=8,
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        raw_index = raw_lookup[base_id]
        raw_delta = (
            raw["recurrent"][raw_index],
            raw["conv"][raw_index],
            raw["kv"][raw_index],
        )
        teacher_cache = apply_decoded_state(
            clean["cache"],
            raw_delta,
            recurrent_layers=recurrent_layers,
            attention_layers=attention_layers,
            prompt_length=clean["prompt_length"],
        )
        teacher_trajectory = _teacher_forced_trajectory(
            bundle,
            teacher_cache,
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        target = coefficients[index]
        nearest = int(np.argmin(np.linalg.norm(anchor_scores - target[None], axis=1)))
        candidates: list[tuple[str, float | None, np.ndarray | None, list[int]]] = [
            ("raw_teacher_intervention", None, None, []),
            ("global_pca_oracle", None, target * pca_mask, []),
            ("global_causal_basis_oracle", None, target * causal_mask, []),
            (
                "static_local_causal_oracle",
                None,
                _project_static(target, atlas[nearest]),
                [nearest],
            ),
        ]
        for alpha in alphas:
            for moving_method in moving_methods:
                value, visited = _moving_coefficients(
                    target,
                    atlas,
                    anchor_scores,
                    alpha=alpha,
                    steps=int(section["moving_tangent"]["maximum_steps"]),
                    neighbors=(
                        3
                        if moving_method == "moving_tangent_interpolated_oracle"
                        else 1
                    ),
                )
                candidates.append((moving_method, alpha, value, visited))
        for method, alpha, candidate, visited in candidates:
            if candidate is None:
                trajectory = teacher_trajectory
            else:
                decoded = _decoded_delta(directions, candidate)
                cache = apply_decoded_state(
                    clean["cache"],
                    decoded,
                    recurrent_layers=recurrent_layers,
                    attention_layers=attention_layers,
                    prompt_length=clean["prompt_length"],
                )
                trajectory = _teacher_forced_trajectory(
                    bundle,
                    cache,
                    tokens,
                    prompt_length=clean["prompt_length"],
                    measured_layers=measured,
                    dense_map=dense_map,
                )
            for horizon in (1, 2, 4, 8):
                semantic_index = min(horizon - 1, len(task.semantic_actions) - 1)
                metrics = _effect_metrics(
                    clean_trajectory,
                    teacher_trajectory,
                    trajectory,
                    main_layer=main_layer,
                    horizon=horizon,
                    semantic_ids=_semantic_ids(
                        bundle.tokenizer, task.semantic_actions[semantic_index]
                    ),
                )
                teacher_effect = (
                    teacher_trajectory["j"][main_layer][horizon - 1]
                    - clean_trajectory["j"][main_layer][horizon - 1]
                )
                effect = (
                    trajectory["j"][main_layer][horizon - 1]
                    - clean_trajectory["j"][main_layer][horizon - 1]
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION_V13,
                        "protocol_version": PROTOCOL_V13,
                        "source_freeze_digest": freeze["freeze_digest"],
                        "split": split,
                        "base_trial_id": base_id,
                        "family": pair["family"],
                        "method": method,
                        "alpha": alpha,
                        "horizon": horizon,
                        "visited_atlas_states": len(set(visited)),
                        "visited_atlas_indices": sorted(set(visited)),
                        **metrics,
                        **_continuous_semantic(teacher_effect, effect, 10),
                    }
                )
        write_json_atomic(
            context.raw_dir / context.run_id / f"oracle_{split}_progress.json",
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(panel),
            },
        )
    frame = pd.DataFrame(rows)
    suffix = "development" if split == "validation" else "confirmatory"
    path = (
        context.root
        / f"results/v13/processed/moving_tangent_oracle_{suffix}_v13.parquet"
    )
    frame.to_parquet(path, index=False, compression="zstd")
    summary_rows = (
        frame.groupby(["method", "alpha", "horizon"], dropna=False)
        .agg(
            direction=("direction_cosine", "mean"),
            magnitude=("magnitude_ratio", "mean"),
            semantic_continuous=("semantic_vector_cosine", "mean"),
            semantic_legacy=("semantic_delta_agreement", "mean"),
            output=("output_direction_cosine", "mean"),
            sign=("task_decision_sign_agreement", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "split": split,
        "independent_confirmatory": split == "final_test",
        "panel_count": len(panel),
        "records": str(path.relative_to(context.root)),
        "records_sha256": sha256_file(path),
        "summary": summary_rows,
    }
    write_json_atomic(
        context.root / f"results/v13/processed/moving_tangent_oracle_{suffix}_v13.json",
        summary,
    )
    return summary


def _freeze_finalists(context: Any) -> dict[str, Any]:
    path = (
        context.root
        / "results/v13/processed/moving_tangent_oracle_development_v13.parquet"
    )
    frame = pd.read_parquet(path)
    moving = frame[
        frame.method.str.startswith("moving_tangent") & frame.horizon.isin([1, 2, 4, 8])
    ].copy()
    moving["score"] = (
        moving["direction_cosine"]
        + moving["output_direction_cosine"]
        + moving["semantic_vector_cosine"]
    ) / 3
    scores = (
        moving.groupby(["method", "alpha"])["score"].mean().sort_values(ascending=False)
    )
    method, alpha = scores.index[0]
    return build_stage_freeze(
        context.root,
        context.config,
        path=Path("artifacts/causal_geometry_v13_finalists.freeze.json"),
        purpose="freeze V13 moving-tangent alpha before independent confirmation",
        inputs=[
            Path("results/v13/processed/moving_tangent_oracle_development_v13.parquet")
        ],
        payload={
            "moving_alpha": float(alpha),
            "moving_method": str(method),
            "finalists": [
                "global_pca_oracle",
                "global_causal_basis_oracle",
                "static_local_causal_oracle",
                str(method),
                "raw_teacher_intervention",
            ],
            "confirmatory_used_for_selection": False,
        },
    )


def _run_linearity(context: Any) -> dict[str, Any]:
    verify_stage_freeze(context.root, context.config, GEOMETRY_FREEZE)
    data = _load_features(context.root)
    ids = data["base_trial_id"].astype(str)
    lookup = {value: index for index, value in enumerate(ids)}
    directions = torch.load(
        context.root / DIRECTIONS, map_location="cpu", weights_only=False
    )
    coefficients, _ = _coefficients(data, directions)
    atlas = _atlas(context.root)
    anchor_indices = np.asarray([lookup[row["base_trial_id"]] for row in atlas])
    anchor_scores = coefficients[anchor_indices]
    metadata = _pair_metadata(context.root)
    tasks = {
        task.example_id: task for _, task in load_tasks(context.root / SELECTION_PATH)
    }
    raw, raw_ids, _, _ = _load_raw_deltas(context.root)
    raw_lookup = {value: index for index, value in enumerate(raw_ids)}
    panel = _panel_ids(
        data,
        "validation",
        int(context.config["causal_geometry_v13"]["evaluation_panel_per_family"]),
    )
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    scales = [
        float(value)
        for value in context.config["causal_geometry_v13"]["linearity_scales"]
    ]
    rows = []
    for case_index, base_id in enumerate(panel):
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=4,
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        source = raw_lookup[base_id]
        raw_delta = (
            raw["recurrent"][source],
            raw["conv"][source],
            raw["kv"][source],
        )
        target_coefficients = coefficients[lookup[base_id]]
        nearest = int(
            np.argmin(np.linalg.norm(anchor_scores - target_coefficients[None], axis=1))
        )
        atlas_row = atlas[nearest]
        with np.load(
            context.root / atlas_row["matrix_path"], allow_pickle=False
        ) as payload:
            matrix = payload["matrix"].astype(np.float32)
            slices = json.loads(str(payload["slices_json"].item()))
            selected_j = payload["selected_j"].astype(int)
        for scale in scales:
            scaled = tuple(
                (value.float() * scale).to(torch.bfloat16) for value in raw_delta
            )
            cache = apply_decoded_state(
                clean["cache"],
                scaled,
                recurrent_layers=recurrent_layers,
                attention_layers=attention_layers,
                prompt_length=clean["prompt_length"],
            )
            trajectory = _teacher_forced_trajectory(
                bundle,
                cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            current_nearest = int(
                np.argmin(
                    np.linalg.norm(
                        anchor_scores - scale * target_coefficients[None], axis=1
                    )
                )
            )
            tangent_angles = _principal_angles(
                atlas_row["vh"][:16], atlas[current_nearest]["vh"][:16]
            )
            for horizon in (1, 2, 4):
                actual_j = (
                    trajectory["j"][main_layer][horizon - 1]
                    - clean_trajectory["j"][main_layer][horizon - 1]
                )[selected_j]
                j_indices = np.asarray(slices[f"j_h{horizon}"], dtype=int)
                predicted_j = matrix[j_indices] @ (scale * target_coefficients)
                actual_logits = (
                    trajectory["logits"][horizon - 1]
                    - clean_trajectory["logits"][horizon - 1]
                )
                logit_indices = np.asarray(slices[f"logits_h{horizon}"], dtype=int)
                with np.load(
                    context.root / atlas_row["matrix_path"], allow_pickle=False
                ) as payload:
                    selected_logits = payload["selected_logits"].astype(int)
                predicted_logits = matrix[logit_indices] @ (scale * target_coefficients)
                actual_logits = actual_logits[selected_logits]
                rows.append(
                    {
                        "base_trial_id": base_id,
                        "family": pair["family"],
                        "scale": scale,
                        "horizon": horizon,
                        "j_direction": _safe_cosine(actual_j, predicted_j),
                        "j_relative_linearity_error": float(
                            np.linalg.norm(actual_j - predicted_j)
                            / max(float(np.linalg.norm(actual_j)), 1e-20)
                        ),
                        "output_direction": _safe_cosine(
                            actual_logits, predicted_logits
                        ),
                        "output_relative_linearity_error": float(
                            np.linalg.norm(actual_logits - predicted_logits)
                            / max(float(np.linalg.norm(actual_logits)), 1e-20)
                        ),
                        "tangent_drift_degrees": float(tangent_angles.mean()),
                    }
                )
        write_json_atomic(
            context.raw_dir / context.run_id / "linearity_progress.json",
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(panel),
            },
        )
    frame = pd.DataFrame(rows)
    path = context.root / LINEARITY_RECORDS
    frame.to_parquet(path, index=False, compression="zstd")
    pooled = (
        frame.groupby(["scale", "horizon"])
        .agg(
            j_direction=("j_direction", "mean"),
            output_direction=("output_direction", "mean"),
            j_relative_error=("j_relative_linearity_error", "mean"),
            output_relative_error=("output_relative_linearity_error", "mean"),
            tangent_drift=("tangent_drift_degrees", "mean"),
        )
        .reset_index()
    )
    valid = pooled[
        (pooled.horizon == 1)
        & (pooled.j_direction >= 0.8)
        & (pooled.output_direction >= 0.8)
    ]
    radius = None if valid.empty else float(valid.scale.max())
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "local_linear_radius": radius,
        "criterion": "largest alpha with pooled h1 J and output direction >= 0.8",
        "records": str(LINEARITY_RECORDS),
        "records_sha256": sha256_file(path),
        "pooled": pooled.to_dict("records"),
    }
    write_json_atomic(context.root / LINEARITY_SUMMARY, summary)
    return summary


def _principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(left @ right.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def _analyze(context: Any) -> dict[str, Any]:
    data = _load_features(context.root)
    atlas = _atlas(context.root)
    lookup = {
        value: index for index, value in enumerate(data["base_trial_id"].astype(str))
    }
    tangent_rows = []
    for left_index in range(len(atlas)):
        for right_index in range(left_index + 1, len(atlas)):
            left, right = atlas[left_index], atlas[right_index]
            relation = (
                "same_prompt_successive_position"
                if left["base_trial_id"] == right["base_trial_id"]
                else "same_family"
                if left["family"] == right["family"]
                else "across_family"
            )
            for rank in (4, 8, 16, 32):
                actual = min(rank, left["vh"].shape[0], right["vh"].shape[0])
                angles = _principal_angles(left["vh"][:actual], right["vh"][:actual])
                li, ri = lookup[left["base_trial_id"]], lookup[right["base_trial_id"]]
                left_loading = np.sum(left["vh"][:actual] ** 2, axis=0)
                right_loading = np.sum(right["vh"][:actual] ** 2, axis=0)
                left_top = set(np.argsort(-left_loading)[:actual].tolist())
                right_top = set(np.argsort(-right_loading)[:actual].tolist())
                tangent_rows.append(
                    {
                        "relation": relation,
                        "rank": rank,
                        "mean_principal_angle_degrees": float(angles.mean()),
                        "maximum_principal_angle_degrees": float(angles.max()),
                        "grassmann_distance": float(
                            np.linalg.norm(np.sin(np.radians(angles)))
                        ),
                        "topk_overlap": float(
                            len(left_top & right_top) / max(actual, 1)
                        ),
                        "token_distance": abs(
                            left["token_position"] - right["token_position"]
                        ),
                        "j_state_distance": float(
                            np.linalg.norm(
                                data["endpoint__current_j_perturbed"][li].astype(float)
                                - data["endpoint__current_j_perturbed"][ri].astype(
                                    float
                                )
                            )
                        ),
                        "raw_score_distance": float(
                            np.linalg.norm(
                                data["layerwise"][li].astype(float)
                                - data["layerwise"][ri].astype(float)
                            )
                        ),
                        "semantic_state_distance": float(
                            np.linalg.norm(
                                (
                                    data["endpoint__perturbed_j_h1"][li]
                                    - data["endpoint__clean_j_h1"][li]
                                ).astype(float)
                                - (
                                    data["endpoint__perturbed_j_h1"][ri]
                                    - data["endpoint__clean_j_h1"][ri]
                                ).astype(float)
                            )
                        ),
                    }
                )
    tangent = pd.DataFrame(tangent_rows)
    tangent_path = (
        context.root / "results/v13/processed/causal_tangent_atlas_v13.parquet"
    )
    tangent.to_parquet(tangent_path, index=False, compression="zstd")
    same = tangent[
        (tangent.relation == "same_prompt_successive_position")
        & (tangent["rank"] == 16)
    ]
    across = tangent[(tangent.relation == "across_family") & (tangent["rank"] == 16)]
    rank16 = tangent[tangent["rank"] == 16].copy()
    rank16["same_family_indicator"] = (rank16["relation"] != "across_family").astype(
        float
    )
    tangent_predictors = {
        name: float(
            rank16["mean_principal_angle_degrees"].corr(rank16[name], method="spearman")
        )
        for name in (
            "token_distance",
            "j_state_distance",
            "raw_score_distance",
            "semantic_state_distance",
            "same_family_indicator",
        )
    }

    target_rows = []
    for row in atlas:
        with np.load(context.root / row["matrix_path"], allow_pickle=False) as payload:
            matrix = payload["matrix"].astype(np.float32)
            slices = json.loads(str(payload["slices_json"].item()))
        bundles = {
            "j_only": [
                index
                for name, indices in slices.items()
                if name.startswith("j_")
                for index in indices
            ],
            "j_logits": [
                index
                for name, indices in slices.items()
                if name.startswith(("j_", "logits_"))
                for index in indices
            ],
            "j_logits_semantic": [
                index
                for name, indices in slices.items()
                if name.startswith(("j_", "logits_", "semantic_"))
                for index in indices
            ],
            "complete_workspace": list(range(matrix.shape[0])),
        }
        for name, indices in bundles.items():
            spectrum, _ = _spectrum(matrix[np.asarray(indices)], [0.90, 0.95, 0.99])
            target_rows.append(
                {
                    "base_trial_id": row["base_trial_id"],
                    "token_position": row["token_position"],
                    "target_bundle": name,
                    **{
                        key: spectrum[key]
                        for key in (
                            "stable_rank",
                            "effective_rank",
                            "rank_90",
                            "rank_95",
                            "rank_99",
                        )
                    },
                }
            )
    target_frame = pd.DataFrame(target_rows)
    target_path = (
        context.root / "results/v13/processed/causal_target_completeness_v13.parquet"
    )
    target_frame.to_parquet(target_path, index=False, compression="zstd")

    channel_energy = np.stack(
        [
            np.asarray(
                json.loads((context.root / DIRECTION_SUMMARY).read_text())[
                    "channel_score_energy"
                ][name]
            )
            for name in ("recurrent", "conv", "kv")
        ],
        axis=1,
    )
    fractions = channel_energy / np.maximum(
        channel_energy.sum(axis=1, keepdims=True), 1e-20
    )
    mixed_fraction = float(np.mean(np.sum(fractions > 0.10, axis=1) >= 2))
    channel_rotation: dict[str, float | None] = {}
    for channel_index, channel in enumerate(("recurrent", "conv", "kv")):
        indices = np.flatnonzero(np.argmax(fractions, axis=1) == channel_index)
        angles_by_channel = []
        for left in atlas:
            for right in atlas:
                if (
                    left["base_trial_id"] != right["base_trial_id"]
                    or left["token_position"] >= right["token_position"]
                ):
                    continue
                with np.load(
                    context.root / left["matrix_path"], allow_pickle=False
                ) as payload:
                    left_matrix = payload["matrix"][:, indices]
                with np.load(
                    context.root / right["matrix_path"], allow_pickle=False
                ) as payload:
                    right_matrix = payload["matrix"][:, indices]
                _, _, left_vh = np.linalg.svd(left_matrix, full_matrices=False)
                _, _, right_vh = np.linalg.svd(right_matrix, full_matrices=False)
                rank = min(8, left_vh.shape[0], right_vh.shape[0])
                angles_by_channel.extend(
                    _principal_angles(left_vh[:rank], right_vh[:rank]).tolist()
                )
        channel_rotation[channel] = (
            None if not angles_by_channel else float(np.mean(angles_by_channel))
        )

    finalist = json.loads(
        (
            context.root / "artifacts/causal_geometry_v13_finalists.freeze.json"
        ).read_text()
    )
    development_paths = pd.read_parquet(
        context.root
        / "results/v13/processed/moving_tangent_oracle_development_v13.parquet"
    )
    development_paths = development_paths[
        (development_paths.method == finalist["moving_method"])
        & (development_paths.alpha == float(finalist["moving_alpha"]))
        & (development_paths.horizon == 1)
    ].drop_duplicates("base_trial_id")
    path_rows = []
    for _, path_record in development_paths.iterrows():
        visited = [int(value) for value in path_record["visited_atlas_indices"]]
        union = [
            atlas[index]["vh"][: min(32, atlas[index]["r95"])] for index in visited
        ]
        if not union:
            continue
        matrix = np.concatenate(union, axis=0)
        singular = np.linalg.svd(matrix, compute_uv=False)
        energy = singular**2
        cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-20)
        path_rows.append(
            {
                "base_trial_id": str(path_record["base_trial_id"]),
                "visited_atlas_states": len(visited),
                "instantaneous_r95": float(
                    np.median([atlas[index]["r95"] for index in visited])
                ),
                "cumulative_path_r95": int(np.searchsorted(cumulative, 0.95) + 1),
                "cumulative_effective_rank": float(
                    np.exp(
                        -np.sum(
                            (energy / energy.sum())
                            * np.log(np.maximum(energy / energy.sum(), 1e-20))
                        )
                    )
                ),
            }
        )
    path_frame = pd.DataFrame(path_rows)
    path_path = context.root / "results/v13/processed/causal_path_dimension_v13.parquet"
    path_frame.to_parquet(path_path, index=False, compression="zstd")

    oracle_confirm = json.loads(
        (
            context.root
            / "results/v13/processed/moving_tangent_oracle_confirmatory_v13.json"
        ).read_text()
    )
    confirm_frame = pd.read_parquet(context.root / oracle_confirm["records"])
    gates = context.config["causal_geometry_v13"]["causal_gates"]
    aggregated = (
        confirm_frame.groupby(["method", "horizon"])
        .agg(
            direction=("direction_cosine", "mean"),
            magnitude=("magnitude_ratio", "mean"),
            semantic_continuous=("semantic_vector_cosine", "mean"),
            semantic_legacy=("semantic_delta_agreement", "mean"),
            output=("output_direction_cosine", "mean"),
            sign=("task_decision_sign_agreement", "mean"),
        )
        .reset_index()
    )
    aggregated["gate_pass"] = (
        (aggregated.direction >= gates["direction_cosine_minimum"])
        & (aggregated.magnitude >= gates["magnitude_ratio_minimum"])
        & (aggregated.magnitude <= gates["magnitude_ratio_maximum"])
        & (aggregated.semantic_legacy >= gates["semantic_delta_agreement_minimum"])
        & (aggregated.semantic_continuous >= gates["semantic_delta_agreement_minimum"])
        & (aggregated.output >= gates["output_direction_minimum"])
        & (aggregated.sign >= gates["task_decision_sign_minimum"])
    )
    selected_moving_method = str(finalist["moving_method"])
    moving = aggregated[aggregated.method == selected_moving_method]
    static = aggregated[aggregated.method == "static_local_causal_oracle"]
    global_causal = aggregated[aggregated.method == "global_causal_basis_oracle"]
    moving_pass = len(moving) == 4 and bool(moving.gate_pass.all())
    global_causal_pass = len(global_causal) == 4 and bool(global_causal.gate_pass.all())
    jvp = json.loads((context.root / JVP_SUMMARY).read_text())
    sensitivity = jvp["probe_family_mean_column_sensitivity"]
    low_variance_high_causal = bool(
        sensitivity.get("low_variance", 0.0)
        >= 0.25 * max(sensitivity.get("high_variance_pca", 0.0), 1e-20)
    )
    r95_curve = [
        jvp["probe_scaling"][str(size)]["median_r95"] for size in (64, 128, 256, 512)
    ]
    stable_low_rank = r95_curve[-1] < 0.25 * 512 and r95_curve[-1] < max(
        32, 2 * r95_curve[0]
    )
    high_path = float(path_frame.cumulative_path_r95.median()) > 2 * float(
        path_frame.instantaneous_r95.median()
    )
    scaling = json.loads((context.root / SCALING_SUMMARY).read_text())
    d512 = scaling["curves"].get("global_causal_weighted/d512", {})
    d512_deltas = d512.get("doubling_deltas", [])
    d512_ci = d512.get("doubling_bootstrap_ci", [])
    data_limited = bool(
        d512.get("status") != "SATURATION_OBSERVED"
        and d512_deltas
        and d512_ci
        and float(d512_deltas[-1]) > scaling["saturation_epsilon"]
        and float(d512_ci[-1]["lower"]) > 0.0
    )
    moving_improvement = (
        float(moving.direction.mean()) > float(static.direction.mean()) + 0.02
    )
    target_medians = target_frame.groupby("target_bundle")["rank_95"].median().to_dict()
    semantic_failure = bool(
        not moving.empty
        and float(moving.semantic_continuous.mean())
        < gates["semantic_delta_agreement_minimum"]
    )
    target_omission_material = bool(
        semantic_failure
        and target_medians.get("complete_workspace", 0)
        > 1.25 * max(target_medians.get("j_logits", 0), 1)
    )
    if not stable_low_rank:
        outcome = "V13-A — PROBE-LIMITED"
    elif data_limited:
        outcome = "V13-B — DATA-LIMITED"
    elif global_causal_pass:
        outcome = "V13-C — GLOBAL LOW-RANK CAUSAL STATE"
    elif high_path and not moving_pass:
        outcome = "V13-E — HIGH PATH DIMENSION"
    elif (
        moving_pass
        and moving_improvement
        and float(same.mean_principal_angle_degrees.mean()) >= 30.0
    ):
        outcome = "V13-D — LOCAL CURVED CAUSAL MANIFOLD"
    elif moving_pass:
        outcome = "V13-G — CAUSAL SUFFICIENT STATE"
    else:
        outcome = "V13-F — HIGH-DIMENSIONAL WRITABLE STATE"
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "probe_r95_curve": dict(zip((64, 128, 256, 512), r95_curve, strict=True)),
        "stable_low_local_causal_rank": stable_low_rank,
        "probe_family_mean_column_sensitivity": sensitivity,
        "low_variance_high_causal_directions": low_variance_high_causal,
        "same_prompt_rank16_mean_angle": float(
            same.mean_principal_angle_degrees.mean()
        ),
        "across_family_rank16_mean_angle": float(
            across.mean_principal_angle_degrees.mean()
        ),
        "tangent_angle_spearman_predictors": tangent_predictors,
        "median_instantaneous_r95": float(path_frame.instantaneous_r95.median()),
        "median_cumulative_path_r95": float(path_frame.cumulative_path_r95.median()),
        "high_path_dimension": high_path,
        "target_bundle_median_r95": target_medians,
        "semantic_failure_present": semantic_failure,
        "semantic_failure_consistent_with_target_omission": target_omission_material,
        "cross_channel_mixed_direction_fraction": mixed_fraction,
        "dominant_directions_cross_channel": mixed_fraction > 0.5,
        "same_prompt_tangent_rotation_by_channel_degrees": channel_rotation,
        "moving_tangent_improves_static": moving_improvement,
        "selected_moving_tangent_method": selected_moving_method,
        "selected_moving_tangent_alpha": float(finalist["moving_alpha"]),
        "moving_tangent_passes_all_horizons": moving_pass,
        "global_causal_basis_passes_all_horizons": global_causal_pass,
        "independent_confirmatory_summary": aggregated.to_dict("records"),
        "formal_outcome": outcome,
        "fixed_512_data_limited": data_limited,
        "candidate_causal_sufficient_state": moving_pass,
        "smallest_independently_validated_writable_dimension": (
            512 if moving_pass else None
        ),
        "candidate_complete_replacement_state": False,
        "strict_replacement_status": "ABSOLUTE_REPLACEMENT_NOT_YET_TESTABLE_FROM_CURRENT_DELTA_REPRESENTATION",
        "h2_remains": not moving_pass,
        "h3_authorized": moving_pass,
        "autonomous_controller_authorized": moving_pass,
        "records": {
            "tangent_atlas": str(tangent_path.relative_to(context.root)),
            "target_completeness": str(target_path.relative_to(context.root)),
            "path_dimension": str(path_path.relative_to(context.root)),
        },
    }
    write_json_atomic(context.root / ANALYSIS_SUMMARY, summary)
    return summary


def main() -> None:
    parser = standard_parser("V13 causal geometry", "configs/causal_geometry_v13.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "features",
            "scaling",
            "directions",
            "freeze-jvp",
            "jvp",
            "oracle-development",
            "freeze-finalists",
            "linearity",
            "oracle-confirmatory",
            "analyze",
        ),
    )
    args = parser.parse_args()
    context = initialize_context("geometry-v13", args)
    try:
        verify_base_freeze(context.root, context.config)
        if args.stage == "features":
            result = _build_features(context)
        elif args.stage == "scaling":
            result = _scaling(context)
        elif args.stage == "directions":
            result = _prepare_directions(context)
        elif args.stage == "freeze-jvp":
            result = _freeze_jvp(context)
        elif args.stage == "jvp":
            result = _run_jvp(context)
        elif args.stage == "oracle-development":
            result = _run_oracle(context, "validation")
        elif args.stage == "freeze-finalists":
            result = _freeze_finalists(context)
        elif args.stage == "linearity":
            result = _run_linearity(context)
        elif args.stage == "oracle-confirmatory":
            result = _run_oracle(context, "final_test")
        else:
            result = _analyze(context)
        context.finish("COMPLETED_V13_GEOMETRY_STAGE", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
