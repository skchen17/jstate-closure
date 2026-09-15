"""Structured-sketch compression and conditional-sufficiency screen for v8."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from jclosure.experiments.arch_compression_v7 import _cosine_rows, _paired_ci, _ridge
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v8 import verify_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v8 import CompressionSweepRecord

FEATURES = Path("artifacts/persistent/v8/structured_features_v8.npz")
SUMMARY = Path("results/v8/processed/persistent_state_compression_v8.json")
_SVD_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def structured_sketch(clean: dict[str, Any], perturbed: dict[str, Any]) -> torch.Tensor:
    """Return an 8192D, architecture-aligned deterministic sketch.

    Recurrent matrices contribute low 4x4 two-dimensional Fourier coefficients
    per head, convolution states contribute 64 spatial bins per channel, and the
    predeclared layer-27/head-3 final-token K/V contributes its exact 512 values.
    """

    recurrent = perturbed["recurrent"].float() - clean["recurrent"].float()
    spectrum = torch.fft.rfft2(recurrent, norm="ortho")[:, :, :4, :4]
    recurrent_features = torch.view_as_real(spectrum).reshape(-1)
    conv = perturbed["conv"].float() - clean["conv"].float()
    conv_features = F.adaptive_avg_pool1d(conv.permute(0, 2, 1), 64).reshape(-1)
    primary = []
    for name in ("keys", "values"):
        left = clean["kv"]["27"][name].float()
        right = perturbed["kv"]["27"][name].float()
        primary.append((right[3, -1] - left[3, -1]).reshape(-1))
    output = torch.cat((recurrent_features, conv_features, *primary))
    if output.numel() != 8192:
        raise RuntimeError(
            f"structured v8 sketch has {output.numel()} rather than 8192 values"
        )
    return output


def _capture_manifests(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(
            (
                root / f"results/v8/processed/persistent_capture_{split}_v8.json"
            ).read_text()
        )
        for split in ("train", "validation", "final_test")
    ]


def _raw_selection(
    screen: dict[str, Any], gate: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    pooled = screen["effects"]["pooled"]
    diagnostics = {}
    for label in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        direction = pooled["direction_cosine_to_full"][label]
        magnitude = pooled["magnitude_ratio_to_full"][label]
        output_sign = pooled["output_sign_agreement_to_full"][label]
        passed = bool(
            direction["estimate"] >= float(gate["minimum_causal_direction_cosine"])
            and magnitude["estimate"] >= float(gate["minimum_full_gap_fraction"])
            and output_sign["estimate"] >= 1.0 - float(gate["maximum_output_sign_loss"])
        )
        diagnostics[label] = {
            "direction_cosine": direction,
            "magnitude_ratio": magnitude,
            "output_sign_agreement": output_sign,
            "passed": passed,
        }
        if passed:
            return label, diagnostics
    return None, diagnostics


def _load_raw_deltas(
    context: Any, captures: list[dict[str, Any]]
) -> tuple[
    dict[str, torch.Tensor], list[str], list[str], list[str], list[dict[str, Any]]
]:
    recurrent_rows, conv_rows, kv_rows = [], [], []
    ids, families, splits, shard_hashes = [], [], [], []
    maximum_tokens = 0
    temporary_kv = []
    for capture in captures:
        for declaration in capture["state_shards"]:
            path = context.root / declaration["path"]
            if sha256_file(path) != declaration["sha256"]:
                raise RuntimeError(f"state shard hash mismatch: {path}")
            shard_hashes.append(declaration)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                clean, perturbed = row["clean"], row["perturbed"]
                recurrent_rows.append(perturbed["recurrent"] - clean["recurrent"])
                conv_rows.append(perturbed["conv"] - clean["conv"])
                current_kv = []
                for layer in ("27", "31"):
                    current_kv.append(
                        torch.stack(
                            [
                                perturbed["kv"][layer][name] - clean["kv"][layer][name]
                                for name in ("keys", "values")
                            ]
                        )
                    )
                current = torch.stack(current_kv)
                maximum_tokens = max(maximum_tokens, int(current.shape[-2]))
                temporary_kv.append(current)
                ids.append(str(row["base_trial_id"]))
                families.append(str(row["family"]))
                splits.append(str(row["split"]))
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
            "kv_full": torch.stack(kv_rows),
        },
        ids,
        families,
        splits,
        shard_hashes,
    )


def _dual_pca_scores(
    blocks: list[torch.Tensor], train: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Exact block-normalized dual PCA without materializing a global flatten."""

    count = int(blocks[0].shape[0])
    fit_count = int(len(train))
    total_gram = torch.zeros((fit_count, fit_count), device=device, dtype=torch.float32)
    total_cross = torch.zeros((count, fit_count), device=device, dtype=torch.float32)
    block_products: list[tuple[torch.Tensor, torch.Tensor]] = []
    metadata = []
    for block_index, block in enumerate(blocks):
        flat = block.reshape(count, -1)
        dimension = int(flat.shape[1])
        sum_values = torch.zeros(dimension, dtype=torch.float64)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            sum_values[start:stop] = flat[train, start:stop].double().sum(dim=0)
        mean = (sum_values / fit_count).float()
        squared = 0.0
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            centered = flat[train, start:stop].float() - mean[start:stop]
            squared += float(torch.sum(centered.square()).item())
        scale = max((squared / (fit_count * dimension)) ** 0.5, 1e-12)
        gram = torch.zeros_like(total_gram)
        cross = torch.zeros_like(total_cross)
        normalization = scale * (dimension**0.5)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            values = (flat[:, start:stop].float() - mean[start:stop]) / normalization
            fit = values[train].to(device)
            held = values.to(device)
            gram.add_(fit @ fit.T)
            cross.add_(held @ fit.T)
        total_gram.add_(gram)
        total_cross.add_(cross)
        block_products.append((gram, cross))
        metadata.append(
            {
                "block_index": block_index,
                "elements": dimension,
                "train_rms": scale,
            }
        )

    def scores(
        gram: torch.Tensor, cross: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(values, descending=True)
        values = values[order].clamp_min(0)
        vectors = vectors[:, order]
        keep = values > values[0].clamp_min(1e-20) * 1e-8
        values, vectors = values[keep], vectors[:, keep]
        return cross @ vectors / torch.sqrt(values).clamp_min(1e-12), values

    combined, eigenvalues = scores(total_gram, total_cross)
    structured = [scores(gram, cross)[0] for gram, cross in block_products]
    return (
        combined.cpu().numpy().astype(np.float32),
        torch.cat(structured, dim=1).cpu().numpy().astype(np.float32),
        [
            *metadata,
            {
                "combined_rank": int(combined.shape[1]),
                "largest_eigenvalue": float(eigenvalues[0]),
                "smallest_retained_eigenvalue": float(eigenvalues[-1]),
            },
        ],
    )


def _features(context: Any, freeze: dict[str, Any]) -> None:
    screen = json.loads(
        (
            context.root / "results/v8/processed/structured_component_screen_v8.json"
        ).read_text()
    )
    gate = context.config["persistent_state_v8"]["raw_screen"]
    selected, diagnostics = _raw_selection(screen, gate)
    if selected is None:
        context.finish(
            "GATED_RAW_STATE_SCREEN",
            source_freeze_digest=freeze["freeze_digest"],
            selected_raw_state=None,
            diagnostics=diagnostics,
        )
        return
    captures = _capture_manifests(context.root)
    for capture in captures:
        if capture["source_freeze_digest"] != freeze["freeze_digest"]:
            raise RuntimeError("v8 capture freeze mismatch")
    raw, ids, families, splits, shard_hashes = _load_raw_deltas(context, captures)
    train = np.flatnonzero(np.asarray(splits) == "train")
    recurrent_layers = captures[0]["architecture"]["recurrent_layers"]
    layer_to_index = {int(layer): index for index, layer in enumerate(recurrent_layers)}
    primary = raw["kv_full"][:, 0, :, 3, -1, :]
    definitions: dict[str, list[torch.Tensor]] = {
        "R1": [raw["recurrent"][:, [layer_to_index[30]]]],
        "R2": [
            raw["recurrent"][:, [layer_to_index[layer] for layer in (28, 29, 30)]],
        ],
        "R3": [raw["conv"]],
        "R4": [raw["recurrent"], raw["conv"]],
        "R5": [primary],
        "R6": [raw["recurrent"], raw["conv"], primary],
        "R7": [raw["recurrent"], raw["conv"], raw["kv_full"]],
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_features, layerwise_features, selected_meta = _dual_pca_scores(
        definitions[selected], train, device=device
    )
    full_features, _, full_meta = _dual_pca_scores(
        definitions["R7"], train, device=device
    )
    path = context.root / FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=selected_features,
        layerwise_features=layerwise_features,
        full_features=full_features,
        base_trial_id=np.asarray(ids),
        family=np.asarray(families),
        split=np.asarray(splits),
    )
    manifest = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "feature_artifact": str(FEATURES),
        "feature_artifact_sha256": sha256_file(path),
        "count": len(ids),
        "dimension": int(selected_features.shape[1]),
        "layerwise_dimension": int(layerwise_features.shape[1]),
        "full_reference_dimension": int(full_features.shape[1]),
        "selected_raw_state": selected,
        "raw_state_authorized": True,
        "raw_selection_diagnostics": diagnostics,
        "definition": {
            "encoder": "exact block-normalized dual PCA over architecture-aligned raw tensors",
            "selected_blocks": selected_meta,
            "full_reference_blocks": full_meta,
            "kv_padding": "zero padding beyond each actual prompt length; exact observed token positions retained",
        },
        "source_shards": shard_hashes,
    }
    write_json_atomic(
        context.root / "results/v8/processed/structured_features_v8.json", manifest
    )
    context.finish("COMPLETED_FEATURES", summary=manifest)


def _endpoints(root: Path) -> dict[str, np.ndarray]:
    combined: dict[str, list[np.ndarray]] = defaultdict(list)
    full = "rec_matrix_all+conv_all+kv_l27_h3+kv_other"
    for capture in _capture_manifests(root):
        with np.load(root / capture["endpoint_artifact"], allow_pickle=False) as data:
            for key in ("base_trial_id", "family", "current_j_perturbed"):
                combined[key].append(data[key])
            combined["target_delta"].append(
                data[f"next_j__{full}"].astype(np.float32)
                - data["next_j__none"].astype(np.float32)
            )
            combined["target_absolute"].append(
                data[f"next_j__{full}"].astype(np.float32)
            )
            combined["future_delta"].append(
                (
                    data[f"future_j__{full}"].astype(np.float32)
                    - data["future_j__none"].astype(np.float32)
                ).reshape(len(data["base_trial_id"]), -1)
            )
            combined["output_delta"].append(
                data[f"output_log_odds__{full}"].astype(np.float32)[:, None]
            )
    return {key: np.concatenate(values) for key, values in combined.items()}


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def _svd_scores(
    train: np.ndarray, all_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fingerprint = hashlib.sha256()
    fingerprint.update(np.asarray(train.shape, dtype=np.int64).tobytes())
    fingerprint.update(np.asarray(all_values.shape, dtype=np.int64).tobytes())
    fingerprint.update(np.ascontiguousarray(train[[0, -1]]).tobytes())
    fingerprint.update(np.ascontiguousarray(all_values[[0, -1]]).tobytes())
    key = fingerprint.hexdigest()
    if key in _SVD_CACHE:
        return _SVD_CACHE[key]
    fit, values = _standardize(train, all_values)
    _, singular, vh = np.linalg.svd(fit, full_matrices=False)
    result = (values @ vh.T, singular)
    _SVD_CACHE[key] = result
    return result


def _order(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = target - target.mean(axis=0, keepdims=True)
    covariance = scores.T @ centered
    energy = np.sum(covariance * covariance, axis=1) / np.maximum(
        np.sum(scores * scores, axis=0), 1e-12
    )
    return np.argsort(-energy)


def _latent(
    method: str,
    dimension: int,
    train_features: np.ndarray,
    all_features: np.ndarray,
    train_absolute: np.ndarray,
    train_delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores, singular = _svd_scores(train_features, all_features)
    rank = scores.shape[1]
    dimension = min(int(dimension), rank)
    if method == "pca":
        order = np.arange(rank)
    elif method == "predictive_bottleneck":
        order = _order(scores[: len(train_features)], train_absolute)
    elif method == "causal_bottleneck":
        order = _order(scores[: len(train_features)], train_delta)
    elif method == "layerwise_fusion":
        # The sketch itself is layer-factorized; interleave high-variance quarters
        # so recurrent, conv, and KV blocks cannot be starved by global variance.
        quarters = [np.arange(start, rank, 4) for start in range(4)]
        order = np.asarray(
            [value for group in zip(*quarters, strict=False) for value in group]
        )
        order = order[order < rank]
    elif method == "nonlinear_encoder":
        # Nonlinear autoencoding is fit once for each declared dimension.
        torch.manual_seed(20260828 + dimension)
        fit = torch.from_numpy(scores[: len(train_features)]).float()
        all_tensor = torch.from_numpy(scores).float()
        width = max(64, 2 * dimension)
        encoder = torch.nn.Sequential(
            torch.nn.Linear(rank, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, dimension),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(dimension, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, rank),
        )
        optimizer = torch.optim.AdamW(
            [*encoder.parameters(), *decoder.parameters()], lr=3e-3, weight_decay=1e-4
        )
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(encoder(fit))
            loss = torch.mean((prediction - fit) ** 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return encoder(all_tensor).numpy(), {
                "rank": rank,
                "train_reconstruction_loss": float(loss),
                "singular_value_max": float(singular[0]),
                "singular_value_min": float(singular[-1]),
            }
    else:
        raise ValueError(method)
    return scores[:, order[:dimension]], {
        "rank": rank,
        "singular_value_max": float(singular[0]),
        "singular_value_min": float(singular[-1]),
    }


def _top10_agreement(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    k = min(10, predicted.shape[1])
    left = np.argpartition(predicted, -k, axis=1)[:, -k:]
    right = np.argpartition(target, -k, axis=1)[:, -k:]
    return np.asarray(
        [len(set(a) & set(b)) / k for a, b in zip(left, right, strict=True)]
    )


def _evaluate_candidate(
    latent: np.ndarray,
    current_j: np.ndarray,
    full_features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    baseline = _ridge(current_j[train], current_j[test], target[train], alpha=alpha)
    candidate_x = np.concatenate((current_j, latent), axis=1)
    candidate = _ridge(
        candidate_x[train], candidate_x[test], target[train], alpha=alpha
    )
    full_x = np.concatenate((current_j, full_features), axis=1)
    full = _ridge(full_x[train], full_x[test], target[train], alpha=alpha)
    residual_x = np.concatenate((current_j, latent, full_features), axis=1)
    residual = _ridge(residual_x[train], residual_x[test], target[train], alpha=alpha)
    truth = target[test]
    base_cos = _cosine_rows(baseline, truth)
    candidate_cos = _cosine_rows(candidate, truth)
    full_cos = _cosine_rows(full, truth)
    residual_cos = _cosine_rows(residual, truth)
    denominator = float(np.mean(full_cos - base_cos))
    gap = float(np.mean(candidate_cos - base_cos) / max(denominator, 1e-12))
    conditional = residual_cos - candidate_cos
    return {
        "j_only_cosine": _paired_ci(base_cos, seed, resamples),
        "candidate_cosine": _paired_ci(candidate_cos, seed, resamples),
        "full_state_cosine": _paired_ci(full_cos, seed, resamples),
        "predictive_gap_closed": gap,
        "conditional_residual_gain": _paired_ci(conditional, seed, resamples),
        "semantic_top10_agreement": _paired_ci(
            _top10_agreement(candidate, truth), seed, resamples
        ),
    }


def _evaluate_scalar(
    latent: np.ndarray,
    current_j: np.ndarray,
    full_features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    baseline = _ridge(current_j[train], current_j[test], target[train], alpha=alpha)
    candidate_x = np.concatenate((current_j, latent), axis=1)
    candidate = _ridge(
        candidate_x[train], candidate_x[test], target[train], alpha=alpha
    )
    full_x = np.concatenate((current_j, full_features), axis=1)
    full = _ridge(full_x[train], full_x[test], target[train], alpha=alpha)
    residual_x = np.concatenate((current_j, latent, full_features), axis=1)
    residual = _ridge(residual_x[train], residual_x[test], target[train], alpha=alpha)
    truth = target[test]
    baseline_error = np.sqrt(np.mean((baseline - truth) ** 2, axis=1))
    candidate_error = np.sqrt(np.mean((candidate - truth) ** 2, axis=1))
    full_error = np.sqrt(np.mean((full - truth) ** 2, axis=1))
    residual_error = np.sqrt(np.mean((residual - truth) ** 2, axis=1))
    denominator = float(np.mean(baseline_error - full_error))
    gap = float(np.mean(baseline_error - candidate_error) / max(denominator, 1e-12))
    return {
        "j_only_rmse": _paired_ci(baseline_error, seed, resamples),
        "candidate_rmse": _paired_ci(candidate_error, seed, resamples),
        "full_state_rmse": _paired_ci(full_error, seed, resamples),
        "predictive_gap_closed": gap,
        "conditional_residual_rmse_improvement": _paired_ci(
            candidate_error - residual_error, seed, resamples
        ),
        "output_sign_agreement": _paired_ci(
            (np.sign(candidate.reshape(-1)) == np.sign(truth.reshape(-1))).astype(
                float
            ),
            seed,
            resamples,
        ),
    }


def _analysis(context: Any, freeze: dict[str, Any]) -> None:
    feature_manifest = json.loads(
        (context.root / "results/v8/processed/structured_features_v8.json").read_text()
    )
    if feature_manifest["source_freeze_digest"] != freeze["freeze_digest"]:
        raise RuntimeError("v8 feature freeze mismatch")
    path = context.root / feature_manifest["feature_artifact"]
    if sha256_file(path) != feature_manifest["feature_artifact_sha256"]:
        raise RuntimeError("v8 feature artifact hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        layerwise_features = data["layerwise_features"].astype(np.float32)
        full_features = data["full_features"].astype(np.float32)
        ids = data["base_trial_id"].astype(str)
        families = data["family"].astype(str)
        splits = data["split"].astype(str)
    endpoint = _endpoints(context.root)
    lookup = {
        value: index
        for index, value in enumerate(endpoint["base_trial_id"].astype(str))
    }
    order = np.asarray([lookup[value] for value in ids])
    current_j = endpoint["current_j_perturbed"][order].astype(np.float32)
    target = endpoint["target_delta"][order].astype(np.float32)
    absolute = endpoint["target_absolute"][order].astype(np.float32)
    future_target = endpoint["future_delta"][order].astype(np.float32)
    output_target = endpoint["output_delta"][order].astype(np.float32)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    final = np.flatnonzero(splits == "final_test")
    section = context.config["persistent_state_v8"]["compression"]
    methods = [str(value) for value in section["methods"]]
    dimensions = [int(value) for value in section["dimensions"]]
    alpha = float(context.config["persistent_state_v8"]["kernel_ridge_alphas"][-1])
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_state_v8"]["bootstrap_resamples"])
    records = []
    for regime in ("universal", "family_specific", "shared_family_residual"):
        for method in methods:
            for dimension in dimensions:
                source_features = (
                    layerwise_features if method == "layerwise_fusion" else features
                )
                if regime == "universal":
                    latent, meta = _latent(
                        method,
                        dimension,
                        source_features[train],
                        source_features,
                        absolute[train],
                        target[train],
                    )
                else:
                    latent = np.zeros((len(features), dimension), dtype=np.float32)
                    meta = {"families": {}}
                    for family in sorted(set(families)):
                        family_train = train[families[train] == family]
                        family_all = np.flatnonzero(families == family)
                        values, detail = _latent(
                            method,
                            dimension,
                            source_features[family_train],
                            source_features[family_all],
                            absolute[family_train],
                            target[family_train],
                        )
                        if regime == "shared_family_residual":
                            shared, _ = _latent(
                                method,
                                max(1, dimension // 2),
                                source_features[train],
                                source_features[family_all],
                                absolute[train],
                                target[train],
                            )
                            keep = min(values.shape[1], dimension - shared.shape[1])
                            values = np.concatenate((shared, values[:, :keep]), axis=1)
                        latent[family_all, : values.shape[1]] = values
                        meta["families"][family] = detail
                evaluation_groups = [("pooled", validation, final)] + [
                    (
                        str(family),
                        validation[families[validation] == family],
                        final[families[final] == family],
                    )
                    for family in sorted(set(families))
                ]
                for family, validation_rows, final_rows in evaluation_groups:
                    validation_metrics = _evaluate_candidate(
                        latent,
                        current_j,
                        full_features,
                        target,
                        train,
                        validation_rows,
                        alpha=alpha,
                        seed=seed,
                        resamples=resamples,
                    )
                    final_metrics = _evaluate_candidate(
                        latent,
                        current_j,
                        full_features,
                        target,
                        train,
                        final_rows,
                        alpha=alpha,
                        seed=seed,
                        resamples=resamples,
                    )
                    future_metrics = _evaluate_candidate(
                        latent,
                        current_j,
                        full_features,
                        future_target,
                        train,
                        final_rows,
                        alpha=alpha,
                        seed=seed,
                        resamples=resamples,
                    )
                    output_metrics = _evaluate_scalar(
                        latent,
                        current_j,
                        full_features,
                        output_target,
                        train,
                        final_rows,
                        alpha=alpha,
                        seed=seed,
                        resamples=resamples,
                    )
                    conditional_threshold = float(
                        section["maximum_conditional_residual_gain"]
                    )
                    conditional_pass = bool(
                        final_metrics["conditional_residual_gain"]["upper"]
                        <= conditional_threshold
                        and future_metrics["conditional_residual_gain"]["upper"]
                        <= conditional_threshold
                        and output_metrics["conditional_residual_rmse_improvement"][
                            "upper"
                        ]
                        <= conditional_threshold
                    )
                    predictive_pass = min(
                        final_metrics["predictive_gap_closed"],
                        future_metrics["predictive_gap_closed"],
                        output_metrics["predictive_gap_closed"],
                    ) >= float(section["predictive_gap_closed"])
                    semantic_pass = final_metrics["semantic_top10_agreement"][
                        "estimate"
                    ] >= float(section["semantic_agreement_minimum"])
                    joint_gap = min(
                        final_metrics["predictive_gap_closed"],
                        future_metrics["predictive_gap_closed"],
                        output_metrics["predictive_gap_closed"],
                    )
                    # Interventional reconstruction is intentionally gated behind both screens.
                    records.append(
                        CompressionSweepRecord(
                            run_id=context.run_id,
                            method=method,
                            regime=regime,
                            dimension=dimension,
                            family=family,
                            split="final_test",
                            count=len(final_rows),
                            predictive_gap_closed=float(joint_gap),
                            conditional_residual_gain=float(
                                final_metrics["conditional_residual_gain"]["estimate"]
                            ),
                            causal_gap_closed=None,
                            direction_cosine=None,
                            magnitude_ratio=None,
                            semantic_agreement=float(
                                final_metrics["semantic_top10_agreement"]["estimate"]
                            ),
                            output_sign_agreement=None,
                            authorized=False,
                            metadata={
                                "validation": validation_metrics,
                                "final": final_metrics,
                                "future": future_metrics,
                                "output": output_metrics,
                                "predictive_pass": predictive_pass,
                                "conditional_pass": conditional_pass,
                                "semantic_pass": semantic_pass,
                                "causal_status": "AUTHORIZED_FOR_INTERVENTIONAL_DECODING"
                                if predictive_pass
                                and conditional_pass
                                and semantic_pass
                                else "GATED_BY_PREDICTIVE_OR_CONDITIONAL_SCREEN",
                                "encoder": meta,
                            },
                        ).to_dict()
                    )
    # Leave-one-family-out transfer is a preregistered sensitivity, never pooled
    # into the within-family authorization decision.
    for held_family in sorted(set(families)):
        cross_train = train[families[train] != held_family]
        cross_test = final[families[final] == held_family]
        for method in methods:
            for dimension in dimensions:
                source_features = (
                    layerwise_features if method == "layerwise_fusion" else features
                )
                latent, meta = _latent(
                    method,
                    dimension,
                    source_features[cross_train],
                    source_features,
                    absolute[cross_train],
                    target[cross_train],
                )
                metrics = _evaluate_candidate(
                    latent,
                    current_j,
                    full_features,
                    target,
                    cross_train,
                    cross_test,
                    alpha=alpha,
                    seed=seed,
                    resamples=resamples,
                )
                records.append(
                    CompressionSweepRecord(
                        run_id=context.run_id,
                        method=method,
                        regime=f"cross_family_leave_{held_family}_out",
                        dimension=dimension,
                        family=held_family,
                        split="final_test",
                        count=len(cross_test),
                        predictive_gap_closed=float(metrics["predictive_gap_closed"]),
                        conditional_residual_gain=float(
                            metrics["conditional_residual_gain"]["estimate"]
                        ),
                        causal_gap_closed=None,
                        direction_cosine=None,
                        magnitude_ratio=None,
                        semantic_agreement=float(
                            metrics["semantic_top10_agreement"]["estimate"]
                        ),
                        output_sign_agreement=None,
                        authorized=False,
                        metadata={
                            "final": metrics,
                            "predictive_pass": False,
                            "conditional_pass": False,
                            "causal_status": "TRANSFER_SENSITIVITY_ONLY",
                            "encoder": meta,
                        },
                    ).to_dict()
                )
    record_path = context.processed_dir / "persistent_state_compression_v8.parquet"
    frame = []
    for record in records:
        flat = dict(record)
        flat["metadata"] = json.dumps(flat["metadata"], sort_keys=True)
        frame.append(flat)
    import pandas as pd

    pd.DataFrame(frame).to_parquet(record_path, index=False, compression="zstd")
    pass_by_candidate: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    for record in records:
        if record["regime"].startswith("cross_family_"):
            continue
        pass_by_candidate[
            (record["method"], record["regime"], record["dimension"])
        ].append(
            bool(
                record["metadata"]["predictive_pass"]
                and record["metadata"]["conditional_pass"]
                and record["metadata"]["semantic_pass"]
            )
        )
    screen_pass = [
        key for key, values in pass_by_candidate.items() if values and all(values)
    ]
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "status": "COMPLETED_SCREEN_CAUSAL_GATED"
        if not screen_pass
        else "PENDING_INTERVENTIONAL_DECODING",
        "candidate_count": len(pass_by_candidate),
        "family_evaluation_count": len(records),
        "predictive_conditional_pass_count": len(screen_pass),
        "smallest_predictive_conditional": min(
            (key[2] for key in screen_pass), default=None
        ),
        "smallest_authorized": None,
        "controller_authorized": False,
        "causal_gate_note": "No candidate is sufficient until decoded cache interventions pass direction, magnitude, semantic, and output-sign gates.",
        "records": str(record_path.relative_to(context.root)),
        "feature_artifact": str(FEATURES),
        "feature_artifact_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / SUMMARY, summary)
    context.finish("COMPLETED_COMPRESSION_SCREEN", summary=summary)


def main() -> None:
    parser = standard_parser(
        "structured persistent-state compression v8", "configs/persistent_state_v8.yaml"
    )
    parser.add_argument("--stage", required=True, choices=("features", "analyze"))
    args = parser.parse_args()
    context = initialize_context("compress-persistent-v8", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
        elif args.stage == "features":
            _features(context, freeze)
        else:
            _analysis(context, freeze)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
