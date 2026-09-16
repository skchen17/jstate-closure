"""Decode frozen compact candidates into cache state and test causal fidelity."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import (
    _capture_manifests,
    _load_raw_deltas,
)
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _safe_cosine,
    _tasks,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import _semantic_ids
from jclosure.experiments.sufficiency_v9 import _bootstrap_ci, _load_data
from jclosure.metrics import jensen_shannon_from_logits
from jclosure.model import load_model_bundle
from jclosure.protocol_v10 import (
    PROTOCOL_V10,
    SCHEMA_VERSION_V10,
    verify_candidate_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.single_arm_v4 import multiple_token_log_odds

DECODER_SUMMARY = Path("results/v10/processed/compact_decoder_v10.json")
CAUSAL = Path("results/v10/processed/decoded_causal_state_validation_v10.parquet")
CAUSAL_SUMMARY = Path("results/v10/processed/decoded_causal_state_validation_v10.json")


def _latent_decoder(
    features: np.ndarray,
    train: np.ndarray,
    targets: dict[str, np.ndarray],
    weights: dict[str, float],
) -> dict[str, Any]:
    mean = features[train].mean(axis=0, keepdims=True)
    scale = features[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    _, singular, vh = np.linalg.svd(standardized[train], full_matrices=False)
    keep = singular > max(float(singular[0]), 1e-20) * 1e-8
    vh = vh[keep]
    scores = standardized @ vh.T

    def energy(target: np.ndarray) -> np.ndarray:
        centered = target[train] - target[train].mean(axis=0, keepdims=True)
        covariance = scores[train].T @ centered
        values = np.sum(covariance * covariance, axis=1) / np.maximum(
            np.sum(scores[train] * scores[train], axis=0), 1e-12
        )
        maximum = float(values.max())
        return values / maximum if maximum > 0 else values

    next_absolute = energy(targets["next_absolute"])
    next_delta = energy(targets["next_delta"])
    future_delta = energy(targets["future_delta_h4"])
    orders = {
        "pca": np.arange(scores.shape[1]),
        "predictive_bottleneck": np.argsort(-next_absolute),
        "causal_bottleneck": np.argsort(-next_delta),
        "semantic_causal_bottleneck": np.argsort(
            -(
                float(weights["next_absolute"]) * next_absolute
                + float(weights["next_causal_delta"]) * next_delta
                + float(weights["future_causal_delta"]) * future_delta
            )
        ),
    }
    return {
        "mean": mean,
        "scale": scale,
        "vh": vh,
        "scores": scores,
        "orders": orders,
        "singular": singular[keep],
    }


def _decoded_feature_scores(
    model: dict[str, Any], method: str, dimension: int, indices: np.ndarray
) -> np.ndarray:
    retained = np.zeros((len(indices), model["scores"].shape[1]), dtype=np.float32)
    selected = model["orders"][method][:dimension]
    retained[:, selected] = model["scores"][indices][:, selected]
    standardized = retained @ model["vh"]
    return (standardized * model["scale"] + model["mean"]).astype(np.float32)


def _dual_reconstruction_alpha(
    train_scores: np.ndarray, decoded_scores: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    fit = train_scores.astype(np.float64)
    inverse = np.linalg.pinv(fit, rcond=1e-10)
    alpha = decoded_scores.astype(np.float64) @ inverse
    reconstructed = alpha @ fit
    singular = np.linalg.svd(fit, compute_uv=False)
    return alpha.astype(np.float32), {
        "largest_singular_value": float(singular.max()),
        "smallest_singular_value": float(singular.min()),
        "score_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - decoded_scores))
        ),
    }


def _decode_block(
    block: torch.Tensor,
    train: np.ndarray,
    alpha: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    flat = block.reshape(block.shape[0], -1)
    output = torch.empty((len(alpha), flat.shape[1]), dtype=torch.bfloat16)
    alpha_device = torch.from_numpy(alpha).to(device=device, dtype=torch.float32)
    for start in range(0, flat.shape[1], chunk_size):
        stop = min(start + chunk_size, flat.shape[1])
        fit = flat[train, start:stop].to(dtype=torch.float32)
        mean = fit.double().mean(dim=0).float()
        centered = (fit - mean).to(device)
        decoded = alpha_device @ centered + mean.to(device)
        output[:, start:stop] = decoded.to("cpu", dtype=torch.bfloat16)
        del centered, decoded
    return output.reshape((len(alpha), *block.shape[1:]))


def _raw_order(
    raw: dict[str, torch.Tensor], raw_ids: list[str], feature_ids: np.ndarray
) -> dict[str, torch.Tensor]:
    lookup = {value: index for index, value in enumerate(raw_ids)}
    order = torch.tensor([lookup[str(value)] for value in feature_ids], dtype=torch.long)
    return {name: values.index_select(0, order) for name, values in raw.items()}


def _prepare_decoder(context: Any, candidate_freeze: dict[str, Any]) -> None:
    section = context.config["causal_sufficiency_v10"]
    data = _load_data(context.root)
    train = np.flatnonzero(data["splits"] == "train")
    selected_ids = [str(value) for value in candidate_freeze["causal_confirmatory_base_trial_ids"]]
    id_lookup = {value: index for index, value in enumerate(data["ids"].tolist())}
    selected = np.asarray([id_lookup[value] for value in selected_ids], dtype=int)
    model = _latent_decoder(
        data["features"],
        train,
        data["targets"],
        context.config["sufficiency_v9"]["semantic_objective_weights"],
    )
    method = str(section["primary_method"])
    dimensions = [int(value) for value in candidate_freeze["candidate_dimensions"]]
    decoded_feature_sets = [
        _decoded_feature_scores(model, method, dimension, selected)
        for dimension in dimensions
    ]
    alphas = []
    alpha_details = []
    for values in decoded_feature_sets:
        alpha, detail = _dual_reconstruction_alpha(
            data["features"][train], values
        )
        alphas.append(alpha)
        alpha_details.append(detail)
    stacked_alpha = np.concatenate(alphas, axis=0)
    raw, raw_ids, _, _, _ = _load_raw_deltas(
        context, _capture_manifests(context.root)
    )
    raw = _raw_order(raw, raw_ids, data["ids"])
    decoder_config = section["decoder"]
    requested_device = int(context.config["model"].get("device", 0))
    device = torch.device(
        f"cuda:{requested_device}" if torch.cuda.is_available() else "cpu"
    )
    decoded = {
        "recurrent": _decode_block(
            raw["recurrent"],
            train,
            stacked_alpha,
            device=device,
            chunk_size=int(decoder_config["reconstruction_chunk"]),
        ),
        "conv": _decode_block(
            raw["conv"],
            train,
            stacked_alpha,
            device=device,
            chunk_size=int(decoder_config["reconstruction_chunk"]),
        ),
        "kv": _decode_block(
            raw["kv_full"],
            train,
            stacked_alpha,
            device=device,
            chunk_size=int(decoder_config["reconstruction_chunk"]),
        ),
    }
    artifact_root = context.root / "artifacts/causal/v10" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    shard_size = int(decoder_config["shard_size"])
    declarations: list[dict[str, Any]] = []
    count = len(selected)
    for dimension_index, dimension in enumerate(dimensions):
        offset = dimension_index * count
        for start in range(0, count, shard_size):
            stop = min(start + shard_size, count)
            rows = []
            for local in range(start, stop):
                source = offset + local
                rows.append(
                    {
                        "base_trial_id": selected_ids[local],
                        "dimension": dimension,
                        "recurrent": decoded["recurrent"][source],
                        "conv": decoded["conv"][source],
                        "kv": decoded["kv"][source],
                    }
                )
            path = artifact_root / f"decoded_d{dimension}_{start:03d}_{stop:03d}.pt"
            torch.save(
                {
                    "format": "decoded_compact_persistent_state_v10_bf16",
                    "rows": rows,
                },
                path,
            )
            declarations.append(
                {
                    "dimension": dimension,
                    "path": str(path.relative_to(context.root)),
                    "sha256": sha256_file(path),
                    "base_trial_ids": selected_ids[start:stop],
                }
            )
    full_reconstruction = _decoded_feature_scores(
        model, method, data["features"].shape[1], selected
    )
    feature_error = float(
        np.max(np.abs(full_reconstruction - data["features"][selected]))
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "run_id": context.run_id,
        "candidate_freeze_digest": candidate_freeze["freeze_digest"],
        "method": method,
        "dimensions": dimensions,
        "base_trial_ids": selected_ids,
        "feature_rank": int(data["features"].shape[1]),
        "full_rank_feature_reconstruction_max_abs_error": feature_error,
        "alpha_diagnostics": {
            str(dimension): detail
            for dimension, detail in zip(dimensions, alpha_details, strict=True)
        },
        "decoder_definition": (
            "train-fitted second-level ordered latent inverse followed by exact "
            "block-normalized dual-PCA raw-state reconstruction"
        ),
        "state_dtype": "bfloat16",
        "shards": declarations,
    }
    write_json_atomic(context.root / DECODER_SUMMARY, summary)
    context.finish("COMPLETED_V10_DECODER", summary=summary)


def apply_decoded_state(
    clean_cache: Any,
    decoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    recurrent_layers: list[int],
    attention_layers: list[int],
    prompt_length: int,
) -> Any:
    """Construct a strict decoded-only cache from clean state and decoded deltas."""

    output = clone_hybrid_cache(clean_cache)
    recurrent, conv, kv = decoded
    for index, layer in enumerate(recurrent_layers):
        target = output.layers[layer]
        target.recurrent_states = (
            target.recurrent_states.float()
            + recurrent[index].to(target.recurrent_states.device).float()
        ).to(target.recurrent_states.dtype)
        target.conv_states = (
            target.conv_states.float()
            + conv[index].to(target.conv_states.device).float()
        ).to(target.conv_states.dtype)
    for index, layer in enumerate(attention_layers):
        target = output.layers[layer]
        length = min(prompt_length, kv.shape[-2], target.keys.shape[-2])
        target.keys[:, :, :length, :] = (
            target.keys[:, :, :length, :].float()
            + kv[index, 0, :, :length, :].to(target.keys.device).float()[None]
        ).to(target.keys.dtype)
        target.values[:, :, :length, :] = (
            target.values[:, :, :length, :].float()
            + kv[index, 1, :, :length, :].to(target.values.device).float()[None]
        ).to(target.values.dtype)
    return output


def _load_decoded(root: Path, summary: dict[str, Any]) -> dict[tuple[int, str], Any]:
    output: dict[tuple[int, str], Any] = {}
    for declaration in summary["shards"]:
        path = root / declaration["path"]
        if sha256_file(path) != declaration["sha256"]:
            raise RuntimeError(f"decoded-state shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload["rows"]:
            output[(int(row["dimension"]), str(row["base_trial_id"]))] = row
    return output


def _pair_metadata(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for capture in _capture_manifests(root):
        for line in (root / capture["pair_records"]).read_text().splitlines():
            row = json.loads(line)
            if row["valid"]:
                output[str(row["base_trial_id"])] = row
    return output


def _effect_metrics(
    clean: dict[str, Any],
    teacher: dict[str, Any],
    decoded: dict[str, Any],
    *,
    main_layer: int,
    horizon: int,
    semantic_ids: list[int],
) -> dict[str, Any]:
    index = horizon - 1
    clean_j = clean["j"][main_layer][index]
    teacher_delta = teacher["j"][main_layer][index] - clean_j
    decoded_delta = decoded["j"][main_layer][index] - clean_j
    teacher_norm = float(np.linalg.norm(teacher_delta))
    decoded_norm = float(np.linalg.norm(decoded_delta))
    direction = _safe_cosine(decoded_delta, teacher_delta)
    magnitude = decoded_norm / max(teacher_norm, 1e-20)
    k = min(10, len(teacher_delta))
    teacher_top = set(np.argpartition(teacher_delta, -k)[-k:].tolist())
    decoded_top = set(np.argpartition(decoded_delta, -k)[-k:].tolist())
    semantic_delta = len(teacher_top & decoded_top) / k
    clean_logits = clean["logits"][index]
    teacher_logits = teacher["logits"][index]
    decoded_logits = decoded["logits"][index]
    teacher_output_delta = teacher_logits - clean_logits
    decoded_output_delta = decoded_logits - clean_logits
    output_direction = _safe_cosine(decoded_output_delta, teacher_output_delta)
    output_magnitude = float(
        np.linalg.norm(decoded_output_delta)
        / max(float(np.linalg.norm(teacher_output_delta)), 1e-20)
    )
    teacher_js = jensen_shannon_from_logits(
        torch.from_numpy(clean_logits), torch.from_numpy(teacher_logits)
    )
    decoded_js = jensen_shannon_from_logits(
        torch.from_numpy(clean_logits), torch.from_numpy(decoded_logits)
    )
    teacher_odds = None
    decoded_odds = None
    sign_agreement = None
    if semantic_ids:
        clean_odds = multiple_token_log_odds(
            torch.from_numpy(clean_logits), semantic_ids
        )
        teacher_odds = multiple_token_log_odds(
            torch.from_numpy(teacher_logits), semantic_ids
        ) - clean_odds
        decoded_odds = multiple_token_log_odds(
            torch.from_numpy(decoded_logits), semantic_ids
        ) - clean_odds
        if abs(teacher_odds) > 1e-8:
            sign_agreement = float(np.sign(teacher_odds) == np.sign(decoded_odds))
    clean_argmax = int(np.argmax(clean_logits))
    teacher_argmax = int(np.argmax(teacher_logits))
    decoded_argmax = int(np.argmax(decoded_logits))
    return {
        "direction_cosine": direction,
        "magnitude_ratio": magnitude,
        "j_l2_error": float(np.linalg.norm(decoded_delta - teacher_delta)),
        "semantic_delta_agreement": float(semantic_delta),
        "output_direction_cosine": output_direction,
        "output_magnitude_ratio": output_magnitude,
        "teacher_output_js": float(teacher_js),
        "decoded_output_js": float(decoded_js),
        "teacher_target_log_odds_delta": teacher_odds,
        "decoded_target_log_odds_delta": decoded_odds,
        "task_decision_sign_agreement": sign_agreement,
        "teacher_answer_change": bool(teacher_argmax != clean_argmax),
        "decoded_answer_change": bool(decoded_argmax != clean_argmax),
        "answer_change_agreement": float(
            (teacher_argmax != clean_argmax) == (decoded_argmax != clean_argmax)
        ),
        "teacher_j_effect_norm": teacher_norm,
        "decoded_j_effect_norm": decoded_norm,
    }


def _aggregate(
    frame: pd.DataFrame,
    seeds: list[int],
    resamples: int,
) -> dict[str, Any]:
    metrics = (
        "direction_cosine",
        "magnitude_ratio",
        "semantic_delta_agreement",
        "output_direction_cosine",
        "output_magnitude_ratio",
        "task_decision_sign_agreement",
        "answer_change_agreement",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        values = frame[metric].dropna().to_numpy(dtype=float)
        output[metric] = _bootstrap_ci(values, seeds, resamples) if len(values) else None
    weights = frame["teacher_j_effect_norm"].to_numpy(dtype=float)
    direction = frame["direction_cosine"].fillna(0).to_numpy(dtype=float)
    output["effect_weighted_direction_cosine"] = float(
        np.sum(weights * direction) / max(float(weights.sum()), 1e-20)
    )
    output["count"] = int(len(frame))
    output["identifiable_task_decision_count"] = int(
        frame["task_decision_sign_agreement"].notna().sum()
    )
    return output


def _causal_gate(metrics: dict[str, Any], section: dict[str, Any]) -> bool:
    direction = metrics["direction_cosine"]
    magnitude = metrics["magnitude_ratio"]
    semantic = metrics["semantic_delta_agreement"]
    output_direction = metrics["output_direction_cosine"]
    sign = metrics["task_decision_sign_agreement"]
    if any(value is None for value in (direction, magnitude, semantic, output_direction)):
        return False
    passed = bool(
        direction["estimate"] >= float(section["direction_cosine_minimum"])
        and float(section["magnitude_ratio_minimum"])
        <= magnitude["estimate"]
        <= float(section["magnitude_ratio_maximum"])
        and semantic["estimate"]
        >= float(section["semantic_delta_agreement_minimum"])
        and output_direction["estimate"]
        >= float(section["output_direction_minimum"])
    )
    if sign is not None and metrics["identifiable_task_decision_count"] >= 5:
        passed &= bool(
            sign["estimate"] >= float(section["task_decision_sign_minimum"])
        )
    return passed


def _causal(context: Any, candidate_freeze: dict[str, Any]) -> None:
    decoder_summary = json.loads((context.root / DECODER_SUMMARY).read_text())
    if decoder_summary["candidate_freeze_digest"] != candidate_freeze["freeze_digest"]:
        raise RuntimeError("v10 decoder belongs to another candidate freeze")
    decoded_rows = _load_decoded(context.root, decoder_summary)
    data = _load_data(context.root)
    raw, raw_ids, _, _, _ = _load_raw_deltas(
        context, _capture_manifests(context.root)
    )
    raw = _raw_order(raw, raw_ids, data["ids"])
    id_lookup = {value: index for index, value in enumerate(data["ids"].tolist())}
    metadata = _pair_metadata(context.root)
    tasks = _tasks(context)
    bundle = load_model_bundle(context.config)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    intervention_layer = int(v8["intervention"]["layer"])
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    causal_section = context.config["causal_sufficiency_v10"]["causal"]
    horizons = [int(value) for value in causal_section["horizons"]]
    maximum_horizon = max(horizons)
    dimensions = [int(value) for value in candidate_freeze["candidate_dimensions"]]
    ids = [str(value) for value in candidate_freeze["causal_confirmatory_base_trial_ids"]]
    enriched = set(candidate_freeze["effect_enriched_base_trial_ids"])
    strict_hash = hashlib.sha256(inspect.getsource(apply_decoded_state).encode()).hexdigest()
    records: list[dict[str, Any]] = []
    progress = context.raw_dir / context.run_id / "causal_progress.json"
    for case_index, base_id in enumerate(ids):
        source_index = id_lookup[base_id]
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=intervention_layer,
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=maximum_horizon,
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
        teacher_delta = (
            raw["recurrent"][source_index],
            raw["conv"][source_index],
            raw["kv_full"][source_index],
        )
        teacher_cache = apply_decoded_state(
            clean["cache"],
            teacher_delta,
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
        del teacher_cache
        for dimension in dimensions:
            row = decoded_rows[(dimension, base_id)]
            decoded_cache = apply_decoded_state(
                clean["cache"],
                (row["recurrent"], row["conv"], row["kv"]),
                recurrent_layers=recurrent_layers,
                attention_layers=attention_layers,
                prompt_length=clean["prompt_length"],
            )
            decoded_trajectory = _teacher_forced_trajectory(
                bundle,
                decoded_cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            del decoded_cache
            for horizon in horizons:
                semantic_index = min(horizon - 1, len(task.semantic_actions) - 1)
                semantic_ids = _semantic_ids(
                    bundle.tokenizer, task.semantic_actions[semantic_index]
                )
                metrics = _effect_metrics(
                    clean_trajectory,
                    teacher_trajectory,
                    decoded_trajectory,
                    main_layer=main_layer,
                    horizon=horizon,
                    semantic_ids=semantic_ids,
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V10,
                        "protocol_version": PROTOCOL_V10,
                        "record_type": "decoded_causal_state_validation",
                        "run_id": context.run_id,
                        "candidate_freeze_digest": candidate_freeze["freeze_digest"],
                        "base_trial_id": base_id,
                        "prompt_id": str(pair["prompt_id"]),
                        "family": str(pair["family"]),
                        "dimension": dimension,
                        "horizon": horizon,
                        "effect_enriched": base_id in enriched,
                        "teacher_forced": True,
                        "strict_interface": True,
                        "strict_interface_function_sha256": strict_hash,
                        **metrics,
                    }
                )
        write_json_atomic(
            progress,
            {
                "status": "RUNNING",
                "completed_cases": case_index + 1,
                "total_cases": len(ids),
                "last_base_trial_id": base_id,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    context.root.joinpath(CAUSAL).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(context.root / CAUSAL, index=False, compression="zstd")
    seeds = [
        int(value)
        for value in context.config["causal_sufficiency_v10"]["confirmation_seeds"]
    ]
    resamples = int(
        context.config["causal_sufficiency_v10"]["bootstrap_resamples"]
    )
    aggregates: dict[str, Any] = {}
    authorization: dict[str, Any] = {}
    auth_horizons = [int(value) for value in causal_section["authorization_horizons"]]
    for dimension in dimensions:
        dimension_gates = []
        aggregates[str(dimension)] = {}
        for horizon in horizons:
            current = frame[(frame["dimension"] == dimension) & (frame["horizon"] == horizon)]
            groups = [("pooled", current)] + [
                (str(family), values)
                for family, values in current.groupby("family", sort=True)
            ]
            groups.append(("effect_enriched", current[current["effect_enriched"]]))
            aggregates[str(dimension)][str(horizon)] = {}
            for name, values in groups:
                metrics = _aggregate(values, seeds, resamples)
                gate = _causal_gate(metrics, causal_section)
                metrics["causal_gate_pass"] = gate
                aggregates[str(dimension)][str(horizon)][name] = metrics
                if horizon in auth_horizons:
                    dimension_gates.append(gate)
        authorized = bool(dimension_gates and all(dimension_gates))
        authorization[str(dimension)] = {
            "teacher_forced_causal_authorized": authorized,
            "strict_interface_pass": True,
            "free_continuation_status": (
                "NOT_RUN_REQUIRES_SEPARATE_FROZEN_PROTOCOL"
                if authorized
                else "GATED_BY_TEACHER_FORCED_CAUSAL_FIDELITY"
            ),
        }
    authorized_dimensions = [
        int(dimension)
        for dimension, value in authorization.items()
        if value["teacher_forced_causal_authorized"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "run_id": context.run_id,
        "candidate_freeze_digest": candidate_freeze["freeze_digest"],
        "decoder_run_id": decoder_summary["run_id"],
        "strict_interface": {
            "verified": True,
            "function_sha256": strict_hash,
            "allowed_inputs": ["clean_cache", "decoded_delta", "teacher_forced_token"],
            "raw_test_cache_available_to_decoded_continuation": False,
        },
        "teacher_forced_horizons": horizons,
        "aggregates": aggregates,
        "authorization": authorization,
        "authorized_dimensions": authorized_dimensions,
        "smallest_candidate_causal_sufficient_dimension": (
            min(authorized_dimensions) if authorized_dimensions else None
        ),
        "controller_authorized": bool(authorized_dimensions),
        "free_continuation_executed": False,
        "records": str(CAUSAL),
        "records_sha256": sha256_file(context.root / CAUSAL),
    }
    write_json_atomic(context.root / CAUSAL_SUMMARY, summary)
    write_json_atomic(
        progress,
        {"status": "COMPLETED", "completed_cases": len(ids), "total_cases": len(ids)},
    )
    context.finish("COMPLETED_V10_CAUSAL", summary=summary)


def main() -> None:
    parser = standard_parser(
        "decoded compact-state causal validation v10",
        "configs/causal_sufficiency_v10.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("prepare-decoder", "causal"))
    args = parser.parse_args()
    context = initialize_context("decoded-causal-v10", args)
    try:
        candidate_freeze = verify_candidate_freeze(context.root, context.config)
        if args.dry_run:
            context.finish(
                "DRY_RUN", candidate_freeze_digest=candidate_freeze["freeze_digest"]
            )
            return
        if args.stage == "prepare-decoder":
            _prepare_decoder(context, candidate_freeze)
        else:
            _causal(context, candidate_freeze)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
