"""Raw-channel ceiling and causal compression for the v7 mixed cache state."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.arch_compression_v7 import (
    apply_delta_blocks,
    component_order,
    fit_dual_pca,
    flatten_blocks,
    linear_reconstruction_alpha,
    nonlinear_reconstruction_alpha,
    pca_scores,
    reconstruct_blocks,
)
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.persistent_channels_v7 import (
    _candidate_states,
    _prefill,
    _source_items,
    _tasks,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.metrics import jensen_shannon_from_logits, token_log_odds
from jclosure.protocol_v7 import FREEZE_PATH as ATTRIBUTION_FREEZE_PATH
from jclosure.protocol_v7_stage2 import build_stage_freeze, verify_stage_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.records_v7 import PROTOCOL_V7, ChannelCompressionRecord
from jclosure.statistics import clustered_bootstrap_ci


def _cache_delta(
    clean: Any,
    perturbed: Any,
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recurrent = torch.stack(
        [
            perturbed.layers[layer].recurrent_states.detach().float().cpu()
            - clean.layers[layer].recurrent_states.detach().float().cpu()
            for layer in recurrent_layers
        ]
    )[:, 0]
    conv = torch.stack(
        [
            perturbed.layers[layer].conv_states.detach().float().cpu()
            - clean.layers[layer].conv_states.detach().float().cpu()
            for layer in recurrent_layers
        ]
    )[:, 0]
    kv = torch.stack(
        [
            torch.stack(
                (
                    perturbed.layers[layer].keys[0, :, -1].detach().float().cpu()
                    - clean.layers[layer].keys[0, :, -1].detach().float().cpu(),
                    perturbed.layers[layer].values[0, :, -1].detach().float().cpu()
                    - clean.layers[layer].values[0, :, -1].detach().float().cpu(),
                )
            )
            for layer in attention_layers
        ]
    )
    return recurrent, conv, kv


def _paired_ci(values: np.ndarray, seed: int, resamples: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    boot = np.asarray(
        [
            float(np.mean(generator.choice(values, len(values), replace=True)))
            for _ in range(resamples)
        ]
    )
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
        "n_clusters": int(len(values)),
        "n_resamples": int(resamples),
    }


def _cosine_rows(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(predicted * target, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20
    )


def _ridge(
    train: np.ndarray,
    test: np.ndarray,
    train_target: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    mean_x = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    fit = (train - mean_x) / scale
    held = (test - mean_x) / scale
    mean_y = train_target.mean(axis=0, keepdims=True)
    centered_y = train_target - mean_y
    kernel = fit @ fit.T / max(1, fit.shape[1])
    cross = held @ fit.T / max(1, fit.shape[1])
    coefficients = np.linalg.solve(kernel + alpha * np.eye(len(fit)), centered_y)
    return (mean_y + cross @ coefficients).astype(np.float32)


def _raw_ceiling(
    context: Any,
    payload: dict[str, Any],
    blocks: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    split = np.asarray(payload["split"])
    train = np.flatnonzero(split == "localization_fit").tolist()
    test = np.flatnonzero(split == "attribution_test").tolist()
    model = fit_dual_pca(blocks, train)
    scores = pca_scores(model, blocks, list(range(len(split)))).numpy()
    source_path = (
        context.root / context.config["persistent_channels_v7"]["source_v6_artifact"]
    )
    with np.load(source_path, allow_pickle=False) as source:
        source_index = {
            str(value): index for index, value in enumerate(source["base_trial_id"])
        }
        source_order = [source_index[str(value)] for value in payload["base_trial_id"]]
        current_delta = (
            source["j_intervened"][source_order, 0] - source["j_clean"][source_order, 0]
        )
    endpoint_path = (
        context.root
        / json.loads(
            (
                context.root
                / "results/v7/processed/persistent_channel_attribution_v7.json"
            ).read_text()
        )["endpoint_artifact"]
    )
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        endpoint_index = {
            str(value): index for index, value in enumerate(endpoint["base_trial_id"])
        }
        endpoint_order = [
            endpoint_index[str(value)] for value in payload["base_trial_id"]
        ]
        target_delta = (
            endpoint["full_next_j"][endpoint_order]
            - endpoint["clean_next_j"][endpoint_order]
        )
    baseline = _ridge(current_delta[train], current_delta[test], target_delta[train])
    full_features = np.concatenate((current_delta, scores), axis=1)
    full = _ridge(full_features[train], full_features[test], target_delta[train])
    target = target_delta[test]
    baseline_cosine = _cosine_rows(baseline, target)
    full_cosine = _cosine_rows(full, target)
    baseline_rmse = np.sqrt(np.mean((baseline - target) ** 2, axis=1))
    full_rmse = np.sqrt(np.mean((full - target) ** 2, axis=1))
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    cosine_gain = _paired_ci(full_cosine - baseline_cosine, seed, resamples)
    rmse_gain = _paired_ci(baseline_rmse - full_rmse, seed, resamples)
    attribution = json.loads(
        (
            context.root / "results/v7/processed/persistent_channel_attribution_v7.json"
        ).read_text()
    )
    authorized = bool(
        attribution["classification"] != "transient_or_unresolved"
        and (cosine_gain["lower"] > 0 or rmse_gain["lower"] > 0)
    )
    return {
        "effective_raw_rank": model.rank,
        "train_count": len(train),
        "test_count": len(test),
        "j_only_delta_cosine": _paired_ci(baseline_cosine, seed, resamples),
        "full_channel_delta_cosine": _paired_ci(full_cosine, seed, resamples),
        "delta_cosine_gain": cosine_gain,
        "j_only_delta_rmse": _paired_ci(baseline_rmse, seed, resamples),
        "full_channel_delta_rmse": _paired_ci(full_rmse, seed, resamples),
        "delta_rmse_improvement": rmse_gain,
        "direct_causal_swap_ceiling": attribution["effects"]["pooled"],
        "raw_channel_ceiling_authorized": authorized,
        "scope_note": "Predictive ceiling is intervention-delta prediction on 33 held-out pairs; direct cache swapping is the causal ceiling.",
    }


def _capture(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    parent = json.loads((context.root / ATTRIBUTION_FREEZE_PATH).read_text())
    items = _source_items(context, parent)
    tasks = _tasks(context)
    _, candidates = _candidate_states(context)
    section = context.config["persistent_channels_v7"]
    rec_layers = [int(value) for value in section["downstream_recurrent_layers"]]
    kv_layers = [int(value) for value in section["downstream_attention_layers"]]
    measured = [int(value) for value in section["measured_layers"]]
    _, _, dense_map = _load_encoder(context, bundle)
    device = next(bundle.hf_model.parameters()).device
    recurrent_values = []
    conv_values = []
    kv_values = []
    meta = []
    for item in items:
        task = tasks[item["prompt_id"]]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        perturbed = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=torch.from_numpy(candidates[int(item["artifact_index"])]).to(
                device
            ),
        )
        recurrent, conv, kv = _cache_delta(
            clean["cache"], perturbed["cache"], rec_layers, kv_layers
        )
        recurrent_values.append(recurrent)
        conv_values.append(conv)
        kv_values.append(kv)
        meta.append(item)
    tensor_payload = {
        "recurrent": torch.stack(recurrent_values),
        "conv": torch.stack(conv_values),
        "kv": torch.stack(kv_values),
    }
    payload: dict[str, Any] = {
        **tensor_payload,
        "base_trial_id": [item["base_trial_id"] for item in meta],
        "prompt_id": [item["prompt_id"] for item in meta],
        "family": [item["family"] for item in meta],
        "split": [item["role"] for item in meta],
        "recurrent_layers": rec_layers,
        "attention_layers": kv_layers,
    }
    artifact_dir = context.root / "artifacts/persistent/v7" / context.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / "architecture_channel_delta_f32.pt"
    torch.save(payload, artifact)
    blocks = flatten_blocks(tensor_payload)
    raw = _raw_ceiling(context, payload, blocks)
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "artifact": str(artifact.relative_to(context.root)),
        "artifact_sha256": sha256_file(artifact),
        "shapes": {
            key: list(tensor_payload[key].shape) for key in ("recurrent", "conv", "kv")
        },
        "storage_dtype": "float32",
        "raw_ceiling": raw,
    }
    write_json_atomic(
        context.processed_dir / "raw_arch_channel_ceiling_v7.json", summary
    )
    context.finish("COMPLETED_CHANNEL_CAPTURE", **summary)


def _compression_alpha(
    method: str,
    dimension: int,
    model: Any,
    train_scores: torch.Tensor,
    test_scores: torch.Tensor,
    predictive_order: torch.Tensor,
    causal_order: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    effective = min(int(dimension), model.rank)
    if method == "pca":
        selected = torch.arange(effective)
    elif method == "predictive_bottleneck":
        selected = predictive_order[:effective]
    elif method == "causal_bottleneck":
        selected = causal_order[:effective]
    elif method == "nonlinear_encoder":
        return nonlinear_reconstruction_alpha(
            model, train_scores, test_scores, effective, seed=seed
        )
    else:
        raise ValueError(method)
    return linear_reconstruction_alpha(model, test_scores, selected), {
        "effective_dimension": effective,
        "rank_limited": dimension > model.rank,
        "selected_components": selected.tolist(),
    }


def _run(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    capture = json.loads(
        (context.processed_dir / "raw_arch_channel_ceiling_v7.json").read_text()
    )
    if not capture["raw_ceiling"]["raw_channel_ceiling_authorized"]:
        context.finish(
            "GATED_NO_STRONG_RAW_CEILING", source_freeze_digest=freeze["freeze_digest"]
        )
        return
    artifact = context.root / capture["artifact"]
    if sha256_file(artifact) != capture["artifact_sha256"]:
        raise RuntimeError("architecture-channel artifact hash mismatch")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    blocks = flatten_blocks(payload)
    split = np.asarray(payload["split"])
    train = np.flatnonzero(split == "localization_fit").tolist()
    test = np.flatnonzero(split == "attribution_test").tolist()
    model = fit_dual_pca(blocks, train)
    train_scores = pca_scores(model, blocks, train)
    test_scores = pca_scores(model, blocks, test)
    attribution = json.loads(
        (context.processed_dir / "persistent_channel_attribution_v7.json").read_text()
    )
    endpoint_path = context.root / attribution["endpoint_artifact"]
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        endpoint_ids = [str(value) for value in endpoint["base_trial_id"]]
        endpoint_index = {value: index for index, value in enumerate(endpoint_ids)}
        endpoint_order = [
            endpoint_index[str(value)] for value in payload["base_trial_id"]
        ]
        full_next = endpoint["full_next_j"][endpoint_order].astype(np.float32)
        clean_next = endpoint["clean_next_j"][endpoint_order].astype(np.float32)
    target_delta = torch.from_numpy(full_next - clean_next)
    target_absolute = torch.from_numpy(full_next)
    predictive_order = component_order(train_scores, target_absolute[train])
    causal_order = component_order(train_scores, target_delta[train])
    section = context.config["persistent_channels_v7"]
    rec_layers = [int(value) for value in payload["recurrent_layers"]]
    kv_layers = [int(value) for value in payload["attention_layers"]]
    methods = [str(value) for value in section["compression"]["methods"]]
    dimensions = [int(value) for value in section["compression"]["dimensions"]]
    coefficients = {}
    metadata = {}
    seed = int(context.seed)
    for method in methods:
        for dimension in dimensions:
            key = (method, dimension)
            coefficients[key], metadata[key] = _compression_alpha(
                method,
                dimension,
                model,
                train_scores,
                test_scores,
                predictive_order,
                causal_order,
                seed,
            )
    parent = json.loads((context.root / ATTRIBUTION_FREEZE_PATH).read_text())
    items = {item["base_trial_id"]: item for item in _source_items(context, parent)}
    tasks = _tasks(context)
    measured = [int(value) for value in section["measured_layers"]]
    main_layer = max(measured)
    _, _, dense_map = _load_encoder(context, bundle)
    records = []
    for test_row, global_index in enumerate(test):
        base_id = str(payload["base_trial_id"][global_index])
        item = items[base_id]
        task = tasks[item["prompt_id"]]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle, clean, count=1, measured_layers=measured, dense_map=dense_map
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        observed = tuple(block[global_index] for block in blocks)
        full_trajectory = _teacher_forced_trajectory(
            bundle,
            apply_delta_blocks(
                clean["cache"],
                observed,
                recurrent_layers=rec_layers,
                attention_layers=kv_layers,
            ),
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_j = clean_trajectory["j"][main_layer][0]
        full_j = full_trajectory["j"][main_layer][0]
        full_delta = full_j - clean_j
        full_norm = float(np.linalg.norm(full_delta))
        clean_logits = clean_trajectory["logits"][0]
        full_logits = full_trajectory["logits"][0]
        target_token = int(np.argmax(clean_logits))
        full_odds_delta = token_log_odds(
            torch.from_numpy(full_logits), target_token
        ) - token_log_odds(torch.from_numpy(clean_logits), target_token)
        for method in methods:
            for dimension in dimensions:
                key = (method, dimension)
                reconstructed = reconstruct_blocks(model, coefficients[key], test_row)
                trajectory = _teacher_forced_trajectory(
                    bundle,
                    apply_delta_blocks(
                        clean["cache"],
                        reconstructed,
                        recurrent_layers=rec_layers,
                        attention_layers=kv_layers,
                    ),
                    tokens,
                    prompt_length=clean["prompt_length"],
                    measured_layers=measured,
                    dense_map=dense_map,
                )
                candidate_j = trajectory["j"][main_layer][0]
                delta = candidate_j - clean_j
                norm = float(np.linalg.norm(delta))
                cosine = float(np.dot(delta, full_delta) / max(norm * full_norm, 1e-20))
                magnitude = norm / max(full_norm, 1e-20)
                error_ratio = float(
                    np.linalg.norm(candidate_j - full_j) / max(full_norm, 1e-20)
                )
                gap = 1 - error_ratio
                fidelity = cosine * float(np.exp(-abs(np.log(max(magnitude, 1e-20)))))
                candidate_logits = trajectory["logits"][0]
                odds_delta = token_log_odds(
                    torch.from_numpy(candidate_logits), target_token
                ) - token_log_odds(torch.from_numpy(clean_logits), target_token)
                record = ChannelCompressionRecord(
                    run_id=context.run_id,
                    channel="mixed_kv_recurrent_conv",
                    method=method,
                    dimension=dimension,
                    split="attribution_test",
                    predictive_gap_closed=gap,
                    causal_gap_closed=fidelity,
                    conditional_residual_gain=error_ratio,
                    causal_direction_cosine=cosine,
                    magnitude_ratio=magnitude,
                    semantic_delta_agreement=float(
                        (np.argmax(candidate_logits) != np.argmax(clean_logits))
                        == (np.argmax(full_logits) != np.argmax(clean_logits))
                    ),
                    output_sign_agreement=float(
                        np.sign(odds_delta) == np.sign(full_odds_delta)
                    ),
                    metadata={
                        **metadata[key],
                        "base_trial_id": base_id,
                        "prompt_id": item["prompt_id"],
                        "family": item["family"],
                        "full_delta_l2": full_norm,
                        "output_js_to_full": jensen_shannon_from_logits(
                            torch.from_numpy(candidate_logits),
                            torch.from_numpy(full_logits),
                        ),
                    },
                )
                records.append(
                    {**record.to_dict(), "record_type": "channel_compression_trial"}
                )
    path = context.raw_dir / context.run_id / "arch_compression_v7.jsonl"
    append_jsonl(path, records)
    context.finish(
        "COMPLETED_COMPRESSION",
        source_freeze_digest=freeze["freeze_digest"],
        records=str(path.relative_to(context.root)),
        record_count=len(records),
        effective_sample_rank=model.rank,
    )


def _ci(frame: pd.DataFrame, column: str, seed: int, resamples: int) -> dict[str, Any]:
    return asdict(
        clustered_bootstrap_ci(
            frame,
            cluster_col="prompt_id",
            value_col=column,
            seed=seed,
            n_resamples=resamples,
        )
    )


def _analyze(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("arch-compression-v7-*/manifest.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "COMPLETED_COMPRESSION":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed architecture compression run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"]).read_text().splitlines()
    ]
    frame = pd.DataFrame(rows)
    frame["prompt_id"] = frame["metadata"].map(lambda value: value["prompt_id"])
    frame["family"] = frame["metadata"].map(lambda value: value["family"])
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    summaries = []
    metrics = (
        "predictive_gap_closed",
        "causal_gap_closed",
        "conditional_residual_gain",
        "causal_direction_cosine",
        "magnitude_ratio",
        "semantic_delta_agreement",
        "output_sign_agreement",
    )
    for (method, dimension), values in frame.groupby(
        ["method", "dimension"], sort=True
    ):
        summaries.append(
            {
                "method": method,
                "dimension": int(dimension),
                "effective_dimension": int(
                    values.iloc[0]["metadata"]["effective_dimension"]
                ),
                **{metric: _ci(values, metric, seed, resamples) for metric in metrics},
            }
        )
    rules = context.config["persistent_channels_v7"]["compression"]
    passing = [
        value
        for value in summaries
        if value["predictive_gap_closed"]["lower"]
        >= float(rules["predictive_gap_closed"])
        and value["causal_gap_closed"]["lower"] >= float(rules["causal_gap_closed"])
        and value["conditional_residual_gain"]["upper"]
        <= float(rules["maximum_conditional_gain"])
        and value["causal_direction_cosine"]["lower"]
        >= float(rules["causal_direction_minimum"])
    ]
    selected = min(passing, key=lambda value: value["dimension"]) if passing else None
    output = frame.copy()
    output["metadata"] = output["metadata"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    path = context.processed_dir / "arch_compression_v7.parquet"
    output.to_parquet(path, index=False, compression="zstd")
    capture = json.loads(
        (context.processed_dir / "raw_arch_channel_ceiling_v7.json").read_text()
    )
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "source_freeze_digest": source["source_freeze_digest"],
        "effective_sample_rank": source["effective_sample_rank"],
        "raw_ceiling": capture["raw_ceiling"],
        "summaries": summaries,
        "passing_count": len(passing),
        "selected": selected,
        "candidate_sufficient_persistent_state": selected is not None,
        "autonomous_controller_authorized": selected is not None,
        "records": str(path.relative_to(context.root)),
        "limitation": "All cache-delta encoders are fit on 33 pairs, so declared dimensions above the empirical rank are rank-limited and cannot establish a 64-512D state.",
    }
    write_json_atomic(context.processed_dir / "arch_compression_v7.json", summary)
    context.finish("COMPLETED_COMPRESSION_ANALYSIS", summary=summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "v7 architecture-aligned compression", "configs/persistent_channels_v7.yaml"
    )
    parser.add_argument(
        "--stage", choices=("freeze", "capture", "run", "analyze"), required=True
    )
    args = parser.parse_args()
    context = initialize_context("arch-compression-v7", args)
    try:
        if args.stage == "freeze":
            freeze = build_stage_freeze(context.root, context.config, "compression")
            context.finish("COMPLETED_FREEZE", freeze=freeze)
            return
        freeze = verify_stage_freeze(context.root, context.config, "compression")
        if args.stage == "analyze":
            _analyze(context, freeze)
            return
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        from jclosure.model import load_model_bundle

        bundle = load_model_bundle(context.config)
        if args.stage == "capture":
            _capture(context, bundle, freeze)
        else:
            _run(context, bundle, freeze)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
