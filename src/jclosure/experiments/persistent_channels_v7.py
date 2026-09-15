"""Qwen3.5 hybrid-cache restoration and KV/REC causal attribution."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from jclosure.cache_v7 import (
    CONV_ATTRIBUTES,
    KV_ATTRIBUTES,
    RECURRENT_ATTRIBUTES,
    cache_component_differences,
    caches_exact,
    clone_hybrid_cache,
    describe_cache,
    make_chimeric_cache,
)
from jclosure.datasets_v5 import load_replication_tasks
from jclosure.experiments.causal_single_v4 import _load_encoder, _replacement
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.metrics import jensen_shannon_from_logits, token_log_odds
from jclosure.protocol_v7 import (
    build_freeze,
    build_v6_guard,
    verify_freeze,
)
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.records_v7 import (
    PROTOCOL_V7,
    CacheRestoreRecord,
    ChannelAttributionRecord,
)
from jclosure.runtime_v3_1 import encode_direct_prompt
from jclosure.statistics import clustered_bootstrap_ci


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-20:
        return None
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def _model_layer_types(bundle: Any) -> list[str]:
    return [
        str(value) for value in bundle.hf_model.config.get_text_config().layer_types
    ]


def _source_items(context: Any, freeze: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads(
        (context.root / freeze["split_manifest"]).read_text(encoding="utf-8")
    )
    return list(manifest["items"])


def _tasks(context: Any) -> dict[str, Any]:
    path = context.root / context.config["persistent_channels_v7"]["source_tasks"]
    return {task.example_id: task for task in load_replication_tasks(path)}


def _candidate_states(context: Any) -> tuple[np.ndarray, np.ndarray]:
    section = context.config["persistent_channels_v7"]
    path = context.root / section["source_v6_artifact"]
    if sha256_file(path) != section["source_v6_artifact_sha256"]:
        raise RuntimeError("source v6 endpoint artifact hash mismatch")
    with np.load(path, allow_pickle=False) as values:
        return (
            values["hidden_clean"][:, 0].astype(np.float32),
            values["hidden_intervened"][:, 0].astype(np.float32),
        )


@torch.no_grad()
def _prefill(
    bundle: Any,
    prompt: str,
    *,
    measured_layers: list[int],
    dense_map: Any,
    intervention_layer: int,
    candidate: torch.Tensor | None,
) -> dict[str, Any]:
    input_ids = encode_direct_prompt(bundle, prompt)
    transforms = (
        {intervention_layer: _replacement(candidate)} if candidate is not None else {}
    )
    with (
        ResidualEditor(bundle.layers, transforms),
        ActivationRecorder(bundle.layers, at=measured_layers) as recorder,
    ):
        output = bundle.hf_model(input_ids=input_ids, use_cache=True)
    if output.past_key_values is None:
        raise RuntimeError("model did not return a hybrid cache")
    return {
        "input_ids": input_ids,
        "prompt_length": int(input_ids.shape[1]),
        "logits": output.logits[0, -1].detach().float(),
        "cache": clone_hybrid_cache(output.past_key_values),
        "hidden": {
            layer: recorder.activations[layer][0, -1].detach().float()
            for layer in measured_layers
        },
        "j": {
            layer: dense_map.dense_state(
                recorder.activations[layer][0, -1].detach().float(), layer
            )
            for layer in measured_layers
        },
    }


@torch.no_grad()
def _continue_one(
    bundle: Any,
    cache: Any,
    token: int,
    *,
    prompt_length: int,
    step: int,
    measured_layers: list[int],
    dense_map: Any,
) -> dict[str, Any]:
    device = next(bundle.hf_model.parameters()).device
    token_tensor = torch.tensor([[int(token)]], device=device)
    attention_mask = torch.ones(
        (1, prompt_length + step + 1), dtype=torch.long, device=device
    )
    with ActivationRecorder(bundle.layers, at=measured_layers) as recorder:
        output = bundle.hf_model(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
    if output.past_key_values is None:
        raise RuntimeError("one-token continuation did not return a cache")
    hidden = {
        layer: recorder.activations[layer][0, -1].detach().float()
        for layer in measured_layers
    }
    return {
        "logits": output.logits[0, -1].detach().float(),
        "hidden": hidden,
        "j": {
            layer: dense_map.dense_state(hidden[layer], layer)
            for layer in measured_layers
        },
        "cache": output.past_key_values,
    }


@torch.no_grad()
def _teacher_tokens(
    bundle: Any,
    prefill: dict[str, Any],
    *,
    count: int,
    measured_layers: list[int],
    dense_map: Any,
) -> list[int]:
    cache = clone_hybrid_cache(prefill["cache"])
    logits = prefill["logits"]
    tokens: list[int] = []
    for step in range(count):
        token = int(torch.argmax(logits).item())
        tokens.append(token)
        continuation = _continue_one(
            bundle,
            cache,
            token,
            prompt_length=prefill["prompt_length"],
            step=step,
            measured_layers=measured_layers,
            dense_map=dense_map,
        )
        logits = continuation["logits"]
        cache = continuation["cache"]
    return tokens


@torch.no_grad()
def _teacher_forced_trajectory(
    bundle: Any,
    cache: Any,
    tokens: list[int],
    *,
    prompt_length: int,
    measured_layers: list[int],
    dense_map: Any,
) -> dict[str, Any]:
    working = clone_hybrid_cache(cache)
    logits: list[np.ndarray] = []
    j_states: dict[int, list[np.ndarray]] = {layer: [] for layer in measured_layers}
    hidden_states: dict[int, list[np.ndarray]] = {
        layer: [] for layer in measured_layers
    }
    for step, token in enumerate(tokens):
        result = _continue_one(
            bundle,
            working,
            token,
            prompt_length=prompt_length,
            step=step,
            measured_layers=measured_layers,
            dense_map=dense_map,
        )
        working = result["cache"]
        logits.append(result["logits"].cpu().numpy().astype(np.float32))
        for layer in measured_layers:
            j_states[layer].append(result["j"][layer].cpu().numpy().astype(np.float32))
            hidden_states[layer].append(
                result["hidden"][layer].cpu().numpy().astype(np.float32)
            )
    return {
        "logits": np.stack(logits),
        "j": {layer: np.stack(values) for layer, values in j_states.items()},
        "hidden": {layer: np.stack(values) for layer, values in hidden_states.items()},
        "cache": working,
    }


def _trajectory_metrics(
    clean: dict[str, Any],
    candidate: dict[str, Any],
    *,
    main_layer: int,
) -> dict[str, Any]:
    js_curve = [
        jensen_shannon_from_logits(torch.from_numpy(left), torch.from_numpy(right))
        for left, right in zip(clean["logits"], candidate["logits"], strict=True)
    ]
    layer_curve: dict[str, list[float]] = {}
    for layer in clean["j"]:
        layer_curve[str(layer)] = [
            float(
                1
                - F.cosine_similarity(
                    torch.from_numpy(left)[None], torch.from_numpy(right)[None]
                ).item()
            )
            for left, right in zip(
                clean["j"][layer], candidate["j"][layer], strict=True
            )
        ]
    clean_argmax = np.argmax(clean["logits"], axis=1)
    candidate_argmax = np.argmax(candidate["logits"], axis=1)
    target_odds = [
        token_log_odds(torch.from_numpy(candidate["logits"][index]), int(token))
        - token_log_odds(torch.from_numpy(clean["logits"][index]), int(token))
        for index, token in enumerate(clean_argmax)
    ]
    next_delta = candidate["j"][main_layer][0] - clean["j"][main_layer][0]
    return {
        "next_j_l2": float(np.linalg.norm(next_delta)),
        "future_j_trajectory_divergence": float(
            np.mean([value for values in layer_curve.values() for value in values])
        ),
        "future_j_layer_curve": layer_curve,
        "output_js_divergence": float(np.mean(js_curve)),
        "output_js_curve": js_curve,
        "target_log_odds_change": float(np.mean(target_odds)),
        "target_log_odds_abs_change": float(abs(np.mean(target_odds))),
        "answer_flip": bool(candidate_argmax[-1] != clean_argmax[-1]),
        "token_flip_rate": float(np.mean(candidate_argmax != clean_argmax)),
        "next_j_delta": next_delta.astype(np.float32),
        "output_delta": (candidate["logits"][0] - clean["logits"][0]).astype(
            np.float32
        ),
    }


def _write_cache_schema(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    items = _source_items(context, freeze)
    tasks = _tasks(context)
    _, _, dense_map = _load_encoder(context, bundle)
    section = context.config["persistent_channels_v7"]
    layers = [int(value) for value in section["measured_layers"]]
    item = items[0]
    prefill = _prefill(
        bundle,
        tasks[item["prompt_id"]].prompt,
        measured_layers=layers,
        dense_map=dense_map,
        intervention_layer=int(section["intervention_layer"]),
        candidate=None,
    )
    rows = describe_cache(prefill["cache"], _model_layer_types(bundle))
    frame = pd.DataFrame(rows)
    output = context.processed_dir / "cache_schema_v7.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "prompt_id": item["prompt_id"],
        "sequence_length": prefill["prompt_length"],
        "layer_types": _model_layer_types(bundle),
        "component_counts": frame.groupby("component").size().to_dict(),
        "component_bytes": frame.groupby("component")["bytes"].sum().to_dict(),
        "total_bytes": int(frame["bytes"].sum()),
        "records": str(output.relative_to(context.root)),
    }
    write_json_atomic(context.processed_dir / "cache_schema_v7.json", summary)
    context.finish("COMPLETED_SCHEMA", **summary)


def _run_restore(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    items = _source_items(context, freeze)
    section = context.config["persistent_channels_v7"]
    count = int(section["restore_test_count"])
    selected = [item for item in items if item["role"] == "attribution_test"][:count]
    tasks = _tasks(context)
    _, _, dense_map = _load_encoder(context, bundle)
    layers = [int(value) for value in section["measured_layers"]]
    tolerances = section["cache_restore_tolerances"]
    records = []
    for item in selected:
        task = tasks[item["prompt_id"]]
        prefill = _prefill(
            bundle,
            task.prompt,
            measured_layers=layers,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        token = int(torch.argmax(prefill["logits"]).item())
        left_cache = clone_hybrid_cache(prefill["cache"])
        right_cache = clone_hybrid_cache(prefill["cache"])
        cloned_kv = caches_exact(left_cache, right_cache, KV_ATTRIBUTES)
        cloned_rec = caches_exact(left_cache, right_cache, RECURRENT_ATTRIBUTES)
        cloned_conv = caches_exact(left_cache, right_cache, CONV_ATTRIBUTES)
        left = _continue_one(
            bundle,
            left_cache,
            token,
            prompt_length=prefill["prompt_length"],
            step=0,
            measured_layers=layers,
            dense_map=dense_map,
        )
        right = _continue_one(
            bundle,
            right_cache,
            token,
            prompt_length=prefill["prompt_length"],
            step=0,
            measured_layers=layers,
            dense_map=dense_map,
        )
        logits_error = float(
            torch.max(torch.abs(left["logits"] - right["logits"])).item()
        )
        hidden_error = max(
            float(
                torch.max(
                    torch.abs(left["hidden"][layer] - right["hidden"][layer])
                ).item()
            )
            for layer in layers
        )
        j_error = max(
            float(torch.max(torch.abs(left["j"][layer] - right["j"][layer])).item())
            for layer in layers
        )
        cache_kv = caches_exact(left["cache"], right["cache"], KV_ATTRIBUTES)
        cache_rec = caches_exact(left["cache"], right["cache"], RECURRENT_ATTRIBUTES)
        cache_conv = caches_exact(left["cache"], right["cache"], CONV_ATTRIBUTES)
        passed = bool(
            cloned_kv
            and cloned_rec
            and cloned_conv
            and cache_kv
            and cache_rec
            and cache_conv
            and logits_error <= float(tolerances["logits_max_abs"])
            and hidden_error <= float(tolerances["hidden_max_abs"])
            and j_error <= float(tolerances["j_max_abs"])
        )
        records.append(
            {
                **CacheRestoreRecord(
                    run_id=context.run_id,
                    prompt_id=item["prompt_id"],
                    family=item["family"],
                    token_id=token,
                    passed=passed,
                    logits_max_abs=logits_error,
                    hidden_max_abs=hidden_error,
                    j_max_abs=j_error,
                    recurrent_exact=bool(cloned_rec and cache_rec),
                    conv_exact=bool(cloned_conv and cache_conv),
                    kv_exact=bool(cloned_kv and cache_kv),
                    metadata={"one_token_only": True, "measured_layers": layers},
                    exclusion_reason=None if passed else "restore_mismatch",
                ).to_dict(),
                "record_type": "cache_restore_test",
            }
        )
    raw = context.raw_dir / context.run_id / "cache_restore_v7.jsonl"
    append_jsonl(raw, records)
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "attempted": len(records),
        "passed": sum(bool(value["passed"]) for value in records),
        "all_passed": bool(records and all(value["passed"] for value in records)),
        "maximum_logits_error": max(value["logits_max_abs"] for value in records),
        "maximum_hidden_error": max(value["hidden_max_abs"] for value in records),
        "maximum_j_error": max(value["j_max_abs"] for value in records),
        "records": str(raw.relative_to(context.root)),
    }
    write_json_atomic(context.processed_dir / "cache_restore_v7.json", summary)
    context.finish(
        "COMPLETED_RESTORE" if summary["all_passed"] else "FAILED_RESTORE", **summary
    )


def _condition_cache(clean: Any, perturbed: Any, condition: str) -> Any:
    if condition == "clean":
        return clone_hybrid_cache(clean)
    if condition == "kv_only":
        return make_chimeric_cache(clean, perturbed, kv_from_perturbed=True)
    if condition == "recurrent_only":
        return make_chimeric_cache(
            clean,
            perturbed,
            recurrent_from_perturbed=True,
            conv_from_perturbed=True,
        )
    if condition == "full":
        return make_chimeric_cache(
            clean,
            perturbed,
            kv_from_perturbed=True,
            recurrent_from_perturbed=True,
            conv_from_perturbed=True,
        )
    raise ValueError(f"unknown cache condition {condition}")


def _run_attribution(
    context: Any, bundle: Any, freeze: dict[str, Any], limit: int | None
) -> None:
    restore = json.loads(
        (context.processed_dir / "cache_restore_v7.json").read_text(encoding="utf-8")
    )
    if not restore.get("all_passed"):
        raise RuntimeError("cache restore correctness gate did not pass")
    items = _source_items(context, freeze)
    if limit is not None:
        items = items[: int(limit)]
    tasks = _tasks(context)
    clean_source, candidate_source = _candidate_states(context)
    section = context.config["persistent_channels_v7"]
    layers = [int(value) for value in section["measured_layers"]]
    main_layer = max(layers)
    _, _, dense_map = _load_encoder(context, bundle)
    device = next(bundle.hf_model.parameters()).device
    records: list[dict[str, Any]] = []
    endpoint_meta: list[dict[str, Any]] = []
    endpoint_arrays: dict[str, list[np.ndarray]] = {
        f"{condition}_{value}": []
        for condition in ("clean", "kv_only", "recurrent_only", "full")
        for value in ("next_j", "next_logits")
    }
    for item in items:
        task = tasks[item["prompt_id"]]
        index = int(item["artifact_index"])
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=layers,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        source_error = float(
            np.max(
                np.abs(
                    clean["hidden"][int(section["intervention_layer"])]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    - clean_source[index]
                )
            )
        )
        candidate = torch.from_numpy(candidate_source[index]).to(device)
        perturbed = _prefill(
            bundle,
            task.prompt,
            measured_layers=layers,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=candidate,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=int(section["teacher_forced_tokens"]),
            measured_layers=layers,
            dense_map=dense_map,
        )
        trajectories: dict[str, dict[str, Any]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        differences = cache_component_differences(clean["cache"], perturbed["cache"])
        for condition in ("clean", "kv_only", "recurrent_only", "full"):
            cache = _condition_cache(clean["cache"], perturbed["cache"], condition)
            trajectory = _teacher_forced_trajectory(
                bundle,
                cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=layers,
                dense_map=dense_map,
            )
            trajectories[condition] = trajectory
        clean_trajectory = trajectories["clean"]
        for condition, trajectory in trajectories.items():
            metrics[condition] = _trajectory_metrics(
                clean_trajectory, trajectory, main_layer=main_layer
            )
        full_delta = metrics["full"]["next_j_delta"]
        full_output_delta = metrics["full"]["output_delta"]
        full_norm = float(np.linalg.norm(full_delta))
        for condition in ("clean", "kv_only", "recurrent_only", "full"):
            values = metrics[condition]
            delta = values.pop("next_j_delta")
            output_delta = values.pop("output_delta")
            cosine = _safe_cosine(delta, full_delta)
            magnitude = float(np.linalg.norm(delta) / max(full_norm, 1e-20))
            output_cosine = _safe_cosine(output_delta, full_output_delta)
            values.update(
                {
                    "direction_cosine_to_full": cosine,
                    "magnitude_ratio_to_full": magnitude,
                    "output_delta_cosine_to_full": output_cosine,
                    "full_next_j_delta_l2": full_norm,
                }
            )
            records.append(
                {
                    **ChannelAttributionRecord(
                        run_id=context.run_id,
                        base_trial_id=item["base_trial_id"],
                        prompt_id=item["prompt_id"],
                        family=item["family"],
                        split=item["role"],
                        condition=condition,
                        valid=bool(source_error <= 1e-5 and full_norm > 0),
                        cache_component=condition,
                        metrics=values,
                        cache_differences=differences if condition == "full" else [],
                        metadata={
                            "teacher_forced_tokens": tokens,
                            "teacher_forced_token_surfaces": [
                                bundle.tokenizer.decode([token]) for token in tokens
                            ],
                            "source_clean_max_abs": source_error,
                            "main_measured_layer": main_layer,
                            "position_scope": "final",
                        },
                        exclusion_reason=None
                        if source_error <= 1e-5 and full_norm > 0
                        else "source_reproduction_or_full_delta_failure",
                    ).to_dict(),
                    "record_type": "channel_attribution_trial",
                }
            )
            endpoint_arrays[f"{condition}_next_j"].append(
                trajectory["j"][main_layer][0]
            )
            endpoint_arrays[f"{condition}_next_logits"].append(trajectory["logits"][0])
        interaction_delta = (
            metrics["full"]["next_j_l2"]
            - metrics["kv_only"]["next_j_l2"]
            - metrics["recurrent_only"]["next_j_l2"]
        )
        vector_interaction = (
            trajectories["full"]["j"][main_layer][0]
            - trajectories["kv_only"]["j"][main_layer][0]
            - trajectories["recurrent_only"]["j"][main_layer][0]
            + trajectories["clean"]["j"][main_layer][0]
        )
        records.append(
            {
                "schema_version": 9,
                "protocol_version": PROTOCOL_V7,
                "record_type": "channel_interaction_trial",
                "run_id": context.run_id,
                "base_trial_id": item["base_trial_id"],
                "prompt_id": item["prompt_id"],
                "family": item["family"],
                "split": item["role"],
                "valid": bool(source_error <= 1e-5 and full_norm > 0),
                "scalar_next_j_interaction": float(interaction_delta),
                "vector_interaction_l2": float(np.linalg.norm(vector_interaction)),
                "interaction_ratio": float(
                    np.linalg.norm(vector_interaction) / max(full_norm, 1e-20)
                ),
                "output_js_interaction": float(
                    metrics["full"]["output_js_divergence"]
                    - metrics["kv_only"]["output_js_divergence"]
                    - metrics["recurrent_only"]["output_js_divergence"]
                ),
            }
        )
        endpoint_meta.append(item)
    raw = context.raw_dir / context.run_id / "persistent_channel_v7.jsonl"
    append_jsonl(raw, records)
    artifact_dir = context.root / "artifacts/persistent/v7" / context.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    endpoint = artifact_dir / "channel_endpoints_f32.npz"
    np.savez_compressed(
        endpoint,
        **{
            key: np.stack(values).astype(np.float32)
            for key, values in endpoint_arrays.items()
        },
        base_trial_id=np.asarray([item["base_trial_id"] for item in endpoint_meta]),
        prompt_id=np.asarray([item["prompt_id"] for item in endpoint_meta]),
        family=np.asarray([item["family"] for item in endpoint_meta]),
        split=np.asarray([item["role"] for item in endpoint_meta]),
    )
    context.finish(
        "COMPLETED_ATTRIBUTION",
        records=str(raw.relative_to(context.root)),
        record_count=len(records),
        endpoint_artifact=str(endpoint.relative_to(context.root)),
        endpoint_artifact_sha256=sha256_file(endpoint),
        source_freeze_digest=freeze["freeze_digest"],
    )


def _ci(
    frame: pd.DataFrame, column: str, *, seed: int, resamples: int
) -> dict[str, Any]:
    if frame.empty:
        return {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
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
    for path in context.raw_dir.glob("persistent-channels-v7-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_ATTRIBUTION":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed v7 attribution run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    trials = pd.DataFrame(
        value for value in rows if value["record_type"] == "channel_attribution_trial"
    )
    interactions = pd.DataFrame(
        value for value in rows if value["record_type"] == "channel_interaction_trial"
    )
    for column in (
        "next_j_l2",
        "future_j_trajectory_divergence",
        "output_js_divergence",
        "target_log_odds_change",
        "target_log_odds_abs_change",
        "answer_flip",
        "token_flip_rate",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
        "full_next_j_delta_l2",
    ):
        trials[column] = trials["metrics"].map(lambda value, key=column: value.get(key))
    formal = trials[(trials["split"] == "attribution_test") & trials["valid"]]
    formal_interactions = interactions[
        (interactions["split"] == "attribution_test") & interactions["valid"]
    ]
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    effects: dict[str, Any] = {}
    groups = [("pooled", formal), *list(formal.groupby("family", sort=True))]
    metrics = (
        "next_j_l2",
        "future_j_trajectory_divergence",
        "output_js_divergence",
        "target_log_odds_abs_change",
        "answer_flip",
        "token_flip_rate",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
    )
    for name, group in groups:
        effects[str(name)] = {
            metric: {
                condition: _ci(
                    values.dropna(subset=[metric]),
                    metric,
                    seed=seed,
                    resamples=resamples,
                )
                for condition, values in group.groupby("condition", sort=True)
            }
            for metric in metrics
        }
        related = (
            formal_interactions
            if name == "pooled"
            else formal_interactions[formal_interactions["family"] == name]
        )
        effects[str(name)]["interaction_ratio"] = _ci(
            related, "interaction_ratio", seed=seed, resamples=resamples
        )
        effects[str(name)]["output_js_interaction"] = _ci(
            related, "output_js_interaction", seed=seed, resamples=resamples
        )
    gate = context.config["persistent_channels_v7"]["channel_gate"]
    pooled = effects["pooled"]

    def stable(condition: str) -> bool:
        direction = pooled["direction_cosine_to_full"][condition]
        magnitude = pooled["magnitude_ratio_to_full"][condition]
        return bool(
            direction["estimate"] is not None
            and direction["estimate"] >= float(gate["minimum_delta_cosine"])
            and direction["lower"] > float(gate["minimum_delta_cosine_ci_lower"])
            and magnitude["estimate"] >= float(gate["minimum_magnitude_ratio"])
        )

    kv_stable = stable("kv_only")
    recurrent_stable = stable("recurrent_only")
    interaction = effects["pooled"]["interaction_ratio"]["estimate"]
    if kv_stable and recurrent_stable:
        classification = (
            "mixed_interacting"
            if interaction is not None
            and interaction > float(gate["maximum_additive_interaction_ratio"])
            else "mixed_factorized"
        )
    elif kv_stable:
        classification = "kv_dominant"
    elif recurrent_stable:
        classification = "recurrent_dominant"
    else:
        classification = "transient_or_unresolved"
    serial = trials.copy()
    for column in ("metrics", "cache_differences", "metadata"):
        serial[column] = serial[column].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    trial_path = context.processed_dir / "persistent_channel_attribution_v7.parquet"
    serial.to_parquet(trial_path, index=False, compression="zstd")
    interaction_path = (
        context.processed_dir / "persistent_channel_interaction_v7.parquet"
    )
    interactions.to_parquet(interaction_path, index=False, compression="zstd")
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "source_freeze_digest": source["source_freeze_digest"],
        "formal_count": int(formal["base_trial_id"].nunique()),
        "fit_count": int(
            trials[trials["split"] == "localization_fit"]["base_trial_id"].nunique()
        ),
        "family_counts": formal.groupby("family")["base_trial_id"].nunique().to_dict(),
        "effects": effects,
        "kv_stable": kv_stable,
        "recurrent_stable": recurrent_stable,
        "classification": classification,
        "channel_gate": gate,
        "endpoint_artifact": source["endpoint_artifact"],
        "endpoint_artifact_sha256": source["endpoint_artifact_sha256"],
        "trial_records": str(trial_path.relative_to(context.root)),
        "interaction_records": str(interaction_path.relative_to(context.root)),
    }
    write_json_atomic(
        context.processed_dir / "persistent_channel_attribution_v7.json", summary
    )
    context.finish("COMPLETED_ANALYSIS", summary=summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Qwen3.5 persistent-channel causal attribution",
        "configs/persistent_channels_v7.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("guard", "freeze", "schema", "restore", "attribution", "analyze"),
    )
    args = parser.parse_args()
    context = initialize_context("persistent-channels-v7", args)
    try:
        if args.stage == "guard":
            value = build_v6_guard(context.root)
            context.finish("COMPLETED_GUARD", guard=value)
            return
        if args.stage == "freeze":
            value = build_freeze(context.root, context.config)
            context.finish("COMPLETED_FREEZE", freeze=value)
            return
        freeze = verify_freeze(context.root, context.config)
        if args.stage == "analyze":
            _analyze(context, freeze)
            return
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        from jclosure.model import load_model_bundle

        bundle = load_model_bundle(context.config)
        if args.stage == "schema":
            _write_cache_schema(context, bundle, freeze)
        elif args.stage == "restore":
            _run_restore(context, bundle, freeze)
        else:
            _run_attribution(context, bundle, freeze, args.limit)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
