"""Float32 causal endpoint and persistent-restoration null control for v6."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.datasets_v5 import load_replication_tasks
from jclosure.experiments.causal_single_v4 import (
    _load_encoder,
    _quality,
    _replacement,
    _semantic_token,
    _thresholds,
    _token_ids,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.experiments.h2_replication_v5 import (
    _load_bank,
    _positive_direction,
)
from jclosure.experiments.mediation_v4 import (
    _clean_activations,
    _restoration_transform,
)
from jclosure.experiments.peripheral_v5 import load_domain
from jclosure.experiments.traces_v4 import ACTION_TO_ID
from jclosure.peripheral_v5 import RemainderTransform, build_u
from jclosure.peripheral_v6 import precision_audit, restoration_corrected_effects
from jclosure.protocol_v6 import build_freeze, verify_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.records_v6 import (
    PROTOCOL_V6,
    RestorationNullRecord,
    TeacherCausalEndpointRecord,
)
from jclosure.runtime_v3_1 import encode_direct_prompt
from jclosure.single_arm_v4 import (
    aligned_rollout_metrics,
    construct_from_shared_basis,
    matched_controls,
    multiple_token_log_odds,
    shared_low_singular_basis,
)
from jclosure.statistics import clustered_bootstrap_ci


@torch.no_grad()
def _rollout_f32(
    bundle: Any,
    task: Any,
    *,
    layer: int,
    workspace_layers: list[int],
    dense_map: Any,
    transforms: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Autonomous rollout with all scientific endpoint tensors retained as f32."""

    input_ids = encode_direct_prompt(bundle, task.prompt)
    with (
        ResidualEditor(bundle.layers, transforms or {}),
        ActivationRecorder(bundle.layers, at=workspace_layers) as recorder,
    ):
        outputs = bundle.hf_model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits[0, -1].detach().float()
    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("model did not return a KV cache")
    profiles = {
        value: dense_map.dense_state(
            recorder.activations[value][0, -1].detach().float(), value
        )
        for value in workspace_layers
    }
    hidden = recorder.activations[layer][0, -1].detach().float()
    within = {
        str(value): profiles[value].detach().cpu().numpy().astype(np.float32)
        for value in workspace_layers
    }
    within_hidden = {
        str(value): recorder.activations[value][0, -1]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
        for value in workspace_layers
    }
    actions: list[str] = []
    logits_values: list[np.ndarray] = []
    hidden_values: list[np.ndarray] = []
    j_values: list[np.ndarray] = []
    odds: list[float] = []
    error = None
    generated = 0
    maximum = 2 * task.horizon + max(8, round(0.25 * task.horizon))
    while generated < maximum and len(actions) < task.horizon:
        token = int(torch.argmax(logits))
        action, action_error = _semantic_token(bundle.tokenizer, token, task)
        if action_error is not None:
            error = action_error
            break
        if action is not None:
            index = len(actions)
            actions.append(action)
            logits_values.append(logits.detach().cpu().numpy().astype(np.float32))
            hidden_values.append(hidden.detach().cpu().numpy().astype(np.float32))
            j_values.append(profiles[layer].detach().cpu().numpy().astype(np.float32))
            odds.append(
                multiple_token_log_odds(
                    logits, _token_ids(bundle.tokenizer, task.semantic_actions[index])
                )
            )
        generated += 1
        token_tensor = torch.tensor([[token]], device=input_ids.device)
        attention_mask = torch.ones(
            (1, int(input_ids.shape[1]) + generated),
            dtype=torch.long,
            device=input_ids.device,
        )
        with ActivationRecorder(bundle.layers, at=workspace_layers) as recorder:
            continuation = bundle.hf_model(
                input_ids=token_tensor,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
        logits = continuation.logits[0, -1].detach().float()
        cache = continuation.past_key_values
        profiles = {
            value: dense_map.dense_state(
                recorder.activations[value][0, -1].detach().float(), value
            )
            for value in workspace_layers
        }
        hidden = recorder.activations[layer][0, -1].detach().float()
    return {
        "actions": actions,
        "expected_actions": list(task.semantic_actions),
        "parseable": len(actions) == task.horizon and error is None,
        "error": error,
        "logits": logits_values,
        "hidden_states": hidden_values,
        "j_states": j_values,
        "target_log_odds": odds,
        "within_forward_j_states": within,
        "within_forward_hidden_states": within_hidden,
    }


def _transforms(
    *,
    layer: int,
    workspace_layers: list[int],
    candidate: torch.Tensor,
    persistent_mode: str | None,
    clean_by_layer: dict[int, torch.Tensor],
    dense_map: Any,
    thresholds: Any,
    capture: dict[int, dict[str, Any]],
) -> dict[int, Any]:
    output: dict[int, Any] = {layer: _replacement(candidate)}
    if persistent_mode is not None:
        for restore_layer in workspace_layers[1:]:
            output[restore_layer] = partial(
                _restoration_transform,
                clean=clean_by_layer[restore_layer],
                dense_map=dense_map,
                mode=persistent_mode,
                capture=capture,
                thresholds=thresholds,
            )
    return output


def _effect_ci(
    frame: pd.DataFrame,
    value: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if frame.empty:
        return {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
    return asdict(
        clustered_bootstrap_ci(
            frame,
            cluster_col="prompt_id",
            value_col=value,
            seed=seed,
            n_resamples=resamples,
        )
    )


def _paired_difference_ci(
    frame: pd.DataFrame,
    left: str,
    right: str,
    value: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pivot = frame.pivot(index="base_trial_id", columns="condition", values=value)
    pivot = pivot.dropna(subset=[left, right])
    differences = (pivot[left] - pivot[right]).to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    boot = np.asarray(
        [
            np.mean(generator.choice(differences, len(differences), replace=True))
            for _ in range(resamples)
        ]
    )
    return {
        "estimate": float(np.mean(differences)),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
        "n_clusters": int(len(differences)),
        "n_resamples": int(resamples),
    }


def _run(context: Any, bundle: Any, freeze: dict[str, Any], limit: int | None) -> None:
    section = context.config["peripheral_v6"]
    pair_manifest = json.loads(
        (context.root / freeze["pair_manifest"]).read_text(encoding="utf-8")
    )
    selected = pair_manifest["items"]
    if limit is not None:
        selected = selected[: int(limit)]
    v5_freeze = json.loads(
        (context.root / "artifacts/peripheral_v5.freeze.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = {
        value.example_id: value
        for value in load_replication_tasks(
            context.root / v5_freeze["replication_data"]
        )
    }
    bank_rows, bank_states = _load_bank(context.root)
    bank = {
        str(row["example_id"]): torch.as_tensor(bank_states[index])
        for index, row in enumerate(bank_rows)
    }
    layer = int(section["layer"])
    layers = [layer, *[int(value) for value in section["future_workspace_layers"]]]
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    shared = shared_low_singular_basis(
        dense_map,
        layer,
        relative_tolerance=1e-4,
        device=next(bundle.hf_model.parameters()).device,
    )
    thresholds = _thresholds(context.config)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    cache: dict[int, torch.Tensor] = {}
    transform = RemainderTransform.fit(
        load_domain(context.root, v5_freeze, "train", 4).current_j,
        load_domain(context.root, v5_freeze, "train", 4).current_h,
        rank=512,
        seed=int(context.seed),
    )
    artifact_directory = context.root / "artifacts/causal/v6" / context.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    transform_path = artifact_directory / "remainder_transform.npz"
    transform.save(transform_path)
    records: list[dict[str, Any]] = []
    endpoint_meta: list[dict[str, Any]] = []
    tensors: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "hidden_clean",
            "hidden_intervened",
            "j_clean",
            "j_intervened",
            "logits_clean",
            "logits_intervened",
            "remainder_clean",
            "remainder_intervened",
            "u",
            "next_action",
            "teacher_semantic_delta",
        )
    }
    conditions = (
        "clean",
        "identity",
        "matched_random",
        "j_positive",
        "full_perturbation",
        "single",
        "persistent_final",
        "persistent_all",
        "persistent_null_final",
        "persistent_null_all",
    )
    for attempt, item in enumerate(selected):
        task = tasks[item["prompt_id"]]
        clean_by_layer = _clean_activations(bundle, task.prompt, layers)
        clean = clean_by_layer[layer][-1].to(next(bundle.hf_model.parameters()).device)
        donor = bank[item["donor_id"]].to(clean.device).float()
        natural_scale = float(torch.linalg.vector_norm(donor - clean).item())
        candidate, construction, initial_valid = construct_from_shared_basis(
            clean,
            donor,
            shared=shared,
            dense_map=dense_map,
            encoder=encoder,
            natural_scale=natural_scale,
            displacement_fraction=float(section["initial_strength"]),
            thresholds=thresholds,
            naturality=naturality,
        )
        if not initial_valid:
            records.append(
                {
                    "schema_version": 8,
                    "protocol_version": PROTOCOL_V6,
                    "record_type": "attrition",
                    "run_id": context.run_id,
                    **item,
                    "exclusion_reason": "reconstructed_candidate_invalid",
                    "quality": construction,
                }
            )
            continue
        answer_ids = _token_ids(bundle.tokenizer, task.semantic_actions[0])
        direction = _positive_direction(
            bundle, dense_map, vocabulary, layer, answer_ids, cache
        )
        controls = matched_controls(
            clean, donor, candidate, direction, seed=int(context.seed) + attempt
        )
        rollouts: dict[str, dict[str, Any]] = {}
        captures: dict[str, dict[int, dict[str, Any]]] = {}
        for condition in conditions:
            capture: dict[int, dict[str, Any]] = {}
            if condition == "clean":
                transforms = {}
            elif condition == "single":
                transforms = _transforms(
                    layer=layer,
                    workspace_layers=layers,
                    candidate=candidate,
                    persistent_mode=None,
                    clean_by_layer=clean_by_layer,
                    dense_map=dense_map,
                    thresholds=thresholds,
                    capture=capture,
                )
            elif condition.startswith("persistent_null"):
                mode = (
                    "persistent_final"
                    if condition.endswith("final")
                    else "persistent_all"
                )
                transforms = _transforms(
                    layer=layer,
                    workspace_layers=layers,
                    candidate=clean,
                    persistent_mode=mode,
                    clean_by_layer=clean_by_layer,
                    dense_map=dense_map,
                    thresholds=thresholds,
                    capture=capture,
                )
            elif condition.startswith("persistent"):
                transforms = _transforms(
                    layer=layer,
                    workspace_layers=layers,
                    candidate=candidate,
                    persistent_mode=condition,
                    clean_by_layer=clean_by_layer,
                    dense_map=dense_map,
                    thresholds=thresholds,
                    capture=capture,
                )
            else:
                transforms = _transforms(
                    layer=layer,
                    workspace_layers=layers,
                    candidate=controls[
                        "identity" if condition == "identity" else condition
                    ],
                    persistent_mode=None,
                    clean_by_layer=clean_by_layer,
                    dense_map=dense_map,
                    thresholds=thresholds,
                    capture=capture,
                )
            rollouts[condition] = _rollout_f32(
                bundle,
                task,
                layer=layer,
                workspace_layers=layers,
                dense_map=dense_map,
                transforms=transforms,
            )
            captures[condition] = capture
        clean_rollout = rollouts["clean"]
        if tuple(clean_rollout["actions"]) != task.semantic_actions:
            records.append(
                {
                    "schema_version": 8,
                    "protocol_version": PROTOCOL_V6,
                    "record_type": "attrition",
                    "run_id": context.run_id,
                    **item,
                    "exclusion_reason": "continuous_teacher_trajectory_incorrect",
                }
            )
            continue
        for condition in conditions:
            capture = captures[condition]
            restoration_expected = condition.startswith("persistent")
            restoration_valid = not restoration_expected or bool(
                len(capture) == len(layers) - 1
                and all(value["passed"] for value in capture.values())
            )
            hook_map = (
                []
                if condition == "clean"
                else [
                    [layer, "final", "identity" if "null" in condition else "initial"]
                ]
            )
            if restoration_expected:
                scope = "final" if condition.endswith("final") else "all_non_padding"
                hook_map.extend(
                    [[value, scope, "measured_j_restoration"] for value in layers[1:]]
                )
            records.append(
                {
                    **RestorationNullRecord(
                        run_id=context.run_id,
                        base_trial_id=item["base_trial_id"],
                        prompt_id=task.example_id,
                        family=task.family,
                        horizon=task.horizon,
                        condition=condition,
                        valid=restoration_valid,
                        metrics=aligned_rollout_metrics(
                            clean_rollout, rollouts[condition]
                        ),
                        restoration_events=[capture[key] for key in sorted(capture)],
                        hook_execution_map=hook_map,
                        exclusion_reason=None
                        if restoration_valid
                        else "runtime_restoration_invalid",
                    ).to_dict(),
                    "record_type": "restoration_null_trial",
                    "split": item["role"],
                    "donor_id": item["donor_id"],
                    "initial_quality": _quality(
                        clean,
                        candidate
                        if condition in {"single", "persistent_final", "persistent_all"}
                        else controls.get(condition, clean),
                        layer=layer,
                        dense_map=dense_map,
                        natural_scale=natural_scale,
                    ),
                }
            )
        single = rollouts["single"]
        if len(clean_rollout["j_states"]) < 2 or len(single["j_states"]) < 2:
            records.append(
                {
                    "schema_version": 8,
                    "protocol_version": PROTOCOL_V6,
                    "record_type": "attrition",
                    "run_id": context.run_id,
                    **item,
                    "exclusion_reason": "insufficient_semantic_steps_for_causal_endpoint",
                }
            )
            continue
        next_layer = layers[1]
        clean_h = np.stack(
            (
                clean_rollout["within_forward_hidden_states"][str(layer)],
                clean_rollout["within_forward_hidden_states"][str(next_layer)],
            )
        ).astype(np.float32)
        single_h = np.stack(
            (
                single["within_forward_hidden_states"][str(layer)],
                single["within_forward_hidden_states"][str(next_layer)],
            )
        ).astype(np.float32)
        clean_j = np.stack(
            (
                clean_rollout["within_forward_j_states"][str(layer)],
                clean_rollout["within_forward_j_states"][str(next_layer)],
            )
        ).astype(np.float32)
        single_j = np.stack(
            (
                single["within_forward_j_states"][str(layer)],
                single["within_forward_j_states"][str(next_layer)],
            )
        ).astype(np.float32)
        clean_logits = np.stack(clean_rollout["logits"][:1]).astype(np.float32)
        single_logits = np.stack(single["logits"][:1]).astype(np.float32)
        tensors["hidden_clean"].append(clean_h)
        tensors["hidden_intervened"].append(single_h)
        tensors["j_clean"].append(clean_j)
        tensors["j_intervened"].append(single_j)
        tensors["logits_clean"].append(clean_logits)
        tensors["logits_intervened"].append(single_logits)
        tensors["remainder_clean"].append(
            transform.transform(clean_j[:1], clean_h[:1])[0]
        )
        tensors["remainder_intervened"].append(
            transform.transform(single_j[:1], single_h[:1])[0]
        )
        tensors["u"].append(
            build_u(
                np.asarray([ACTION_TO_ID[task.semantic_actions[0]]]),
                np.asarray([task.family]),
                np.asarray([0]),
                np.asarray([task.horizon]),
            )[0]
        )
        tensors["next_action"].append(
            np.asarray(ACTION_TO_ID[task.semantic_actions[0]], dtype=np.int64)
        )
        tensors["teacher_semantic_delta"].append(
            np.asarray(
                clean_rollout["actions"][0] != single["actions"][0], dtype=np.int64
            )
        )
        endpoint_meta.append({**item, "task": task, "quality": construction})
    if not endpoint_meta:
        raise RuntimeError("no valid float32 causal endpoints")
    packed = {
        key: np.stack(value).astype(
            np.int64 if key in {"next_action", "teacher_semantic_delta"} else np.float32
        )
        for key, value in tensors.items()
    }
    causal_write_layer = layers[1]
    centered = (
        dense_map.centered_map(causal_write_layer, device="cpu", dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    audit = precision_audit(
        packed["hidden_clean"][:, 1], packed["hidden_intervened"][:, 1], centered
    )
    for index, item in enumerate(endpoint_meta):
        task = item["task"]
        delta_hidden = (
            packed["hidden_intervened"][index, 1] - packed["hidden_clean"][index, 1]
        )
        delta_remainder = (
            packed["remainder_intervened"][index] - packed["remainder_clean"][index]
        )
        delta_logits = (
            packed["logits_intervened"][index, 0] - packed["logits_clean"][index, 0]
        )
        target_ids = _token_ids(bundle.tokenizer, task.semantic_actions[0])
        odds_delta = multiple_token_log_odds(
            torch.from_numpy(packed["logits_intervened"][index, 0]), target_ids
        ) - multiple_token_log_odds(
            torch.from_numpy(packed["logits_clean"][index, 0]), target_ids
        )
        records.append(
            {
                **TeacherCausalEndpointRecord(
                    run_id=context.run_id,
                    base_trial_id=item["base_trial_id"],
                    prompt_id=item["prompt_id"],
                    family=item["family"],
                    horizon=item["horizon"],
                    split=item["role"],
                    valid=True,
                    delta_j_l2_f32=float(audit["norm_f32"][index]),
                    delta_j_l2_f64_projection=float(audit["norm_f64"][index]),
                    delta_j_f32_f64_cosine=float(audit["cosine_f32_f64"][index])
                    if audit["norm_f32"][index] > 0 and audit["norm_f64"][index] > 0
                    else None,
                    delta_hidden_l2=float(np.linalg.norm(delta_hidden)),
                    delta_remainder_l2=float(np.linalg.norm(delta_remainder)),
                    output_logit_delta_l2=float(np.linalg.norm(delta_logits)),
                    semantic_delta=bool(
                        clean_rollout["actions"][0] != single["actions"][0]
                    ),
                    output_sign=int(np.sign(odds_delta)),
                    metadata={
                        "storage_dtype": "float32",
                        "projection_sensitivity": "same float32 hidden; float32 versus float64 map arithmetic",
                        "layer": layer,
                        "causal_write_layer": causal_write_layer,
                        "position_scope": "final",
                        "dictionary_size": len(vocabulary.token_ids),
                        "dictionary_hash": vocabulary.digest,
                        "intervention_quality": item["quality"],
                        "endpoint_definition": "same-forward layer-23 intervention to layer-24 measured-J write",
                    },
                ).to_dict(),
                "record_type": "teacher_causal_endpoint",
                "artifact_index": index,
            }
        )
    artifact = artifact_directory / "causal_endpoint_f32.npz"
    np.savez_compressed(
        artifact,
        **packed,
        teacher_delta_j_f32=audit["delta_f32"],
        teacher_delta_j_f64_projection=audit["delta_f64"],
        prompt_id=np.asarray([value["prompt_id"] for value in endpoint_meta]),
        base_trial_id=np.asarray([value["base_trial_id"] for value in endpoint_meta]),
        family=np.asarray([value["family"] for value in endpoint_meta]),
        split=np.asarray([value["role"] for value in endpoint_meta]),
    )
    records_path = context.raw_dir / context.run_id / "peripheral_foundations_v6.jsonl"
    append_jsonl(records_path, records)
    serializable_audit = {
        key: value for key, value in audit.items() if not isinstance(value, np.ndarray)
    }
    context.finish(
        "COMPLETED_CAUSAL_AND_NULL",
        records=str(records_path.relative_to(context.root)),
        record_count=len(records),
        causal_endpoint_artifact=str(artifact.relative_to(context.root)),
        causal_endpoint_artifact_sha256=sha256_file(artifact),
        remainder_transform=str(transform_path.relative_to(context.root)),
        remainder_transform_sha256=sha256_file(transform_path),
        precision_audit=serializable_audit,
        source_freeze_digest=freeze["freeze_digest"],
    )


def _merge(context: Any) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("causal-endpoint-v6-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_CAUSAL_AND_NULL":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed v6 causal endpoint run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    endpoints = pd.DataFrame(
        value for value in rows if value["record_type"] == "teacher_causal_endpoint"
    )
    trials = pd.DataFrame(
        value for value in rows if value["record_type"] == "restoration_null_trial"
    )
    for metric in (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_change",
        "answer_flip",
        "task_accuracy_change",
    ):
        trials[metric] = trials["metrics"].map(lambda value, key=metric: value[key])
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["peripheral_v6"]["bootstrap_resamples"])
    effects: dict[str, Any] = {}
    groups = [("pooled", trials), *list(trials.groupby("family", sort=True))]
    for name, group in groups:
        effects[str(name)] = {}
        for metric in (
            "output_js_divergence",
            "future_j_trajectory_divergence",
            "target_log_odds_change",
            "answer_flip",
            "task_accuracy_change",
        ):
            effects[str(name)][metric] = {
                str(condition): _effect_ci(
                    values.dropna(subset=[metric]),
                    metric,
                    seed=seed,
                    resamples=resamples,
                )
                for condition, values in group.groupby("condition", sort=True)
            }
        effects[str(name)]["artifact_corrected"] = {}
        for scope in ("final", "all"):
            persistent = f"persistent_{scope}"
            null = f"persistent_null_{scope}"
            effects[str(name)]["artifact_corrected"][scope] = {
                "persistent_minus_null": _paired_difference_ci(
                    group,
                    persistent,
                    null,
                    "output_js_divergence",
                    seed=seed,
                    resamples=resamples,
                ),
                "null_artifact": effects[str(name)]["output_js_divergence"].get(null),
            }
            corrected = restoration_corrected_effects(
                group,
                effect_column="output_js_divergence",
                persistent_condition=persistent,
                null_condition=null,
            )
            if len(corrected):
                denominator = float(corrected["single"].mean())
                corrected_persistent = float(corrected["corrected_persistent"].mean())
                effects[str(name)]["artifact_corrected"][scope][
                    "mediation_point_estimate"
                ] = 1 - corrected_persistent / max(denominator, 1e-20)
    endpoint_records = context.processed_dir / "causal_endpoint_precision_v6.parquet"
    endpoints.to_parquet(endpoint_records, index=False, compression="zstd")
    trial_records = context.processed_dir / "restoration_null_v6.parquet"
    serial = trials.copy()
    for column in (
        "metrics",
        "restoration_events",
        "hook_execution_map",
        "initial_quality",
    ):
        serial[column] = serial[column].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    serial.to_parquet(trial_records, index=False, compression="zstd")
    precision = {
        "count": int(len(endpoints)),
        "storage_dtype": "float32",
        "nonzero_fraction_f32": float(np.mean(endpoints["delta_j_l2_f32"] > 0)),
        "nonzero_fraction_f64_projection": float(
            np.mean(endpoints["delta_j_l2_f64_projection"] > 0)
        ),
        "delta_norm_f32": {
            "min": float(endpoints["delta_j_l2_f32"].min()),
            "median": float(endpoints["delta_j_l2_f32"].median()),
            "max": float(endpoints["delta_j_l2_f32"].max()),
        },
        "delta_norm_f64_projection": {
            "min": float(endpoints["delta_j_l2_f64_projection"].min()),
            "median": float(endpoints["delta_j_l2_f64_projection"].median()),
            "max": float(endpoints["delta_j_l2_f64_projection"].max()),
        },
        "median_f32_f64_cosine": float(
            endpoints["delta_j_f32_f64_cosine"].dropna().median()
        ),
    }
    summary = {
        "schema_version": 8,
        "protocol_version": PROTOCOL_V6,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "source_freeze_digest": source["source_freeze_digest"],
        "precision_audit": precision,
        "valid_endpoints_by_family": endpoints.groupby("family").size().to_dict(),
        "valid_trials_by_condition_family": trials.groupby(["condition", "family"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
        "effects": effects,
        "causal_endpoint_artifact": source["causal_endpoint_artifact"],
        "causal_endpoint_artifact_sha256": source["causal_endpoint_artifact_sha256"],
        "remainder_transform": source["remainder_transform"],
        "remainder_transform_sha256": source["remainder_transform_sha256"],
        "endpoint_records": str(endpoint_records.relative_to(context.root)),
        "trial_records": str(trial_records.relative_to(context.root)),
    }
    write_json_atomic(context.processed_dir / "causal_endpoint_v6.json", summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Float32 causal endpoint and restoration null", "configs/peripheral_v6.yaml"
    )
    parser.add_argument("--stage", choices=("freeze", "run", "merge"), required=True)
    args = parser.parse_args()
    context = initialize_context("causal-endpoint-v6", args)
    try:
        if args.stage == "freeze":
            freeze = build_freeze(context.root, context.config)
            context.finish("COMPLETED_FREEZE", freeze=freeze)
        elif args.stage == "run":
            freeze = verify_freeze(context.root, context.config)
            if args.dry_run:
                context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
                return
            from jclosure.model import load_model_bundle

            _run(context, load_model_bundle(context.config), freeze, args.limit)
        else:
            verify_freeze(context.root, context.config)
            summary = _merge(context)
            context.finish("COMPLETED", summary=summary)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
