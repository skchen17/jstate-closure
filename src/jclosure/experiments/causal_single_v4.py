"""Basic paired same-J/different-hidden-state causal experiment (protocol v4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.datasets_v4 import ProgramTraceTask
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.metrics import rms_drift, topk_overlap
from jclosure.model import load_model_bundle
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.records_v4 import SingleArmTrialRecord
from jclosure.runtime_v3_1 import encode_direct_prompt
from jclosure.single_arm_v4 import (
    aligned_rollout_metrics,
    construct_from_shared_basis,
    matched_controls,
    multiple_token_log_odds,
    shared_low_singular_basis,
)
from jclosure.statistics import clustered_bootstrap_ci, numerical_null_threshold

PROTOCOL = "single_arm_causal_v4"


def _load_tasks(
    root: Path, freeze: dict[str, Any], domain: str
) -> list[ProgramTraceTask]:
    payload = json.loads(
        (root / freeze["formal_data"][domain]).read_text(encoding="utf-8")
    )
    tasks = []
    for raw in payload["items"]:
        item = dict(raw)
        item["semantic_actions"] = tuple(item["semantic_actions"])
        tasks.append(ProgramTraceTask(**item))
    return tasks


def _competence_records(context: Any) -> list[dict[str, Any]]:
    summary = json.loads(
        (context.processed_dir / "teacher_formal_v4.json").read_text(encoding="utf-8")
    )
    path = context.root / summary["records"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_encoder(context: Any, bundle: Any) -> tuple[Any, Any, DenseJMap]:
    size = int(context.config["single_arm_v4"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=PROTOCOL,
        direction_chunk_size=int(
            context.config["jstate"].get("direction_chunk_size", 512)
        ),
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _thresholds(config: dict[str, Any]) -> V3ClampThresholds:
    section = config["single_arm_v4"]
    return V3ClampThresholds(
        dense_cosine=float(section["dense_cosine_threshold"]),
        dense_top10_overlap=float(section["top10_overlap_threshold"]),
        rms_drift=float(section["rms_drift_threshold"]),
        formal_displacement=float(section["formal_displacement_fraction"]),
        sensitivity_displacement=0.05,
    )


@torch.no_grad()
def _initial_state(bundle: Any, prompt: str, layer: int) -> torch.Tensor:
    input_ids = encode_direct_prompt(bundle, prompt)
    with ActivationRecorder(bundle.layers, at=[layer]) as recorder:
        bundle.forward_logits(input_ids)
    return recorder.activations[layer][0, -1].detach().float()


def _build_bank(context: Any, bundle: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    layer = int(context.config["single_arm_v4"]["layer"])
    correct = {
        record["example_id"]: record
        for record in _competence_records(context)
        if record["domain"] in {"train", "validation"}
        and record["full_trajectory_correct"]
    }
    tasks = [
        task
        for domain in ("train", "validation")
        for task in _load_tasks(context.root, freeze, domain)
        if task.example_id in correct
    ]
    bank_root = context.root / "artifacts/causal/v4" / context.run_id
    bank_root.mkdir(parents=True, exist_ok=True)
    rows = []
    states = []
    for index, task in enumerate(tasks):
        prompt = task.prompt
        state = _initial_state(bundle, prompt, layer)
        states.append(state.detach().cpu().numpy().astype(np.float32))
        rows.append(
            {
                "schema_version": 6,
                "protocol_version": PROTOCOL,
                "run_id": context.run_id,
                "bank_index": index,
                "example_id": task.example_id,
                "family": task.family,
                "horizon": task.horizon,
                "variant": task.variant,
                "program_hash": task.program_hash,
                "prompt": prompt,
            }
        )
    state_path = bank_root / "initial_states.npz"
    np.savez_compressed(state_path, states=np.stack(states))
    manifest = context.raw_dir / context.run_id / "single_arm_bank.jsonl"
    append_jsonl(manifest, rows)
    payload = {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "layer": layer,
        "records": str(manifest.relative_to(context.root)),
        "states": str(state_path.relative_to(context.root)),
        "states_sha256": sha256_file(state_path),
        "count": len(rows),
        "program_freeze_digest": freeze["freeze_digest"],
    }
    output = context.processed_dir / "single_arm_bank_v4.json"
    write_json_atomic(output, payload)
    return payload


def _load_bank(context: Any) -> tuple[list[dict[str, Any]], np.ndarray]:
    summary_path = context.processed_dir / "single_arm_bank_v4.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sha256_file(context.root / summary["states"]) != summary["states_sha256"]:
        raise RuntimeError("single-arm state bank hash mismatch")
    rows = [
        json.loads(line)
        for line in (context.root / summary["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with np.load(context.root / summary["states"], allow_pickle=False) as payload:
        states = payload["states"].astype(np.float32)
    if len(rows) != len(states):
        raise RuntimeError("single-arm bank metadata/state count mismatch")
    return rows, states


def _token_ids(tokenizer: Any, surface: str) -> list[int]:
    output = []
    for value in (surface, " " + surface):
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) == 1:
            output.append(int(ids[0]))
    return sorted(set(output))


def _replacement(candidate: torch.Tensor):
    def transform(activation: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        output = activation.clone()
        output[0, -1] = candidate.to(output.device, output.dtype)
        return output

    return transform


def _semantic_token(
    tokenizer: Any, token_id: int, task: ProgramTraceTask
) -> tuple[str | None, str | None]:
    if int(token_id) in set(getattr(tokenizer, "all_special_ids", ()) or ()):
        return None, "special_token_before_complete"
    pieces = tokenizer.decode([int(token_id)], skip_special_tokens=True).split()
    if not pieces:
        return None, None
    allowed = (
        set("0123456789")
        if all(value.isdigit() for value in task.semantic_actions)
        else set("ABCDEF")
    )
    if len(pieces) != 1 or len(pieces[0]) != 1 or pieces[0] not in allowed:
        return None, "nonsemantic_generated_token"
    return pieces[0], None


@torch.no_grad()
def _rollout(
    bundle: Any,
    task: ProgramTraceTask,
    *,
    layer: int,
    workspace_layers: list[int],
    dense_map: DenseJMap,
    candidate: torch.Tensor | None,
) -> dict[str, Any]:
    actions: list[str] = []
    logits_values: list[np.ndarray] = []
    j_states: list[np.ndarray] = []
    target_odds: list[float] = []
    layer_states: dict[str, np.ndarray] = {}
    error = None
    input_ids = encode_direct_prompt(bundle, task.prompt)
    transforms = {layer: _replacement(candidate)} if candidate is not None else {}
    with (
        ResidualEditor(bundle.layers, transforms),
        ActivationRecorder(bundle.layers, at=workspace_layers) as recorder,
    ):
        outputs = bundle.hf_model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits[0, -1].detach().float()
    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError("model did not return a KV cache for free continuation")
    current_profiles = {
        current_layer: dense_map.dense_state(
            recorder.activations[current_layer][0, -1].detach().float(),
            current_layer,
        )
        for current_layer in workspace_layers
    }
    generated_count = 0
    maximum_tokens = 2 * task.horizon + max(8, round(0.25 * task.horizon))
    while generated_count < maximum_tokens and len(actions) < task.horizon:
        token = int(torch.argmax(logits))
        action, action_error = _semantic_token(bundle.tokenizer, token, task)
        if action_error is not None:
            error = action_error
            break
        if action is not None:
            step_index = len(actions)
            j_states.append(
                current_profiles[layer].detach().cpu().numpy().astype(np.float16)
            )
            logits_values.append(logits.detach().cpu().numpy().astype(np.float16))
            target_odds.append(
                multiple_token_log_odds(
                    logits,
                    _token_ids(bundle.tokenizer, task.semantic_actions[step_index]),
                )
            )
            if step_index == 0:
                for current_layer, value in current_profiles.items():
                    layer_states[str(current_layer)] = (
                        value.detach().cpu().numpy().astype(np.float16)
                    )
            actions.append(action)
        generated_count += 1
        next_token = torch.tensor([[token]], device=input_ids.device)
        attention_mask = torch.ones(
            (1, int(input_ids.shape[1]) + generated_count),
            dtype=torch.long,
            device=input_ids.device,
        )
        with ActivationRecorder(bundle.layers, at=workspace_layers) as recorder:
            continuation = bundle.hf_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = continuation.logits[0, -1].detach().float()
        past_key_values = continuation.past_key_values
        current_profiles = {
            current_layer: dense_map.dense_state(
                recorder.activations[current_layer][0, -1].detach().float(),
                current_layer,
            )
            for current_layer in workspace_layers
        }
    return {
        "actions": actions,
        "expected_actions": list(task.semantic_actions),
        "parseable": len(actions) == task.horizon and error is None,
        "error": error,
        "logits": logits_values,
        "j_states": j_states,
        "target_log_odds": target_odds,
        "within_forward_j_states": layer_states,
    }


def _quality(
    clean: torch.Tensor,
    candidate: torch.Tensor,
    *,
    layer: int,
    dense_map: DenseJMap,
    natural_scale: float,
) -> dict[str, float]:
    clean_dense = dense_map.dense_state(clean.float(), layer)
    candidate_dense = dense_map.dense_state(candidate.float(), layer)
    return {
        "dense_cosine": float(
            torch.nn.functional.cosine_similarity(
                clean_dense[None], candidate_dense[None]
            ).item()
        ),
        "top10_overlap": topk_overlap(
            dense_map.raw_scores(clean.float(), layer),
            dense_map.raw_scores(candidate.float(), layer),
            10,
        ),
        "rms_drift": rms_drift(clean, candidate),
        "displacement_fraction": float(
            torch.linalg.vector_norm((candidate - clean).float()).item()
            / max(natural_scale, 1e-20)
        ),
    }


def _run(
    context: Any, bundle: Any, freeze: dict[str, Any], limit: int | None
) -> list[dict[str, Any]]:
    section = context.config["single_arm_v4"]
    layer = int(section["layer"])
    workspace_layers = [
        layer,
        *[int(value) for value in section["future_workspace_layers"]],
    ]
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    bank_rows, bank_states = _load_bank(context)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    shared = shared_low_singular_basis(
        dense_map,
        layer,
        relative_tolerance=1e-4,
        device=next(bundle.hf_model.parameters()).device,
    )
    thresholds = _thresholds(context.config)
    tasks = _load_tasks(context.root, freeze, "causal_test")
    correct_ids = {
        record["example_id"]
        for record in _competence_records(context)
        if record["domain"] == "causal_test" and record["full_trajectory_correct"]
    }
    tasks = sorted(
        (task for task in tasks if task.example_id in correct_ids),
        key=lambda task: hashlib.sha256(task.example_id.encode()).hexdigest(),
    )
    target = int(section["pilot_valid_base_trials"])
    if limit is not None:
        target = min(target, int(limit))
    maximum_attempts = min(len(tasks), target * int(section["max_attempt_multiplier"]))
    token_to_index = {
        int(token): index for index, token in enumerate(vocabulary.token_ids)
    }
    raw_map = dense_map.raw_map(
        layer, device=next(bundle.hf_model.parameters()).device, dtype=torch.float32
    )
    positive_direction_cache: dict[int, torch.Tensor] = {}
    output: list[dict[str, Any]] = []
    completed = 0
    for attempt, task in enumerate(tasks[:maximum_attempts]):
        prompt = task.prompt
        clean = _initial_state(bundle, prompt, layer)
        matches = [
            (row, bank_states[index])
            for index, row in enumerate(bank_rows)
            if row["family"] == task.family and int(row["horizon"]) == task.horizon
        ]
        if not matches:
            output.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "exclusion_reason": "no_matched_donor",
                }
            )
            continue
        donor_index = int(
            hashlib.sha256(task.example_id.encode()).hexdigest(), 16
        ) % len(matches)
        donor_row, donor_array = matches[donor_index]
        donor = torch.as_tensor(donor_array, device=clean.device, dtype=torch.float32)
        natural_scale = float(torch.linalg.vector_norm(donor - clean).item())
        preserving, construction_quality, valid = construct_from_shared_basis(
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
        if not valid:
            output.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "horizon": task.horizon,
                    "donor_id": donor_row["example_id"],
                    "exclusion_reason": "initial_state_invalid",
                    "quality": construction_quality,
                }
            )
            continue
        answer_ids = _token_ids(bundle.tokenizer, task.semantic_actions[0])
        answer_index = next(
            (token_to_index[value] for value in answer_ids if value in token_to_index),
            None,
        )
        if not answer_ids:
            output.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "exclusion_reason": "answer_not_single_token",
                }
            )
            continue
        if answer_index is not None:
            j_direction = raw_map[answer_index]
            positive_control_provenance = "measured_dictionary_answer_direction"
        else:
            answer_token_id = int(answer_ids[0])
            if answer_token_id not in positive_direction_cache:
                selected_unembedding = bundle.unembedding_weight[answer_token_id].detach()
                jacobian = bundle.lens.jacobians[layer].detach()
                positive_direction_cache[answer_token_id] = (
                    selected_unembedding.float().cpu() @ jacobian.float().cpu()
                ).to(clean.device)
            j_direction = positive_direction_cache[answer_token_id]
            positive_control_provenance = "full_vocabulary_answer_j_direction"
        controls = matched_controls(
            clean,
            donor,
            preserving,
            j_direction,
            seed=int(context.seed) + attempt,
        )
        clean_rollout = _rollout(
            bundle,
            task,
            layer=layer,
            workspace_layers=workspace_layers,
            dense_map=dense_map,
            candidate=None,
        )
        if tuple(clean_rollout["actions"]) != task.semantic_actions:
            output.append(
                {
                    "schema_version": 6,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "horizon": task.horizon,
                    "exclusion_reason": "continuous_teacher_trajectory_incorrect",
                    "generated_actions": clean_rollout["actions"],
                }
            )
            continue
        base_id = hashlib.sha256(
            f"{task.example_id}\x1f{donor_row['example_id']}\x1f{layer}".encode()
        ).hexdigest()
        trial_rows = []
        for condition in section["controls"]:
            replacement = controls[str(condition)]
            rollout = (
                clean_rollout
                if condition == "clean"
                else _rollout(
                    bundle,
                    task,
                    layer=layer,
                    workspace_layers=workspace_layers,
                    dense_map=dense_map,
                    candidate=replacement,
                )
            )
            metrics = aligned_rollout_metrics(clean_rollout, rollout)
            quality: dict[str, Any] = _quality(
                clean,
                replacement,
                layer=layer,
                dense_map=dense_map,
                natural_scale=natural_scale,
            )
            if condition == "j_preserving":
                quality["construction"] = construction_quality
            record = SingleArmTrialRecord(
                run_id=context.run_id,
                base_trial_id=base_id,
                condition=str(condition),
                prompt_id=task.example_id,
                family=task.family,
                horizon=task.horizon,
                layer=layer,
                valid=True,
                metrics=metrics,
                quality=quality,
            ).to_dict()
            record.update(
                {
                    "record_type": "causal_trial",
                    "prompt": prompt,
                    "answer": task.final_answer,
                    "donor_id": donor_row["example_id"],
                    "dictionary_size": len(vocabulary.token_ids),
                    "dictionary_hash": vocabulary.digest,
                    "positive_control_provenance": positive_control_provenance,
                    "initial_scope": "final",
                    "hook_execution_map": (
                        [] if condition == "clean" else [[layer, "final"]]
                    ),
                }
            )
            trial_rows.append(record)
        output.extend(trial_rows)
        completed += 1
        if completed >= target:
            break
    return output


def _summary(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    trials = frame[(frame["record_type"] == "causal_trial") & frame["valid"]].copy()
    attrition = frame[frame["record_type"] == "attrition"].copy()
    for metric in (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_change",
        "answer_flip",
        "task_accuracy_change",
    ):
        trials[metric] = trials["metrics"].map(lambda value, key=metric: value.get(key))
    required = set(config["single_arm_v4"]["controls"])
    common_ids = [
        base_id
        for base_id, group in trials.groupby("base_trial_id")
        if set(group["condition"]) == required
    ]
    common = trials[trials["base_trial_id"].isin(common_ids)].copy()
    effects: dict[str, Any] = {}
    for metric in (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_change",
        "answer_flip",
        "task_accuracy_change",
    ):
        effects[metric] = {}
        for condition, group in common.groupby("condition", sort=True):
            values = group.dropna(subset=[metric])
            if values.empty:
                continue
            effects[metric][str(condition)] = asdict(
                clustered_bootstrap_ci(
                    values,
                    cluster_col="prompt_id",
                    value_col=metric,
                    n_resamples=int(config["single_arm_v4"]["bootstrap_resamples"]),
                    confidence=float(config["single_arm_v4"]["confidence"]),
                    seed=int(config["reproducibility"]["bootstrap_seed"]),
                )
            )
    effects_by_family: dict[str, Any] = {}
    for family, family_frame in common.groupby("family", sort=True):
        effects_by_family[str(family)] = {}
        for metric in (
            "output_js_divergence",
            "future_j_trajectory_divergence",
            "target_log_odds_change",
            "answer_flip",
            "task_accuracy_change",
        ):
            values = family_frame[family_frame["condition"] == "j_preserving"].dropna(
                subset=[metric]
            )
            if values.empty:
                continue
            effects_by_family[str(family)][metric] = asdict(
                clustered_bootstrap_ci(
                    values,
                    cluster_col="prompt_id",
                    value_col=metric,
                    n_resamples=int(config["single_arm_v4"]["bootstrap_resamples"]),
                    confidence=float(config["single_arm_v4"]["confidence"]),
                    seed=int(config["reproducibility"]["bootstrap_seed"]),
                )
            )
    null = common[common["condition"].isin(["clean", "identity"])][
        "output_js_divergence"
    ].dropna()
    threshold = numerical_null_threshold(
        null,
        floor=float(config["single_arm_v4"]["null_js_floor"]),
        quantile=float(config["single_arm_v4"]["null_quantile"]),
    )
    preserving = effects.get("output_js_divergence", {}).get("j_preserving")
    causal_gate = bool(preserving and preserving["lower"] > threshold)
    paired_clean = common[common["condition"] == "clean"]
    paired_cells = []
    for (family, horizon), group in paired_clean.groupby(
        ["family", "horizon"], sort=True
    ):
        paired_cells.append(
            {"family": str(family), "horizon": int(horizon), "n": int(len(group))}
        )
    preserving_quality = common[common["condition"] == "j_preserving"]["quality"]
    quality_summary = {
        key: float(np.median([value[key] for value in preserving_quality]))
        for key in (
            "dense_cosine",
            "top10_overlap",
            "rms_drift",
            "displacement_fraction",
        )
        if len(preserving_quality) and all(key in value for value in preserving_quality)
    }
    return {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "attempted_records": int(len(frame)),
        "valid_records": int(len(trials)),
        "complete_paired_base_trials": int(len(common_ids)),
        "attempted_base_prompts": int(frame["prompt_id"].nunique()),
        "attrition_counts": {
            str(key): int(value)
            for key, value in attrition["exclusion_reason"].value_counts().items()
        }
        if not attrition.empty
        else {},
        "paired_task_cells": paired_cells,
        "j_preserving_quality_medians": quality_summary,
        "null_threshold": threshold,
        "effects": effects,
        "j_preserving_effects_by_family": effects_by_family,
        "single_arm_effect_above_noise": causal_gate,
        "persistent_mediation_authorized": causal_gate,
    }


def main() -> None:
    parser = standard_parser(
        "Run the basic protocol-v4 single-arm causal test",
        "configs/predictive_v4.yaml",
    )
    parser.add_argument("--stage", choices=("bank", "run", "merge"), required=True)
    args = parser.parse_args()
    context = initialize_context("causal-single-v4", args)
    try:
        freeze = verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "bank":
            bundle = load_model_bundle(context.config)
            summary = _build_bank(context, bundle, freeze)
            context.finish("COMPLETED_BANK", summary=summary)
            return
        if args.stage == "run":
            bundle = load_model_bundle(context.config)
            rows = _run(context, bundle, freeze, args.limit)
            path = context.raw_dir / context.run_id / "single_arm_trials.jsonl"
            append_jsonl(path, rows)
            context.finish(
                "COMPLETED_TRIALS",
                records=str(path.relative_to(context.root)),
                record_count=len(rows),
            )
            return
        manifests = []
        for path in context.raw_dir.glob("causal-single-v4-*/manifest.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "COMPLETED_TRIALS":
                manifests.append(value)
        if not manifests:
            raise RuntimeError("no completed v4 single-arm trial run")
        source = sorted(manifests, key=lambda value: value["run_id"])[-1]
        rows = [
            json.loads(line)
            for line in (context.root / source["records"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        frame = pd.DataFrame(rows)
        summary = _summary(frame, context.config)
        records_path = context.processed_dir / "single_arm_v4.parquet"
        parquet_frame = frame.copy()
        for column in ("metrics", "quality", "hook_execution_map"):
            if column in parquet_frame:
                parquet_frame[column] = parquet_frame[column].map(
                    lambda value: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict | list)
                    else value
                )
        parquet_frame.to_parquet(records_path, index=False, compression="zstd")
        summary.update(
            {
                "run_id": context.run_id,
                "source_run_id": source["run_id"],
                "records": str(records_path.relative_to(context.root)),
            }
        )
        output = context.processed_dir / "single_arm_v4.json"
        write_json_atomic(output, summary)
        context.finish("COMPLETED", summary=str(output.relative_to(context.root)))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
