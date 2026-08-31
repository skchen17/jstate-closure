"""Paired causal closure and mediation runner for frozen protocol v3.2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp_v3_2 import build_v32_schedule, schedules_share_initial_perturbation
from jclosure.experiments.calibrate_v3_1 import _chain_transform, _read_jsonl
from jclosure.experiments.calibrate_v3_2 import _load_encoder, _payload
from jclosure.experiments.closure_v3_1 import _control_candidate
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.metrics import (
    answer_flip,
    jensen_shannon_from_logits,
    token_log_odds,
    token_probability,
)
from jclosure.model import load_model_bundle
from jclosure.protocol_v3_2 import verify_closure_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_2 import (
    PROTOCOL_V32,
    construct_initial_sequence,
    fit_naturality_models_v32,
    load_v32_domain,
    match_donor,
    replacement_transform,
    restoration_is_optimized,
    teacher_preanswer_prefix,
    tensor_digest,
    v32_thresholds,
)
from jclosure.statistics import clustered_bootstrap_ci, numerical_null_threshold
from jclosure.statistics_v3_1 import (
    common_valid_base_trials,
    paired_mediation_bootstrap,
)


def _bank_from_freeze(root: Path, freeze: dict[str, Any]) -> Path:
    paths = [path for path in freeze["calibration_hashes"] if path.endswith("activation_bank_manifest.jsonl")]
    if len(paths) != 1:
        raise RuntimeError("v3.2 freeze must name one activation bank")
    return root / paths[0]


def _run_condition(
    *,
    bundle,
    input_ids: torch.Tensor,
    clean_logits: torch.Tensor,
    clean_by_layer: dict[int, torch.Tensor],
    donor_by_layer: dict[int, torch.Tensor],
    candidate: torch.Tensor,
    condition: str,
    mode: str,
    restoration_scope: str,
    restoration_method: str,
    l1: int,
    restoration_layers: list[int],
    dense_map,
    encoder,
    naturality,
    tolerance: float,
    thresholds,
    answer_id: int,
) -> dict[str, Any]:
    length = int(input_ids.shape[-1])
    schedule = build_v32_schedule(
        mode=mode, initial_layer=l1, restoration_layers=restoration_layers,
        sequence_length=length, initial_scope="final", restoration_scope=restoration_scope,
    )
    transforms: dict[int, Any] = {}
    if condition != "clean":
        transforms[l1] = replacement_transform(candidate, schedule.initial_positions)
    capture: dict[int, dict[str, Any]] = {}
    if condition == "state_preserving" and mode != "single":
        optimized = restoration_is_optimized(restoration_method)
        for layer in restoration_layers:
            transforms[layer] = partial(
                _chain_transform, clean=clean_by_layer[layer], donor=donor_by_layer[layer],
                positions=schedule.restoration_positions, dense_map=dense_map,
                encoder=encoder, naturality=naturality[restoration_scope][layer],
                tolerance=tolerance, optimized=optimized, thresholds=thresholds,
                capture=capture,
            )
    record_layers = [l1, *restoration_layers]
    with ResidualEditor(bundle.layers, transforms), ActivationRecorder(bundle.layers, at=record_layers) as recorder:
        with torch.no_grad():
            logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
    future_final: dict[str, float] = {}
    future_all: dict[str, float] = {}
    for layer in restoration_layers:
        clean_sequence = clean_by_layer[layer].to(bundle.hf_model.device).float()
        current_sequence = recorder.activations[layer][0].detach().float()
        distances = []
        for position in range(length):
            clean_state = dense_map.dense_state(clean_sequence[position], layer)
            current_state = dense_map.dense_state(current_sequence[position], layer)
            distances.append(float(1 - torch.nn.functional.cosine_similarity(clean_state[None], current_state[None]).item()))
        future_final[str(layer)] = distances[-1]
        future_all[str(layer)] = float(np.mean(distances))
    valid = condition != "state_preserving" or mode == "single" or (
        len(capture) == len(restoration_layers) and all(value["passed"] for value in capture.values())
    )
    actual_events = [] if condition == "clean" else [asdict(event) for event in schedule.events]
    return {
        "valid": valid,
        "exclusion_reason": None if valid else "runtime_restoration_invalid",
        "hook_schedule": asdict(schedule),
        "hook_execution_map": actual_events,
        "restoration_events": [capture[layer] for layer in sorted(capture)],
        "metrics": {
            "js_divergence": jensen_shannon_from_logits(clean_logits, logits),
            "target_probability": token_probability(logits, answer_id),
            "target_probability_clean": token_probability(clean_logits, answer_id),
            "target_log_odds": token_log_odds(logits, answer_id),
            "target_log_odds_clean": token_log_odds(clean_logits, answer_id),
            "target_log_odds_delta": token_log_odds(logits, answer_id) - token_log_odds(clean_logits, answer_id),
            "answer_flip": answer_flip(clean_logits, logits),
            "task_correct": int(torch.argmax(logits)) == answer_id,
            "task_accuracy_change": float((int(torch.argmax(logits)) == answer_id) - (int(torch.argmax(clean_logits)) == answer_id)),
            "future_j_distances_final": future_final,
            "future_j_distances_all_positions": future_all,
            "mean_future_j_distance": float(np.mean(list(future_final.values()))) if future_final else None,
        },
        "_logits": logits.numpy().astype(np.float16),
    }


def _run_trials(context, bundle, freeze, *, domain: str, shard_index: int, shard_count: int, limit: int | None) -> list[dict[str, Any]]:
    protocol = freeze["selected_protocol"]
    l1 = int(protocol["l1"])
    restoration_layers = [int(value) for value in protocol["restoration_layers"]]
    restoration_method = str(protocol["restoration_method"])
    bank = _bank_from_freeze(context.root, freeze)
    bank_records = _read_jsonl(bank)
    fit = [row for row in bank_records if row["domain"] == "naturality_fit"]
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    naturality = {
        scope: fit_naturality_models_v32(context.root, fit, [l1, *restoration_layers], scope=scope, config=context.config)
        for scope in ("final", "all_non_padding")
    }
    thresholds = v32_thresholds(context.config)
    tolerance = float(context.config["geometry"]["formal_null_tolerance"])
    examples = load_v32_domain(context.root, domain)
    target = int(context.config["run"]["pilot_valid_base_trials" if domain == "pilot" else "confirm_valid_base_trials"])
    if limit is not None:
        target = min(target, int(limit))
    shard_target = target // shard_count + int(shard_index < target % shard_count)
    maximum_attempts = min(len(examples), shard_target * int(context.config["run"]["max_attempt_multiplier"]))
    rows: list[dict[str, Any]] = []
    completed = attempts = 0
    answer_map = dense_map.raw_map(l1, device=bundle.hf_model.device, dtype=torch.float32)
    token_to_index = {int(token): index for index, token in enumerate(vocabulary.token_ids)}
    logits_root = context.root / "artifacts/causal/v3_2" / context.run_id
    logits_root.mkdir(parents=True, exist_ok=True)
    for example in examples:
        if int(hashlib.sha256(example.example_id.encode()).hexdigest(), 16) % shard_count != shard_index:
            continue
        if attempts >= maximum_attempts or completed >= shard_target:
            break
        attempts += 1
        input_ids, answer_id, teacher_correct, exclusion = teacher_preanswer_prefix(bundle, example)
        attrition = {
            "schema_version": 5, "protocol_version": PROTOCOL_V32, "run_id": context.run_id,
            "prompt_id": example.example_id, "domain": domain, "record_type": "attrition",
            "teacher_correct": teacher_correct, "exclusion_reason": exclusion,
        }
        if not teacher_correct or answer_id is None:
            rows.append(attrition)
            continue
        with ActivationRecorder(bundle.layers, at=[l1, *restoration_layers]) as recorder:
            with torch.no_grad():
                clean_logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
        clean_by_layer = {layer: recorder.activations[layer][0].detach().float() for layer in [l1, *restoration_layers]}
        anchor = {
            "prompt_id": example.example_id,
            "prompt_hash": hashlib.sha256(example.prompt.strip().encode()).hexdigest(),
            "template_id": example.template_id, "sequence_length": int(input_ids.shape[-1]),
        }
        donor = match_donor(anchor, fit)
        if donor is None:
            rows.append({**attrition, "exclusion_reason": "donor_length_or_template_mismatch"})
            continue
        donor_payload = _payload(context.root, donor, bundle.hf_model.device)
        donor_by_layer = {layer: donor_payload["activations"][layer].to(bundle.hf_model.device).float() for layer in [l1, *restoration_layers]}
        initial_positions = (int(input_ids.shape[-1]) - 1,)
        candidate, initial_quality, initial_valid, displacement = construct_initial_sequence(
            clean_by_layer[l1], donor_by_layer[l1], positions=initial_positions, layer=l1,
            dense_map=dense_map, encoder=encoder, naturality=naturality["final"][l1],
            tolerance=tolerance, strength=float(context.config["v3_2"]["initial_strength"]),
            thresholds=thresholds, require_per_position_displacement=True,
        )
        if not initial_valid:
            rows.append({**attrition, "exclusion_reason": "initial_intervention_invalid", "initial_quality": initial_quality, "initial_aggregate_displacement_fraction": displacement})
            continue
        if answer_id not in token_to_index:
            rows.append({**attrition, "exclusion_reason": "answer_not_in_dictionary"})
            continue
        answer_direction = answer_map[token_to_index[answer_id]]
        base_id = hashlib.sha256(f"{domain}\x1f{example.example_id}\x1f{donor['prompt_id']}\x1f{l1}".encode()).hexdigest()
        variants: list[tuple[str, str, str, torch.Tensor]] = [
            ("clean", "single", "none", clean_by_layer[l1]),
            ("state_preserving", "single", "none", candidate),
            ("state_preserving", "persistent_final", "final", candidate),
            ("state_preserving", "persistent_all", "all_non_padding", candidate),
        ]
        for control in context.config["v3_2"]["controls"]:
            if control == "clean":
                continue
            control_candidate = _control_candidate(
                str(control), clean=clean_by_layer[l1], donor=donor_by_layer[l1],
                preserving=candidate, answer_direction=answer_direction,
                seed=int(context.seed) + attempts,
            )
            variants.append((str(control), "single", "none", control_candidate))
        trial_rows = []
        logits_payload: dict[str, np.ndarray] = {}
        schedules = []
        for condition, mode, restoration_scope, replacement in variants:
            result = _run_condition(
                bundle=bundle, input_ids=input_ids, clean_logits=clean_logits,
                clean_by_layer=clean_by_layer, donor_by_layer=donor_by_layer,
                candidate=replacement, condition=condition, mode=mode,
                restoration_scope=restoration_scope, restoration_method=restoration_method,
                l1=l1, restoration_layers=restoration_layers, dense_map=dense_map,
                encoder=encoder, naturality=naturality, tolerance=tolerance,
                thresholds=thresholds, answer_id=answer_id,
            )
            logits_payload[f"{condition}__{mode}"] = result.pop("_logits")
            if condition == "state_preserving":
                schedules.append(result["hook_schedule"])
            trial_rows.append({
                "schema_version": 5, "protocol_version": PROTOCOL_V32,
                "run_id": context.run_id, "record_type": "causal_trial",
                "base_trial_id": base_id,
                "paired_trial_id": hashlib.sha256(f"{base_id}\x1f{condition}\x1f{mode}".encode()).hexdigest(),
                "prompt_id": example.example_id, "prompt_hash": anchor["prompt_hash"],
                "prompt": example.prompt, "answer": example.answer, "answer_token_id": answer_id,
                "donor_id": donor["prompt_id"], "domain": domain, "condition": condition,
                "mode": mode, "l1": l1, "initial_scope": "final",
                "restoration_scope": restoration_scope, "restoration_method": restoration_method,
                "dictionary_size": len(vocabulary.token_ids), "dictionary_hash": vocabulary.digest,
                "initial_quality": initial_quality,
                "initial_candidate_sha256": tensor_digest(candidate),
                "initial_aggregate_displacement_fraction": displacement,
                **result,
            })
        logit_path = logits_root / f"{base_id}.npz"
        np.savez_compressed(logit_path, **logits_payload)
        for row in trial_rows:
            row["output_logits_path"] = str(logit_path.relative_to(context.root))
            row["output_logits_sha256"] = sha256_file(logit_path)
        schedule_objects = [
            build_v32_schedule(
                mode=row["mode"], initial_layer=l1, restoration_layers=restoration_layers,
                sequence_length=int(input_ids.shape[-1]), initial_scope="final",
                restoration_scope=row["restoration_scope"],
            )
            for row in trial_rows if row["condition"] == "state_preserving"
        ]
        initial_shared = schedules_share_initial_perturbation(schedule_objects)
        for row in trial_rows:
            row["paired_initial_perturbation_identical"] = initial_shared
        if all(row["valid"] for row in trial_rows):
            completed += 1
        rows.extend(trial_rows)
    return rows


def _summary(frame: pd.DataFrame, config: dict[str, Any], *, domain: str) -> dict[str, Any]:
    trials = frame[(frame["record_type"] == "causal_trial") & frame["valid"]].copy()
    metric_names = ["js_divergence", "mean_future_j_distance", "target_log_odds_delta", "answer_flip", "task_accuracy_change"]
    for metric in metric_names:
        trials[metric] = trials["metrics"].map(lambda value, name=metric: value.get(name))
    required_pairs = {
        ("clean", "single"), ("state_preserving", "single"),
        ("state_preserving", "persistent_final"), ("state_preserving", "persistent_all"),
        ("identity", "single"), ("zero_strength", "single"),
        ("matched_random", "single"), ("j_positive", "single"), ("full_patch", "single"),
    }
    common = common_valid_base_trials(trials, required_pairs=required_pairs)
    effects: dict[str, Any] = {}
    for metric in metric_names:
        effects[metric] = {}
        for (condition, mode), group in common.dropna(subset=[metric]).groupby(["condition", "mode"]):
            ci = clustered_bootstrap_ci(
                group, cluster_col="prompt_id", value_col=metric,
                n_resamples=int(config["v3_2"]["bootstrap_resamples"]),
                confidence=float(config["v3_2"]["confidence"]),
                seed=int(config["reproducibility"]["bootstrap_seed"]),
            )
            effects[metric][f"{condition}:{mode}"] = asdict(ci)
    null = common[common["condition"].isin(["identity", "zero_strength"])]
    null_threshold = numerical_null_threshold(
        null["js_divergence"], floor=float(config["v3_2"]["null_js_floor"]),
        quantile=float(config["v3_2"]["null_quantile"]),
    )
    try:
        mediation = paired_mediation_bootstrap(
            common, cluster_col="prompt_id", value_col="js_divergence",
            n_resamples=int(config["v3_2"]["bootstrap_resamples"]),
            confidence=float(config["v3_2"]["confidence"]),
            seed=int(config["reproducibility"]["bootstrap_seed"]),
            null_threshold=null_threshold,
        )
    except ValueError:
        mediation = None
    j_positive = effects["js_divergence"].get("j_positive:single")
    expected = int(config["run"]["pilot_valid_base_trials" if domain == "pilot" else "confirm_valid_base_trials"])
    state_rows = common[common["condition"] == "state_preserving"]
    map_counts = state_rows.groupby("mode")["hook_execution_map"].apply(lambda values: len({json.dumps(value, sort_keys=True) for value in values})) if not state_rows.empty else pd.Series(dtype=int)
    instrumentation = {
        "complete_paired_target_met": common["base_trial_id"].nunique() >= expected,
        "positive_control_above_null": bool(j_positive and j_positive["lower"] > null_threshold),
        "all_initial_candidates_identical": bool(common["paired_initial_perturbation_identical"].all()) if not common.empty else False,
        "three_hook_modes_present": set(state_rows["mode"]) == {"single", "persistent_final", "persistent_all"},
        "hook_map_variants_by_mode": map_counts.to_dict(),
    }
    instrumentation["passed"] = all(value for key, value in instrumentation.items() if key != "hook_map_variants_by_mode")
    return {
        "schema_version": 5, "protocol_version": PROTOCOL_V32, "domain": domain,
        "attempted_records": len(frame), "valid_records": len(trials),
        "complete_paired_base_trials": int(common["base_trial_id"].nunique()),
        "null_threshold": null_threshold, "effects": effects,
        "mediation": mediation, "instrumentation_gate": instrumentation,
    }


def main() -> None:
    parser = standard_parser("Run frozen closure protocol v3.2", "configs/closure_v3_2.yaml")
    parser.add_argument("--domain", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--stage", choices=("run", "merge"), default="run")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID", "single"))
    args = parser.parse_args()
    context = initialize_context("closure-v3-2", args)
    try:
        freeze = verify_closure_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", domain=args.domain, stage=args.stage)
            return
        if args.stage == "run":
            bundle = load_model_bundle(context.config)
            rows = _run_trials(context, bundle, freeze, domain=args.domain, shard_index=args.shard_index, shard_count=args.shard_count, limit=args.limit)
            output = context.raw_dir / context.run_id / f"causal-{args.domain}-shard-{args.shard_index:03d}.jsonl"
            append_jsonl(output, rows)
            context.finish("COMPLETED_SHARD", domain=args.domain, shard_group_id=args.shard_group_id, shard_index=args.shard_index, shard_count=args.shard_count, records=str(output.relative_to(context.root)), record_count=len(rows))
            return
        manifests = []
        for path in context.raw_dir.glob("closure-v3-2-*/manifest.json"):
            value = json.loads(path.read_text())
            if value.get("status") == "COMPLETED_SHARD" and value.get("shard_group_id") == args.shard_group_id and value.get("domain") == args.domain:
                manifests.append(value)
        by_index = {int(value["shard_index"]): value for value in manifests}
        if set(by_index) != set(range(args.shard_count)):
            raise RuntimeError("v3.2 causal merge is missing shards")
        rows = []
        for index in sorted(by_index):
            rows.extend(_read_jsonl(context.root / by_index[index]["records"]))
        frame = pd.DataFrame(rows)
        records_path = context.processed_dir / f"closure_v3_2_{args.domain}.parquet"
        frame.to_parquet(records_path, index=False, compression="zstd")
        summary = _summary(frame, context.config, domain=args.domain)
        summary.update({"run_id": context.run_id, "records": str(records_path.relative_to(context.root)), "source_shards": [by_index[index]["run_id"] for index in sorted(by_index)]})
        output = context.processed_dir / f"closure_v3_2_{args.domain}.json"
        write_json_atomic(output, summary)
        context.finish("COMPLETED", domain=args.domain, stage="merge", summary=str(output.relative_to(context.root)))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
