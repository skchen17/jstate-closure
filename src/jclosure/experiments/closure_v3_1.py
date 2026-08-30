"""Frozen direct-L1 causal closure and mediation runner for protocol v3.1."""

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

from jclosure.clamp_v3_1 import build_v31_schedule
from jclosure.experiments.calibrate_v3_1 import (
    _chain_transform,
    _load_encoder,
    _read_jsonl,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.interventions import matched_random_direction
from jclosure.metrics import (
    answer_flip,
    jensen_shannon_from_logits,
    token_log_odds,
    token_probability,
)
from jclosure.model import load_model_bundle
from jclosure.protocol_v3_1 import verify_freeze
from jclosure.provenance import append_jsonl, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_1 import (
    PROTOCOL_V31,
    construct_initial_sequence,
    fit_naturality_models,
    load_v31_domain,
    match_donor,
    replacement_transform,
    select_positions,
    teacher_preanswer_prefix,
    v31_thresholds,
)
from jclosure.statistics import (
    clustered_bootstrap_ci,
    numerical_null_threshold,
)
from jclosure.statistics_v3_1 import (
    common_valid_base_trials,
    paired_mediation_bootstrap,
)


def _payload(
    root: Path, record: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    value = torch.load(root / record["activation_path"], map_location="cpu")
    value["input_ids"] = value["input_ids"].to(device)
    return value


def _bank_from_freeze(root: Path, freeze: dict[str, Any]) -> Path:
    paths = [
        path
        for path in freeze["data_hashes"]
        if path.endswith("activation_bank_manifest.jsonl")
    ]
    if len(paths) != 1:
        raise RuntimeError("closure v3.1 freeze must name one activation bank")
    return root / paths[0]


def _scaled(vector: torch.Tensor, target_norm: float) -> torch.Tensor:
    return vector.float() * (
        float(target_norm) / torch.linalg.vector_norm(vector.float()).clamp_min(1e-20)
    )


def _sphere_candidate(
    clean: torch.Tensor, direction: torch.Tensor, target: float
) -> torch.Tensor:
    tangent = direction.float() - clean.float() * (
        torch.dot(direction.float(), clean.float())
        / torch.dot(clean.float(), clean.float()).clamp_min(1e-20)
    )
    step = _scaled(tangent, target)
    candidate = clean.float() + step
    return candidate * (
        torch.linalg.vector_norm(clean.float())
        / torch.linalg.vector_norm(candidate).clamp_min(1e-20)
    )


def _control_candidate(
    condition: str,
    *,
    clean: torch.Tensor,
    donor: torch.Tensor,
    preserving: torch.Tensor,
    answer_direction: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    if condition in {"identity", "zero_strength"}:
        return clean.clone().float()
    output = clean.clone().float()
    for position in range(clean.shape[0]):
        target = float(
            torch.linalg.vector_norm(
                preserving[position].float() - clean[position].float()
            )
        )
        if condition == "matched_random":
            direction = matched_random_direction(
                clean[position], seed=seed + position, norm=1.0
            )
        elif condition == "j_positive":
            direction = answer_direction
        elif condition == "full_patch":
            direction = donor[position].float() - clean[position].float()
        else:
            raise ValueError(f"unknown v3.1 control: {condition}")
        output[position] = _sphere_candidate(clean[position], direction, target)
    return output


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
    l1: int,
    restoration_layers: list[int],
    initial_positions: tuple[int, ...],
    dense_map,
    encoder,
    naturality,
    tolerance: float,
    thresholds,
    answer_id: int,
) -> dict[str, Any]:
    schedule = build_v31_schedule(
        mode=mode,
        initial_layer=l1,
        restoration_layers=restoration_layers,
        initial_positions=initial_positions,
        final_position=input_ids.shape[-1] - 1,
    )
    transforms: dict[int, Any] = {}
    if condition != "clean":
        transforms[l1] = replacement_transform(candidate, initial_positions)
    capture: dict[int, dict[str, Any]] = {}
    if condition == "state_preserving" and mode != "single":
        by_layer: dict[int, tuple[int, ...]] = {}
        for layer, position in schedule.modified_layer_positions:
            if layer > l1:
                by_layer.setdefault(layer, ())
                by_layer[layer] = (*by_layer[layer], position)
        for layer, positions in by_layer.items():
            transforms[layer] = partial(
                _chain_transform,
                clean=clean_by_layer[layer],
                donor=donor_by_layer[layer],
                positions=positions,
                dense_map=dense_map,
                encoder=encoder,
                naturality=naturality[layer],
                tolerance=tolerance,
                optimized=False,
                thresholds=thresholds,
                capture=capture,
            )
    record_layers = [l1, *restoration_layers]
    with (
        ResidualEditor(bundle.layers, transforms),
        ActivationRecorder(bundle.layers, at=record_layers) as recorder,
    ):
        with torch.no_grad():
            logits = bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
    future = {}
    for layer in restoration_layers:
        clean_state = dense_map.dense_state(
            clean_by_layer[layer][-1].to(bundle.hf_model.device).float(), layer
        )
        observed = recorder.activations[layer][0, -1].detach().float()
        current_state = dense_map.dense_state(observed, layer)
        future[str(layer)] = float(
            1
            - torch.nn.functional.cosine_similarity(
                clean_state[None], current_state[None]
            ).item()
        )
    valid = condition != "state_preserving" or (
        mode == "single"
        or (
            len(capture) == len(restoration_layers)
            and all(value["passed"] for value in capture.values())
        )
    )
    return {
        "valid": valid,
        "exclusion_reason": None if valid else "runtime_restoration_invalid",
        "hook_schedule": asdict(schedule),
        "restoration_events": [capture[layer] for layer in sorted(capture)],
        "metrics": {
            "js_divergence": jensen_shannon_from_logits(clean_logits, logits),
            "target_probability": token_probability(logits, answer_id),
            "target_probability_clean": token_probability(clean_logits, answer_id),
            "target_log_odds": token_log_odds(logits, answer_id),
            "target_log_odds_clean": token_log_odds(clean_logits, answer_id),
            "answer_flip": answer_flip(clean_logits, logits),
            "task_correct": int(torch.argmax(logits)) == answer_id,
            "future_j_distances": future,
            "mean_future_j_distance": float(np.mean(list(future.values())))
            if future
            else None,
        },
    }


def _run_trials(
    context,
    bundle,
    freeze,
    *,
    domain: str,
    shard_index: int,
    shard_count: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    protocol = freeze["selected_protocol"]
    l1 = int(protocol["l1"])
    restoration_layers = [int(value) for value in protocol["restoration_layers"]]
    bank = _bank_from_freeze(context.root, freeze)
    bank_records = _read_jsonl(bank)
    fit = [row for row in bank_records if row["domain"] == "naturality_fit"]
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    naturality = fit_naturality_models(
        context.root,
        fit,
        [l1, *restoration_layers],
        scope="all_non_padding",
        config=context.config,
    )
    threshold = v31_thresholds(context.config)
    tolerance = float(context.config["geometry"]["formal_null_tolerance"])
    examples = load_v31_domain(context.root, domain)
    target = int(
        context.config["run"][
            "pilot_valid_base_trials"
            if domain == "pilot"
            else "confirm_valid_base_trials"
        ]
    )
    if limit is not None:
        target = min(target, limit)
    shard_target = target // shard_count + int(shard_index < target % shard_count)
    maximum_attempts = min(
        len(examples),
        shard_target * int(context.config["run"]["max_attempt_multiplier"]),
    )
    rows: list[dict[str, Any]] = []
    completed = 0
    attempts = 0
    answer_map = dense_map.raw_map(
        l1, device=bundle.hf_model.device, dtype=torch.float32
    )
    token_to_index = {
        int(token): index for index, token in enumerate(vocabulary.token_ids)
    }
    for example in examples:
        if (
            int(hashlib.sha256(example.example_id.encode()).hexdigest(), 16)
            % shard_count
            != shard_index
        ):
            continue
        if attempts >= maximum_attempts or completed >= shard_target:
            break
        attempts += 1
        input_ids, answer_id, teacher_correct, exclusion = teacher_preanswer_prefix(
            bundle, example
        )
        attrition = {
            "schema_version": 4,
            "protocol_version": PROTOCOL_V31,
            "run_id": context.run_id,
            "prompt_id": example.example_id,
            "domain": domain,
            "record_type": "attrition",
            "teacher_correct": teacher_correct,
            "exclusion_reason": exclusion,
        }
        if not teacher_correct or answer_id is None:
            rows.append(attrition)
            continue
        with ActivationRecorder(
            bundle.layers, at=[l1, *restoration_layers]
        ) as recorder:
            with torch.no_grad():
                clean_logits = (
                    bundle.forward_logits(input_ids)[0, -1].detach().float().cpu()
                )
        clean_by_layer = {
            layer: recorder.activations[layer][0].detach().float()
            for layer in [l1, *restoration_layers]
        }
        anchor = {
            "prompt_id": example.example_id,
            "prompt_hash": hashlib.sha256(example.prompt.strip().encode()).hexdigest(),
            "template_id": example.template_id,
            "sequence_length": int(input_ids.shape[-1]),
        }
        donor = match_donor(anchor, fit)
        if donor is None:
            rows.append(
                {**attrition, "exclusion_reason": "donor_length_or_template_mismatch"}
            )
            continue
        donor_payload = _payload(context.root, donor, bundle.hf_model.device)
        donor_by_layer = {
            layer: donor_payload["activations"][layer]
            .to(bundle.hf_model.device)
            .float()
            for layer in [l1, *restoration_layers]
        }
        positions = select_positions(int(input_ids.shape[-1]), "all_non_padding")
        candidate, initial_positions, initial_valid, aggregate = (
            construct_initial_sequence(
                clean_by_layer[l1],
                donor_by_layer[l1],
                positions=positions,
                layer=l1,
                dense_map=dense_map,
                encoder=encoder,
                naturality=naturality[l1],
                tolerance=tolerance,
                strength=float(context.config["v3_1"]["initial_strength"]),
                thresholds=threshold,
                require_per_position_displacement=False,
            )
        )
        if not initial_valid:
            rows.append(
                {
                    **attrition,
                    "exclusion_reason": "initial_intervention_invalid",
                    "initial_positions": initial_positions,
                    "initial_aggregate_displacement_fraction": aggregate,
                }
            )
            continue
        if answer_id not in token_to_index:
            rows.append({**attrition, "exclusion_reason": "answer_not_in_dictionary"})
            continue
        answer_direction = answer_map[token_to_index[answer_id]]
        base_id = hashlib.sha256(
            f"{domain}\x1f{example.example_id}\x1f{donor['prompt_id']}\x1f{l1}".encode()
        ).hexdigest()
        variants: list[tuple[str, str, torch.Tensor]] = [
            ("clean", "single", clean_by_layer[l1]),
            ("state_preserving", "single", candidate),
            ("state_preserving", "persistent_final", candidate),
            ("state_preserving", "persistent_all", candidate),
        ]
        for control in context.config["v3_1"]["controls"]:
            if control == "clean":
                continue
            control_candidate = _control_candidate(
                str(control),
                clean=clean_by_layer[l1],
                donor=donor_by_layer[l1],
                preserving=candidate,
                answer_direction=answer_direction,
                seed=int(context.seed) + attempts,
            )
            variants.append((str(control), "single", control_candidate))
        trial_rows = []
        for condition, mode, replacement in variants:
            result = _run_condition(
                bundle=bundle,
                input_ids=input_ids,
                clean_logits=clean_logits,
                clean_by_layer=clean_by_layer,
                donor_by_layer=donor_by_layer,
                candidate=replacement,
                condition=condition,
                mode=mode,
                l1=l1,
                restoration_layers=restoration_layers,
                initial_positions=positions,
                dense_map=dense_map,
                encoder=encoder,
                naturality=naturality,
                tolerance=tolerance,
                thresholds=threshold,
                answer_id=answer_id,
            )
            trial_rows.append(
                {
                    "schema_version": 4,
                    "protocol_version": PROTOCOL_V31,
                    "run_id": context.run_id,
                    "record_type": "causal_trial",
                    "base_trial_id": base_id,
                    "paired_trial_id": hashlib.sha256(
                        f"{base_id}\x1f{condition}\x1f{mode}".encode()
                    ).hexdigest(),
                    "prompt_id": example.example_id,
                    "prompt_hash": anchor["prompt_hash"],
                    "prompt": example.prompt,
                    "answer": example.answer,
                    "answer_token_id": answer_id,
                    "donor_id": donor["prompt_id"],
                    "domain": domain,
                    "condition": condition,
                    "mode": mode,
                    "l1": l1,
                    "position_scope": "all_non_padding",
                    "dictionary_size": len(vocabulary.token_ids),
                    "dictionary_hash": vocabulary.digest,
                    "initial_positions": initial_positions,
                    "initial_aggregate_displacement_fraction": aggregate,
                    **result,
                }
            )
        if all(row["valid"] for row in trial_rows):
            completed += 1
        rows.extend(trial_rows)
    return rows


def _summary(
    frame: pd.DataFrame, config: dict[str, Any], *, domain: str
) -> dict[str, Any]:
    trials = frame[(frame["record_type"] == "causal_trial") & frame["valid"]].copy()
    trials["js_divergence"] = trials["metrics"].map(
        lambda value: value["js_divergence"]
    )
    required_pairs = {
        ("clean", "single"),
        ("state_preserving", "single"),
        ("state_preserving", "persistent_final"),
        ("state_preserving", "persistent_all"),
        ("identity", "single"),
        ("zero_strength", "single"),
        ("matched_random", "single"),
        ("j_positive", "single"),
        ("full_patch", "single"),
    }
    common = common_valid_base_trials(trials, required_pairs=required_pairs)
    common_ids = set(common["base_trial_id"])
    estimates = {}
    for (condition, mode), group in common.groupby(["condition", "mode"]):
        ci = clustered_bootstrap_ci(
            group,
            cluster_col="prompt_id",
            value_col="js_divergence",
            n_resamples=int(config["v3_1"]["bootstrap_resamples"]),
            confidence=float(config["v3_1"]["confidence"]),
            seed=int(config["reproducibility"]["bootstrap_seed"]),
        )
        estimates[f"{condition}:{mode}"] = asdict(ci)
    null = common[common["condition"].isin(["identity", "zero_strength"])]
    null_threshold = numerical_null_threshold(
        null["js_divergence"],
        floor=float(config["v3_1"]["null_js_floor"]),
        quantile=float(config["v3_1"]["null_quantile"]),
    )
    try:
        mediation = paired_mediation_bootstrap(
            common,
            cluster_col="prompt_id",
            value_col="js_divergence",
            n_resamples=int(config["v3_1"]["bootstrap_resamples"]),
            confidence=float(config["v3_1"]["confidence"]),
            seed=int(config["reproducibility"]["bootstrap_seed"]),
            null_threshold=null_threshold,
        )
    except ValueError:
        mediation = None
    return {
        "schema_version": 4,
        "protocol_version": PROTOCOL_V31,
        "domain": domain,
        "attempted_records": len(frame),
        "valid_records": len(trials),
        "complete_paired_base_trials": len(common_ids),
        "null_threshold": null_threshold,
        "effects": estimates,
        "mediation": mediation,
    }


def main() -> None:
    parser = standard_parser(
        "Run frozen closure protocol v3.1", "configs/closure_v3_1.yaml"
    )
    parser.add_argument("--domain", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--stage", choices=("run", "merge"), default="run")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID", "single")
    )
    args = parser.parse_args()
    context = initialize_context("closure-v3-1", args)
    try:
        freeze = verify_freeze(context.root, kind="closure", config=context.config)
        if args.dry_run:
            context.finish("DRY_RUN", domain=args.domain, stage=args.stage)
            return
        if args.stage == "run":
            bundle = load_model_bundle(context.config)
            rows = _run_trials(
                context,
                bundle,
                freeze,
                domain=args.domain,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                limit=args.limit,
            )
            output = (
                context.raw_dir
                / context.run_id
                / f"causal-{args.domain}-shard-{args.shard_index:03d}.jsonl"
            )
            append_jsonl(output, rows)
            context.finish(
                "COMPLETED_SHARD",
                domain=args.domain,
                shard_group_id=args.shard_group_id,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                records=str(output.relative_to(context.root)),
                record_count=len(rows),
            )
            return
        manifests = []
        for path in context.raw_dir.glob("closure-v3-1-*/manifest.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("status") == "COMPLETED_SHARD"
                and value.get("shard_group_id") == args.shard_group_id
                and value.get("domain") == args.domain
            ):
                manifests.append(value)
        by_index = {int(value["shard_index"]): value for value in manifests}
        if set(by_index) != set(range(args.shard_count)):
            raise RuntimeError("causal merge is missing shards")
        rows = []
        for index in sorted(by_index):
            rows.extend(_read_jsonl(context.root / by_index[index]["records"]))
        frame = pd.DataFrame(rows)
        records_path = context.processed_dir / f"closure_v3_1_{args.domain}.parquet"
        frame.to_parquet(records_path, index=False)
        summary = _summary(frame, context.config, domain=args.domain)
        summary.update(
            {
                "run_id": context.run_id,
                "records": str(records_path.relative_to(context.root)),
                "source_shards": [
                    by_index[index]["run_id"] for index in sorted(by_index)
                ],
            }
        )
        summary_path = context.processed_dir / f"closure_v3_1_{args.domain}.json"
        write_json_atomic(summary_path, summary)
        context.finish(
            "COMPLETED",
            domain=args.domain,
            stage="merge",
            summary=str(summary_path.relative_to(context.root)),
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
