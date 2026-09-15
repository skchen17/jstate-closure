"""Teacher screening, exact state capture, and raw causal screen for protocol v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.datasets_v8 import FAMILIES, load_tasks
from jclosure.experiments.calibrate_v8 import teacher_actions
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _safe_cosine,
    _teacher_forced_trajectory,
    _teacher_tokens,
    _trajectory_metrics,
)
from jclosure.persistent_state_v8 import (
    ATOM_ORDER,
    compose_condition,
    extract_architecture_state,
    factorial_conditions,
    leave_one_out_masks,
    mobius_interactions,
    named_raw_conditions,
    pairwise_masks,
    save_state_shard,
)
from jclosure.protocol_v8 import (
    SELECTION_PATH,
    build_freeze,
    build_guard,
    verify_freeze,
)
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.records_v8 import PersistentPairRecord, RawStateEffectRecord
from jclosure.single_arm_v4 import (
    construct_from_shared_basis,
    multiple_token_log_odds,
    shared_low_singular_basis,
)
from jclosure.statistics import clustered_bootstrap_ci

FORMAL_DATA = Path("data/v8/persistent_causal_formal.json")
TEACHER_SUMMARY = Path("results/v8/processed/teacher_formal_v8.json")
BANK_SUMMARY = Path("results/v8/processed/persistent_bank_v8.json")


def _parser() -> argparse.ArgumentParser:
    parser = standard_parser(
        "structured persistent causal-state experiment v8",
        "configs/persistent_state_v8.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "guard",
            "teacher",
            "teacher-reparse",
            "freeze",
            "bank",
            "capture",
            "analyze",
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "final_test"))
    return parser


def _teacher(context: Any, bundle: Any, limit: int | None) -> None:
    tasks = load_tasks(context.root / FORMAL_DATA)
    if limit is not None:
        tasks = tasks[: int(limit)]
    rows = []
    progress = context.raw_dir / context.run_id / "teacher_progress.json"
    for index, (split, task) in enumerate(tasks):
        actions, error = teacher_actions(bundle, task)
        rows.append(
            {
                "schema_version": 11,
                "protocol_version": "structured_persistent_state_protocol_v8",
                "record_type": "teacher_formal_trial",
                "run_id": context.run_id,
                "prompt_id": task.example_id,
                "program_hash": task.program_hash,
                "family": task.family,
                "split": split,
                "horizon": task.horizon,
                "attempted": True,
                "parseable": error is None and len(actions) == task.horizon,
                "full_trajectory_correct": tuple(actions) == task.semantic_actions,
                "final_answer_correct": bool(
                    actions and actions[-1] == task.final_answer
                ),
                "generated_actions": actions,
                "expected_actions": list(task.semantic_actions),
                "error": error,
            }
        )
        if index % 25 == 0 or index + 1 == len(tasks):
            write_json_atomic(
                progress,
                {"status": "RUNNING", "completed": index + 1, "total": len(tasks)},
            )
    path = context.raw_dir / context.run_id / "teacher_formal_v8.jsonl"
    append_jsonl(path, rows)
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["split", "family", "horizon"], sort=True)
        .agg(
            attempted=("attempted", "sum"),
            parseable=("parseable", "sum"),
            full_trajectory_correct=("full_trajectory_correct", "sum"),
            final_answer_correct=("final_answer_correct", "sum"),
        )
        .reset_index()
    )
    for numerator, rate in (
        ("parseable", "parseable_rate"),
        ("full_trajectory_correct", "full_trajectory_accuracy"),
        ("final_answer_correct", "final_answer_accuracy"),
    ):
        grouped[rate] = grouped[numerator] / grouped["attempted"]
    targets = context.config["persistent_state_v8"]["formal"]["target_valid_per_family"]
    correct_counts = (
        frame.groupby(["split", "family"])["full_trajectory_correct"].sum().to_dict()
    )
    authorized = all(
        int(correct_counts.get((split, family), 0)) >= int(targets[split])
        for split in targets
        for family in FAMILIES
    )
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "records": str(path.relative_to(context.root)),
        "records_sha256": sha256_file(path),
        "formal_data": str(FORMAL_DATA),
        "formal_data_sha256": sha256_file(context.root / FORMAL_DATA),
        "rows": grouped.to_dict("records"),
        "teacher_correct_counts": {
            f"{split}/{family}": int(value)
            for (split, family), value in correct_counts.items()
        },
        "target_valid_per_family": targets,
        "selection_authorized": authorized,
    }
    write_json_atomic(context.root / TEACHER_SUMMARY, summary)
    write_json_atomic(
        progress, {"status": "COMPLETED", "completed": len(tasks), "total": len(tasks)}
    )
    context.finish("COMPLETED_TEACHER_SCREEN", summary=summary)


def _teacher_reparse(context: Any, bundle: Any) -> None:
    source_summary = json.loads(
        (
            context.root
            / "results/v8/processed/teacher_formal_v8_competence_r2_strict_parser_attempt.json"
        ).read_text()
    )
    source_rows = [
        json.loads(line)
        for line in (context.root / source_summary["records"]).read_text().splitlines()
    ]
    tasks = {task.example_id: task for _, task in load_tasks(context.root / FORMAL_DATA)}
    rows = []
    reparsed = 0
    progress = context.raw_dir / context.run_id / "teacher_reparse_progress.json"
    for index, row in enumerate(source_rows):
        value = dict(row)
        value["run_id"] = context.run_id
        if not value["parseable"]:
            task = tasks[str(value["prompt_id"])]
            actions, error = teacher_actions(bundle, task)
            value.update(
                {
                    "parseable": error is None and len(actions) == task.horizon,
                    "full_trajectory_correct": tuple(actions)
                    == task.semantic_actions,
                    "final_answer_correct": bool(
                        actions and actions[-1] == task.final_answer
                    ),
                    "generated_actions": actions,
                    "expected_actions": list(task.semantic_actions),
                    "error": error,
                }
            )
            reparsed += 1
        rows.append(value)
        if index % 25 == 0 or index + 1 == len(source_rows):
            write_json_atomic(
                progress,
                {
                    "status": "RUNNING",
                    "completed": index + 1,
                    "total": len(source_rows),
                    "reparsed": reparsed,
                },
            )
    path = context.raw_dir / context.run_id / "teacher_formal_v8_reparsed.jsonl"
    append_jsonl(path, rows)
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["split", "family", "horizon"], sort=True)
        .agg(
            attempted=("attempted", "sum"),
            parseable=("parseable", "sum"),
            full_trajectory_correct=("full_trajectory_correct", "sum"),
            final_answer_correct=("final_answer_correct", "sum"),
        )
        .reset_index()
    )
    for numerator, rate in (
        ("parseable", "parseable_rate"),
        ("full_trajectory_correct", "full_trajectory_accuracy"),
        ("final_answer_correct", "final_answer_accuracy"),
    ):
        grouped[rate] = grouped[numerator] / grouped["attempted"]
    targets = context.config["persistent_state_v8"]["formal"][
        "target_valid_per_family"
    ]
    correct_counts = (
        frame.groupby(["split", "family"])["full_trajectory_correct"].sum().to_dict()
    )
    authorized = all(
        int(correct_counts.get((split, family), 0)) >= int(targets[split])
        for split in targets
        for family in FAMILIES
    )
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "records": str(path.relative_to(context.root)),
        "records_sha256": sha256_file(path),
        "formal_data": str(FORMAL_DATA),
        "formal_data_sha256": sha256_file(context.root / FORMAL_DATA),
        "rows": grouped.to_dict("records"),
        "teacher_correct_counts": {
            f"{split}/{family}": int(value)
            for (split, family), value in correct_counts.items()
        },
        "target_valid_per_family": targets,
        "selection_authorized": authorized,
        "parser_correction": {
            "source_summary": "results/v8/processed/teacher_formal_v8_competence_r2_strict_parser_attempt.json",
            "reparsed_count": reparsed,
            "reason": "strict expected-answer alphabet underestimated parseability",
        },
    }
    write_json_atomic(context.root / TEACHER_SUMMARY, summary)
    write_json_atomic(
        progress,
        {
            "status": "COMPLETED",
            "completed": len(source_rows),
            "total": len(source_rows),
            "reparsed": reparsed,
        },
    )
    context.finish("COMPLETED_TEACHER_REPARSE", summary=summary)


def _selected(root: Path) -> list[tuple[str, Any]]:
    manifest = json.loads((root / SELECTION_PATH).read_text(encoding="utf-8"))
    output = []
    for raw in manifest["items"]:
        item = dict(raw)
        split = str(item.pop("split"))
        item["semantic_actions"] = tuple(item["semantic_actions"])
        from jclosure.datasets_v4 import ProgramTraceTask

        output.append((split, ProgramTraceTask(**item)))
    return output


@torch.no_grad()
def _bank(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    layer = int(context.config["persistent_state_v8"]["intervention"]["layer"])
    tasks = [
        (split, task) for split, task in _selected(context.root) if split == "train"
    ]
    rows, states = [], []
    progress = context.raw_dir / context.run_id / "bank_progress.json"
    from jclosure.experiments.causal_single_v4 import _initial_state

    for index, (_, task) in enumerate(tasks):
        state = _initial_state(bundle, task.prompt, layer)
        states.append(state.cpu().numpy().astype(np.float32))
        rows.append(
            {
                "bank_index": index,
                "prompt_id": task.example_id,
                "family": task.family,
                "horizon": task.horizon,
                "program_hash": task.program_hash,
            }
        )
        if index % 25 == 0 or index + 1 == len(tasks):
            write_json_atomic(
                progress,
                {"status": "RUNNING", "completed": index + 1, "total": len(tasks)},
            )
    directory = context.root / "artifacts/persistent/v8" / context.run_id
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "train_states_f32.npz"
    np.savez_compressed(artifact, states=np.stack(states))
    records = context.raw_dir / context.run_id / "persistent_bank_v8.jsonl"
    append_jsonl(records, rows)
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "layer": layer,
        "count": len(rows),
        "records": str(records.relative_to(context.root)),
        "records_sha256": sha256_file(records),
        "artifact": str(artifact.relative_to(context.root)),
        "artifact_sha256": sha256_file(artifact),
    }
    write_json_atomic(context.root / BANK_SUMMARY, summary)
    write_json_atomic(
        progress, {"status": "COMPLETED", "completed": len(tasks), "total": len(tasks)}
    )
    context.finish("COMPLETED_BANK", summary=summary)


def _load_bank(root: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    summary = json.loads((root / BANK_SUMMARY).read_text(encoding="utf-8"))
    if sha256_file(root / summary["artifact"]) != summary["artifact_sha256"]:
        raise RuntimeError("v8 bank tensor hash mismatch")
    rows = [
        json.loads(line)
        for line in (root / summary["records"]).read_text().splitlines()
    ]
    with np.load(root / summary["artifact"], allow_pickle=False) as data:
        states = data["states"].astype(np.float32)
    return rows, states


def _thresholds(section: dict[str, Any]) -> V3ClampThresholds:
    return V3ClampThresholds(
        dense_cosine=float(section["dense_cosine_threshold"]),
        dense_top10_overlap=float(section["top10_overlap_threshold"]),
        rms_drift=float(section["rms_drift_threshold"]),
        formal_displacement=float(section["formal_displacement_fraction"]),
        sensitivity_displacement=0.05,
    )


def _mask_for(condition: str) -> int | None:
    reverse = {name: mask for name, mask in factorial_conditions().items()}
    return reverse.get(condition)


def _condition_names() -> list[str]:
    names = list(factorial_conditions())
    names.extend(("top_rec_l30", "top_rec_l28_l29_l30"))
    return names


def _semantic_ids(tokenizer: Any, surface: str) -> list[int]:
    output = []
    for value in (surface, " " + surface):
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) == 1:
            output.append(int(ids[0]))
    return sorted(set(output))


@torch.no_grad()
def _capture(
    context: Any, bundle: Any, freeze: dict[str, Any], split: str, limit: int | None
) -> None:
    section = context.config["persistent_state_v8"]
    intervention = section["intervention"]
    layer = int(intervention["layer"])
    measured = [int(value) for value in intervention["measured_layers"]]
    main_layer = max(measured)
    recurrent = [
        int(value) for value in section["persistent_components"]["recurrent_layers"]
    ]
    attention = [
        int(value) for value in section["persistent_components"]["attention_layers"]
    ]
    primary_layer = int(section["persistent_components"]["kv_primary_layer"])
    primary_head = int(section["persistent_components"]["kv_primary_head"])
    target_per_family = int(section["formal"]["target_valid_per_family"][split])
    if limit is not None:
        target_per_family = min(target_per_family, int(limit))
    tasks = [(role, task) for role, task in _selected(context.root) if role == split]
    bank_rows, bank_states = _load_bank(context.root)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    _, encoder, dense_map = _load_encoder(context, bundle)
    device = next(bundle.hf_model.parameters()).device
    shared = shared_low_singular_basis(
        dense_map, layer, relative_tolerance=1e-4, device=device
    )
    threshold = _thresholds(intervention)
    family_counts: dict[str, int] = defaultdict(int)
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    endpoint: dict[str, list[Any]] = defaultdict(list)
    shard_rows: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    artifact_dir = context.root / "artifacts/persistent/v8" / context.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    progress = context.raw_dir / context.run_id / f"capture_{split}_progress.json"

    def flush_shard() -> None:
        if not shard_rows:
            return
        index = len(shard_manifest)
        path = artifact_dir / f"state_shard_{split}_{index:04d}.pt"
        save_state_shard(path, shard_rows)
        shard_manifest.append(
            {
                "path": str(path.relative_to(context.root)),
                "sha256": sha256_file(path),
                "count": len(shard_rows),
                "base_trial_ids": [row["base_trial_id"] for row in shard_rows],
            }
        )
        shard_rows.clear()

    ordered = sorted(
        tasks,
        key=lambda value: hashlib.sha256(value[1].example_id.encode()).hexdigest(),
    )
    for attempt, (_, task) in enumerate(ordered):
        if all(family_counts[family] >= target_per_family for family in FAMILIES):
            break
        if family_counts[task.family] >= target_per_family:
            continue
        matches = [
            (row, bank_states[index])
            for index, row in enumerate(bank_rows)
            if row["family"] == task.family
            and int(row["horizon"]) == task.horizon
            and row["prompt_id"] != task.example_id
        ]
        if not matches:
            pair_rows.append(
                PersistentPairRecord(
                    run_id=context.run_id,
                    base_trial_id=f"v8-{split}-{task.example_id}",
                    prompt_id=task.example_id,
                    family=task.family,
                    split=split,
                    donor_id="",
                    valid=False,
                    intervention_quality={},
                    exclusion_reason="no_matched_donor",
                ).to_dict()
            )
            continue
        donor_row, donor_array = matches[
            int(hashlib.sha256(task.example_id.encode()).hexdigest(), 16) % len(matches)
        ]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=layer,
            candidate=None,
        )
        donor = torch.from_numpy(donor_array).to(device)
        natural_scale = float(
            torch.linalg.vector_norm(donor - clean["hidden"][layer]).item()
        )
        candidate, quality, valid = construct_from_shared_basis(
            clean["hidden"][layer],
            donor,
            shared=shared,
            dense_map=dense_map,
            encoder=encoder,
            natural_scale=natural_scale,
            displacement_fraction=float(intervention["initial_strength"]),
            thresholds=threshold,
            naturality=naturality,
        )
        singular_values = quality["construction"].pop("singular_values", [])
        quality["construction"]["singular_value_count"] = len(singular_values)
        quality["construction"]["singular_value_sha256"] = hashlib.sha256(
            np.asarray(singular_values, dtype=np.float32).tobytes()
        ).hexdigest()
        base_id = (
            f"v8-{split}-{hashlib.sha256(task.example_id.encode()).hexdigest()[:20]}"
        )
        if not valid:
            pair_rows.append(
                PersistentPairRecord(
                    run_id=context.run_id,
                    base_trial_id=base_id,
                    prompt_id=task.example_id,
                    family=task.family,
                    split=split,
                    donor_id=str(donor_row["prompt_id"]),
                    valid=False,
                    intervention_quality=quality,
                    exclusion_reason="intervention_ineligible",
                ).to_dict()
            )
            continue
        perturbed = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=layer,
            candidate=candidate,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=int(intervention["teacher_forced_tokens"]),
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
        semantic_index = min(1, task.horizon - 1)
        semantic_ids = _semantic_ids(
            bundle.tokenizer, task.semantic_actions[semantic_index]
        )
        clean_next_logits = torch.from_numpy(clean_trajectory["logits"][0])
        clean_ground_truth_odds = multiple_token_log_odds(
            clean_next_logits, semantic_ids
        )
        clean_ground_truth_correct = (
            int(torch.argmax(clean_next_logits)) in semantic_ids
        )
        raw_named = {value: key for key, value in named_raw_conditions().items()}
        condition_states: dict[str, np.ndarray] = {}
        condition_outputs: dict[str, np.ndarray] = {}
        for condition in _condition_names():
            named_condition = raw_named.get(condition)
            cache = compose_condition(
                clean["cache"],
                perturbed["cache"],
                condition,
                recurrent_layers=recurrent,
                attention_layers=attention,
                primary_layer=primary_layer,
                primary_head=primary_head,
                primary_position=clean["prompt_length"] - 1,
            )
            trajectory = _teacher_forced_trajectory(
                bundle,
                cache,
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured,
                dense_map=dense_map,
            )
            values = _trajectory_metrics(
                clean_trajectory, trajectory, main_layer=main_layer
            )
            next_delta = values.pop("next_j_delta")
            condition_outputs[condition] = values.pop("output_delta")
            candidate_next_logits = torch.from_numpy(trajectory["logits"][0])
            values["ground_truth_log_odds_change"] = (
                multiple_token_log_odds(candidate_next_logits, semantic_ids)
                - clean_ground_truth_odds
            )
            condition_correct = int(torch.argmax(candidate_next_logits)) in semantic_ids
            values["ground_truth_accuracy"] = float(condition_correct)
            values["clean_ground_truth_accuracy"] = float(clean_ground_truth_correct)
            values["ground_truth_accuracy_change"] = float(condition_correct) - float(
                clean_ground_truth_correct
            )
            endpoint[f"output_log_odds__{condition}"].append(
                np.float32(values["ground_truth_log_odds_change"])
            )
            condition_states[condition] = next_delta
            endpoint[f"next_j__{condition}"].append(
                trajectory["j"][main_layer][0].astype(np.float16)
            )
            endpoint[f"future_j__{condition}"].append(
                trajectory["j"][main_layer].astype(np.float16)
            )
            effect_rows.append(
                RawStateEffectRecord(
                    run_id=context.run_id,
                    base_trial_id=base_id,
                    prompt_id=task.example_id,
                    family=task.family,
                    split=split,
                    condition=named_condition or condition,
                    atom_mask=_mask_for(condition),
                    valid=True,
                    metrics=values,
                    metadata={
                        "condition_definition": condition,
                        "teacher_forced_tokens": tokens,
                        "main_measured_layer": main_layer,
                        "semantic_target": task.semantic_actions[semantic_index],
                        "semantic_target_token_ids": semantic_ids,
                    },
                ).to_dict()
            )
        full_delta = condition_states["+".join(ATOM_ORDER)]
        full_output_delta = condition_outputs["+".join(ATOM_ORDER)]
        current_effect_rows = effect_rows[-len(_condition_names()) :]
        full_effect_row = next(
            row
            for row in current_effect_rows
            if row["metadata"]["condition_definition"] == "+".join(ATOM_ORDER)
        )
        full_odds_sign = np.sign(full_effect_row["metrics"]["target_log_odds_change"])
        for row in current_effect_rows:
            condition = row["metadata"]["condition_definition"]
            delta = condition_states[condition]
            full_norm = float(np.linalg.norm(full_delta))
            row["metrics"]["direction_cosine_to_full"] = _safe_cosine(delta, full_delta)
            row["metrics"]["magnitude_ratio_to_full"] = float(
                float(np.linalg.norm(delta)) / max(full_norm, 1e-20)
            )
            row["metrics"]["output_delta_cosine_to_full"] = _safe_cosine(
                condition_outputs[condition], full_output_delta
            )
            condition_sign = np.sign(row["metrics"]["target_log_odds_change"])
            row["metrics"]["output_sign_agreement_to_full"] = float(
                condition_sign == full_odds_sign
            )
        clean_state = extract_architecture_state(
            clean["cache"], recurrent_layers=recurrent, attention_layers=attention
        )
        perturbed_state = extract_architecture_state(
            perturbed["cache"], recurrent_layers=recurrent, attention_layers=attention
        )
        artifact_row = len(shard_rows)
        shard_rows.append(
            {
                "base_trial_id": base_id,
                "prompt_id": task.example_id,
                "family": task.family,
                "split": split,
                "prompt_length": clean["prompt_length"],
                "teacher_tokens": tokens,
                "clean": clean_state,
                "perturbed": perturbed_state,
            }
        )
        pair_rows.append(
            PersistentPairRecord(
                run_id=context.run_id,
                base_trial_id=base_id,
                prompt_id=task.example_id,
                family=task.family,
                split=split,
                donor_id=str(donor_row["prompt_id"]),
                valid=True,
                intervention_quality=quality,
                artifact_shard=f"pending:{len(shard_manifest)}",
                artifact_row=artifact_row,
                metadata={"natural_scale": natural_scale, "teacher_tokens": tokens},
            ).to_dict()
        )
        endpoint["base_trial_id"].append(base_id)
        endpoint["prompt_id"].append(task.example_id)
        endpoint["family"].append(task.family)
        endpoint["current_j_clean"].append(
            clean["j"][layer].cpu().numpy().astype(np.float16)
        )
        endpoint["current_j_perturbed"].append(
            perturbed["j"][layer].cpu().numpy().astype(np.float16)
        )
        family_counts[task.family] += 1
        if len(shard_rows) >= 5:
            flush_shard()
        if attempt % 10 == 0:
            write_json_atomic(
                progress,
                {
                    "status": "RUNNING",
                    "attempted": attempt + 1,
                    "valid_counts": dict(family_counts),
                },
            )
    flush_shard()
    # Resolve shard references from the manifest after buffered writes.
    location = {
        base_id: (entry["path"], row)
        for entry in shard_manifest
        for row, base_id in enumerate(entry["base_trial_ids"])
    }
    for row in pair_rows:
        if row["valid"]:
            row["artifact_shard"], row["artifact_row"] = location[row["base_trial_id"]]
    pair_path = context.raw_dir / context.run_id / f"persistent_pairs_{split}_v8.jsonl"
    effect_path = (
        context.raw_dir / context.run_id / f"raw_state_effects_{split}_v8.jsonl"
    )
    append_jsonl(pair_path, pair_rows)
    append_jsonl(effect_path, effect_rows)
    endpoint_path = artifact_dir / f"endpoints_{split}_f16.npz"
    np.savez_compressed(
        endpoint_path, **{key: np.asarray(values) for key, values in endpoint.items()}
    )
    complete = all(family_counts[family] >= target_per_family for family in FAMILIES)
    artifact_manifest = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "split": split,
        "source_freeze_digest": freeze["freeze_digest"],
        "target_per_family": target_per_family,
        "valid_counts": dict(family_counts),
        "complete": complete,
        "pair_records": str(pair_path.relative_to(context.root)),
        "pair_records_sha256": sha256_file(pair_path),
        "effect_records": str(effect_path.relative_to(context.root)),
        "effect_records_sha256": sha256_file(effect_path),
        "endpoint_artifact": str(endpoint_path.relative_to(context.root)),
        "endpoint_artifact_sha256": sha256_file(endpoint_path),
        "state_shards": shard_manifest,
        "architecture": {
            "recurrent_layers": recurrent,
            "attention_layers": attention,
            "kv_primary": {"layer": primary_layer, "head": primary_head},
            "state_dtype": "bfloat16",
            "kv_token_scope": "all_prefill_positions",
        },
    }
    manifest_path = context.processed_dir / f"persistent_capture_{split}_v8.json"
    write_json_atomic(manifest_path, artifact_manifest)
    write_json_atomic(
        progress,
        {
            "status": "COMPLETED" if complete else "INCOMPLETE",
            "valid_counts": dict(family_counts),
        },
    )
    context.finish(
        "COMPLETED_CAPTURE" if complete else "INCOMPLETE_CAPTURE",
        summary=artifact_manifest,
    )


def _ci(frame: pd.DataFrame, column: str, seed: int, resamples: int) -> dict[str, Any]:
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


def _analyze(context: Any, freeze: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    captures: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "final_test"):
        capture = json.loads(
            (context.processed_dir / f"persistent_capture_{split}_v8.json").read_text()
        )
        if capture["source_freeze_digest"] != freeze["freeze_digest"]:
            raise RuntimeError("v8 capture freeze mismatch")
        captures[split] = capture
        rows.extend(
            json.loads(line)
            for line in (context.root / capture["effect_records"])
            .read_text()
            .splitlines()
        )
    frame = pd.DataFrame(rows)
    for metric in (
        "next_j_l2",
        "future_j_trajectory_divergence",
        "output_js_divergence",
        "target_log_odds_change",
        "target_log_odds_abs_change",
        "ground_truth_log_odds_change",
        "ground_truth_accuracy",
        "ground_truth_accuracy_change",
        "answer_flip",
        "token_flip_rate",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
        "output_sign_agreement_to_full",
    ):
        frame[metric] = frame["metrics"].map(lambda value, key=metric: value.get(key))
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_state_v8"]["bootstrap_resamples"])
    effects: dict[str, Any] = {}
    final = frame[(frame["split"] == "final_test") & frame["valid"]]
    groups = [("pooled", final), *list(final.groupby("family", sort=True))]
    for family, group in groups:
        effects[str(family)] = {
            metric: {
                condition: _ci(values.dropna(subset=[metric]), metric, seed, resamples)
                for condition, values in group.groupby("condition", sort=True)
            }
            for metric in (
                "next_j_l2",
                "future_j_trajectory_divergence",
                "output_js_divergence",
                "target_log_odds_change",
                "target_log_odds_abs_change",
                "ground_truth_log_odds_change",
                "ground_truth_accuracy",
                "ground_truth_accuracy_change",
                "answer_flip",
                "token_flip_rate",
                "direction_cosine_to_full",
                "magnitude_ratio_to_full",
                "output_delta_cosine_to_full",
                "output_sign_agreement_to_full",
            )
        }
    factorial = final[final["atom_mask"].notna()].copy()
    endpoint_vectors: dict[str, dict[int, np.ndarray]] = {}
    for split, capture in captures.items():
        if split != "final_test":
            continue
        with np.load(context.root / capture["endpoint_artifact"], allow_pickle=False) as data:
            names = data["base_trial_id"].astype(str)
            clean_values = data["next_j__none"].astype(np.float32)
            for index, base_id in enumerate(names):
                endpoint_vectors[base_id] = {
                    mask: data[f"next_j__{condition}"][index].astype(np.float32)
                    - clean_values[index]
                    for condition, mask in factorial_conditions().items()
                }
    interactions = []
    for base_id, group in factorial.groupby("base_trial_id", sort=True):
        values = {
            int(row.atom_mask): np.asarray([row.next_j_l2, row.output_js_divergence])
            for row in group.itertuples()
        }
        if set(values) != set(range(16)):
            continue
        decomposed = mobius_interactions(values)
        vector_decomposed = mobius_interactions(endpoint_vectors[str(base_id)])
        full_vector_norm = float(
            np.linalg.norm(endpoint_vectors[str(base_id)][15])
        )
        source = group.iloc[0]
        for mask, effect in decomposed.items():
            interactions.append(
                {
                    "base_trial_id": base_id,
                    "prompt_id": source["prompt_id"],
                    "family": source["family"],
                    "order": int(mask.bit_count()),
                    "mask": mask,
                    "atoms": [
                        ATOM_ORDER[index] for index in range(4) if mask & (1 << index)
                    ],
                    "next_j_interaction": float(effect[0]),
                    "vector_next_j_interaction_l2": float(
                        np.linalg.norm(vector_decomposed[mask])
                    ),
                    "vector_interaction_ratio_to_full": float(
                        np.linalg.norm(vector_decomposed[mask])
                        / max(full_vector_norm, 1e-20)
                    ),
                    "output_js_interaction": float(effect[1]),
                }
            )
    interaction_frame = pd.DataFrame(interactions)
    interaction_effects = {
        str(mask): {
            "atoms": group.iloc[0]["atoms"],
            "order": int(group.iloc[0]["order"]),
            "vector_interaction_ratio_to_full": _ci(
                group,
                "vector_interaction_ratio_to_full",
                seed,
                resamples,
            ),
            "output_js_interaction": _ci(
                group,
                "output_js_interaction",
                seed,
                resamples,
            ),
        }
        for mask, group in interaction_frame.groupby("mask", sort=True)
    }
    trial_path = context.processed_dir / "structured_component_screen_v8.parquet"
    serial = frame.copy()
    for column in ("metrics", "metadata"):
        serial[column] = serial[column].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    serial.to_parquet(trial_path, index=False, compression="zstd")
    interaction_path = (
        context.processed_dir / "structured_component_interactions_v8.parquet"
    )
    interaction_frame.to_parquet(interaction_path, index=False, compression="zstd")
    raw_gate = context.config["persistent_state_v8"]["raw_screen"]
    selected_raw_state = None
    for label in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        if (
            effects["pooled"]["direction_cosine_to_full"][label]["estimate"]
            >= float(raw_gate["minimum_causal_direction_cosine"])
            and effects["pooled"]["magnitude_ratio_to_full"][label]["estimate"]
            >= float(raw_gate["minimum_full_gap_fraction"])
            and effects["pooled"]["output_sign_agreement_to_full"][label]["estimate"]
            >= 1.0 - float(raw_gate["maximum_output_sign_loss"])
        ):
            selected_raw_state = label
            break
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "captures": captures,
        "valid_pair_count": int(frame.groupby(["split", "base_trial_id"]).ngroups),
        "final_test_pair_count": int(final["base_trial_id"].nunique()),
        "family_counts": final.groupby("family")["base_trial_id"].nunique().to_dict(),
        "raw_labels": named_raw_conditions(),
        "raw_screen_gate": raw_gate,
        "selected_raw_state": selected_raw_state,
        "factorial_atoms": list(ATOM_ORDER),
        "standalone_masks": {name: 1 << index for index, name in enumerate(ATOM_ORDER)},
        "leave_one_out_masks": leave_one_out_masks(),
        "pairwise_masks": pairwise_masks(),
        "effects": effects,
        "interaction_effects": interaction_effects,
        "trial_records": str(trial_path.relative_to(context.root)),
        "interaction_records": str(interaction_path.relative_to(context.root)),
    }
    write_json_atomic(
        context.processed_dir / "structured_component_screen_v8.json", summary
    )
    context.finish("COMPLETED_RAW_SCREEN", summary=summary)


def main() -> None:
    args = _parser().parse_args()
    context = initialize_context("persistent-state-v8", args)
    try:
        if args.stage == "guard":
            context.finish("COMPLETED_GUARD", guard=build_guard(context.root))
            return
        if args.stage == "freeze":
            context.finish(
                "COMPLETED_FREEZE", freeze=build_freeze(context.root, context.config)
            )
            return
        if args.stage in {"teacher", "teacher-reparse"}:
            if args.dry_run:
                context.finish("DRY_RUN")
                return
            from jclosure.model import load_model_bundle

            bundle = load_model_bundle(context.config)
            if args.stage == "teacher":
                _teacher(context, bundle, args.limit)
            else:
                _teacher_reparse(context, bundle)
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
        if args.stage == "bank":
            _bank(context, bundle, freeze)
        else:
            if args.split is None:
                raise ValueError("capture requires --split")
            _capture(context, bundle, freeze, args.split, args.limit)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
