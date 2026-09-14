"""Independent family-wise H2 replication and later-J mediation for v5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.datasets_v5 import load_replication_tasks
from jclosure.experiments.causal_single_v4 import (
    _initial_state,
    _load_encoder,
    _quality,
    _rollout,
    _thresholds,
    _token_ids,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.experiments.mediation_v4 import (
    _clean_activations,
    _persistent_rollout,
)
from jclosure.experiments.traces_v4 import ACTION_TO_ID
from jclosure.peripheral_v5 import build_u
from jclosure.protocol_v5 import verify_peripheral_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.records_v5 import H2ReplicationRecord
from jclosure.single_arm_v4 import (
    aligned_rollout_metrics,
    construct_from_shared_basis,
    matched_controls,
    shared_low_singular_basis,
)
from jclosure.statistics import clustered_bootstrap_ci

PROTOCOL = "h2_family_replication_v5"


def _load_bank(root: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    summary = json.loads(
        (root / "results/v4/processed/single_arm_bank_v4.json").read_text(
            encoding="utf-8"
        )
    )
    state_path = root / summary["states"]
    if sha256_file(state_path) != summary["states_sha256"]:
        raise RuntimeError("frozen v4 donor bank hash mismatch")
    rows = [
        json.loads(line)
        for line in (root / summary["records"]).read_text(encoding="utf-8").splitlines()
    ]
    with np.load(state_path, allow_pickle=False) as payload:
        states = payload["states"].astype(np.float32)
    return rows, states


def _positive_direction(
    bundle: Any,
    dense_map: Any,
    vocabulary: Any,
    layer: int,
    answer_ids: list[int],
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    token_to_index = {
        int(value): index for index, value in enumerate(vocabulary.token_ids)
    }
    raw_map = dense_map.raw_map(
        layer, device=next(bundle.hf_model.parameters()).device, dtype=torch.float32
    )
    answer_index = next(
        (token_to_index[value] for value in answer_ids if value in token_to_index), None
    )
    if answer_index is not None:
        return raw_map[answer_index]
    token_id = int(answer_ids[0])
    if token_id not in cache:
        unembedding = bundle.unembedding_weight[token_id].detach()
        jacobian = bundle.lens.jacobians[layer].detach()
        cache[token_id] = (unembedding.float().cpu() @ jacobian.float().cpu()).to(
            next(bundle.hf_model.parameters()).device
        )
    return cache[token_id]


def _run_single(
    context: Any,
    bundle: Any,
    freeze: dict[str, Any],
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    tasks = load_replication_tasks(context.root / freeze["replication_data"])
    if limit is not None:
        tasks = tasks[: int(limit)]
    section = context.config["h2_replication_v5"]
    layer = int(section["layer"])
    layers = [layer, *[int(value) for value in section["future_workspace_layers"]]]
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    bank_rows, bank_states = _load_bank(context.root)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    shared = shared_low_singular_basis(
        dense_map,
        layer,
        relative_tolerance=1e-4,
        device=next(bundle.hf_model.parameters()).device,
    )
    thresholds = _thresholds(context.config)
    targets = {
        str(key): int(value) for key, value in section["target_valid_by_family"].items()
    }
    completed = {key: 0 for key in targets}
    cache: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    arrays: dict[str, list[Any]] = {
        "clean_hidden": [],
        "intervened_hidden": [],
        "current_j_clean": [],
        "current_j_intervened": [],
        "next_j_clean": [],
        "next_j_intervened": [],
        "u": [],
        "next_action": [],
        "teacher_next_action_clean": [],
        "teacher_next_action_intervened": [],
        "teacher_output_delta": [],
        "family": [],
        "prompt_id": [],
        "base_trial_id": [],
    }
    for attempt, task in enumerate(tasks):
        if completed[task.family] >= targets[task.family]:
            continue
        clean = _initial_state(bundle, task.prompt, layer)
        matches = [
            (row, bank_states[index])
            for index, row in enumerate(bank_rows)
            if row["family"] == task.family and int(row["horizon"]) == task.horizon
        ]
        if not matches:
            records.append(
                {
                    "schema_version": 7,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "family": task.family,
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
        candidate, construction, valid = construct_from_shared_basis(
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
            records.append(
                {
                    "schema_version": 7,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "horizon": task.horizon,
                    "exclusion_reason": "initial_state_invalid",
                    "quality": construction,
                }
            )
            continue
        answer_ids = _token_ids(bundle.tokenizer, task.semantic_actions[0])
        if not answer_ids:
            records.append(
                {
                    "schema_version": 7,
                    "protocol_version": PROTOCOL,
                    "record_type": "attrition",
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "exclusion_reason": "answer_not_single_token",
                }
            )
            continue
        direction = _positive_direction(
            bundle, dense_map, vocabulary, layer, answer_ids, cache
        )
        controls = matched_controls(
            clean, donor, candidate, direction, seed=int(context.seed) + attempt
        )
        clean_rollout = _rollout(
            bundle,
            task,
            layer=layer,
            workspace_layers=layers,
            dense_map=dense_map,
            candidate=None,
        )
        if tuple(clean_rollout["actions"]) != task.semantic_actions:
            records.append(
                {
                    "schema_version": 7,
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
        candidate_rollout: dict[str, Any] | None = None
        for condition in section["controls"]:
            replacement = controls[str(condition)]
            rollout = (
                clean_rollout
                if condition == "clean"
                else _rollout(
                    bundle,
                    task,
                    layer=layer,
                    workspace_layers=layers,
                    dense_map=dense_map,
                    candidate=replacement,
                )
            )
            if condition == "j_preserving":
                candidate_rollout = rollout
            quality: dict[str, Any] = _quality(
                clean,
                replacement,
                layer=layer,
                dense_map=dense_map,
                natural_scale=natural_scale,
            )
            if condition == "j_preserving":
                quality["construction"] = construction
            records.append(
                {
                    **H2ReplicationRecord(
                        run_id=context.run_id,
                        base_trial_id=base_id,
                        prompt_id=task.example_id,
                        family=task.family,
                        horizon=task.horizon,
                        condition=str(condition),
                        valid=True,
                        metrics=aligned_rollout_metrics(clean_rollout, rollout),
                        quality=quality,
                    ).to_dict(),
                    "record_type": "causal_trial",
                    "prompt": task.prompt,
                    "donor_id": donor_row["example_id"],
                }
            )
        if candidate_rollout is None:
            raise RuntimeError("J-preserving condition was not executed")
        if (
            len(clean_rollout["j_states"]) >= 2
            and len(candidate_rollout["j_states"]) >= 2
        ):
            current_action = np.asarray([ACTION_TO_ID[task.semantic_actions[0]]])
            family = np.asarray([task.family], dtype="U32")
            u = build_u(
                current_action, family, np.asarray([0]), np.asarray([task.horizon])
            )[0]
            arrays["clean_hidden"].append(
                clean.detach().cpu().numpy().astype(np.float16)
            )
            arrays["intervened_hidden"].append(
                candidate.detach().cpu().numpy().astype(np.float16)
            )
            arrays["current_j_clean"].append(clean_rollout["j_states"][0])
            arrays["current_j_intervened"].append(candidate_rollout["j_states"][0])
            arrays["next_j_clean"].append(clean_rollout["j_states"][1])
            arrays["next_j_intervened"].append(candidate_rollout["j_states"][1])
            arrays["u"].append(u)
            next_surface = task.semantic_actions[1]
            next_id = ACTION_TO_ID[next_surface]
            arrays["next_action"].append(next_id)
            arrays["teacher_next_action_clean"].append(
                ACTION_TO_ID[clean_rollout["actions"][1]]
            )
            arrays["teacher_next_action_intervened"].append(
                ACTION_TO_ID[candidate_rollout["actions"][1]]
            )
            arrays["teacher_output_delta"].append(
                float(
                    candidate_rollout["target_log_odds"][1]
                    - clean_rollout["target_log_odds"][1]
                )
            )
            arrays["family"].append(task.family)
            arrays["prompt_id"].append(task.example_id)
            arrays["base_trial_id"].append(base_id)
        completed[task.family] += 1
        if all(completed[key] >= targets[key] for key in targets):
            break
    output_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    return records, output_arrays


def _metric_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    trials = frame[(frame["record_type"] == "causal_trial") & frame["valid"]].copy()
    for metric in (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_change",
        "answer_flip",
        "task_accuracy_change",
    ):
        trials[metric] = trials["metrics"].map(lambda value, key=metric: value[key])
    return trials


def _effects(
    frame: pd.DataFrame, context: Any, *, group_column: str = "family"
) -> dict[str, Any]:
    section = context.config["h2_replication_v5"]
    output: dict[str, Any] = {}
    groups = [("pooled", frame), *list(frame.groupby(group_column, sort=True))]
    for group_name, group in groups:
        output[str(group_name)] = {}
        for metric in (
            "output_js_divergence",
            "future_j_trajectory_divergence",
            "target_log_odds_change",
            "answer_flip",
            "task_accuracy_change",
        ):
            if metric not in group:
                continue
            output[str(group_name)][metric] = {}
            for condition, values in group.groupby("condition", sort=True):
                if values[metric].isna().any():
                    continue
                ci = clustered_bootstrap_ci(
                    values,
                    cluster_col="prompt_id",
                    value_col=metric,
                    n_resamples=int(section["bootstrap_resamples"]),
                    seed=int(context.config["reproducibility"]["bootstrap_seed"]),
                )
                output[str(group_name)][metric][str(condition)] = asdict(ci)
    return output


def _merge_single(context: Any) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("h2-replication-v5-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_SINGLE":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed v5 H2 single-arm run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    trials = _metric_frame(rows)
    complete = [
        value
        for value, group in trials.groupby("base_trial_id")
        if set(group["condition"])
        == {
            "clean",
            "identity",
            "matched_random",
            "j_positive",
            "full_perturbation",
            "j_preserving",
        }
    ]
    common = trials[trials["base_trial_id"].isin(complete)].copy()
    effects = _effects(common, context)
    floor = float(context.config["h2_replication_v5"]["null_js_floor"])
    passed_families = []
    for family, metrics in effects.items():
        value = metrics["output_js_divergence"].get("j_preserving")
        if family != "pooled" and value and value["lower"] > floor:
            passed_families.append(family)
    parquet = context.processed_dir / "h2_replication_v5.parquet"
    serialized = trials.copy()
    for column in ("metrics", "quality"):
        serialized[column] = serialized[column].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    serialized.to_parquet(parquet, index=False, compression="zstd")
    summary = {
        "schema_version": 7,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "attempted_items": int(
            sum(value.get("record_type") == "attrition" for value in rows)
            + len(complete)
        ),
        "complete_paired_base_trials": len(complete),
        "valid_by_family": common[common["condition"] == "j_preserving"]
        .groupby("family")
        .size()
        .to_dict(),
        "attrition": pd.Series(
            [
                value.get("exclusion_reason")
                for value in rows
                if value.get("record_type") == "attrition"
            ]
        )
        .value_counts()
        .to_dict(),
        "effects": effects,
        "families_above_frozen_noise": passed_families,
        "mediation_authorized": bool(passed_families),
        "records": str(parquet.relative_to(context.root)),
        "causal_state_artifact": source["causal_state_artifact"],
        "causal_state_artifact_sha256": source["causal_state_artifact_sha256"],
    }
    write_json_atomic(context.processed_dir / "h2_replication_v5.json", summary)
    return summary


def _run_mediation(
    context: Any, bundle: Any, freeze: dict[str, Any], limit: int | None
) -> list[dict[str, Any]]:
    summary = json.loads(
        (context.processed_dir / "h2_replication_v5.json").read_text(encoding="utf-8")
    )
    if not summary["mediation_authorized"]:
        raise RuntimeError("family-wise H2 effect did not authorize mediation")
    artifact = context.root / summary["causal_state_artifact"]
    with np.load(artifact, allow_pickle=False) as payload:
        prompts = payload["prompt_id"].astype("U64")
        candidates = payload["intervened_hidden"].astype(np.float32)
        base_ids = payload["base_trial_id"].astype("U64")
    tasks = {
        value.example_id: value
        for value in load_replication_tasks(context.root / freeze["replication_data"])
    }
    section = context.config["h2_replication_v5"]
    layer = int(section["layer"])
    layers = [layer, *[int(value) for value in section["future_workspace_layers"]]]
    _, _, dense_map = _load_encoder(context, bundle)
    thresholds = _thresholds(context.config)
    rows = []
    count = len(prompts) if limit is None else min(len(prompts), int(limit))
    for index in range(count):
        task = tasks[str(prompts[index])]
        candidate = torch.as_tensor(
            candidates[index],
            device=next(bundle.hf_model.parameters()).device,
            dtype=torch.float32,
        )
        clean_by_layer = _clean_activations(bundle, task.prompt, layers)
        clean_rollout = _rollout(
            bundle,
            task,
            layer=layer,
            workspace_layers=layers,
            dense_map=dense_map,
            candidate=None,
        )
        for mode in ("persistent_final", "persistent_all"):
            rollout, capture = _persistent_rollout(
                bundle,
                task,
                layer=layer,
                workspace_layers=layers,
                dense_map=dense_map,
                candidate=candidate,
                mode=mode,
                clean_by_layer=clean_by_layer,
                thresholds=thresholds,
            )
            valid = bool(
                len(capture) == len(layers) - 1
                and all(value["passed"] for value in capture.values())
            )
            rows.append(
                {
                    "schema_version": 7,
                    "protocol_version": PROTOCOL,
                    "record_type": "mediation_trial",
                    "run_id": context.run_id,
                    "base_trial_id": str(base_ids[index]),
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "horizon": task.horizon,
                    "mode": mode,
                    "valid": valid,
                    "exclusion_reason": None
                    if valid
                    else "runtime_restoration_invalid",
                    "metrics": aligned_rollout_metrics(clean_rollout, rollout),
                    "restoration_events": [capture[key] for key in sorted(capture)],
                }
            )
    return rows


def _merge_mediation(context: Any) -> dict[str, Any]:
    single = json.loads(
        (context.processed_dir / "h2_replication_v5.json").read_text(encoding="utf-8")
    )
    manifests = []
    for path in context.raw_dir.glob("h2-mediation-v5-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_MEDIATION":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed v5 H2 mediation run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    frame = pd.DataFrame(rows)
    frame = frame[(frame["record_type"] == "mediation_trial") & frame["valid"]].copy()
    for metric in ("output_js_divergence", "future_j_trajectory_divergence"):
        frame[metric] = frame["metrics"].map(lambda value, key=metric: value[key])
    effects = _effects(frame.assign(condition=frame["mode"]), context)
    fractions: dict[str, dict[str, float]] = {}
    for family in single["effects"]:
        single_effect = single["effects"][family]["output_js_divergence"].get(
            "j_preserving"
        )
        if not single_effect or single_effect["estimate"] <= 0:
            continue
        fractions[family] = {}
        for mode in ("persistent_final", "persistent_all"):
            persistent = (
                effects.get(family, {}).get("output_js_divergence", {}).get(mode)
            )
            if persistent:
                fractions[family][mode] = (
                    1 - persistent["estimate"] / single_effect["estimate"]
                )
    summary = {
        "schema_version": 7,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "valid_records": int(len(frame)),
        "effects": effects,
        "mediation_fraction_point_estimates": fractions,
    }
    write_json_atomic(context.processed_dir / "h2_mediation_v5.json", summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Independent family-wise H2 replication", "configs/peripheral_v5.yaml"
    )
    parser.add_argument(
        "--stage",
        choices=("single", "merge", "mediation", "merge_mediation"),
        required=True,
    )
    args = parser.parse_args()
    kind = "h2-mediation-v5" if "mediation" in args.stage else "h2-replication-v5"
    context = initialize_context(kind, args)
    try:
        freeze = verify_peripheral_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "single":
            from jclosure.model import load_model_bundle

            bundle = load_model_bundle(context.config)
            rows, arrays = _run_single(context, bundle, freeze, args.limit)
            records = context.raw_dir / context.run_id / "h2_single_trials.jsonl"
            append_jsonl(records, rows)
            artifact_dir = context.root / "artifacts/causal/v5" / context.run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact = artifact_dir / "causal_state_pairs.npz"
            np.savez_compressed(artifact, **arrays)
            context.finish(
                "COMPLETED_SINGLE",
                records=str(records.relative_to(context.root)),
                record_count=len(rows),
                causal_state_artifact=str(artifact.relative_to(context.root)),
                causal_state_artifact_sha256=sha256_file(artifact),
            )
        elif args.stage == "merge":
            summary = _merge_single(context)
            context.finish("COMPLETED", summary=summary)
        elif args.stage == "mediation":
            from jclosure.model import load_model_bundle

            bundle = load_model_bundle(context.config)
            rows = _run_mediation(context, bundle, freeze, args.limit)
            records = context.raw_dir / context.run_id / "h2_mediation_trials.jsonl"
            append_jsonl(records, rows)
            context.finish(
                "COMPLETED_MEDIATION",
                records=str(records.relative_to(context.root)),
                record_count=len(rows),
            )
        else:
            summary = _merge_mediation(context)
            context.finish("COMPLETED", summary=summary)
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
