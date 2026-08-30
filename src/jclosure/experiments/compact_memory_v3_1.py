"""Token-time trace, representation screen, and compact-memory experiment."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from jclosure.compact_memory_v3_1 import (
    LinearRepresentation,
    autonomous_rollout,
    build_parameter_matched_controller,
    fit_representation,
    parameter_count,
    row_cosine,
    scheduled_feedback_probability,
)
from jclosure.datasets import upstream_multihop, upstream_order_ops
from jclosure.experiments.calibrate_v3_1 import _load_encoder, _read_jsonl
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.validate_lens import _single_token_ids, _swap_transform
from jclosure.geometry import DenseJMap
from jclosure.model import load_model_bundle
from jclosure.phase0 import single_token_candidates, synonym_surfaces
from jclosure.protocol_v3_1 import MEMORY_PROTOCOL, verify_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder, ResidualEditor
from jclosure.runtime_v3_1 import encode_direct_prompt

ACTION_SURFACES = tuple(str(value) for value in range(10)) + tuple("ABCDEF")
ACTION_TO_ID = {value: index for index, value in enumerate(ACTION_SURFACES)}


def _memory_items(root: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(
        (root / "artifacts/v3_1_data_manifest.json").read_text(encoding="utf-8")
    )
    path = root / manifest["memory_splits"][split]["path"]
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _macrostate(
    activations: dict[int, torch.Tensor], dense_map: DenseJMap, layers: list[int]
) -> tuple[torch.Tensor, float]:
    states = torch.stack(
        [
            dense_map.dense_state(activations[layer][0, -1].detach().float(), layer)
            for layer in layers
        ]
    )
    pooled = F.normalize(states.mean(0), dim=0)
    dispersion = float(
        (1 - F.cosine_similarity(states, pooled[None], dim=1)).mean().item()
    )
    return pooled, dispersion


@torch.no_grad()
def _extract_trace(
    bundle,
    encoder,
    dense_map,
    task: dict[str, Any],
    layers: list[int],
    maximum_extra: int = 8,
) -> dict[str, Any]:
    input_ids = encode_direct_prompt(bundle, str(task["prompt"]))
    semantic: list[str] = []
    states: list[np.ndarray] = []
    dispersions: list[float] = []
    remainders: list[np.ndarray] = []
    sparse_states: list[np.ndarray] = []
    generated: list[int] = []
    error = None
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    maximum = 2 * int(task["length"]) + maximum_extra
    allowed = (
        set("0123456789")
        if task["family"] == "iterated_modular_arithmetic"
        else set("ABCDEF")
    )
    for _ in range(maximum):
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            logits = bundle.forward_logits(input_ids)[0, -1]
        token = int(torch.argmax(logits))
        if eos is not None and token == int(eos):
            error = "eos_before_complete_trajectory"
            break
        surface = bundle.tokenizer.decode([token], skip_special_tokens=False)
        stripped = surface.strip()
        if stripped:
            if stripped not in allowed or len(stripped) != 1:
                error = "nonsemantic_generated_token"
                break
            pooled, dispersion = _macrostate(recorder.activations, dense_map, layers)
            final_h = recorder.activations[layers[-1]][0, -1].detach().float()
            decomposition = encoder.decompose(final_h, layers[-1])
            sparse_state = np.zeros(len(encoder.vocabulary.token_ids), dtype=np.float16)
            sparse_state[decomposition.atom_indices.detach().cpu().numpy()] = (
                decomposition.coefficients.detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.float16)
            )
            states.append(pooled.detach().cpu().numpy().astype(np.float16))
            dispersions.append(dispersion)
            remainders.append(
                decomposition.remainder.detach().cpu().numpy().astype(np.float16)
            )
            sparse_states.append(sparse_state)
            semantic.append(stripped)
        generated.append(token)
        input_ids = torch.cat(
            (input_ids, torch.tensor([[token]], device=input_ids.device)), dim=1
        )
        if len(semantic) == int(task["length"]):
            break
    parseable = len(semantic) == int(task["length"]) and error is None
    expected = [str(value) for value in task["semantic_actions"]]
    teacher_correct = parseable and semantic == expected
    if not parseable and error is None:
        error = "trajectory_length_mismatch"
    return {
        "states": np.stack(states)
        if states
        else np.empty((0, len(encoder.vocabulary.token_ids)), dtype=np.float16),
        "remainders": np.stack(remainders)
        if remainders
        else np.empty((0, bundle.lens_model.d_model), dtype=np.float16),
        "sparse_states": np.stack(sparse_states)
        if sparse_states
        else np.empty((0, len(encoder.vocabulary.token_ids)), dtype=np.float16),
        "dispersion": np.asarray(dispersions, dtype=np.float32),
        "actions": np.asarray(
            [ACTION_TO_ID[value] for value in semantic], dtype=np.int16
        ),
        "generated_token_ids": generated,
        "generated_semantic": semantic,
        "parseable": parseable,
        "teacher_correct": teacher_correct,
        "error": error,
    }


def _trace_rows(
    context,
    bundle,
    *,
    split: str,
    shard_index: int,
    shard_count: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    _, encoder, dense_map = _load_encoder(context, bundle)
    layers = [
        int(value) for value in context.config["compact_memory"]["workspace_layers"]
    ]
    tasks = _memory_items(context.root, split)
    if limit is not None:
        tasks = tasks[:limit]
    tensor_root = context.root / "artifacts/traces/v3_1" / context.run_id
    tensor_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        if int(task["program_hash"], 16) % shard_count != shard_index:
            continue
        trace = _extract_trace(bundle, encoder, dense_map, task, layers)
        tensor_path = tensor_root / f"{task['example_id']}.npz"
        np.savez_compressed(
            tensor_path,
            states=trace.pop("states"),
            remainders=trace.pop("remainders"),
            sparse_states=trace.pop("sparse_states"),
            dispersion=trace.pop("dispersion"),
            actions=trace.pop("actions"),
        )
        rows.append(
            {
                "schema_version": 4,
                "protocol_version": MEMORY_PROTOCOL,
                "run_id": context.run_id,
                "record_type": "teacher_trace",
                "split": split,
                "example_id": task["example_id"],
                "family": task["family"],
                "template_id": task["template_id"],
                "program_hash": task["program_hash"],
                "length": int(task["length"]),
                "expected_actions": task["semantic_actions"],
                "final_answer": task["final_answer"],
                "trace_path": str(tensor_path.relative_to(context.root)),
                "trace_sha256": sha256_file(tensor_path),
                **trace,
            }
        )
    return rows


def _merge_traces(context, *, shard_group_id: str, shard_count: int) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("compact-memory-v3-1-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("status") == "COMPLETED_TRACE_SHARD"
            and value.get("shard_group_id") == shard_group_id
        ):
            manifests.append(value)
    expected = {
        (split, shard)
        for split in ("train", "validation", "test")
        for shard in range(shard_count)
    }
    by_key = {(value["split"], int(value["shard_index"])): value for value in manifests}
    if set(by_key) != expected:
        raise RuntimeError(
            f"compact-memory trace merge is missing shards: {sorted(expected - set(by_key))}"
        )
    rows = []
    for key in sorted(by_key):
        rows.extend(_read_jsonl(context.root / by_key[key]["records"]))
    frame = pd.DataFrame(rows)
    records_path = context.processed_dir / "compact_memory_trace_records.parquet"
    frame.to_parquet(records_path, index=False)
    counts = []
    for (split, family), group in frame.groupby(["split", "family"]):
        counts.append(
            {
                "split": split,
                "family": family,
                "attempted": len(group),
                "parseable": int(group["parseable"].sum()),
                "teacher_correct": int(group["teacher_correct"].sum()),
                "errors": group["error"].fillna("none").value_counts().to_dict(),
            }
        )
    per_family = frame.groupby("family")["parseable"].sum().to_dict()
    target = int(context.config["compact_memory"]["target_parseable_per_family"])
    authorized = all(
        int(per_family.get(family, 0)) >= target
        for family in context.config["compact_memory"]["task_families"]
    )
    summary = {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "run_id": context.run_id,
        "records": str(records_path.relative_to(context.root)),
        "counts": counts,
        "parseable_by_family": {key: int(value) for key, value in per_family.items()},
        "target_parseable_per_family": target,
        "representation_screen_authorized": authorized,
        "source_shards": [by_key[key]["run_id"] for key in sorted(by_key)],
    }
    output = context.processed_dir / "compact_memory_trace_summary.json"
    write_json_atomic(output, summary)
    return summary


def _load_trace_frame(context) -> pd.DataFrame:
    summary = json.loads(
        (context.processed_dir / "compact_memory_trace_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not summary.get("representation_screen_authorized"):
        raise RuntimeError("teacher trace parseability gate failed")
    return pd.read_parquet(context.root / summary["records"])


@torch.no_grad()
def _phase0_regression_profiles(
    context, bundle, dense_map, vocabulary
) -> list[dict[str, Any]]:
    examples = [
        *upstream_multihop(context.root / context.config["data"]["upstream_root"]),
        *upstream_order_ops(context.root / context.config["data"]["upstream_root"]),
    ]
    layers = [
        int(value) for value in context.config["compact_memory"]["workspace_layers"]
    ]
    token_to_index = {
        int(token): index for index, token in enumerate(vocabulary.token_ids)
    }
    output = []
    for example in examples:
        input_ids = bundle.lens_model.encode(
            example.prompt, max_length=int(context.config["model"]["max_seq_len"])
        )
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            bundle.forward_logits(input_ids)
        profile, _ = _macrostate(recorder.activations, dense_map, layers)
        concepts = []
        for concept in example.intermediates:
            candidates = single_token_candidates(
                bundle.tokenizer,
                synonym_surfaces(concept, family=example.family),
            )
            indices = sorted(
                {
                    token_to_index[candidate.token_id]
                    for candidate in candidates
                    if candidate.token_id in token_to_index
                }
            )
            if indices:
                concepts.append(indices)
        if concepts:
            output.append(
                {
                    "example_id": example.example_id,
                    "family": example.family,
                    "profile": profile.detach().cpu().numpy(),
                    "concept_indices": concepts,
                }
            )
    return output


def _profile_pass10(profile: np.ndarray, concepts: list[list[int]]) -> float:
    values = np.asarray(profile)
    indicators = []
    for indices in concepts:
        best = max(float(values[index]) for index in indices)
        rank = int(np.count_nonzero(values > best)) + 1
        indicators.append(rank <= 10)
    return float(np.mean(indicators))


@torch.no_grad()
def _causal_regression_pairs(
    context, bundle, encoder, dense_map
) -> list[tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(
        (
            context.root
            / "data/upstream/anthropic/experiments/flexible-generalization.json"
        ).read_text(encoding="utf-8")
    )
    layers = [
        int(value) for value in context.config["compact_memory"]["workspace_layers"]
    ]
    token_to_index = {
        int(token): index for index, token in enumerate(encoder.vocabulary.token_ids)
    }
    pairs = []
    for category in payload["categories"]:
        arguments = list(category["args"])
        for function in category["funcs"]:
            for source_index, source in enumerate(arguments):
                target = arguments[(source_index + 1) % len(arguments)]
                prompt = str(function["template"]).format(arg=source)
                source_ids = _single_token_ids(bundle.tokenizer, source)
                target_ids = _single_token_ids(bundle.tokenizer, target)
                answer_ids = _single_token_ids(
                    bundle.tokenizer, function["answers"][source]
                )
                source_id = next(
                    (value for value in source_ids if value in token_to_index), None
                )
                target_id = next(
                    (value for value in target_ids if value in token_to_index), None
                )
                if source_id is None or target_id is None or not answer_ids:
                    continue
                input_ids = bundle.lens_model.encode(prompt, max_length=512)
                with ActivationRecorder(bundle.layers, at=layers) as clean_recorder:
                    clean_logits = bundle.forward_logits(input_ids)[0, -1]
                if int(torch.argmax(clean_logits)) not in answer_ids:
                    continue
                transforms = {}
                for layer in layers:
                    dictionary = encoder.dictionary(layer)
                    transforms[layer] = partial(
                        _swap_transform,
                        source_direction=dictionary[token_to_index[source_id]],
                        target_direction=dictionary[token_to_index[target_id]],
                        strength=1.0,
                        positions=(-1,),
                    )
                with (
                    ResidualEditor(bundle.layers, transforms),
                    ActivationRecorder(bundle.layers, at=layers) as swap_recorder,
                ):
                    bundle.forward_logits(input_ids)
                clean_profile, _ = _macrostate(
                    clean_recorder.activations, dense_map, layers
                )
                swapped_profile, _ = _macrostate(
                    swap_recorder.activations, dense_map, layers
                )
                pairs.append(
                    (
                        clean_profile.detach().cpu().numpy(),
                        swapped_profile.detach().cpu().numpy(),
                    )
                )
                if len(pairs) >= int(
                    context.config["validation"]["positive_control_trials"]
                ):
                    return pairs
    return pairs


def _pairs(
    context, frame: pd.DataFrame
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {
        f"{split}_{kind}": []
        for split in ("train", "validation", "test")
        for kind in ("x", "y", "r", "s", "a")
    }
    trajectories = []
    for row in frame[frame["parseable"]].itertuples():
        payload = np.load(context.root / row.trace_path)
        state = payload["states"].astype(np.float32)
        remainder = payload["remainders"].astype(np.float32)
        sparse_state = payload["sparse_states"].astype(np.float32)
        actions = payload["actions"].astype(np.int64)
        if len(state) < 2:
            continue
        arrays[f"{row.split}_x"].append(state[:-1])
        arrays[f"{row.split}_y"].append(state[1:])
        arrays[f"{row.split}_r"].append(remainder[:-1])
        arrays[f"{row.split}_s"].append(sparse_state[:-1])
        arrays[f"{row.split}_a"].append(actions[1:])
        trajectories.append(
            {
                "example_id": row.example_id,
                "family": row.family,
                "split": row.split,
                "states": state,
                "remainders": remainder,
                "sparse_states": sparse_state,
                "actions": actions,
                "teacher_correct": bool(row.teacher_correct),
            }
        )
    concatenated = {
        key: np.concatenate(value) if value else np.empty((0,))
        for key, value in arrays.items()
    }
    return concatenated, trajectories


def _screen(context, bundle) -> dict[str, Any]:
    frame = _load_trace_frame(context)
    arrays, _ = _pairs(context, frame)
    vocabulary, encoder, dense_map = _load_encoder(context, bundle)
    phase0_profiles = _phase0_regression_profiles(
        context, bundle, dense_map, vocabulary
    )
    causal_pairs = _causal_regression_pairs(context, bundle, encoder, dense_map)
    original_phase0 = float(
        np.mean(
            [
                _profile_pass10(value["profile"], value["concept_indices"])
                for value in phase0_profiles
            ]
        )
    )
    rows = []
    selected = []
    representation_root = context.root / "artifacts/traces/v3_1/representations"
    representation_root.mkdir(parents=True, exist_ok=True)
    for dimension in context.config["compact_memory"]["state_dimensions"]:
        candidates = []
        for family in context.config["compact_memory"]["representation_families"]:
            representation = fit_representation(
                str(family),
                int(dimension),
                arrays["train_x"],
                arrays["train_y"],
                feature_importance=(
                    np.mean(np.abs(arrays["train_s"]), axis=0)
                    if family == "sparse_j_centered"
                    else None
                ),
            )
            train_z = representation.encode(arrays["train_x"])
            validation_z = representation.encode(arrays["validation_x"])
            predictor = Ridge(alpha=1.0).fit(train_z, arrays["train_y"])
            prediction = predictor.predict(validation_z)
            cosine = float(np.median(row_cosine(prediction, arrays["validation_y"])))
            reconstruction = representation.decode(validation_z)
            reconstruction_cosine = float(
                np.median(row_cosine(reconstruction, arrays["validation_x"]))
            )
            reconstructed_phase0 = float(
                np.mean(
                    [
                        _profile_pass10(
                            representation.decode(
                                representation.encode(value["profile"][None])
                            )[0],
                            value["concept_indices"],
                        )
                        for value in phase0_profiles
                    ]
                )
            )
            phase0_retention = reconstructed_phase0 / max(original_phase0, 1e-12)
            delta_cosines = []
            magnitude_retentions = []
            for clean, swapped in causal_pairs:
                original_delta = swapped - clean
                reconstructed_clean = representation.decode(
                    representation.encode(clean[None])
                )[0]
                reconstructed_swapped = representation.decode(
                    representation.encode(swapped[None])
                )[0]
                reconstructed_delta = reconstructed_swapped - reconstructed_clean
                cosine_value = float(
                    row_cosine(reconstructed_delta[None], original_delta[None])[0]
                )
                ratio = float(
                    np.linalg.norm(reconstructed_delta)
                    / max(np.linalg.norm(original_delta), 1e-12)
                )
                delta_cosines.append(cosine_value)
                magnitude_retentions.append(min(ratio, 1 / max(ratio, 1e-12)))
            direction_agreement = float(np.mean(np.asarray(delta_cosines) >= 0.8))
            magnitude_retention = float(np.median(magnitude_retentions))
            path = representation_root / f"{family}-{representation.dimension}.npz"
            np.savez_compressed(path, **representation.state_dict())
            row = {
                "schema_version": 4,
                "protocol_version": MEMORY_PROTOCOL,
                "representation_family": family,
                "dimension": representation.dimension,
                "validation_next_state_cosine": cosine,
                "validation_reconstruction_cosine": reconstruction_cosine,
                "phase0_pass10_original": original_phase0,
                "phase0_pass10_reconstructed": reconstructed_phase0,
                "phase0_pass10_retention": phase0_retention,
                "causal_trials": len(causal_pairs),
                "causal_delta_cosine_median": float(np.median(delta_cosines)),
                "causal_direction_retention": direction_agreement,
                "causal_magnitude_retention": magnitude_retention,
                "representation_path": str(path.relative_to(context.root)),
                "representation_sha256": sha256_file(path),
                "screen_status": "COMPLETE",
            }
            candidates.append(row)
            rows.append(row)
        candidates.sort(
            key=lambda value: (
                -value["validation_next_state_cosine"],
                -value["phase0_pass10_retention"],
                -value["causal_direction_retention"],
                str(value["representation_family"]),
            )
        )
        winner = dict(candidates[0])
        winner["selection_status"] = "SELECTED_ON_VALIDATION_ONLY"
        selected.append(winner)
    records_path = (
        context.processed_dir / "compact_memory_representation_screen.parquet"
    )
    pd.DataFrame(rows).to_parquet(records_path, index=False)
    summary = {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "records": str(records_path.relative_to(context.root)),
        "selected_by_dimension": selected,
        "phase0_regression_items": len(phase0_profiles),
        "causal_regression_trials": len(causal_pairs),
        "temporal_training_authorized": bool(
            phase0_profiles and causal_pairs and selected
        ),
        "authorization_reason": "complete"
        if phase0_profiles and causal_pairs
        else "regression_endpoint_missing",
    }
    path = context.processed_dir / "compact_memory_representation_screen.json"
    write_json_atomic(path, summary)
    return summary


def _load_representation(root: Path, path: str) -> LinearRepresentation:
    payload = np.load(root / path, allow_pickle=False)
    family = str(payload["family"].item())
    return LinearRepresentation(
        family,
        int(payload["dimension"]),
        payload["mean"],
        payload["encoder"],
        payload["decoder"],
    )


def _latent_trajectories(
    trajectories: list[dict[str, Any]], representation: LinearRepresentation
) -> list[dict[str, Any]]:
    output = []
    for trajectory in trajectories:
        output.append(
            {
                **trajectory,
                "latent": representation.encode(trajectory["states"]).astype(
                    np.float32
                ),
            }
        )
    return output


def _train_epoch(
    model,
    trajectories,
    optimizer,
    *,
    family: str,
    history: int,
    memory_dim: int,
    feedback_probability: float,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(trajectories), generator=generator).tolist()
    losses = []
    model.train()
    for index in order:
        trajectory = trajectories[index]
        states = torch.from_numpy(trajectory["latent"]).to(device)
        actions = torch.from_numpy(trajectory["actions"]).long().to(device)
        if len(states) < 2:
            continue
        optimizer.zero_grad(set_to_none=True)
        current = states[0:1]
        predicted_history = [current]
        memory = torch.zeros(1, memory_dim, device=device) if family == "gru" else None
        total = torch.zeros((), device=device)
        for step in range(len(states) - 1):
            if family == "markov":
                predicted_state, action_logits = model(current)
            elif family == "history":
                from jclosure.compact_memory_v3_1 import _history_input

                window, mask = _history_input(predicted_history, history)
                predicted_state, action_logits = model(window, mask)
            else:
                assert memory is not None
                predicted_state, action_logits, memory = model(current, memory)
            target_state = states[step + 1 : step + 2]
            target_action = actions[step + 1 : step + 2]
            total = (
                total
                + F.mse_loss(predicted_state, target_state)
                + F.cross_entropy(action_logits, target_action)
            )
            use_prediction = bool(
                torch.rand((), generator=generator).item() < feedback_probability
            )
            current = predicted_state if use_prediction else target_state
            predicted_history.append(current)
        loss = total / (len(states) - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate(
    model,
    trajectories,
    representation,
    *,
    family: str,
    history: int,
    memory_dim: int,
    device: torch.device,
    horizons: list[int],
) -> dict[str, Any]:
    rows = []
    model.eval()
    for trajectory in trajectories:
        states = trajectory["latent"]
        maximum = min(max(horizons), len(states) - 1)
        if maximum < 1:
            continue
        initial = torch.from_numpy(states[0:1]).to(device)
        predicted, actions = autonomous_rollout(
            model,
            initial,
            steps=maximum,
            family=family,
            history=history,
            memory_dim=memory_dim,
        )
        predicted_np = predicted[0].detach().cpu().numpy()
        decoded = representation.decode(predicted_np)
        target = trajectory["states"][1 : maximum + 1]
        cosine = row_cosine(decoded, target)
        action_ids = actions[0].argmax(-1).detach().cpu().numpy()
        target_actions = trajectory["actions"][1 : maximum + 1]
        for horizon in horizons:
            if horizon <= maximum:
                rows.append(
                    {
                        "example_id": trajectory["example_id"],
                        "family": trajectory["family"],
                        "teacher_correct": trajectory["teacher_correct"],
                        "horizon": horizon,
                        "decoded_cosine": float(cosine[horizon - 1]),
                        "trajectory_distance": float(np.mean(1 - cosine[:horizon])),
                        "semantic_action_accuracy": float(
                            np.mean(action_ids[:horizon] == target_actions[:horizon])
                        ),
                        "answer_correct": bool(
                            action_ids[horizon - 1] == target_actions[horizon - 1]
                        ),
                        "finite": bool(np.isfinite(predicted_np[:horizon]).all()),
                        "variance": float(np.var(predicted_np[:horizon])),
                        "time_to_divergence": int(
                            next(
                                (
                                    i + 1
                                    for i, value in enumerate(cosine)
                                    if value < 0.8
                                ),
                                maximum + 1,
                            )
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    summaries = []
    for horizon, group in frame.groupby("horizon"):
        summaries.append(
            {
                "horizon": int(horizon),
                "n": len(group),
                "decoded_cosine_median": float(group["decoded_cosine"].median()),
                "trajectory_distance_mean": float(group["trajectory_distance"].mean()),
                "semantic_action_accuracy": float(
                    group["semantic_action_accuracy"].mean()
                ),
                "answer_accuracy": float(group["answer_correct"].mean()),
                "finite_fraction": float(group["finite"].mean()),
                "variance_median": float(group["variance"].median()),
            }
        )
    return {"rows": rows, "horizons": summaries}


def _train(
    context,
    *,
    family: str,
    dimension: int,
    history: int,
    memory_dim: int,
    controller_seed: int,
) -> dict[str, Any]:
    screen = json.loads(
        (context.processed_dir / "compact_memory_representation_screen.json").read_text(
            encoding="utf-8"
        )
    )
    if not screen.get("temporal_training_authorized"):
        raise RuntimeError("representation screen did not authorize temporal training")
    selected = next(
        value
        for value in screen["selected_by_dimension"]
        if int(value["dimension"]) == dimension
    )
    representation = _load_representation(context.root, selected["representation_path"])
    frame = _load_trace_frame(context)
    _, raw_trajectories = _pairs(context, frame)
    trajectories = _latent_trajectories(raw_trajectories, representation)
    train = [value for value in trajectories if value["split"] == "train"]
    validation = [value for value in trajectories if value["split"] == "validation"]
    test = [value for value in trajectories if value["split"] == "test"]
    torch.manual_seed(controller_seed)
    target = int(context.config["compact_memory"]["target_parameter_count"])
    tolerance = float(context.config["compact_memory"]["parameter_tolerance"])
    model = build_parameter_matched_controller(
        family,
        state_dim=dimension,
        action_count=len(ACTION_SURFACES),
        target=target,
        tolerance=tolerance,
        history=history,
        memory_dim=memory_dim,
    ).to(
        bundle_device := torch.device(
            context.config["model"].get("device", 0)
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    maximum_epochs = int(context.config["compact_memory"]["maximum_epochs"])
    patience = int(context.config["compact_memory"]["early_stopping_patience"])
    best = -float("inf")
    best_state = None
    stale = 0
    training = []
    for epoch in range(maximum_epochs):
        feedback = scheduled_feedback_probability(
            epoch,
            maximum_epochs,
            warmup_fraction=float(
                context.config["compact_memory"]["teacher_forcing_fraction"]
            ),
            maximum_feedback=float(
                context.config["compact_memory"]["maximum_predicted_feedback"]
            ),
        )
        loss = _train_epoch(
            model,
            train,
            optimizer,
            family=family,
            history=history,
            memory_dim=memory_dim,
            feedback_probability=feedback,
            device=bundle_device,
            seed=controller_seed + epoch,
        )
        validation_result = _evaluate(
            model,
            validation,
            representation,
            family=family,
            history=history,
            memory_dim=memory_dim,
            device=bundle_device,
            horizons=[8],
        )
        score = (
            validation_result["horizons"][0]["decoded_cosine_median"]
            if validation_result["horizons"]
            else -float("inf")
        )
        training.append(
            {
                "epoch": epoch,
                "loss": loss,
                "feedback_probability": feedback,
                "validation_horizon8_cosine": score,
            }
        )
        if score > best:
            best, stale = score, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_result = _evaluate(
        model,
        test,
        representation,
        family=family,
        history=history,
        memory_dim=memory_dim,
        device=bundle_device,
        horizons=[int(value) for value in context.config["compact_memory"]["horizons"]],
    )
    checkpoint = (
        context.root
        / "artifacts/checkpoints/v3_1"
        / f"{family}-d{dimension}-h{history}-m{memory_dim}-s{controller_seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "training": training}, checkpoint)
    return {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "run_id": context.run_id,
        "model_family": family,
        "state_representation": selected["representation_family"],
        "state_dimension": dimension,
        "history_length": history if family == "history" else None,
        "memory_dimension": memory_dim if family == "gru" else None,
        "seed": controller_seed,
        "parameter_count": parameter_count(model),
        "best_validation_horizon8_cosine": best,
        "epochs": len(training),
        "training": training,
        "test": test_result,
        "checkpoint_path": str(checkpoint.relative_to(context.root)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def main() -> None:
    parser = standard_parser(
        "Run compact-memory protocol v3.1", "configs/compact_memory_v3_1.yaml"
    )
    parser.add_argument(
        "--stage",
        choices=("traces", "merge-traces", "screen", "train"),
        default="traces",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--shard-group-id", default=os.environ.get("JCLOSURE_SHARD_GROUP_ID", "single")
    )
    parser.add_argument("--family", choices=("markov", "history", "gru"), default="gru")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--memory-dimension", type=int, default=128)
    parser.add_argument("--controller-seed", type=int, default=20260828)
    args = parser.parse_args()
    context = initialize_context("compact-memory-v3-1", args)
    try:
        verify_freeze(context.root, kind="memory", config=context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "traces":
            bundle = load_model_bundle(context.config)
            rows = _trace_rows(
                context,
                bundle,
                split=args.split,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                limit=args.limit,
            )
            output = (
                context.raw_dir
                / context.run_id
                / f"traces-{args.split}-shard-{args.shard_index:03d}.jsonl"
            )
            append_jsonl(output, rows)
            context.finish(
                "COMPLETED_TRACE_SHARD",
                stage=args.stage,
                split=args.split,
                shard_group_id=args.shard_group_id,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                records=str(output.relative_to(context.root)),
                record_count=len(rows),
            )
        elif args.stage == "merge-traces":
            summary = _merge_traces(
                context,
                shard_group_id=args.shard_group_id,
                shard_count=args.shard_count,
            )
            context.finish("COMPLETED", stage=args.stage, summary=summary)
        elif args.stage == "screen":
            bundle = load_model_bundle(context.config)
            summary = _screen(context, bundle)
            context.finish("COMPLETED", stage=args.stage, summary=summary)
        else:
            result = _train(
                context,
                family=args.family,
                dimension=args.dimension,
                history=args.history,
                memory_dim=args.memory_dimension,
                controller_seed=args.controller_seed,
            )
            output = context.raw_dir / context.run_id / "controller_result.json"
            write_json_atomic(output, result)
            context.finish(
                "COMPLETED",
                stage=args.stage,
                result=str(output.relative_to(context.root)),
            )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
