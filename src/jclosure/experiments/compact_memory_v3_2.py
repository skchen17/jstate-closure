"""Canonical trace audit, representation screen, and compact-memory training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from jclosure.compact_memory_v3_1 import (
    LinearRepresentation,
    _closest_width,
    _history_input,
    autonomous_rollout,
    build_parameter_matched_controller,
    fit_representation,
    parameter_count,
    row_cosine,
    scheduled_feedback_probability,
)
from jclosure.compact_memory_v3_2 import (
    action_metrics,
    audit_trace_payload,
    representation_gate_reasons,
    time_to_divergence,
)
from jclosure.experiments.calibrate_v3_1 import _read_jsonl
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_v3_1 import (
    ACTION_SURFACES,
    ACTION_TO_ID,
    _causal_regression_pairs,
    _phase0_regression_profiles,
    _profile_pass10,
)
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.model import load_model_bundle
from jclosure.protocol_v3_2 import verify_memory_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.runtime_v3_2 import MEMORY_PROTOCOL_V32


class HistoryControllerV32(torch.nn.Module):
    """History controller whose transition head returns the compact state size."""

    def __init__(
        self, state_dim: int, history: int, action_count: int, width: int
    ) -> None:
        super().__init__()
        self.history = int(history)
        input_dim = state_dim * history + history
        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
        )
        self.state_head = torch.nn.Linear(width, state_dim)
        self.action_head = torch.nn.Linear(width, action_count)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(torch.cat((states.flatten(-2), mask), dim=-1))
        return self.state_head(hidden), self.action_head(hidden)


def _build_controller_v32(
    family: str,
    *,
    state_dim: int,
    action_count: int,
    target: int,
    tolerance: float,
    history: int,
    memory_dim: int,
):
    if family == "history":
        return _closest_width(
            lambda width: HistoryControllerV32(
                state_dim, history, action_count, width
            ),
            target,
            tolerance,
        )
    return build_parameter_matched_controller(
        family,
        state_dim=state_dim,
        action_count=action_count,
        target=target,
        tolerance=tolerance,
        history=history,
        memory_dim=memory_dim,
    )


def _load_memory_encoder(context, bundle):
    size = int(context.config["compact_memory"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    encoder = JStateEncoder.from_lens(
        bundle.lens,
        bundle.unembedding_weight,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        lazy=True,
        protocol_version=MEMORY_PROTOCOL_V32,
        direction_chunk_size=int(context.config["jstate"].get("direction_chunk_size", 512)),
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _source_manifests(context, freeze: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    group = freeze["source_shard_group_id"]
    manifests = []
    for path in context.root.glob("results/v3_1/raw/compact-memory-v3-1-*/manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "COMPLETED_TRACE_SHARD" and value.get("shard_group_id") == group:
            manifests.append(value)
    by_key = {(str(value["split"]), int(value["shard_index"])): value for value in manifests}
    expected = {(split, shard) for split in ("train", "validation", "test") for shard in range(2)}
    if set(by_key) != expected:
        raise RuntimeError(f"compact-memory v3.2 source shards mismatch: {sorted(expected - set(by_key))}")
    return by_key


def _merge_and_audit(context, freeze: dict[str, Any]) -> dict[str, Any]:
    by_key = _source_manifests(context, freeze)
    rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        rows.extend(_read_jsonl(context.root / by_key[key]["records"]))
    frame = pd.DataFrame(rows)
    required = {
        "example_id", "split", "family", "template_id", "program_hash", "length",
        "expected_actions", "generated_semantic", "parseable", "teacher_correct",
        "trace_path", "trace_sha256", "error",
    }
    missing_columns = sorted(required - set(frame.columns))
    duplicate_ids = sorted(frame.loc[frame["example_id"].duplicated(keep=False), "example_id"].astype(str).unique()) if "example_id" in frame else []
    split_overlap_ids: list[str] = []
    program_overlap: list[str] = []
    if not frame.empty:
        split_counts = frame.groupby("example_id")["split"].nunique()
        split_overlap_ids = sorted(split_counts[split_counts > 1].index.astype(str))
        program_counts = frame.groupby("program_hash")["split"].nunique()
        program_overlap = sorted(program_counts[program_counts > 1].index.astype(str))
    audit_reasons: list[list[str]] = []
    hash_failures = 0
    for row in frame.itertuples():
        reasons: list[str] = []
        path = context.root / row.trace_path
        if not path.is_file():
            reasons.append("trace_tensor_missing")
        else:
            if sha256_file(path) != row.trace_sha256:
                reasons.append("trace_hash_mismatch")
                hash_failures += 1
            try:
                with np.load(path, allow_pickle=False) as payload:
                    reasons.extend(audit_trace_payload(row, payload))
            except Exception as exc:
                reasons.append(f"corrupted_trace:{type(exc).__name__}")
        audit_reasons.append(sorted(set(reasons)))
    frame = frame.copy()
    frame["audit_reasons"] = audit_reasons
    frame["trace_valid"] = frame["audit_reasons"].map(lambda value: not value)
    frame["parseable_but_wrong"] = frame["parseable"].astype(bool) & ~frame["teacher_correct"].astype(bool)
    frame["invalid_unparseable"] = ~frame["parseable"].astype(bool)
    output = context.processed_dir / "compact_memory_trace_records_v3_2.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    counts = []
    for (split, family, length), group in frame.groupby(["split", "family", "length"], sort=True):
        counts.append({
            "split": str(split), "family": str(family), "length": int(length),
            "attempted": len(group), "parseable": int(group["parseable"].sum()),
            "teacher_correct": int(group["teacher_correct"].sum()),
            "parseable_but_wrong": int(group["parseable_but_wrong"].sum()),
            "invalid_unparseable": int(group["invalid_unparseable"].sum()),
            "trace_valid": int(group["trace_valid"].sum()),
        })
    parseable_by_family = frame[frame["trace_valid"]].groupby("family")["parseable"].sum().to_dict()
    target = int(context.config["compact_memory_v3_2"]["target_parseable_per_family"])
    authorized = (
        not missing_columns and not duplicate_ids and not split_overlap_ids and not program_overlap
        and bool(frame["trace_valid"].all())
        and all(int(parseable_by_family.get(family, 0)) >= target for family in context.config["compact_memory"]["task_families"])
    )
    summary = {
        "schema_version": 5, "protocol_version": MEMORY_PROTOCOL_V32,
        "run_id": context.run_id, "records": str(output.relative_to(context.root)),
        "source_shards": [by_key[key]["run_id"] for key in sorted(by_key)],
        "total_records": len(frame), "counts": counts,
        "parseable_by_family": {str(key): int(value) for key, value in parseable_by_family.items()},
        "target_parseable_per_family": target, "missing_columns": missing_columns,
        "duplicate_example_ids": duplicate_ids, "split_overlap_ids": split_overlap_ids,
        "program_split_overlap": program_overlap, "trace_hash_failures": hash_failures,
        "corrupted_or_invalid_traces": int((~frame["trace_valid"]).sum()),
        "representation_screen_authorized": authorized,
    }
    path = context.processed_dir / "compact_memory_trace_audit_v3_2.json"
    write_json_atomic(path, summary)
    return summary


def _load_audited_frame(context) -> pd.DataFrame:
    summary = json.loads((context.processed_dir / "compact_memory_trace_audit_v3_2.json").read_text())
    if not summary.get("representation_screen_authorized"):
        raise RuntimeError("compact-memory trace audit did not authorize representation screen")
    return pd.read_parquet(context.root / summary["records"])


def _pairs(context, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {
        f"{split}_{kind}": []
        for split in ("train", "validation", "test")
        for kind in ("x", "y", "r", "s", "a", "g")
    }
    trajectories: list[dict[str, Any]] = []
    selected = frame[frame["trace_valid"].astype(bool) & frame["parseable"].astype(bool)]
    for row in selected.itertuples():
        with np.load(context.root / row.trace_path, allow_pickle=False) as payload:
            states = payload["states"].astype(np.float32)
            remainder = payload["remainders"].astype(np.float32)
            sparse = payload["sparse_states"].astype(np.float32)
            actions = payload["actions"].astype(np.int64)
        ground_truth = np.asarray([ACTION_TO_ID[str(value)] for value in row.expected_actions], dtype=np.int64)
        if len(states) < 2:
            continue
        arrays[f"{row.split}_x"].append(states[:-1])
        arrays[f"{row.split}_y"].append(states[1:])
        arrays[f"{row.split}_r"].append(remainder[:-1])
        arrays[f"{row.split}_s"].append(sparse[:-1])
        arrays[f"{row.split}_a"].append(actions[1:])
        arrays[f"{row.split}_g"].append(ground_truth[1:])
        trajectories.append({
            "example_id": row.example_id, "family": row.family, "split": row.split,
            "states": states, "remainders": remainder, "sparse_states": sparse,
            "actions": actions, "ground_truth_actions": ground_truth,
            "teacher_correct": bool(row.teacher_correct), "length": int(row.length),
        })
    concatenated = {key: np.concatenate(value) if value else np.empty((0,)) for key, value in arrays.items()}
    return concatenated, trajectories


def _screen(context, bundle) -> dict[str, Any]:
    frame = _load_audited_frame(context)
    arrays, trajectories = _pairs(context, frame)
    correct_frame = frame[frame["teacher_correct"].astype(bool)]
    correct_arrays, _ = _pairs(context, correct_frame)
    vocabulary, encoder, dense_map = _load_memory_encoder(context, bundle)
    phase0_profiles = _phase0_regression_profiles(context, bundle, dense_map, vocabulary)
    causal_pairs = _causal_regression_pairs(context, bundle, encoder, dense_map)
    original_phase0 = float(np.mean([_profile_pass10(value["profile"], value["concept_indices"]) for value in phase0_profiles])) if phase0_profiles else 0.0
    config = context.config["compact_memory_v3_2"]
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    root = context.root / "artifacts/traces/v3_2/representations"
    root.mkdir(parents=True, exist_ok=True)
    for dimension in config["state_dimensions"]:
        candidates = []
        for family in config["representation_families"]:
            representation = fit_representation(
                str(family), int(dimension), arrays["train_x"], arrays["train_y"],
                feature_importance=np.mean(np.abs(arrays["train_s"]), axis=0) if family == "sparse_j_centered" else None,
            )
            train_z = representation.encode(arrays["train_x"])
            validation_z = representation.encode(arrays["validation_x"])
            predictor = Ridge(alpha=1.0).fit(train_z, arrays["train_y"])
            prediction = predictor.predict(validation_z)
            next_cosine = float(np.median(row_cosine(prediction, arrays["validation_y"])))
            reconstruction = representation.decode(validation_z)
            reconstruction_cosine = float(np.median(row_cosine(reconstruction, arrays["validation_x"])))
            reconstructed_phase0 = float(np.mean([
                _profile_pass10(representation.decode(representation.encode(value["profile"][None]))[0], value["concept_indices"])
                for value in phase0_profiles
            ])) if phase0_profiles else 0.0
            phase0_retention = reconstructed_phase0 / max(original_phase0, 1e-12)
            delta_cosines: list[float] = []
            magnitude_retentions: list[float] = []
            for clean, swapped in causal_pairs:
                original_delta = swapped - clean
                reconstructed_delta = (
                    representation.decode(representation.encode(swapped[None]))[0]
                    - representation.decode(representation.encode(clean[None]))[0]
                )
                delta_cosines.append(float(row_cosine(reconstructed_delta[None], original_delta[None])[0]))
                reconstructed_norm = float(np.linalg.norm(reconstructed_delta))
                original_norm = float(np.linalg.norm(original_delta))
                ratio = reconstructed_norm / max(original_norm, 1e-12)
                magnitude_retentions.append(min(ratio, 1 / max(ratio, 1e-12)))
            direction_retention = float(np.mean(np.asarray(delta_cosines) >= 0.8)) if delta_cosines else 0.0
            magnitude_retention = float(np.median(magnitude_retentions)) if magnitude_retentions else 0.0
            correct_sensitivity = None
            if correct_arrays["train_x"].ndim == 2 and correct_arrays["validation_x"].ndim == 2 and len(correct_arrays["train_x"]) > representation.dimension:
                correct_predictor = Ridge(alpha=1.0).fit(
                    representation.encode(correct_arrays["train_x"]), correct_arrays["train_y"]
                )
                correct_prediction = correct_predictor.predict(representation.encode(correct_arrays["validation_x"]))
                correct_sensitivity = float(np.median(row_cosine(correct_prediction, correct_arrays["validation_y"])))
            gate_reasons = list(
                representation_gate_reasons(
                    {
                        "validation_reconstruction_cosine": reconstruction_cosine,
                        "phase0_pass10_retention": phase0_retention,
                        "causal_trials": len(causal_pairs),
                        "causal_direction_retention": direction_retention,
                        "causal_magnitude_retention": magnitude_retention,
                    },
                    config,
                )
            )
            path = root / f"{family}-{representation.dimension}.npz"
            np.savez_compressed(path, **representation.state_dict())
            row = {
                "schema_version": 5, "protocol_version": MEMORY_PROTOCOL_V32,
                "representation_family": str(family), "dimension": representation.dimension,
                "validation_next_state_cosine": next_cosine,
                "validation_reconstruction_cosine": reconstruction_cosine,
                "phase0_pass10_original": original_phase0,
                "phase0_pass10_reconstructed": reconstructed_phase0,
                "phase0_pass10_retention": phase0_retention,
                "causal_trials": len(causal_pairs),
                "causal_delta_cosine_median": float(np.median(delta_cosines)) if delta_cosines else None,
                "causal_direction_retention": direction_retention,
                "causal_magnitude_retention": magnitude_retention,
                "teacher_correct_only_validation_next_state_cosine": correct_sensitivity,
                "teacher_correct_train_trajectories": sum(value["split"] == "train" and value["teacher_correct"] for value in trajectories),
                "teacher_correct_validation_trajectories": sum(value["split"] == "validation" and value["teacher_correct"] for value in trajectories),
                "gate_passed": not gate_reasons, "gate_reasons": gate_reasons,
                "representation_path": str(path.relative_to(context.root)),
                "representation_sha256": sha256_file(path),
            }
            rows.append(row)
            candidates.append(row)
        eligible = [value for value in candidates if value["gate_passed"]]
        if eligible:
            eligible.sort(key=lambda value: (-float(value["validation_next_state_cosine"]), -float(value["phase0_pass10_retention"]), -float(value["causal_direction_retention"]), str(value["representation_family"])))
            winner = dict(eligible[0])
            winner["selection_status"] = "SELECTED_ON_VALIDATION_AFTER_RETENTION_GATES"
            selected.append(winner)
    selected.sort(key=lambda value: (-float(value["validation_next_state_cosine"]), int(value["dimension"]), str(value["representation_family"])))
    overall = selected[0] if selected else None
    records = context.processed_dir / "compact_memory_representation_screen_v3_2.parquet"
    pd.DataFrame(rows).to_parquet(records, index=False, compression="zstd")
    summary = {
        "schema_version": 5, "protocol_version": MEMORY_PROTOCOL_V32,
        "records": str(records.relative_to(context.root)), "selected_by_dimension": selected,
        "overall_selected": overall, "phase0_regression_items": len(phase0_profiles),
        "causal_regression_trials": len(causal_pairs),
        "temporal_training_authorized": bool(overall),
        "authorization_reason": "retention_gates_passed" if overall else "no_candidate_passed_semantic_and_causal_retention",
    }
    path = context.processed_dir / "compact_memory_representation_screen_v3_2.json"
    write_json_atomic(path, summary)
    return summary


def _load_representation(root: Path, path: str) -> LinearRepresentation:
    payload = np.load(root / path, allow_pickle=False)
    return LinearRepresentation(
        str(payload["family"].item()), int(payload["dimension"]),
        payload["mean"], payload["encoder"], payload["decoder"],
    )


def _latent_trajectories(trajectories: list[dict[str, Any]], representation: LinearRepresentation) -> list[dict[str, Any]]:
    return [{**value, "latent": representation.encode(value["states"]).astype(np.float32)} for value in trajectories]


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
    batch_size: int = 64,
) -> float:
    generator = np.random.default_rng(seed)
    groups: dict[int, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        groups.setdefault(len(trajectory["latent"]), []).append(trajectory)
    losses = []
    model.train()
    for length in sorted(groups):
        ordered = list(groups[length])
        generator.shuffle(ordered)
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            states = torch.from_numpy(np.stack([value["latent"] for value in batch])).to(device)
            actions = torch.from_numpy(np.stack([value["actions"] for value in batch])).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            current = states[:, 0]
            predicted_history = [current]
            memory = torch.zeros(len(batch), memory_dim, device=device) if family == "gru" else None
            total = torch.zeros((), device=device)
            for step in range(length - 1):
                if family == "markov":
                    predicted_state, action_logits = model(current)
                elif family == "history":
                    window, mask = _history_input(predicted_history, history)
                    predicted_state, action_logits = model(window, mask)
                else:
                    assert memory is not None
                    predicted_state, action_logits, memory = model(current, memory)
                target_state = states[:, step + 1]
                target_action = actions[:, step + 1]
                total = total + F.mse_loss(predicted_state, target_state) + F.cross_entropy(action_logits, target_action)
                use_prediction = torch.from_numpy(generator.random(len(batch)) < feedback_probability).to(device)
                current = torch.where(use_prediction[:, None], predicted_state, target_state)
                predicted_history.append(current)
            loss = total / (length - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _evaluate(model, trajectories, representation, *, family: str, history: int, memory_dim: int, device: torch.device, horizons: list[int], divergence: float) -> dict[str, Any]:
    rows = []
    model.eval()
    for trajectory in trajectories:
        maximum = min(max(horizons), len(trajectory["latent"]) - 1)
        if maximum < 1:
            continue
        initial = torch.from_numpy(trajectory["latent"][0:1]).to(device)
        predicted, action_logits = autonomous_rollout(model, initial, steps=maximum, family=family, history=history, memory_dim=memory_dim)
        predicted_np = predicted[0].cpu().numpy()
        decoded = representation.decode(predicted_np)
        target = trajectory["states"][1 : maximum + 1]
        cosine = row_cosine(decoded, target)
        action_ids = action_logits[0].argmax(-1).cpu().numpy()
        teacher = trajectory["actions"][1 : maximum + 1]
        ground_truth = trajectory["ground_truth_actions"][1 : maximum + 1]
        for horizon in horizons:
            if horizon > maximum:
                continue
            action_values = action_metrics(action_ids[:horizon], teacher[:horizon], ground_truth[:horizon])
            rows.append({
                "example_id": trajectory["example_id"], "family": trajectory["family"],
                "teacher_correct": trajectory["teacher_correct"], "horizon": horizon,
                "decoded_cosine": float(cosine[horizon - 1]),
                "trajectory_distance": float(np.mean(1 - cosine[:horizon])),
                **action_values,
                "time_to_divergence": time_to_divergence(cosine[:horizon], divergence),
                "rollout_failure": bool(not np.isfinite(predicted_np[:horizon]).all()),
                "variance": float(np.var(predicted_np[:horizon])),
            })
    frame = pd.DataFrame(rows)
    summaries = []
    for horizon, group in frame.groupby("horizon"):
        summaries.append({
            "horizon": int(horizon), "n": len(group),
            "decoded_cosine_median": float(group["decoded_cosine"].median()),
            "trajectory_distance_mean": float(group["trajectory_distance"].mean()),
            "teacher_action_fidelity": float(group["teacher_action_fidelity"].mean()),
            "ground_truth_action_accuracy": float(group["ground_truth_action_accuracy"].mean()),
            "teacher_ground_truth_agreement": float(group["teacher_ground_truth_agreement"].mean()),
            "ground_truth_final_accuracy": float(group["ground_truth_final_accuracy"].mean()),
            "time_to_divergence_median": float(group["time_to_divergence"].median()),
            "rollout_failure_rate": float(group["rollout_failure"].mean()),
            "variance_median": float(group["variance"].median()),
        })
    return {"rows": rows, "horizons": summaries}


def _train(context, *, family: str, history: int, memory_dim: int, seed: int, training_subset: str) -> dict[str, Any]:
    screen = json.loads((context.processed_dir / "compact_memory_representation_screen_v3_2.json").read_text())
    if not screen.get("temporal_training_authorized"):
        raise RuntimeError("representation screen did not authorize temporal training")
    selected = screen["overall_selected"]
    representation = _load_representation(context.root, selected["representation_path"])
    frame = _load_audited_frame(context)
    _, raw = _pairs(context, frame)
    trajectories = _latent_trajectories(raw, representation)
    if training_subset == "teacher_correct_only":
        trajectories = [value for value in trajectories if value["teacher_correct"]]
    train = [value for value in trajectories if value["split"] == "train"]
    validation = [value for value in trajectories if value["split"] == "validation"]
    test = [value for value in trajectories if value["split"] == "test"]
    if not train or not validation or not test:
        raise RuntimeError(f"{training_subset} lacks a nonempty train/validation/test split")
    torch.manual_seed(seed)
    values = context.config["compact_memory_v3_2"]
    model = _build_controller_v32(
        family, state_dim=int(selected["dimension"]), action_count=len(ACTION_SURFACES),
        target=int(values["target_parameter_count"]), tolerance=float(values["parameter_tolerance"]),
        history=history, memory_dim=memory_dim,
    )
    device = torch.device(context.config["model"].get("device", 0) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    maximum_epochs = int(values["maximum_epochs"])
    patience = int(values["early_stopping_patience"])
    best = -float("inf")
    best_state = None
    stale = 0
    training = []
    for epoch in range(maximum_epochs):
        feedback = scheduled_feedback_probability(
            epoch, maximum_epochs,
            warmup_fraction=float(values["teacher_forcing_fraction"]),
            maximum_feedback=float(values["maximum_predicted_feedback"]),
        )
        loss = _train_epoch(model, train, optimizer, family=family, history=history, memory_dim=memory_dim, feedback_probability=feedback, device=device, seed=seed + epoch)
        validation_result = _evaluate(
            model, validation, representation, family=family, history=history,
            memory_dim=memory_dim, device=device, horizons=[8],
            divergence=float(values["divergence_cosine"]),
        )
        score = validation_result["horizons"][0]["decoded_cosine_median"] if validation_result["horizons"] else -float("inf")
        training.append({"epoch": epoch, "loss": loss, "feedback_probability": feedback, "validation_horizon8_cosine": score})
        if score > best:
            best, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_result = _evaluate(
        model, test, representation, family=family, history=history,
        memory_dim=memory_dim, device=device, horizons=[int(value) for value in values["horizons"]],
        divergence=float(values["divergence_cosine"]),
    )
    checkpoint = context.root / "artifacts/checkpoints/v3_2" / f"{family}-h{history}-m{memory_dim}-s{seed}-{training_subset}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "training": training}, checkpoint)
    return {
        "schema_version": 5, "protocol_version": MEMORY_PROTOCOL_V32,
        "run_id": context.run_id, "model_family": family,
        "state_representation": selected["representation_family"],
        "state_dimension": int(selected["dimension"]),
        "history_length": history if family == "history" else None,
        "memory_dimension": memory_dim if family == "gru" else None,
        "training_subset": training_subset, "seed": seed,
        "parameter_count": parameter_count(model), "train_trajectories": len(train),
        "validation_trajectories": len(validation), "test_trajectories": len(test),
        "best_validation_horizon8_cosine": best, "epochs": len(training),
        "training": training, "test": test_result,
        "checkpoint_path": str(checkpoint.relative_to(context.root)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def main() -> None:
    parser = standard_parser("Run compact-memory protocol v3.2", "configs/compact_memory_v3_2.yaml")
    parser.add_argument("--stage", choices=("merge-audit", "screen", "train"), default="merge-audit")
    parser.add_argument("--family", choices=("markov", "history", "gru"), default="gru")
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--memory-dimension", type=int, default=128)
    parser.add_argument("--controller-seed", type=int, default=20260828)
    parser.add_argument("--training-subset", choices=("all_parseable", "teacher_correct_only"), default="all_parseable")
    args = parser.parse_args()
    context = initialize_context("compact-memory-v3-2", args)
    try:
        freeze = verify_memory_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "merge-audit":
            summary = _merge_and_audit(context, freeze)
            context.finish("COMPLETED", stage=args.stage, summary=summary)
        elif args.stage == "screen":
            bundle = load_model_bundle(context.config)
            summary = _screen(context, bundle)
            context.finish("COMPLETED", stage=args.stage, summary=summary)
        else:
            result = _train(
                context, family=args.family, history=args.history,
                memory_dim=args.memory_dimension, seed=args.controller_seed,
                training_subset=args.training_subset,
            )
            output = context.raw_dir / context.run_id / "controller_result.json"
            write_json_atomic(output, result)
            context.finish("COMPLETED", stage=args.stage, result=str(output.relative_to(context.root)))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt: run cancelled")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
