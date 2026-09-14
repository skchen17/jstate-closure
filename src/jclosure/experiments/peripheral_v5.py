"""Strong full-remainder ceiling and compact peripheral-state experiments."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.peripheral_v5 import (
    ACTION_COUNT,
    U_DIM,
    OneStepPredictor,
    PeripheralComposite,
    PeripheralEncoder,
    RecurrentPeripheralController,
    RemainderTransform,
    TransitionArrays,
    bottleneck_regularizer,
    build_u,
    encode_compact,
    make_history,
    parameter_count,
    row_cosine,
)
from jclosure.protocol_v5 import build_peripheral_freeze, verify_peripheral_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v5 import PeripheralReferenceRecord, PeripheralStateRecord
from jclosure.statistics import clustered_bootstrap_ci

PROTOCOL = "jstate_peripheral_protocol_v5"


@torch.no_grad()
def run_prepare(context: Any) -> dict[str, Any]:
    """Derive a layer-23 J trace without mutating the frozen pooled-v4 tensors."""

    from jclosure.model import load_model_bundle

    source_path = context.root / context.config["peripheral_v5"]["source_trace_summary"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bundle = load_model_bundle(context.config)
    vocabulary, _, dense_map = _load_encoder(context, bundle)
    layer = int(context.config["peripheral_v5"]["workspace_layer"])
    device = _device(context)
    centered_map = dense_map.centered_map(layer, device=device, dtype=torch.float32)
    directory = context.root / "artifacts/traces/v5" / context.run_id
    directory.mkdir(parents=True, exist_ok=True)
    domains = []
    for record in source["domains"]:
        source_tensor = context.root / record["tensor_path"]
        if sha256_file(source_tensor) != record["tensor_sha256"]:
            raise RuntimeError(f"source v4 trace hash mismatch: {record['domain']}")
        with np.load(source_tensor, allow_pickle=False) as payload:
            hidden = payload["full_states"].astype(np.float32)
            actions = payload["actions"].astype(np.int16)
            dispersion = payload["dispersion"].astype(np.float32)
        profiles = []
        for start in range(0, len(hidden), 256):
            values = torch.from_numpy(hidden[start : start + 256]).float().to(device)
            scores = values @ centered_map.T
            profiles.append(F.normalize(scores, dim=1).cpu().numpy().astype(np.float16))
        target = directory / f"{record['domain']}.npz"
        np.savez_compressed(
            target,
            j_profiles=np.concatenate(profiles),
            full_states=hidden.astype(np.float16),
            actions=actions,
            source_pooled_dispersion=dispersion,
        )
        domains.append(
            {
                **record,
                "tensor_path": str(target.relative_to(context.root)),
                "tensor_sha256": sha256_file(target),
                "state_definition": f"layer_{layer}_normalized_dense_measured_J",
                "source_tensor_path": record["tensor_path"],
                "source_tensor_sha256": record["tensor_sha256"],
            }
        )
    summary = {
        "schema_version": 7,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "state_definition": f"layer_{layer}_normalized_dense_measured_J",
        "layer": layer,
        "dictionary_size": len(vocabulary.token_ids),
        "dictionary_hash": vocabulary.digest,
        "source_trace_summary": str(source_path.relative_to(context.root)),
        "source_trace_summary_sha256": sha256_file(source_path),
        "source_program_freeze_digest": source["program_freeze_digest"],
        "domains": domains,
    }
    write_json_atomic(context.processed_dir / "peripheral_traces_v5.json", summary)
    return summary


def _record_for_domain(freeze: dict[str, Any], domain: str) -> dict[str, Any]:
    role = freeze["analysis_split_roles"].get(domain, domain)
    return freeze["source_domains"][role]


def load_domain(
    root: Path, freeze: dict[str, Any], domain: str, history: int
) -> TransitionArrays:
    record = _record_for_domain(freeze, domain)
    tensor_path = root / record["tensor_path"]
    with np.load(tensor_path, allow_pickle=False) as payload:
        states = payload["j_profiles"].astype(np.float32)
        hidden = payload["full_states"].astype(np.float32)
        actions = payload["actions"].astype(np.int64)
    trajectories = pd.read_parquet(root / record["trajectories"])
    current_j: list[np.ndarray] = []
    next_j: list[np.ndarray] = []
    current_h: list[np.ndarray] = []
    next_h: list[np.ndarray] = []
    current_actions: list[np.ndarray] = []
    next_actions: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    identifiers: list[str] = []
    families: list[str] = []
    steps: list[int] = []
    horizons: list[int] = []
    for row in trajectories.itertuples():
        start, stop = int(row.start), int(row.stop)
        if stop - start < 2:
            continue
        trajectory_j = states[start:stop]
        trajectory_h = hidden[start:stop]
        trajectory_actions = actions[start:stop]
        history_values, history_mask = make_history(trajectory_j[:-1], history)
        count = len(trajectory_j) - 1
        current_j.append(trajectory_j[:-1])
        next_j.append(trajectory_j[1:])
        current_h.append(trajectory_h[:-1])
        next_h.append(trajectory_h[1:])
        current_actions.append(trajectory_actions[:-1])
        next_actions.append(trajectory_actions[1:])
        histories.append(history_values)
        masks.append(history_mask)
        identifiers.extend([str(row.example_id)] * count)
        families.extend([str(row.family)] * count)
        steps.extend(range(count))
        horizons.extend([int(row.horizon)] * count)
    current_action = np.concatenate(current_actions)
    next_action = np.concatenate(next_actions)
    family = np.asarray(families, dtype="U32")
    step_index = np.asarray(steps, dtype=np.int16)
    horizon_values = np.asarray(horizons, dtype=np.int16)
    return TransitionArrays(
        current_j=np.concatenate(current_j),
        next_j=np.concatenate(next_j),
        current_h=np.concatenate(current_h),
        next_h=np.concatenate(next_h),
        current_action=current_action,
        next_action=next_action,
        u=build_u(current_action, family, step_index, horizon_values),
        next_u=build_u(next_action, family, step_index + 1, horizon_values),
        history_j=np.concatenate(histories),
        history_mask=np.concatenate(masks),
        example_id=np.asarray(identifiers, dtype="U64"),
        family=family,
        step_index=step_index,
        horizon=horizon_values,
    )


def _device(context: Any) -> torch.device:
    return torch.device(
        f"cuda:{int(context.config['model'].get('device', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )


def _batches(
    length: int, size: int, generator: np.random.Generator
) -> list[np.ndarray]:
    indices = np.arange(length)
    generator.shuffle(indices)
    return [indices[start : start + size] for start in range(0, length, size)]


@torch.no_grad()
def _predict(
    model: OneStepPredictor,
    data: TransitionArrays,
    auxiliary: np.ndarray | None,
    *,
    device: torch.device,
    use_history: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for start in range(0, len(data), batch_size):
        stop = start + batch_size
        j = torch.from_numpy(data.current_j[start:stop]).float().to(device)
        u = torch.from_numpy(data.u[start:stop]).float().to(device)
        aux = (
            torch.from_numpy(auxiliary[start:stop]).float().to(device)
            if auxiliary is not None
            else None
        )
        history_j = (
            torch.from_numpy(data.history_j[start:stop]).float().to(device)
            if use_history
            else None
        )
        history_mask = (
            torch.from_numpy(data.history_mask[start:stop]).float().to(device)
            if use_history
            else None
        )
        predicted_j, predicted_action = model(j, u, aux, history_j, history_mask)
        states.append(predicted_j.cpu().numpy())
        actions.append(predicted_action.cpu().numpy())
    return np.concatenate(states), np.concatenate(actions)


def _evaluate_predictions(
    data: TransitionArrays,
    predicted_j: np.ndarray,
    action_logits: np.ndarray,
    *,
    run_id: str,
    architecture: str,
    input_state: str,
    seed: int,
    split: str,
    parameters: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cosine = row_cosine(predicted_j, data.next_j)
    action = np.argmax(action_logits, axis=1)
    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "architecture": architecture,
            "input_state": input_state,
            "seed": seed,
            "split": split,
            "example_id": data.example_id,
            "family": data.family,
            "step_index": data.step_index,
            "next_j_cosine": cosine,
            "next_j_distance": 1 - cosine,
            "semantic_correct": action == data.next_action,
        }
    )
    summaries: list[dict[str, Any]] = []
    for family, group in [("pooled", frame), *list(frame.groupby("family", sort=True))]:
        summaries.append(
            PeripheralReferenceRecord(
                run_id=run_id,
                architecture=architecture,
                input_state=input_state,
                split=split,
                seed=seed,
                next_j_cosine=float(group["next_j_cosine"].mean()),
                next_j_distance=float(group["next_j_distance"].mean()),
                semantic_accuracy=float(group["semantic_correct"].mean()),
                parameter_count=parameters,
                family=str(family),
            ).to_dict()
        )
    return frame, summaries


def _teacher_current_trajectory_metrics(
    data: TransitionArrays,
    next_j_cosine: np.ndarray,
    semantic_correct: np.ndarray,
) -> dict[str, Any]:
    """Aggregate one-step predictions along trajectories with teacher-current inputs.

    This is deliberately distinct from autonomous rollout: every transition is
    conditioned on the observed current teacher state.
    """

    frame = pd.DataFrame(
        {
            "example_id": data.example_id,
            "family": data.family,
            "step_index": data.step_index,
            "next_j_cosine": next_j_cosine,
            "semantic_correct": semantic_correct.astype(float),
        }
    )
    trajectories = (
        frame.sort_values(["example_id", "step_index"])
        .groupby(["example_id", "family"], as_index=False)
        .agg(
            trajectory_mean_cosine=("next_j_cosine", "mean"),
            terminal_j_cosine=("next_j_cosine", "last"),
            action_fidelity=("semantic_correct", "mean"),
            full_action_sequence_correct=("semantic_correct", "min"),
        )
    )
    output: dict[str, Any] = {}
    for family, group in [
        ("pooled", trajectories),
        *list(trajectories.groupby("family", sort=True)),
    ]:
        output[str(family)] = {
            "n_trajectories": int(len(group)),
            "trajectory_mean_cosine": float(group["trajectory_mean_cosine"].mean()),
            "terminal_j_cosine": float(group["terminal_j_cosine"].mean()),
            "action_fidelity": float(group["action_fidelity"].mean()),
            "full_action_sequence_accuracy": float(
                group["full_action_sequence_correct"].mean()
            ),
            "conditioning": "teacher_current_state_at_each_transition",
        }
    return output


def _train_predictor(
    context: Any,
    train: TransitionArrays,
    validation: TransitionArrays,
    *,
    train_auxiliary: np.ndarray | None,
    validation_auxiliary: np.ndarray | None,
    architecture: str,
    use_history: bool,
    seed: int,
) -> tuple[OneStepPredictor, list[dict[str, float]]]:
    section = context.config["peripheral_v5"]
    device = _device(context)
    torch.manual_seed(seed)
    model = OneStepPredictor(
        train.current_j.shape[1],
        train.u.shape[1],
        0 if train_auxiliary is None else train_auxiliary.shape[1],
        int(section["hidden_width"]),
        ACTION_COUNT,
        architecture=architecture,
        history_length=int(section["history_length"]) if use_history else 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    best = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    batch_size = int(section["batch_size"])
    for epoch in range(int(section["maximum_epochs"])):
        model.train()
        losses = []
        for selected in _batches(len(train), batch_size, generator):
            j = torch.from_numpy(train.current_j[selected]).float().to(device)
            target_j = torch.from_numpy(train.next_j[selected]).float().to(device)
            u = torch.from_numpy(train.u[selected]).float().to(device)
            target_action = (
                torch.from_numpy(train.next_action[selected]).long().to(device)
            )
            aux = (
                torch.from_numpy(train_auxiliary[selected]).float().to(device)
                if train_auxiliary is not None
                else None
            )
            hist = (
                torch.from_numpy(train.history_j[selected]).float().to(device)
                if use_history
                else None
            )
            mask = (
                torch.from_numpy(train.history_mask[selected]).float().to(device)
                if use_history
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            predicted_j, logits = model(j, u, aux, hist, mask)
            cosine_loss = (
                1 - F.cosine_similarity(predicted_j, target_j, dim=-1)
            ).mean()
            semantic_loss = F.cross_entropy(logits, target_action)
            loss = cosine_loss + 0.25 * semantic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_j, validation_action = _predict(
            model,
            validation,
            validation_auxiliary,
            device=device,
            use_history=use_history,
            batch_size=batch_size,
        )
        cosine = float(row_cosine(validation_j, validation.next_j).mean())
        accuracy = float(
            np.mean(np.argmax(validation_action, axis=1) == validation.next_action)
        )
        score = cosine + 0.02 * accuracy
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_cosine": cosine,
                "validation_semantic_accuracy": accuracy,
                "selection_score": score,
            }
        )
        if score > best + 1e-5:
            best = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(section["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("one-step predictor produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _save_predictor(
    context: Any,
    model: OneStepPredictor,
    *,
    label: str,
    architecture: str,
    use_history: bool,
    aux_dim: int,
) -> tuple[str, str]:
    directory = context.root / "artifacts/peripheral/v5" / context.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "use_history": use_history,
            "aux_dim": aux_dim,
            "width": int(context.config["peripheral_v5"]["hidden_width"]),
            "history_length": int(context.config["peripheral_v5"]["history_length"]),
        },
        path,
    )
    return str(path.relative_to(context.root)), sha256_file(path)


def _bootstrap_gain(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    keys = ["seed", "example_id", "step_index"]
    paired = left.merge(right, on=keys, suffixes=("_left", "_right"))
    paired["gain"] = paired["next_j_cosine_right"] - paired["next_j_cosine_left"]
    paired["action_gain"] = paired["semantic_correct_right"].astype(float) - paired[
        "semantic_correct_left"
    ].astype(float)
    paired["cluster"] = paired["seed"].astype(str) + ":" + paired["example_id"]
    ci = clustered_bootstrap_ci(
        paired,
        cluster_col="cluster",
        value_col="gain",
        n_resamples=resamples,
        seed=seed,
    )
    by_family = {}
    for family, group in paired.groupby("family_left", sort=True):
        family_ci = clustered_bootstrap_ci(
            group,
            cluster_col="cluster",
            value_col="gain",
            n_resamples=resamples,
            seed=seed,
        )
        by_family[str(family)] = {
            "cosine_gain": asdict(family_ci),
            "semantic_accuracy_gain": float(group["action_gain"].mean()),
            "n_transitions": int(len(group)),
            "n_trajectories": int(group["cluster"].nunique()),
        }
    return {
        "cosine_gain": asdict(ci),
        "semantic_accuracy_gain": float(paired["action_gain"].mean()),
        "n_transitions": int(len(paired)),
        "n_trajectory_seed_clusters": int(paired["cluster"].nunique()),
        "by_family": by_family,
    }


def _fit_remainder(
    context: Any, freeze: dict[str, Any]
) -> tuple[RemainderTransform, dict[str, TransitionArrays]]:
    history = int(context.config["peripheral_v5"]["history_length"])
    domains = {
        domain: load_domain(context.root, freeze, domain, history)
        for domain in ("train", "validation", "causal_fidelity", "rollout_test")
    }
    transform = RemainderTransform.fit(
        domains["train"].current_j,
        domains["train"].current_h,
        rank=int(context.config["peripheral_v5"]["residual_conditioning_rank"]),
        seed=int(context.config["peripheral_v5"]["screen_seed"]),
    )
    directory = context.root / "artifacts/peripheral/v5" / context.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "remainder_transform.npz"
    transform.save(path)
    return transform, domains


def run_ceiling(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    transform, domains = _fit_remainder(context, freeze)
    remainders = {
        key: transform.transform(value.current_j, value.current_h)
        for key, value in domains.items()
    }
    section = context.config["peripheral_v5"]
    screen_seed = int(section["screen_seed"])
    candidates = [
        ("j_only", "mlp", False),
        ("j_history", "attention", True),
        *[
            ("full_remainder", str(value), False)
            for value in section["reference_architectures"]
        ],
    ]
    screen: list[dict[str, Any]] = []
    for input_state, architecture, use_history in candidates:
        train_aux = remainders["train"] if input_state == "full_remainder" else None
        validation_aux = (
            remainders["validation"] if input_state == "full_remainder" else None
        )
        model, training = _train_predictor(
            context,
            domains["train"],
            domains["validation"],
            train_auxiliary=train_aux,
            validation_auxiliary=validation_aux,
            architecture=architecture,
            use_history=use_history,
            seed=screen_seed,
        )
        predicted, actions = _predict(
            model,
            domains["validation"],
            validation_aux,
            device=_device(context),
            use_history=use_history,
            batch_size=int(section["batch_size"]),
        )
        cosine = float(row_cosine(predicted, domains["validation"].next_j).mean())
        accuracy = float(
            np.mean(np.argmax(actions, axis=1) == domains["validation"].next_action)
        )
        screen.append(
            {
                "input_state": input_state,
                "architecture": architecture,
                "use_history": use_history,
                "validation_next_j_cosine": cosine,
                "validation_semantic_accuracy": accuracy,
                "selection_score": cosine + 0.02 * accuracy,
                "parameter_count": parameter_count(model),
                "training": training,
            }
        )
    baseline_choice = max(
        (value for value in screen if value["input_state"] != "full_remainder"),
        key=lambda value: value["selection_score"],
    )
    full_choice = max(
        (value for value in screen if value["input_state"] == "full_remainder"),
        key=lambda value: value["selection_score"],
    )
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    for seed in (int(value) for value in section["confirmation_seeds"]):
        for choice in (baseline_choice, full_choice):
            is_full = choice["input_state"] == "full_remainder"
            train_aux = remainders["train"] if is_full else None
            validation_aux = remainders["validation"] if is_full else None
            test_aux = remainders["rollout_test"] if is_full else None
            model, training = _train_predictor(
                context,
                domains["train"],
                domains["validation"],
                train_auxiliary=train_aux,
                validation_auxiliary=validation_aux,
                architecture=str(choice["architecture"]),
                use_history=bool(choice["use_history"]),
                seed=seed,
            )
            predicted, actions = _predict(
                model,
                domains["rollout_test"],
                test_aux,
                device=_device(context),
                use_history=bool(choice["use_history"]),
                batch_size=int(section["batch_size"]),
            )
            frame, rows = _evaluate_predictions(
                domains["rollout_test"],
                predicted,
                actions,
                run_id=context.run_id,
                architecture=str(choice["architecture"]),
                input_state=str(choice["input_state"]),
                seed=seed,
                split="rollout_test",
                parameters=parameter_count(model),
            )
            all_rows.append(frame)
            summaries.extend(rows)
            path, digest = _save_predictor(
                context,
                model,
                label=f"{choice['input_state']}-{choice['architecture']}-s{seed}",
                architecture=str(choice["architecture"]),
                use_history=bool(choice["use_history"]),
                aux_dim=0 if train_aux is None else train_aux.shape[1],
            )
            confirmations.append(
                {
                    "input_state": choice["input_state"],
                    "architecture": choice["architecture"],
                    "use_history": choice["use_history"],
                    "seed": seed,
                    "checkpoint": path,
                    "checkpoint_sha256": digest,
                    "training": training,
                }
            )
    frame = pd.concat(all_rows, ignore_index=True)
    baseline_frame = frame[frame["input_state"] == baseline_choice["input_state"]]
    full_frame = frame[frame["input_state"] == "full_remainder"]
    gain = _bootstrap_gain(
        baseline_frame,
        full_frame,
        seed=int(context.config["reproducibility"]["bootstrap_seed"]),
        resamples=int(section["bootstrap_resamples"]),
    )
    ceiling_authorized = bool(
        gain["cosine_gain"]["lower"] > 0
        and gain["cosine_gain"]["estimate"]
        >= float(section["full_remainder_minimum_cosine_gain"])
        and gain["semantic_accuracy_gain"]
        >= -float(section["maximum_action_accuracy_loss"])
    )
    records_path = context.processed_dir / "full_remainder_reference_v5.parquet"
    frame.to_parquet(records_path, index=False, compression="zstd")
    transform_path = (
        context.root
        / "artifacts/peripheral/v5"
        / context.run_id
        / "remainder_transform.npz"
    )
    summary = {
        "schema_version": 7,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "operational_remainder_definition": "standardized hidden state minus train-fitted rank-512 J-to-hidden prediction",
        "screen": screen,
        "baseline_choice": baseline_choice,
        "full_remainder_choice": full_choice,
        "confirmations": confirmations,
        "summaries": summaries,
        "gain": gain,
        "full_remainder_ceiling_authorized": ceiling_authorized,
        "records": str(records_path.relative_to(context.root)),
        "remainder_transform": str(transform_path.relative_to(context.root)),
        "remainder_transform_sha256": sha256_file(transform_path),
        "source_freeze_digest": freeze["freeze_digest"],
    }
    output = context.processed_dir / "full_remainder_reference_v5.json"
    write_json_atomic(output, summary)
    return summary


def _load_ceiling(context: Any) -> dict[str, Any]:
    return json.loads(
        (context.processed_dir / "full_remainder_reference_v5.json").read_text(
            encoding="utf-8"
        )
    )


def _load_transform(context: Any, ceiling: dict[str, Any]) -> RemainderTransform:
    path = context.root / ceiling["remainder_transform"]
    if sha256_file(path) != ceiling["remainder_transform_sha256"]:
        raise RuntimeError("v5 remainder transform hash mismatch")
    return RemainderTransform.load(path)


def _train_composite(
    context: Any,
    train: TransitionArrays,
    validation: TransitionArrays,
    train_r: np.ndarray,
    validation_r: np.ndarray,
    *,
    dimension: int,
    nonlinear: bool,
    seed: int,
) -> tuple[PeripheralComposite, list[dict[str, float]]]:
    section = context.config["peripheral_v5"]
    device = _device(context)
    torch.manual_seed(seed)
    encoder = PeripheralEncoder(
        train_r.shape[1],
        dimension,
        train.current_j.shape[1],
        train.u.shape[1],
        nonlinear=nonlinear,
    )
    predictor = OneStepPredictor(
        train.current_j.shape[1],
        train.u.shape[1],
        dimension,
        int(section["hidden_width"]),
        ACTION_COUNT,
        architecture="gated_mlp",
    )
    model = PeripheralComposite(encoder, predictor).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    best = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    batch_size = int(section["batch_size"])
    for epoch in range(int(section["maximum_epochs"])):
        model.train()
        losses = []
        for selected in _batches(len(train), batch_size, generator):
            j = torch.from_numpy(train.current_j[selected]).float().to(device)
            target_j = torch.from_numpy(train.next_j[selected]).float().to(device)
            u = torch.from_numpy(train.u[selected]).float().to(device)
            r = torch.from_numpy(train_r[selected]).float().to(device)
            target_action = (
                torch.from_numpy(train.next_action[selected]).long().to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            predicted, logits, compact = model(j, u, r)
            cosine_loss = (1 - F.cosine_similarity(predicted, target_j, dim=-1)).mean()
            semantic_loss = F.cross_entropy(logits, target_action)
            regularizer = bottleneck_regularizer(compact)
            loss = cosine_loss + 0.25 * semantic_loss + 0.05 * regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predicted, logits, _ = _predict_composite(
            model, validation, validation_r, device=device, batch_size=batch_size
        )
        cosine = float(row_cosine(predicted, validation.next_j).mean())
        accuracy = float(np.mean(np.argmax(logits, axis=1) == validation.next_action))
        score = cosine + 0.02 * accuracy
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_cosine": cosine,
                "validation_semantic_accuracy": accuracy,
                "selection_score": score,
            }
        )
        if score > best + 1e-5:
            best = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(section["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("peripheral bottleneck produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _predict_composite(
    model: PeripheralComposite,
    data: TransitionArrays,
    remainder: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    states, actions, compact = [], [], []
    for start in range(0, len(data), batch_size):
        stop = start + batch_size
        output = model(
            torch.from_numpy(data.current_j[start:stop]).float().to(device),
            torch.from_numpy(data.u[start:stop]).float().to(device),
            torch.from_numpy(remainder[start:stop]).float().to(device),
        )
        states.append(output[0].cpu().numpy())
        actions.append(output[1].cpu().numpy())
        compact.append(output[2].cpu().numpy())
    return np.concatenate(states), np.concatenate(actions), np.concatenate(compact)


def _ceiling_means(ceiling: dict[str, Any]) -> tuple[float, float]:
    pooled = [
        value
        for value in ceiling["summaries"]
        if value["family"] == "pooled" and value["split"] == "rollout_test"
    ]
    baseline_name = ceiling["baseline_choice"]["input_state"]
    baseline = np.mean(
        [
            value["next_j_cosine"]
            for value in pooled
            if value["input_state"] == baseline_name
        ]
    )
    full = np.mean(
        [
            value["next_j_cosine"]
            for value in pooled
            if value["input_state"] == "full_remainder"
        ]
    )
    return float(baseline), float(full)


def run_sweep(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    ceiling = _load_ceiling(context)
    transform = _load_transform(context, ceiling)
    section = context.config["peripheral_v5"]
    history = int(section["history_length"])
    domains = {
        key: load_domain(context.root, freeze, key, history)
        for key in ("train", "validation", "rollout_test")
    }
    remainder = {
        key: transform.transform(value.current_j, value.current_h)
        for key, value in domains.items()
    }
    baseline, full = _ceiling_means(ceiling)
    denominator = full - baseline
    records: list[dict[str, Any]] = []
    artifact_root = context.root / "artifacts/peripheral/v5" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    seed = int(section["screen_seed"])
    for dimension in (int(value) for value in section["dimensions"]):
        for encoder_family in section["encoder_families"]:
            if encoder_family == "pca_remainder":
                pca = PCA(
                    min(dimension, len(remainder["train"]) - 1),
                    svd_solver="randomized",
                    random_state=seed,
                ).fit(remainder["train"])
                compact = {
                    key: pca.transform(value).astype(np.float32)
                    for key, value in remainder.items()
                }
                model, training = _train_predictor(
                    context,
                    domains["train"],
                    domains["validation"],
                    train_auxiliary=compact["train"],
                    validation_auxiliary=compact["validation"],
                    architecture="gated_mlp",
                    use_history=False,
                    seed=seed,
                )
                predicted, actions = _predict(
                    model,
                    domains["rollout_test"],
                    compact["rollout_test"],
                    device=_device(context),
                    use_history=False,
                    batch_size=int(section["batch_size"]),
                )
                path = artifact_root / f"{encoder_family}-{dimension}.pt"
                torch.save(
                    {
                        "kind": "pca",
                        "dimension": dimension,
                        "pca_mean": pca.mean_.astype(np.float32),
                        "pca_components": pca.components_.astype(np.float32),
                        "predictor_state": model.state_dict(),
                        "predictor_architecture": "gated_mlp",
                    },
                    path,
                )
                parameters = parameter_count(model)
                validation_predicted, validation_actions = _predict(
                    model,
                    domains["validation"],
                    compact["validation"],
                    device=_device(context),
                    use_history=False,
                    batch_size=int(section["batch_size"]),
                )
            else:
                nonlinear = encoder_family == "nonlinear_bottleneck"
                composite, training = _train_composite(
                    context,
                    domains["train"],
                    domains["validation"],
                    remainder["train"],
                    remainder["validation"],
                    dimension=dimension,
                    nonlinear=nonlinear,
                    seed=seed,
                )
                predicted, actions, _ = _predict_composite(
                    composite,
                    domains["rollout_test"],
                    remainder["rollout_test"],
                    device=_device(context),
                    batch_size=int(section["batch_size"]),
                )
                validation_predicted, validation_actions, _ = _predict_composite(
                    composite,
                    domains["validation"],
                    remainder["validation"],
                    device=_device(context),
                    batch_size=int(section["batch_size"]),
                )
                path = artifact_root / f"{encoder_family}-{dimension}.pt"
                torch.save(
                    {
                        "kind": "nonlinear" if nonlinear else "linear",
                        "dimension": dimension,
                        "encoder_state": composite.encoder.state_dict(),
                        "predictor_state": composite.predictor.state_dict(),
                    },
                    path,
                )
                parameters = parameter_count(composite)
            cosine = row_cosine(predicted, domains["rollout_test"].next_j)
            validation_cosine = row_cosine(
                validation_predicted, domains["validation"].next_j
            )
            semantic = np.argmax(actions, axis=1) == domains["rollout_test"].next_action
            validation_semantic = (
                np.argmax(validation_actions, axis=1)
                == domains["validation"].next_action
            )
            gap = (
                (float(cosine.mean()) - baseline) / denominator
                if denominator > 1e-8
                else None
            )
            family_metrics = {}
            for family in sorted(set(domains["rollout_test"].family)):
                mask = domains["rollout_test"].family == family
                family_metrics[str(family)] = {
                    "next_j_cosine": float(cosine[mask].mean()),
                    "semantic_accuracy": float(semantic[mask].mean()),
                    "n_transitions": int(mask.sum()),
                    "n_trajectories": int(
                        len(set(domains["rollout_test"].example_id[mask]))
                    ),
                }
            trajectory_metrics = _teacher_current_trajectory_metrics(
                domains["rollout_test"], cosine, semantic
            )
            records.append(
                {
                    **PeripheralStateRecord(
                        run_id=context.run_id,
                        encoder_family=str(encoder_family),
                        state_dimension=dimension,
                        split="rollout_test",
                        next_j_cosine=float(cosine.mean()),
                        semantic_accuracy=float(semantic.mean()),
                        gap_closed=gap,
                        conditional_residual_gain=None,
                        causal_direction_cosine=None,
                        causal_magnitude_ratio=None,
                        output_sign_agreement=None,
                        authorized=False,
                        parameter_count=parameters,
                        family_metrics=family_metrics,
                    ).to_dict(),
                    "validation_next_j_cosine": float(validation_cosine.mean()),
                    "validation_semantic_accuracy": float(validation_semantic.mean()),
                    "teacher_current_trajectory_metrics": trajectory_metrics,
                    "training": training,
                    "artifact": str(path.relative_to(context.root)),
                    "artifact_sha256": sha256_file(path),
                }
            )
    selected_by_dimension = {}
    for dimension in section["dimensions"]:
        candidates = [
            value for value in records if value["state_dimension"] == int(dimension)
        ]
        selected_by_dimension[str(dimension)] = max(
            candidates,
            key=lambda value: (
                value["validation_next_j_cosine"]
                + 0.02 * value["validation_semantic_accuracy"],
                -value["parameter_count"],
            ),
        )["encoder_family"]
    summary = {
        "schema_version": 7,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "full_remainder_ceiling_authorized": ceiling[
            "full_remainder_ceiling_authorized"
        ],
        "j_only_mean": baseline,
        "full_remainder_mean": full,
        "records": records,
        "selected_by_dimension": selected_by_dimension,
        "conditional_sufficiency_complete": False,
        "causal_fidelity_complete": False,
    }
    output = context.processed_dir / "peripheral_dimension_sweep_v5.json"
    write_json_atomic(output, summary)
    return summary


def _load_candidate(
    context: Any, record: dict[str, Any], remainder_dim: int
) -> tuple[Any, OneStepPredictor]:
    path = context.root / record["artifact"]
    if sha256_file(path) != record["artifact_sha256"]:
        raise RuntimeError("peripheral candidate artifact hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    dimension = int(value["dimension"])
    predictor = OneStepPredictor(
        4096,
        U_DIM,
        dimension,
        int(context.config["peripheral_v5"]["hidden_width"]),
        ACTION_COUNT,
        architecture="gated_mlp",
    )
    predictor.load_state_dict(value["predictor_state"])
    if value["kind"] == "pca":
        encoder: Any = {
            "kind": "pca",
            "mean": np.asarray(value["pca_mean"]),
            "components": np.asarray(value["pca_components"]),
        }
    else:
        encoder = PeripheralEncoder(
            remainder_dim,
            dimension,
            4096,
            U_DIM,
            nonlinear=value["kind"] == "nonlinear",
        )
        encoder.load_state_dict(value["encoder_state"])
    return encoder, predictor


def _encode_candidate(
    encoder: Any,
    remainder: np.ndarray,
    measured_j: np.ndarray,
    u: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if isinstance(encoder, dict):
        return ((remainder - encoder["mean"]) @ encoder["components"].T).astype(
            np.float32
        )
    return encode_compact(encoder.to(device), remainder, measured_j, u, device)


def run_conditional(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    sweep_path = context.processed_dir / "peripheral_dimension_sweep_v5.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    ceiling = _load_ceiling(context)
    transform = _load_transform(context, ceiling)
    section = context.config["peripheral_v5"]
    history = int(section["history_length"])
    domains = {
        key: load_domain(context.root, freeze, key, history)
        for key in ("train", "validation", "rollout_test")
    }
    remainder = {
        key: transform.transform(value.current_j, value.current_h)
        for key, value in domains.items()
    }
    updated = []
    conditional_rows = []
    seed = int(section["screen_seed"])
    for record in sweep["records"]:
        if (
            sweep["selected_by_dimension"][str(record["state_dimension"])]
            != record["encoder_family"]
        ):
            updated.append(record)
            continue
        encoder, _ = _load_candidate(context, record, remainder["train"].shape[1])
        compact = {
            key: _encode_candidate(
                encoder,
                remainder[key],
                domains[key].current_j,
                domains[key].u,
                _device(context),
            )
            for key in domains
        }
        decoder = Ridge(alpha=1.0, solver="lsqr").fit(
            compact["train"], remainder["train"]
        )
        residual = {
            key: remainder[key] - decoder.predict(compact[key]) for key in domains
        }
        pca = PCA(
            min(256, len(residual["train"]) - 1, residual["train"].shape[1]),
            svd_solver="randomized",
            random_state=seed,
        ).fit(residual["train"])
        auxiliary = {
            key: np.concatenate(
                (compact[key], pca.transform(residual[key])), axis=1
            ).astype(np.float32)
            for key in domains
        }
        model, training = _train_predictor(
            context,
            domains["train"],
            domains["validation"],
            train_auxiliary=auxiliary["train"],
            validation_auxiliary=auxiliary["validation"],
            architecture="gated_mlp",
            use_history=False,
            seed=seed,
        )
        predicted, actions = _predict(
            model,
            domains["rollout_test"],
            auxiliary["rollout_test"],
            device=_device(context),
            use_history=False,
            batch_size=int(section["batch_size"]),
        )
        cosine_values = row_cosine(predicted, domains["rollout_test"].next_j)
        semantic_values = (
            np.argmax(actions, axis=1) == domains["rollout_test"].next_action
        )
        cosine = float(cosine_values.mean())
        gain = cosine - float(record["next_j_cosine"])
        by_family = {}
        for family in sorted(set(domains["rollout_test"].family)):
            mask = domains["rollout_test"].family == family
            base = record["family_metrics"][str(family)]["next_j_cosine"]
            augmented = float(cosine_values[mask].mean())
            by_family[str(family)] = {
                "base_next_j_cosine": float(base),
                "conditional_augmented_next_j_cosine": augmented,
                "conditional_residual_gain": augmented - float(base),
                "conditional_augmented_semantic_accuracy": float(
                    semantic_values[mask].mean()
                ),
                "n_transitions": int(mask.sum()),
            }
        new_record = {
            **record,
            "conditional_residual_gain": gain,
            "conditional_augmented_next_j_cosine": cosine,
            "conditional_augmented_semantic_accuracy": float(semantic_values.mean()),
            "conditional_by_family": by_family,
        }
        updated.append(new_record)
        conditional_rows.append(
            {
                "state_dimension": int(record["state_dimension"]),
                "encoder_family": record["encoder_family"],
                "base_next_j_cosine": record["next_j_cosine"],
                "conditional_augmented_next_j_cosine": cosine,
                "conditional_residual_gain": gain,
                "by_family": by_family,
                "training": training,
            }
        )
    sweep["records"] = updated
    sweep["conditional_sufficiency_complete"] = True
    sweep["conditional_results"] = conditional_rows
    write_json_atomic(sweep_path, sweep)
    return sweep


def run_fidelity(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    del freeze
    sweep_path = context.processed_dir / "peripheral_dimension_sweep_v5.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    replication = json.loads(
        (context.processed_dir / "h2_replication_v5.json").read_text(encoding="utf-8")
    )
    artifact_path = context.root / replication["causal_state_artifact"]
    if sha256_file(artifact_path) != replication["causal_state_artifact_sha256"]:
        raise RuntimeError("H2 replication causal-state artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as payload:
        clean_h = payload["clean_hidden"].astype(np.float32)
        changed_h = payload["intervened_hidden"].astype(np.float32)
        clean_j = payload["current_j_clean"].astype(np.float32)
        changed_j = payload["current_j_intervened"].astype(np.float32)
        teacher_clean = payload["next_j_clean"].astype(np.float32)
        teacher_changed = payload["next_j_intervened"].astype(np.float32)
        u = payload["u"].astype(np.float32)
        teacher_output_delta = payload["teacher_output_delta"].astype(np.float32)
        causal_families = payload["family"].astype("U32")
        causal_base_ids = payload["base_trial_id"].astype("U64")
        next_actions = payload["next_action"].astype(np.int64)
        teacher_action_clean = payload["teacher_next_action_clean"].astype(np.int64)
        teacher_action_changed = payload["teacher_next_action_intervened"].astype(
            np.int64
        )
    ceiling = _load_ceiling(context)
    transform = _load_transform(context, ceiling)
    clean_r = transform.transform(clean_j, clean_h)
    changed_r = transform.transform(changed_j, changed_h)
    teacher_delta = teacher_changed - teacher_clean
    updated = []
    fidelity_rows = []
    device = _device(context)
    minimum_direction = float(
        context.config["peripheral_v5"]["compact_minimum_causal_direction_cosine"]
    )
    minimum_sign = float(
        context.config["peripheral_v5"]["compact_minimum_output_sign_agreement"]
    )
    for record in sweep["records"]:
        if (
            sweep["selected_by_dimension"].get(str(record["state_dimension"]))
            != record["encoder_family"]
        ):
            updated.append(record)
            continue
        encoder, predictor = _load_candidate(context, record, clean_r.shape[1])
        predictor = predictor.to(device).eval()
        clean_c = _encode_candidate(encoder, clean_r, clean_j, u, device)
        changed_c = _encode_candidate(encoder, changed_r, changed_j, u, device)
        dummy = TransitionArrays(
            current_j=clean_j,
            next_j=teacher_clean,
            current_h=clean_h,
            next_h=clean_h,
            current_action=np.zeros(len(clean_j), dtype=np.int64),
            next_action=np.zeros(len(clean_j), dtype=np.int64),
            u=u,
            next_u=u,
            history_j=np.zeros((len(clean_j), 1, clean_j.shape[1]), dtype=np.float32),
            history_mask=np.ones((len(clean_j), 1), dtype=np.float32),
            example_id=np.arange(len(clean_j)).astype("U16"),
            family=causal_families,
            step_index=np.zeros(len(clean_j), dtype=np.int16),
            horizon=np.ones(len(clean_j), dtype=np.int16),
        )
        clean_prediction, clean_logits = _predict(
            predictor,
            dummy,
            clean_c,
            device=device,
            use_history=False,
            batch_size=256,
        )
        changed_dummy = TransitionArrays(**{**dummy.__dict__, "current_j": changed_j})
        changed_prediction, changed_logits = _predict(
            predictor,
            changed_dummy,
            changed_c,
            device=device,
            use_history=False,
            batch_size=256,
        )
        predicted_delta = changed_prediction - clean_prediction
        directions = row_cosine(predicted_delta, teacher_delta)
        magnitude = np.linalg.norm(predicted_delta, axis=1) / np.maximum(
            np.linalg.norm(teacher_delta, axis=1), 1e-12
        )
        rows = np.arange(len(next_actions))
        predicted_output_delta = (
            changed_logits[rows, next_actions] - clean_logits[rows, next_actions]
        )
        sign_values = (
            np.sign(predicted_output_delta) == np.sign(teacher_output_delta)
        ).astype(float)
        predicted_action_clean = clean_logits.argmax(axis=1)
        predicted_action_changed = changed_logits.argmax(axis=1)
        semantic_delta_values = (
            (predicted_action_clean != predicted_action_changed)
            == (teacher_action_clean != teacher_action_changed)
        ).astype(float)
        sign = np.mean(sign_values)
        direction = float(np.median(directions))
        ratio = float(np.median(magnitude))
        fidelity_frame = pd.DataFrame(
            {
                "base_trial_id": causal_base_ids,
                "family": causal_families,
                "direction_cosine": directions,
                "magnitude_ratio": magnitude,
                "semantic_delta_agreement": semantic_delta_values,
                "output_sign_agreement": sign_values,
            }
        ).replace([np.inf, -np.inf], np.nan)
        by_family: dict[str, Any] = {}
        groups = [
            ("pooled", fidelity_frame),
            *list(fidelity_frame.groupby("family", sort=True)),
        ]
        statistics: dict[str, Callable[[np.ndarray], float]] = {
            "direction_cosine": lambda values: float(np.median(values)),
            "magnitude_ratio": lambda values: float(np.median(values)),
            "semantic_delta_agreement": lambda values: float(np.mean(values)),
            "output_sign_agreement": lambda values: float(np.mean(values)),
        }
        for family, group in groups:
            metric_values = {}
            for metric, statistic in statistics.items():
                metric_values[metric] = asdict(
                    clustered_bootstrap_ci(
                        group,
                        cluster_col="base_trial_id",
                        value_col=metric,
                        statistic=statistic,
                        n_resamples=int(
                            context.config["peripheral_v5"]["bootstrap_resamples"]
                        ),
                        seed=int(context.config["reproducibility"]["bootstrap_seed"]),
                    )
                )
            by_family[str(family)] = metric_values
        new_record = {
            **record,
            "causal_direction_cosine": direction,
            "causal_magnitude_ratio": ratio,
            "output_sign_agreement": float(sign),
            "semantic_delta_agreement": float(semantic_delta_values.mean()),
            "causal_fidelity_by_family": by_family,
        }
        updated.append(new_record)
        fidelity_rows.append(
            {
                "state_dimension": int(record["state_dimension"]),
                "encoder_family": record["encoder_family"],
                "causal_direction_cosine": direction,
                "causal_magnitude_ratio": ratio,
                "output_sign_agreement": float(sign),
                "semantic_delta_agreement": float(semantic_delta_values.mean()),
                "n_interventions": int(len(directions)),
                "by_family": by_family,
            }
        )
    authorized = []
    for record in updated:
        selected = (
            sweep["selected_by_dimension"].get(str(record["state_dimension"]))
            == record["encoder_family"]
        )
        passed = bool(
            selected
            and sweep["full_remainder_ceiling_authorized"]
            and record.get("gap_closed") is not None
            and record["gap_closed"]
            >= float(context.config["peripheral_v5"]["compact_gap_closed"])
            and record.get("conditional_residual_gain") is not None
            and record["conditional_residual_gain"]
            <= float(
                context.config["peripheral_v5"]["compact_maximum_conditional_gain"]
            )
            and record.get("causal_direction_cosine") is not None
            and record["causal_direction_cosine"] >= minimum_direction
            and record.get("output_sign_agreement") is not None
            and record["output_sign_agreement"] >= minimum_sign
        )
        record["authorized"] = passed
        if passed:
            authorized.append(record)
    sweep["records"] = updated
    sweep["causal_fidelity_complete"] = True
    sweep["causal_fidelity"] = fidelity_rows
    sweep["authorized_candidates"] = authorized
    sweep["compact_peripheral_state_authorized"] = bool(authorized)
    write_json_atomic(sweep_path, sweep)
    return sweep


def _matched_recurrent_model(
    context: Any, architecture: str, compact_dim: int
) -> RecurrentPeripheralController:
    section = context.config["peripheral_v5"]
    target = int(section["recurrent_parameter_target"])
    candidates = []
    for width in range(128, 1025, 16):
        model = RecurrentPeripheralController(
            4096,
            compact_dim,
            U_DIM,
            ACTION_COUNT,
            width,
            architecture=architecture,
        )
        candidates.append((abs(parameter_count(model) - target), model))
    model = min(candidates, key=lambda value: value[0])[1]
    relative = abs(parameter_count(model) - target) / target
    if relative > float(section["recurrent_parameter_tolerance"]):
        raise RuntimeError(
            f"cannot parameter-match {architecture}: relative error {relative:.4f}"
        )
    return model


def _compact_domains(
    context: Any,
    freeze: dict[str, Any],
    record: dict[str, Any],
) -> tuple[
    dict[str, TransitionArrays],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    ceiling = _load_ceiling(context)
    transform = _load_transform(context, ceiling)
    history = int(context.config["peripheral_v5"]["history_length"])
    domains = {
        key: load_domain(context.root, freeze, key, history)
        for key in ("train", "validation", "rollout_test")
    }
    encoder, _ = _load_candidate(context, record, domains["train"].current_h.shape[1])
    current: dict[str, np.ndarray] = {}
    following: dict[str, np.ndarray] = {}
    for key, domain in domains.items():
        current_r = transform.transform(domain.current_j, domain.current_h)
        next_r = transform.transform(domain.next_j, domain.next_h)
        current[key] = _encode_candidate(
            encoder, current_r, domain.current_j, domain.u, _device(context)
        )
        following[key] = _encode_candidate(
            encoder, next_r, domain.next_j, domain.next_u, _device(context)
        )
    mean = current["train"].mean(axis=0).astype(np.float32)
    scale = current["train"].std(axis=0).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    for key in current:
        current[key] = ((current[key] - mean) / scale).astype(np.float32)
        following[key] = ((following[key] - mean) / scale).astype(np.float32)
    return domains, current, following, mean, scale


def _group_indices(data: TransitionArrays) -> list[np.ndarray]:
    frame = pd.DataFrame(
        {
            "index": np.arange(len(data)),
            "example_id": data.example_id,
            "step": data.step_index,
        }
    )
    return [
        group.sort_values("step")["index"].to_numpy(dtype=np.int64)
        for _, group in frame.groupby("example_id", sort=True)
    ]


def _train_recurrent(
    context: Any,
    architecture: str,
    compact_dim: int,
    train: TransitionArrays,
    train_c: np.ndarray,
    train_next_c: np.ndarray,
    validation: TransitionArrays,
    validation_c: np.ndarray,
    validation_next_c: np.ndarray,
    *,
    seed: int,
) -> tuple[RecurrentPeripheralController, list[dict[str, float]]]:
    section = context.config["peripheral_v5"]
    device = _device(context)
    torch.manual_seed(seed)
    model = _matched_recurrent_model(context, architecture, compact_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    groups = _group_indices(train)
    generator = np.random.default_rng(seed)
    best = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    epochs = int(section["maximum_epochs"])
    for epoch in range(epochs):
        generator.shuffle(groups)
        warmup = max(1, round(0.2 * epochs))
        feedback = (
            0.0
            if epoch < warmup
            else min(0.8, 0.8 * (epoch - warmup) / max(1, epochs - warmup - 1))
        )
        losses = []
        model.train()
        for indices in groups:
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            predicted_j: torch.Tensor | None = None
            predicted_c: torch.Tensor | None = None
            predicted_action: int | None = None
            for position, index in enumerate(indices):
                use_prediction = (
                    position > 0
                    and predicted_j is not None
                    and predicted_c is not None
                    and generator.random() < feedback
                )
                if use_prediction:
                    current_j = predicted_j
                    compact = predicted_c
                else:
                    current_j = (
                        torch.from_numpy(train.current_j[index : index + 1])
                        .float()
                        .to(device)
                    )
                    compact = (
                        torch.from_numpy(train_c[index : index + 1]).float().to(device)
                    )
                action_for_u = (
                    predicted_action
                    if use_prediction and predicted_action is not None
                    else int(train.current_action[index])
                )
                u_value = build_u(
                    np.asarray([action_for_u]),
                    train.family[index : index + 1],
                    train.step_index[index : index + 1],
                    train.horizon[index : index + 1],
                )
                u = torch.from_numpy(u_value).float().to(device)
                target_j = (
                    torch.from_numpy(train.next_j[index : index + 1]).float().to(device)
                )
                target_c = (
                    torch.from_numpy(train_next_c[index : index + 1]).float().to(device)
                )
                target_action = (
                    torch.from_numpy(train.next_action[index : index + 1])
                    .long()
                    .to(device)
                )
                predicted_j, predicted_c, logits = model(current_j, compact, u)
                assert predicted_j is not None and predicted_c is not None
                predicted_action = int(logits.argmax(-1).item())
                total = total + (
                    1
                    - F.cosine_similarity(predicted_j, target_j, dim=-1).mean()
                    + 0.25 * F.mse_loss(predicted_c, target_c)
                    + 0.25 * F.cross_entropy(logits, target_action)
                )
            loss = total / max(1, len(indices))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_rows = _evaluate_recurrent(
            model,
            validation,
            validation_c,
            validation_next_c,
            horizons=[1, 2, 4, 8],
            device=device,
        )
        available = validation_rows[validation_rows["horizon"] <= 8]
        score = float(
            0.6 * available["j_similarity"].mean()
            + 0.3 * available["c_similarity"].mean()
            + 0.1 * available["teacher_action_fidelity"].mean()
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "feedback_probability": float(feedback),
                "validation_score": score,
            }
        )
        if score > best + 1e-5:
            best = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(section["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("recurrent peripheral controller produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _evaluate_recurrent(
    model: RecurrentPeripheralController,
    data: TransitionArrays,
    compact: np.ndarray,
    next_compact: np.ndarray,
    *,
    horizons: list[int],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for indices in _group_indices(data):
        current_j = torch.from_numpy(data.current_j[indices[:1]]).float().to(device)
        current_c = torch.from_numpy(compact[indices[:1]]).float().to(device)
        predicted_action: int | None = None
        j_cosines: list[float] = []
        c_cosines: list[float] = []
        actions: list[bool] = []
        norms: list[float] = []
        for position, index in enumerate(indices):
            action_for_u = (
                int(data.current_action[index])
                if position == 0 or predicted_action is None
                else predicted_action
            )
            u_value = build_u(
                np.asarray([action_for_u]),
                data.family[index : index + 1],
                data.step_index[index : index + 1],
                data.horizon[index : index + 1],
            )
            u = torch.from_numpy(u_value).float().to(device)
            predicted_j, predicted_c, logits = model(current_j, current_c, u)
            predicted_action = int(logits.argmax(-1).item())
            target_j = data.next_j[index : index + 1]
            target_c = next_compact[index : index + 1]
            j_cosines.append(float(row_cosine(predicted_j.cpu().numpy(), target_j)[0]))
            c_cosines.append(float(row_cosine(predicted_c.cpu().numpy(), target_c)[0]))
            actions.append(predicted_action == int(data.next_action[index]))
            norms.append(float(torch.linalg.vector_norm(predicted_c).item()))
            current_j, current_c = predicted_j, predicted_c
        for horizon in horizons:
            if horizon > len(indices):
                continue
            prefix_j = np.asarray(j_cosines[:horizon])
            prefix_c = np.asarray(c_cosines[:horizon])
            below = np.flatnonzero(prefix_j < 0.8)
            rows.append(
                {
                    "example_id": str(data.example_id[indices[0]]),
                    "family": str(data.family[indices[0]]),
                    "horizon": int(horizon),
                    "j_similarity": float(prefix_j[-1]),
                    "c_similarity": float(prefix_c[-1]),
                    "j_trajectory_divergence": float(np.mean(1 - prefix_j)),
                    "c_trajectory_divergence": float(np.mean(1 - prefix_c)),
                    "teacher_action_fidelity": float(np.mean(actions[:horizon])),
                    "ground_truth_task_accuracy": float(np.mean(actions[:horizon])),
                    "final_accuracy": float(actions[horizon - 1]),
                    "time_to_divergence": int(below[0] + 1)
                    if len(below)
                    else int(horizon + 1),
                    "latent_norm": float(np.mean(norms[:horizon])),
                    "finite": bool(
                        np.isfinite(prefix_j).all() and np.isfinite(prefix_c).all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_recurrent(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    sweep = json.loads(
        (context.processed_dir / "peripheral_dimension_sweep_v5.json").read_text(
            encoding="utf-8"
        )
    )
    maximum = int(context.config["peripheral_v5"]["recurrent_maximum_state_dimension"])
    candidates = [
        value
        for value in sweep.get("authorized_candidates", [])
        if int(value["state_dimension"]) <= maximum
    ]
    if not candidates:
        summary = {
            "schema_version": 7,
            "protocol_version": PROTOCOL,
            "status": "GATED_NOT_AUTHORIZED",
            "reason": "no <=128D peripheral state passed ceiling, conditional-sufficiency, and causal-fidelity gates",
            "autonomous_rollout_executed": False,
        }
    else:
        selected = min(candidates, key=lambda value: value["state_dimension"])
        domains, current_c, next_c, compact_mean, compact_scale = _compact_domains(
            context, freeze, selected
        )
        section = context.config["peripheral_v5"]
        screen_seed = int(section["screen_seed"])
        screen = []
        screen_models = {}
        for architecture in section["recurrent_architectures"]:
            model, training = _train_recurrent(
                context,
                str(architecture),
                int(selected["state_dimension"]),
                domains["train"],
                current_c["train"],
                next_c["train"],
                domains["validation"],
                current_c["validation"],
                next_c["validation"],
                seed=screen_seed,
            )
            validation = _evaluate_recurrent(
                model,
                domains["validation"],
                current_c["validation"],
                next_c["validation"],
                horizons=[int(value) for value in section["recurrent_horizons"]],
                device=_device(context),
            )
            at_eight = validation[validation["horizon"] <= 8]
            score = float(
                0.6 * at_eight["j_similarity"].mean()
                + 0.3 * at_eight["c_similarity"].mean()
                + 0.1 * at_eight["teacher_action_fidelity"].mean()
            )
            screen.append(
                {
                    "architecture": architecture,
                    "validation_score": score,
                    "parameter_count": parameter_count(model),
                    "training": training,
                }
            )
            screen_models[str(architecture)] = model
        choice = max(screen, key=lambda value: value["validation_score"])
        results = []
        raw = []
        artifact_root = context.root / "artifacts/peripheral/v5" / context.run_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        for seed in (int(value) for value in section["confirmation_seeds"]):
            if seed == screen_seed:
                model = screen_models[str(choice["architecture"])]
                training = next(
                    value["training"]
                    for value in screen
                    if value["architecture"] == choice["architecture"]
                )
            else:
                model, training = _train_recurrent(
                    context,
                    str(choice["architecture"]),
                    int(selected["state_dimension"]),
                    domains["train"],
                    current_c["train"],
                    next_c["train"],
                    domains["validation"],
                    current_c["validation"],
                    next_c["validation"],
                    seed=seed,
                )
            test = _evaluate_recurrent(
                model,
                domains["rollout_test"],
                current_c["rollout_test"],
                next_c["rollout_test"],
                horizons=[int(value) for value in section["recurrent_horizons"]],
                device=_device(context),
            )
            test["seed"] = seed
            test["architecture"] = choice["architecture"]
            raw.append(test)
            results.append(
                {
                    "seed": seed,
                    "architecture": choice["architecture"],
                    "parameter_count": parameter_count(model),
                    "training": training,
                    "test_by_horizon": test.groupby("horizon", sort=True)
                    .agg(
                        n=("example_id", "size"),
                        j_similarity=("j_similarity", "median"),
                        c_similarity=("c_similarity", "median"),
                        j_trajectory_divergence=("j_trajectory_divergence", "mean"),
                        teacher_action_fidelity=("teacher_action_fidelity", "mean"),
                        ground_truth_task_accuracy=(
                            "ground_truth_task_accuracy",
                            "mean",
                        ),
                        final_accuracy=("final_accuracy", "mean"),
                        time_to_divergence=("time_to_divergence", "median"),
                        latent_norm=("latent_norm", "median"),
                        finite_rate=("finite", "mean"),
                    )
                    .reset_index()
                    .to_dict(orient="records"),
                }
            )
            torch.save(model.state_dict(), artifact_root / f"recurrent-s{seed}.pt")
        raw_path = context.processed_dir / "peripheral_recurrent_rollout_v5.parquet"
        pd.concat(raw, ignore_index=True).to_parquet(
            raw_path, index=False, compression="zstd"
        )
        summary = {
            "schema_version": 7,
            "protocol_version": PROTOCOL,
            "status": "COMPLETED",
            "selected_candidate": selected,
            "compact_normalization": {
                "mean": compact_mean.tolist(),
                "scale": compact_scale.tolist(),
            },
            "architecture_screen": screen,
            "selected_architecture": choice["architecture"],
            "results": results,
            "records": str(raw_path.relative_to(context.root)),
            "autonomous_rollout_executed": True,
        }
    write_json_atomic(context.processed_dir / "peripheral_recurrent_v5.json", summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Protocol-v5 peripheral-state experiment", "configs/peripheral_v5.yaml"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "freeze",
            "ceiling",
            "sweep",
            "conditional",
            "fidelity",
            "recurrent",
        ),
        required=True,
    )
    args = parser.parse_args()
    context = initialize_context("peripheral-v5", args)
    try:
        if args.stage == "prepare":
            if args.dry_run:
                context.finish("DRY_RUN", stage=args.stage)
                return
            result = run_prepare(context)
            context.finish(
                "COMPLETED",
                stage=args.stage,
                summary_status=result.get("status", "COMPLETED"),
            )
            return
        if args.stage == "freeze":
            freeze = build_peripheral_freeze(context.root, context.config)
            context.finish(
                "COMPLETED_FREEZE",
                freeze="artifacts/peripheral_v5.freeze.json",
                freeze_digest=freeze["freeze_digest"],
            )
            return
        freeze = verify_peripheral_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        runner = {
            "ceiling": run_ceiling,
            "sweep": run_sweep,
            "conditional": run_conditional,
            "fidelity": run_fidelity,
            "recurrent": run_recurrent,
        }[args.stage]
        result = runner(context, freeze)
        context.finish(
            "COMPLETED",
            stage=args.stage,
            summary_status=result.get("status", "COMPLETED"),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
