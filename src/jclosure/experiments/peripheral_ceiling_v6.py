"""Multi-endpoint strong peripheral-information ceiling for protocol v6."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.peripheral_v5 import _batches, load_domain
from jclosure.peripheral_v5 import (
    ACTION_COUNT,
    RemainderTransform,
    TransitionArrays,
    parameter_count,
    row_cosine,
)
from jclosure.peripheral_v6 import (
    PredictivePeripheralBottleneck,
    StrongPeripheralPredictor,
    fit_causal_directions,
    strong_multitask_loss,
)
from jclosure.protocol_v6 import verify_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v6 import PROTOCOL_V6


def _device(context: Any) -> torch.device:
    return torch.device(
        f"cuda:{int(context.config['model'].get('device', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )


def _load_endpoint(context: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary = json.loads(
        (context.processed_dir / "causal_endpoint_v6.json").read_text(encoding="utf-8")
    )
    path = context.root / summary["causal_endpoint_artifact"]
    if sha256_file(path) != summary["causal_endpoint_artifact_sha256"]:
        raise RuntimeError("v6 causal endpoint hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    records = pd.read_parquet(context.root / summary["endpoint_records"]).sort_values(
        "artifact_index"
    )
    arrays["teacher_output_sign"] = records["output_sign"].to_numpy(dtype=np.int8)
    return summary, arrays


def _load_transform(context: Any, summary: dict[str, Any]) -> RemainderTransform:
    path = context.root / summary["remainder_transform"]
    if sha256_file(path) != summary["remainder_transform_sha256"]:
        raise RuntimeError("v6 remainder transform hash mismatch")
    return RemainderTransform.load(path)


def _architecture_inputs(
    architecture: str,
    data: TransitionArrays,
    remainder: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    full = architecture.startswith("full_remainder")
    history = architecture == "j_history_attention"
    return (
        torch.from_numpy(remainder[indices]).float().to(device) if full else None,
        torch.from_numpy(data.history_j[indices]).float().to(device)
        if history
        else None,
        torch.from_numpy(data.history_mask[indices]).float().to(device)
        if history
        else None,
    )


@torch.no_grad()
def _predict(
    model: StrongPeripheralPredictor,
    data: TransitionArrays,
    remainder: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    states, actions = [], []
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(len(data), start + batch_size))
        auxiliary, history, mask = _architecture_inputs(
            model.architecture, data, remainder, indices, device
        )
        predicted, logits = model(
            torch.from_numpy(data.current_j[indices]).float().to(device),
            torch.from_numpy(data.u[indices]).float().to(device),
            auxiliary,
            history,
            mask,
        )
        states.append(predicted.cpu().numpy())
        actions.append(logits.cpu().numpy())
    return np.concatenate(states), np.concatenate(actions)


def _causal_data(arrays: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    selected = arrays["split"].astype("U16") == role
    return {
        "current_j_clean": arrays["j_clean"][selected, 0].astype(np.float32),
        "current_j_intervened": arrays["j_intervened"][selected, 0].astype(np.float32),
        "next_j_clean": arrays["j_clean"][selected, 1].astype(np.float32),
        "next_j_intervened": arrays["j_intervened"][selected, 1].astype(np.float32),
        "remainder_clean": arrays["remainder_clean"][selected].astype(np.float32),
        "remainder_intervened": arrays["remainder_intervened"][selected].astype(
            np.float32
        ),
        "u": arrays["u"][selected].astype(np.float32),
        "next_action": arrays["next_action"][selected].astype(np.int64),
        "family": arrays["family"][selected].astype("U32"),
        "prompt_id": arrays["prompt_id"][selected].astype("U64"),
        "base_trial_id": arrays["base_trial_id"][selected].astype("U64"),
        "teacher_delta": arrays["teacher_delta_j_f32"][selected].astype(np.float32),
        "teacher_semantic_delta": arrays["teacher_semantic_delta"][selected].astype(
            bool
        ),
        "teacher_output_sign": arrays["teacher_output_sign"][selected].astype(np.int8),
        "teacher_output_logit_delta": (
            arrays["logits_intervened"][selected, 0]
            - arrays["logits_clean"][selected, 0]
        ).astype(np.float32),
    }


def _fit_model(
    context: Any,
    architecture: str,
    train: TransitionArrays,
    validation: TransitionArrays,
    train_r: np.ndarray,
    validation_r: np.ndarray,
    causal_fit: dict[str, np.ndarray],
    directions: np.ndarray,
    top_dimensions: np.ndarray,
    *,
    seed: int,
) -> tuple[StrongPeripheralPredictor, list[dict[str, float]]]:
    section = context.config["peripheral_v6"]
    device = _device(context)
    torch.manual_seed(seed)
    model = StrongPeripheralPredictor(
        train.current_j.shape[1],
        train_r.shape[1],
        train.u.shape[1],
        int(section["reference_width"]),
        ACTION_COUNT,
        architecture=architecture,
        history_length=train.history_j.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["reference_learning_rate"]),
        weight_decay=float(section["reference_weight_decay"]),
    )
    direction_tensor = torch.from_numpy(directions).float().to(device)
    top_tensor = (
        torch.from_numpy(top_dimensions[: int(section["top_causal_dimension_count"])])
        .long()
        .to(device)
    )
    weights = section["objective_weights"]
    generator = np.random.default_rng(seed)
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    batch_size = int(section["reference_batch_size"])
    causal_count = len(causal_fit["u"])
    for epoch in range(int(section["reference_epochs"])):
        model.train()
        losses = []
        for indices in _batches(len(train), batch_size, generator):
            auxiliary, historical, mask = _architecture_inputs(
                architecture, train, train_r, indices, device
            )
            predicted, action = model(
                torch.from_numpy(train.current_j[indices]).float().to(device),
                torch.from_numpy(train.u[indices]).float().to(device),
                auxiliary,
                historical,
                mask,
            )
            loss = strong_multitask_loss(
                predicted,
                action,
                torch.from_numpy(train.next_j[indices]).float().to(device),
                torch.from_numpy(train.next_action[indices]).long().to(device),
                direction_tensor,
                top_tensor,
                weights,
            )
            # Counterfactual pairs are deliberately oversampled: their global
            # energy is tiny, but their causal direction is a primary endpoint.
            chosen = generator.integers(
                0, causal_count, size=min(batch_size, 4 * causal_count)
            )
            current = np.concatenate(
                (
                    causal_fit["current_j_clean"][chosen],
                    causal_fit["current_j_intervened"][chosen],
                )
            )
            target = np.concatenate(
                (
                    causal_fit["next_j_clean"][chosen],
                    causal_fit["next_j_intervened"][chosen],
                )
            )
            causal_u = np.concatenate(
                (causal_fit["u"][chosen], causal_fit["u"][chosen])
            )
            causal_action = np.concatenate(
                (causal_fit["next_action"][chosen], causal_fit["next_action"][chosen])
            )
            causal_r = np.concatenate(
                (
                    causal_fit["remainder_clean"][chosen],
                    causal_fit["remainder_intervened"][chosen],
                )
            )
            current_tensor = torch.from_numpy(current).float().to(device)
            u_tensor = torch.from_numpy(causal_u).float().to(device)
            auxiliary_tensor = (
                torch.from_numpy(causal_r).float().to(device)
                if model.uses_remainder
                else None
            )
            causal_predicted, causal_logits = model(
                current_tensor, u_tensor, auxiliary_tensor
            )
            causal_loss = strong_multitask_loss(
                causal_predicted,
                causal_logits,
                torch.from_numpy(target).float().to(device),
                torch.from_numpy(causal_action).long().to(device),
                direction_tensor,
                top_tensor,
                weights,
            )
            pair_count = len(chosen)
            predicted_pair_delta = (
                causal_predicted[pair_count:] - causal_predicted[:pair_count]
            )
            target_pair_delta = (
                torch.from_numpy(causal_fit["teacher_delta"][chosen]).float().to(device)
            )
            predicted_projection = predicted_pair_delta @ direction_tensor.T
            target_projection = target_pair_delta @ direction_tensor.T
            pair_direction_loss = F.mse_loss(
                predicted_projection, target_projection
            ) / target_projection.square().mean().detach().clamp_min(1e-6)
            action_indices = (
                torch.from_numpy(causal_fit["next_action"][chosen]).long().to(device)
            )
            predicted_output_delta = causal_logits[pair_count:, :].gather(
                1, action_indices[:, None]
            ).squeeze(1) - causal_logits[:pair_count, :].gather(
                1, action_indices[:, None]
            ).squeeze(1)
            output_sign = (
                torch.from_numpy(causal_fit["teacher_output_sign"][chosen])
                .float()
                .to(device)
            )
            nonzero_sign = output_sign != 0
            sign_loss = (
                F.softplus(
                    -output_sign[nonzero_sign] * predicted_output_delta[nonzero_sign]
                ).mean()
                if torch.any(nonzero_sign)
                else predicted_output_delta.square().mean() * 0
            )
            pair_loss = pair_direction_loss + sign_loss
            total = loss.total + float(weights["causal_counterfactual"]) * (
                causal_loss.total + pair_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        predicted, actions = _predict(
            model,
            validation,
            validation_r,
            device=device,
            batch_size=batch_size,
        )
        cosine = float(np.mean(row_cosine(predicted, validation.next_j)))
        accuracy = float(np.mean(np.argmax(actions, axis=1) == validation.next_action))
        projection_rmse = float(
            np.sqrt(np.mean(((predicted - validation.next_j) @ directions.T) ** 2))
        )
        score = cosine + 0.02 * accuracy - 0.1 * projection_rmse
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_cosine": cosine,
                "validation_action_accuracy": accuracy,
                "validation_causal_projection_rmse": projection_rmse,
                "selection_score": score,
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(section["reference_patience"]):
                break
    if best_state is None:
        raise RuntimeError("strong reference produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _ordinary_frame(
    model: StrongPeripheralPredictor,
    data: TransitionArrays,
    remainder: np.ndarray,
    directions: np.ndarray,
    top: np.ndarray,
    *,
    run_id: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    predicted, logits = _predict(
        model, data, remainder, device=device, batch_size=batch_size
    )
    log_prob = torch.log_softmax(torch.from_numpy(logits).float(), dim=1).numpy()
    selected = top[:64]
    return pd.DataFrame(
        {
            "run_id": run_id,
            "model": model.architecture,
            "seed": seed,
            "example_id": data.example_id,
            "family": data.family,
            "step_index": data.step_index,
            "next_j_cosine": row_cosine(predicted, data.next_j),
            "action_correct": np.argmax(logits, axis=1) == data.next_action,
            "action_cross_entropy": -log_prob[np.arange(len(data)), data.next_action],
            "causal_projection_rmse": np.sqrt(
                np.mean(((predicted - data.next_j) @ directions.T) ** 2, axis=1)
            ),
            "top_causal_dimension_rmse": np.sqrt(
                np.mean(
                    (predicted[:, selected] - data.next_j[:, selected]) ** 2, axis=1
                )
            ),
            "parameter_count": parameter_count(model),
        }
    )


@torch.no_grad()
def _causal_frame(
    model: StrongPeripheralPredictor,
    data: dict[str, np.ndarray],
    *,
    run_id: str,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    clean_r = (
        torch.from_numpy(data["remainder_clean"]).float().to(device)
        if model.uses_remainder
        else None
    )
    intervened_r = (
        torch.from_numpy(data["remainder_intervened"]).float().to(device)
        if model.uses_remainder
        else None
    )
    u = torch.from_numpy(data["u"]).float().to(device)
    clean_j, clean_logits = model(
        torch.from_numpy(data["current_j_clean"]).float().to(device), u, clean_r
    )
    changed_j, changed_logits = model(
        torch.from_numpy(data["current_j_intervened"]).float().to(device),
        u,
        intervened_r,
    )
    predicted_delta = (changed_j - clean_j).cpu().numpy()
    teacher_delta = data["teacher_delta"]
    predicted_norm = np.linalg.norm(predicted_delta, axis=1)
    teacher_norm = np.linalg.norm(teacher_delta, axis=1)
    direction = row_cosine(predicted_delta, teacher_delta)
    target = data["next_action"]
    clean_logits_np = clean_logits.cpu().numpy()
    changed_logits_np = changed_logits.cpu().numpy()
    predicted_target_delta = (
        changed_logits_np[np.arange(len(target)), target]
        - clean_logits_np[np.arange(len(target)), target]
    )
    teacher_target_delta = data["teacher_output_sign"]
    teacher_semantic_delta = data["teacher_semantic_delta"]
    predicted_semantic_delta = np.argmax(changed_logits_np, axis=1) != np.argmax(
        clean_logits_np, axis=1
    )
    return pd.DataFrame(
        {
            "run_id": run_id,
            "model": model.architecture,
            "seed": seed,
            "base_trial_id": data["base_trial_id"],
            "prompt_id": data["prompt_id"],
            "family": data["family"],
            "teacher_delta_norm": teacher_norm,
            "predicted_delta_norm": predicted_norm,
            "causal_direction_cosine": direction,
            "causal_magnitude_ratio": predicted_norm / np.maximum(teacher_norm, 1e-20),
            "semantic_delta_agreement": predicted_semantic_delta
            == teacher_semantic_delta,
            "output_sign_agreement": np.sign(predicted_target_delta)
            == teacher_target_delta,
            "teacher_target_logit_delta": teacher_target_delta,
            "predicted_target_logit_delta": predicted_target_delta,
        }
    )


def _paired_gain_ci(
    baseline: pd.DataFrame,
    full: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
    seed: int,
    resamples: int,
    cluster: str,
) -> dict[str, Any]:
    keys = [cluster, "seed"] + (
        ["step_index"] if "step_index" in baseline.columns else []
    )
    left = baseline.groupby(keys, as_index=False)[metric].mean()
    right = full.groupby(keys, as_index=False)[metric].mean()
    joined = left.merge(right, on=keys, suffixes=("_baseline", "_full"))
    raw = joined[f"{metric}_full"] - joined[f"{metric}_baseline"]
    if not higher_is_better:
        raw = -raw
    by_cluster = (
        pd.DataFrame({cluster: joined[cluster], "gain": raw})
        .groupby(cluster)["gain"]
        .mean()
    )
    values = by_cluster.to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    boot = np.asarray(
        [
            np.mean(generator.choice(values, len(values), replace=True))
            for _ in range(resamples)
        ]
    )
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
        "n_clusters": int(len(values)),
        "n_resamples": int(resamples),
        "orientation": "positive favors full peripheral",
    }


def _target_summary(
    next_j: np.ndarray,
    next_action: np.ndarray,
    *,
    state_pca: PCA,
    directions: np.ndarray,
    top: np.ndarray,
) -> np.ndarray:
    one_hot = np.zeros((len(next_action), ACTION_COUNT), dtype=np.float32)
    one_hot[np.arange(len(next_action)), next_action.astype(np.int64)] = 1
    return np.concatenate(
        (
            state_pca.transform(next_j).astype(np.float32),
            (next_j @ directions.T).astype(np.float32),
            next_j[:, top[:64]].astype(np.float32),
            one_hot,
        ),
        axis=1,
    )


def _fit_bottleneck(
    context: Any,
    train_r: np.ndarray,
    train_target: np.ndarray,
    causal_r: np.ndarray,
    causal_target: np.ndarray,
    *,
    dimension: int,
    nonlinear: bool,
    seed: int,
) -> PredictivePeripheralBottleneck:
    device = _device(context)
    torch.manual_seed(seed)
    model = PredictivePeripheralBottleneck(
        train_r.shape[1],
        dimension,
        train_target.shape[1],
        nonlinear=nonlinear,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    combined_r = np.concatenate((train_r, np.repeat(causal_r, 8, axis=0)))
    combined_target = np.concatenate(
        (train_target, np.repeat(causal_target, 8, axis=0))
    )
    mean = combined_target.mean(axis=0, keepdims=True)
    scale = combined_target.std(axis=0, keepdims=True)
    scale[scale < 1e-4] = 1
    standardized = (combined_target - mean) / scale
    for _ in range(20):
        model.train()
        for indices in _batches(len(combined_r), 256, generator):
            _, prediction = model(
                torch.from_numpy(combined_r[indices]).float().to(device)
            )
            target = torch.from_numpy(standardized[indices]).float().to(device)
            loss = F.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _encode_bottleneck(
    model: PredictivePeripheralBottleneck,
    values: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    encoded = []
    for start in range(0, len(values), 256):
        latent, _ = model(
            torch.from_numpy(values[start : start + 256]).float().to(device)
        )
        encoded.append(latent.cpu().numpy().astype(np.float32))
    return np.concatenate(encoded)


def _run_compact(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    section = context.config["peripheral_v6"]
    ceiling = json.loads(
        (context.processed_dir / "peripheral_ceiling_v6.json").read_text(
            encoding="utf-8"
        )
    )
    if not ceiling["strong_peripheral_ceiling_authorized"]:
        summary = {
            "schema_version": 8,
            "protocol_version": PROTOCOL_V6,
            "run_id": context.run_id,
            "status": "GATED_NO_STRONG_PERIPHERAL_CEILING",
            "source_freeze_digest": freeze["freeze_digest"],
            "compact_authorized": False,
        }
        write_json_atomic(context.processed_dir / "compact_peripheral_v6.json", summary)
        return summary
    endpoint_summary, endpoint = _load_endpoint(context)
    transform = _load_transform(context, endpoint_summary)
    v5_freeze = json.loads(
        (context.root / "artifacts/peripheral_v5.freeze.json").read_text(
            encoding="utf-8"
        )
    )
    domains = {
        name: load_domain(context.root, v5_freeze, name, 4)
        for name in ("train", "validation", "rollout_test")
    }
    remainder = {
        name: transform.transform(value.current_j, value.current_h)
        for name, value in domains.items()
    }
    causal_fit = _causal_data(endpoint, "causal_fit")
    causal_test = _causal_data(endpoint, "causal_test")
    target_path = context.root / ceiling["causal_targets"]
    if sha256_file(target_path) != ceiling["causal_targets_sha256"]:
        raise RuntimeError("v6 causal target hash mismatch")
    with np.load(target_path, allow_pickle=False) as payload:
        directions = payload["directions"].astype(np.float32)
        top = payload["top_dimensions"].astype(np.int64)
    state_pca = PCA(64, svd_solver="randomized", random_state=int(context.seed)).fit(
        domains["train"].next_j
    )
    train_target = _target_summary(
        domains["train"].next_j,
        domains["train"].next_action,
        state_pca=state_pca,
        directions=directions,
        top=top,
    )
    causal_remainder = np.concatenate(
        (causal_fit["remainder_clean"], causal_fit["remainder_intervened"])
    )
    causal_target = _target_summary(
        np.concatenate((causal_fit["next_j_clean"], causal_fit["next_j_intervened"])),
        np.concatenate((causal_fit["next_action"], causal_fit["next_action"])),
        state_pca=state_pca,
        directions=directions,
        top=top,
    )
    artifact_directory = context.root / "artifacts/peripheral/v6" / context.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    baseline_name = str(ceiling["baseline_choice"]["architecture"])
    full_name = str(ceiling["full_choice"]["architecture"])
    ordinary_reference = pd.read_parquet(context.root / ceiling["ordinary_records"])
    causal_reference = pd.read_parquet(context.root / ceiling["causal_records"])
    baseline_ordinary = ordinary_reference[ordinary_reference["model"] == baseline_name]
    full_ordinary = ordinary_reference[ordinary_reference["model"] == full_name]
    baseline_causal = causal_reference[causal_reference["model"] == baseline_name]
    full_causal = causal_reference[causal_reference["model"] == full_name]
    baseline_cosine = float(baseline_ordinary["next_j_cosine"].mean())
    full_cosine = float(full_ordinary["next_j_cosine"].mean())
    baseline_direction = float(baseline_causal["causal_direction_cosine"].mean())
    full_direction = float(full_causal["causal_direction_cosine"].mean())
    results = []
    ordinary_frames = []
    causal_frames = []
    representations: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for family in section["compact_families"]:
        for dimension in (int(value) for value in section["compact_dimensions"]):
            if family == "pca":
                encoder: Any = PCA(
                    dimension,
                    svd_solver="randomized",
                    random_state=int(section["reference_screen_seed"]),
                ).fit(remainder["train"])
                encoded = {
                    name: encoder.transform(values).astype(np.float32)
                    for name, values in remainder.items()
                }
                causal_fit_encoded = {
                    "clean": encoder.transform(causal_fit["remainder_clean"]).astype(
                        np.float32
                    ),
                    "intervened": encoder.transform(
                        causal_fit["remainder_intervened"]
                    ).astype(np.float32),
                }
                causal_test_encoded = {
                    "clean": encoder.transform(causal_test["remainder_clean"]).astype(
                        np.float32
                    ),
                    "intervened": encoder.transform(
                        causal_test["remainder_intervened"]
                    ).astype(np.float32),
                }
                encoder_path = artifact_directory / f"{family}-{dimension}.npz"
                np.savez_compressed(
                    encoder_path,
                    mean=encoder.mean_.astype(np.float32),
                    components=encoder.components_.astype(np.float32),
                )
            else:
                encoder = _fit_bottleneck(
                    context,
                    remainder["train"],
                    train_target,
                    causal_remainder,
                    causal_target,
                    dimension=dimension,
                    nonlinear=family == "nonlinear_bottleneck",
                    seed=int(section["reference_screen_seed"]),
                )
                encoded = {
                    name: _encode_bottleneck(encoder, values, device=_device(context))
                    for name, values in remainder.items()
                }
                causal_fit_encoded = {
                    "clean": _encode_bottleneck(
                        encoder, causal_fit["remainder_clean"], device=_device(context)
                    ),
                    "intervened": _encode_bottleneck(
                        encoder,
                        causal_fit["remainder_intervened"],
                        device=_device(context),
                    ),
                }
                causal_test_encoded = {
                    "clean": _encode_bottleneck(
                        encoder, causal_test["remainder_clean"], device=_device(context)
                    ),
                    "intervened": _encode_bottleneck(
                        encoder,
                        causal_test["remainder_intervened"],
                        device=_device(context),
                    ),
                }
                encoder_path = artifact_directory / f"{family}-{dimension}.pt"
                torch.save(encoder.state_dict(), encoder_path)
            representations[(str(family), dimension)] = {
                **encoded,
                "causal_fit_clean": causal_fit_encoded["clean"],
                "causal_fit_intervened": causal_fit_encoded["intervened"],
                "causal_test_clean": causal_test_encoded["clean"],
                "causal_test_intervened": causal_test_encoded["intervened"],
            }
            fit_for_model = dict(causal_fit)
            fit_for_model["remainder_clean"] = causal_fit_encoded["clean"]
            fit_for_model["remainder_intervened"] = causal_fit_encoded["intervened"]
            model, training = _fit_model(
                context,
                full_name,
                domains["train"],
                domains["validation"],
                encoded["train"],
                encoded["validation"],
                fit_for_model,
                directions,
                top,
                seed=int(section["reference_screen_seed"]),
            )
            ordinary = _ordinary_frame(
                model,
                domains["rollout_test"],
                encoded["rollout_test"],
                directions,
                top,
                run_id=context.run_id,
                seed=int(section["reference_screen_seed"]),
                device=_device(context),
                batch_size=int(section["reference_batch_size"]),
            )
            label = f"{family}_{dimension}"
            ordinary["model"] = label
            test_for_model = dict(causal_test)
            test_for_model["remainder_clean"] = causal_test_encoded["clean"]
            test_for_model["remainder_intervened"] = causal_test_encoded["intervened"]
            causal = _causal_frame(
                model,
                test_for_model,
                run_id=context.run_id,
                seed=int(section["reference_screen_seed"]),
                device=_device(context),
            )
            causal["model"] = label
            ordinary_frames.append(ordinary)
            causal_frames.append(causal)
            cosine = float(ordinary["next_j_cosine"].mean())
            direction = float(causal["causal_direction_cosine"].mean())
            gap = (cosine - baseline_cosine) / max(full_cosine - baseline_cosine, 1e-20)
            causal_gap = (direction - baseline_direction) / max(
                full_direction - baseline_direction, 1e-20
            )
            results.append(
                {
                    "family": family,
                    "dimension": dimension,
                    "next_j_cosine": cosine,
                    "predictive_gap_closed": gap,
                    "action_accuracy": float(ordinary["action_correct"].mean()),
                    "causal_direction_cosine": direction,
                    "causal_gap_closed": causal_gap,
                    "causal_magnitude_ratio": float(
                        causal["causal_magnitude_ratio"].median()
                    ),
                    "semantic_delta_agreement": float(
                        causal["semantic_delta_agreement"].mean()
                    ),
                    "output_sign_agreement": float(
                        causal["output_sign_agreement"].mean()
                    ),
                    "encoder": str(encoder_path.relative_to(context.root)),
                    "encoder_sha256": sha256_file(encoder_path),
                    "predictor_parameter_count": parameter_count(model),
                    "training": training,
                }
            )
    result_frame = pd.DataFrame(results)
    full_action = float(full_ordinary["action_correct"].mean())
    candidates = result_frame[
        (result_frame["predictive_gap_closed"] >= float(section["compact_gap_closed"]))
        & (result_frame["causal_gap_closed"] >= float(section["compact_gap_closed"]))
        & (result_frame["action_accuracy"] >= full_action - 0.02)
    ].sort_values(["dimension", "predictive_gap_closed"], ascending=[True, False])
    conditional: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None
    if not candidates.empty:
        selected_value: dict[str, Any] = candidates.iloc[0].to_dict()
        selected = selected_value
        key = (str(selected_value["family"]), int(selected_value["dimension"]))
        values = representations[key]
        ridge = Ridge(alpha=10.0, solver="lsqr").fit(
            values["train"], remainder["train"]
        )
        residual = {
            name: remainder[name] - ridge.predict(values[name]).astype(np.float32)
            for name in remainder
        }
        augmented = {
            name: np.concatenate((values[name], residual[name]), axis=1).astype(
                np.float32
            )
            for name in remainder
        }
        fit_for_model = dict(causal_fit)
        fit_residual_clean = causal_fit["remainder_clean"] - ridge.predict(
            values["causal_fit_clean"]
        )
        fit_residual_intervened = causal_fit["remainder_intervened"] - ridge.predict(
            values["causal_fit_intervened"]
        )
        fit_for_model["remainder_clean"] = np.concatenate(
            (values["causal_fit_clean"], fit_residual_clean), axis=1
        ).astype(np.float32)
        fit_for_model["remainder_intervened"] = np.concatenate(
            (values["causal_fit_intervened"], fit_residual_intervened), axis=1
        ).astype(np.float32)
        model, training = _fit_model(
            context,
            full_name,
            domains["train"],
            domains["validation"],
            augmented["train"],
            augmented["validation"],
            fit_for_model,
            directions,
            top,
            seed=int(section["reference_screen_seed"]),
        )
        ordinary = _ordinary_frame(
            model,
            domains["rollout_test"],
            augmented["rollout_test"],
            directions,
            top,
            run_id=context.run_id,
            seed=int(section["reference_screen_seed"]),
            device=_device(context),
            batch_size=int(section["reference_batch_size"]),
        )
        test_for_model = dict(causal_test)
        test_residual_clean = causal_test["remainder_clean"] - ridge.predict(
            values["causal_test_clean"]
        )
        test_residual_intervened = causal_test["remainder_intervened"] - ridge.predict(
            values["causal_test_intervened"]
        )
        test_for_model["remainder_clean"] = np.concatenate(
            (values["causal_test_clean"], test_residual_clean), axis=1
        ).astype(np.float32)
        test_for_model["remainder_intervened"] = np.concatenate(
            (values["causal_test_intervened"], test_residual_intervened), axis=1
        ).astype(np.float32)
        causal = _causal_frame(
            model,
            test_for_model,
            run_id=context.run_id,
            seed=int(section["reference_screen_seed"]),
            device=_device(context),
        )
        conditional = {
            "selected": selected,
            "next_j_cosine_with_residual": float(ordinary["next_j_cosine"].mean()),
            "predictive_residual_gain": float(ordinary["next_j_cosine"].mean())
            - float(selected_value["next_j_cosine"]),
            "action_accuracy_with_residual": float(ordinary["action_correct"].mean()),
            "semantic_residual_gain": float(ordinary["action_correct"].mean())
            - float(selected_value["action_accuracy"]),
            "causal_direction_with_residual": float(
                causal["causal_direction_cosine"].mean()
            ),
            "causal_residual_gain": float(causal["causal_direction_cosine"].mean())
            - float(selected_value["causal_direction_cosine"]),
            "training": training,
        }
        conditional["conditionally_sufficient"] = bool(
            conditional["predictive_residual_gain"]
            <= float(section["compact_maximum_conditional_gain"])
            and conditional["causal_residual_gain"] <= 0.02
            and conditional["semantic_residual_gain"] <= 0.02
        )
    ordinary_all = pd.concat(ordinary_frames, ignore_index=True)
    causal_all = pd.concat(causal_frames, ignore_index=True)
    ordinary_path = context.processed_dir / "compact_peripheral_ordinary_v6.parquet"
    causal_path = context.processed_dir / "compact_peripheral_causal_v6.parquet"
    ordinary_all.to_parquet(ordinary_path, index=False, compression="zstd")
    causal_all.to_parquet(causal_path, index=False, compression="zstd")
    summary = {
        "schema_version": 8,
        "protocol_version": PROTOCOL_V6,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "source_ceiling_run_id": ceiling["run_id"],
        "results": results,
        "screen_passing_count": int(len(candidates)),
        "selected": selected,
        "conditional_sufficiency": conditional,
        "compact_authorized": bool(
            selected is not None
            and conditional is not None
            and conditional["conditionally_sufficient"]
        ),
        "ordinary_records": str(ordinary_path.relative_to(context.root)),
        "causal_records": str(causal_path.relative_to(context.root)),
        "recurrent_controller_trained": False,
    }
    write_json_atomic(context.processed_dir / "compact_peripheral_v6.json", summary)
    return summary


def _run(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    section = context.config["peripheral_v6"]
    endpoint_summary, endpoint = _load_endpoint(context)
    transform = _load_transform(context, endpoint_summary)
    v5_freeze = json.loads(
        (context.root / "artifacts/peripheral_v5.freeze.json").read_text(
            encoding="utf-8"
        )
    )
    domains = {
        name: load_domain(context.root, v5_freeze, name, 4)
        for name in ("train", "validation", "rollout_test")
    }
    remainders = {
        name: transform.transform(data.current_j, data.current_h)
        for name, data in domains.items()
    }
    causal_fit = _causal_data(endpoint, "causal_fit")
    causal_test = _causal_data(endpoint, "causal_test")
    directions, top = fit_causal_directions(
        causal_fit["teacher_delta"], int(section["causal_direction_count"])
    )
    artifact_directory = context.root / "artifacts/peripheral/v6" / context.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    targets_path = artifact_directory / "causal_targets.npz"
    np.savez_compressed(targets_path, directions=directions, top_dimensions=top)
    screen = []
    screen_models: dict[str, StrongPeripheralPredictor] = {}
    for architecture in section["reference_architectures"]:
        model, training = _fit_model(
            context,
            str(architecture),
            domains["train"],
            domains["validation"],
            remainders["train"],
            remainders["validation"],
            causal_fit,
            directions,
            top,
            seed=int(section["reference_screen_seed"]),
        )
        validation = _ordinary_frame(
            model,
            domains["validation"],
            remainders["validation"],
            directions,
            top,
            run_id=context.run_id,
            seed=int(section["reference_screen_seed"]),
            device=_device(context),
            batch_size=int(section["reference_batch_size"]),
        )
        causal = _causal_frame(
            model,
            causal_test,
            run_id=context.run_id,
            seed=int(section["reference_screen_seed"]),
            device=_device(context),
        )
        entry = {
            "architecture": architecture,
            "uses_remainder": model.uses_remainder,
            "validation_next_j_cosine": float(validation["next_j_cosine"].mean()),
            "validation_action_accuracy": float(validation["action_correct"].mean()),
            "validation_causal_projection_rmse": float(
                validation["causal_projection_rmse"].mean()
            ),
            "causal_test_direction_cosine": float(
                causal["causal_direction_cosine"].mean()
            ),
            "selection_score": float(validation["next_j_cosine"].mean())
            + 0.02 * float(validation["action_correct"].mean())
            - 0.1 * float(validation["causal_projection_rmse"].mean()),
            "selection_data": "ordinary_validation_only; causal_test excluded",
            "parameter_count": parameter_count(model),
            "training": training,
        }
        screen.append(entry)
        screen_models[str(architecture)] = model
    baseline_choice = max(
        (value for value in screen if not value["uses_remainder"]),
        key=lambda value: value["selection_score"],
    )
    full_choice = max(
        (value for value in screen if value["uses_remainder"]),
        key=lambda value: value["selection_score"],
    )
    ordinary_frames, causal_frames, checkpoints = [], [], []
    for seed in (int(value) for value in section["reference_confirmation_seeds"]):
        for choice in (baseline_choice, full_choice):
            architecture = str(choice["architecture"])
            model, training = _fit_model(
                context,
                architecture,
                domains["train"],
                domains["validation"],
                remainders["train"],
                remainders["validation"],
                causal_fit,
                directions,
                top,
                seed=seed,
            )
            ordinary_frames.append(
                _ordinary_frame(
                    model,
                    domains["rollout_test"],
                    remainders["rollout_test"],
                    directions,
                    top,
                    run_id=context.run_id,
                    seed=seed,
                    device=_device(context),
                    batch_size=int(section["reference_batch_size"]),
                )
            )
            causal_frames.append(
                _causal_frame(
                    model,
                    causal_test,
                    run_id=context.run_id,
                    seed=seed,
                    device=_device(context),
                )
            )
            checkpoint = artifact_directory / f"{architecture}-s{seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": architecture,
                    "training": training,
                },
                checkpoint,
            )
            checkpoints.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "path": str(checkpoint.relative_to(context.root)),
                    "sha256": sha256_file(checkpoint),
                    "parameter_count": parameter_count(model),
                }
            )
    ordinary = pd.concat(ordinary_frames, ignore_index=True)
    causal = pd.concat(causal_frames, ignore_index=True)
    ordinary_path = context.processed_dir / "peripheral_ceiling_ordinary_v6.parquet"
    causal_path = context.processed_dir / "peripheral_ceiling_causal_v6.parquet"
    ordinary.to_parquet(ordinary_path, index=False, compression="zstd")
    causal.to_parquet(causal_path, index=False, compression="zstd")
    baseline = ordinary[ordinary["model"] == baseline_choice["architecture"]]
    full = ordinary[ordinary["model"] == full_choice["architecture"]]
    baseline_causal = causal[causal["model"] == baseline_choice["architecture"]]
    full_causal = causal[causal["model"] == full_choice["architecture"]]
    bootstrap_seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(section["bootstrap_resamples"])
    ordinary_metrics = {
        metric: _paired_gain_ci(
            baseline,
            full,
            metric,
            higher_is_better=orientation,
            seed=bootstrap_seed,
            resamples=resamples,
            cluster="example_id",
        )
        for metric, orientation in (
            ("next_j_cosine", True),
            ("action_correct", True),
            ("action_cross_entropy", False),
            ("causal_projection_rmse", False),
            ("top_causal_dimension_rmse", False),
        )
    }
    causal_metrics = {
        metric: _paired_gain_ci(
            baseline_causal,
            full_causal,
            metric,
            higher_is_better=True,
            seed=bootstrap_seed,
            resamples=resamples,
            cluster="prompt_id",
        )
        for metric in (
            "causal_direction_cosine",
            "semantic_delta_agreement",
            "output_sign_agreement",
        )
    }
    family_wise = {}
    for family in sorted(set(ordinary["family"])):
        left = baseline[baseline["family"] == family]
        right = full[full["family"] == family]
        family_wise[family] = {
            "next_j_cosine": _paired_gain_ci(
                left,
                right,
                "next_j_cosine",
                higher_is_better=True,
                seed=bootstrap_seed,
                resamples=resamples,
                cluster="example_id",
            ),
            "action_accuracy": _paired_gain_ci(
                left,
                right,
                "action_correct",
                higher_is_better=True,
                seed=bootstrap_seed,
                resamples=resamples,
                cluster="example_id",
            ),
        }
    for family in sorted(set(causal["family"])):
        left = baseline_causal[baseline_causal["family"] == family]
        right = full_causal[full_causal["family"] == family]
        family_wise.setdefault(family, {})["causal_direction_cosine"] = _paired_gain_ci(
            left,
            right,
            "causal_direction_cosine",
            higher_is_better=True,
            seed=bootstrap_seed,
            resamples=resamples,
            cluster="prompt_id",
        )
    rules = section["ceiling_rules"]
    cosine_gate = bool(
        ordinary_metrics["next_j_cosine"]["lower"] > 0
        and ordinary_metrics["next_j_cosine"]["estimate"]
        >= float(rules["global_cosine_minimum_gain"])
        and ordinary_metrics["action_correct"]["estimate"]
        >= -float(rules["maximum_action_accuracy_loss"])
    )
    causal_gate = bool(
        causal_metrics["causal_direction_cosine"]["lower"] > 0
        and causal_metrics["causal_direction_cosine"]["estimate"]
        >= float(rules["causal_direction_minimum_gain"])
        and float(full_causal["causal_direction_cosine"].mean())
        >= float(rules["causal_direction_minimum_absolute"])
    )
    projection_base = float(baseline["causal_projection_rmse"].mean())
    projection_full = float(full["causal_projection_rmse"].mean())
    projection_reduction = (projection_base - projection_full) / max(
        projection_base, 1e-20
    )
    projection_gate = bool(
        ordinary_metrics["causal_projection_rmse"]["lower"] > 0
        and projection_reduction
        >= float(rules["causal_projection_minimum_relative_rmse_reduction"])
    )
    semantic_gate = bool(
        ordinary_metrics["action_correct"]["lower"] > 0
        and ordinary_metrics["action_correct"]["estimate"]
        >= float(rules["semantic_accuracy_minimum_gain"])
    )
    summary = {
        "schema_version": 8,
        "protocol_version": PROTOCOL_V6,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "screen": screen,
        "baseline_choice": baseline_choice,
        "full_choice": full_choice,
        "ordinary_gains": ordinary_metrics,
        "causal_gains": causal_metrics,
        "family_wise_gains": family_wise,
        "absolute_means": {
            "baseline": baseline.mean(numeric_only=True).to_dict(),
            "full": full.mean(numeric_only=True).to_dict(),
            "baseline_causal": baseline_causal.mean(numeric_only=True).to_dict(),
            "full_causal": full_causal.mean(numeric_only=True).to_dict(),
        },
        "causal_projection_relative_rmse_reduction": projection_reduction,
        "authorization_gates": {
            "global_profile": cosine_gate,
            "causal_direction": causal_gate,
            "causal_projection": projection_gate,
            "semantic": semantic_gate,
        },
        "strong_peripheral_ceiling_authorized": bool(
            cosine_gate or causal_gate or projection_gate or semantic_gate
        ),
        "ordinary_records": str(ordinary_path.relative_to(context.root)),
        "causal_records": str(causal_path.relative_to(context.root)),
        "causal_targets": str(targets_path.relative_to(context.root)),
        "causal_targets_sha256": sha256_file(targets_path),
        "checkpoints": checkpoints,
    }
    output = context.processed_dir / "peripheral_ceiling_v6.json"
    write_json_atomic(output, summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "Strong multi-endpoint peripheral ceiling", "configs/peripheral_v6.yaml"
    )
    parser.add_argument("--stage", choices=("run", "compact"), default="run")
    args = parser.parse_args()
    context = initialize_context("peripheral-ceiling-v6", args)
    try:
        freeze = verify_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        summary = (
            _run_compact(context, freeze)
            if args.stage == "compact"
            else _run(context, freeze)
        )
        context.finish("COMPLETED", summary=summary)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
