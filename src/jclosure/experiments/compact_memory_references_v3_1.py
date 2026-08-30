"""Remainder-aware references and conditional fidelity endpoint for v3.1."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from jclosure.compact_memory_references_v3_1 import (
    AutonomousRemainderGRU,
    autonomous_remainder_rollout,
    build_autonomous_remainder_reference,
    build_full_remainder_reference,
    decoded_one_step_metrics,
    fit_linear_current_remainder_reference,
)
from jclosure.compact_memory_v3_1 import (
    LinearRepresentation,
    parameter_count,
    row_cosine,
    scheduled_feedback_probability,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_v3_1 import (
    ACTION_SURFACES,
    _load_representation,
    _load_trace_frame,
    _pairs,
)
from jclosure.protocol_v3_1 import MEMORY_PROTOCOL, verify_freeze
from jclosure.provenance import sha256_file, write_json_atomic


def _selected_representation(context, dimension: int) -> LinearRepresentation:
    screen_path = (
        context.processed_dir / "compact_memory_representation_screen.json"
    )
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not screen.get("temporal_training_authorized"):
        raise RuntimeError("representation screen did not authorize references")
    selected = next(
        value
        for value in screen["selected_by_dimension"]
        if int(value["dimension"]) == dimension
    )
    return _load_representation(context.root, selected["representation_path"])


def _latent_arrays(
    arrays: dict[str, np.ndarray], representation: LinearRepresentation
) -> dict[str, np.ndarray]:
    output = dict(arrays)
    for split in ("train", "validation", "test"):
        output[f"{split}_z"] = representation.encode(arrays[f"{split}_x"]).astype(
            np.float32
        )
        output[f"{split}_next_z"] = representation.encode(
            arrays[f"{split}_y"]
        ).astype(np.float32)
    return output


def _train_full_one_step(
    context,
    arrays: dict[str, np.ndarray],
    representation: LinearRepresentation,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module]:
    torch.manual_seed(seed)
    config = context.config["compact_memory"]
    model = build_full_remainder_reference(
        state_dim=representation.dimension,
        remainder_dim=arrays["train_r"].shape[1],
        target=int(config["target_parameter_count"]),
        tolerance=float(config["parameter_tolerance"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    batch_size = 512
    maximum_epochs = int(config["maximum_epochs"])
    patience = int(config["early_stopping_patience"])
    best = -float("inf")
    best_state = None
    stale = 0
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(maximum_epochs):
        model.train()
        order = torch.randperm(len(arrays["train_z"]), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size].numpy()
            state = torch.from_numpy(arrays["train_z"][index]).to(device)
            remainder = torch.from_numpy(arrays["train_r"][index]).to(device)
            target = torch.from_numpy(arrays["train_next_z"][index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(state, remainder)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(arrays["validation_z"]), batch_size):
                state = torch.from_numpy(
                    arrays["validation_z"][start : start + batch_size]
                ).to(device)
                remainder = torch.from_numpy(
                    arrays["validation_r"][start : start + batch_size]
                ).to(device)
                chunks.append(model(state, remainder).cpu().numpy())
        validation_prediction = np.concatenate(chunks)
        validation_decoded = representation.decode(validation_prediction)
        score = float(
            np.median(row_cosine(validation_decoded, arrays["validation_y"]))
        )
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "validation_cosine": score}
        )
        if score > best:
            best = score
            stale = 0
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
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(arrays["test_z"]), batch_size):
            state = torch.from_numpy(
                arrays["test_z"][start : start + batch_size]
            ).to(device)
            remainder = torch.from_numpy(
                arrays["test_r"][start : start + batch_size]
            ).to(device)
            chunks.append(model(state, remainder).cpu().numpy())
    prediction = np.concatenate(chunks)
    return (
        {
            "reference_type": "nonlinear_full_remainder_teacher_current_one_step",
            "teacher_current_only": True,
            "parameter_count": parameter_count(model),
            "best_validation_cosine": best,
            "epochs": len(history),
            "training": history,
            "test": decoded_one_step_metrics(
                representation.decode(prediction), arrays["test_y"]
            ),
        },
        model,
    )


def _trajectory_latents(
    trajectories: list[dict[str, Any]],
    representation: LinearRepresentation,
    pca: PCA,
) -> list[dict[str, Any]]:
    output = []
    for trajectory in trajectories:
        output.append(
            {
                **trajectory,
                "latent": representation.encode(trajectory["states"]).astype(
                    np.float32
                ),
                "remainder_latent": pca.transform(trajectory["remainders"]).astype(
                    np.float32
                ),
            }
        )
    return output


def _train_recurrent_epoch(
    model: AutonomousRemainderGRU,
    trajectories: list[dict[str, Any]],
    optimizer,
    *,
    feedback_probability: float,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(trajectories), generator=generator).tolist()
    losses = []
    model.train()
    for index in order:
        trajectory = trajectories[index]
        states = torch.from_numpy(trajectory["latent"]).to(device)
        remainders = torch.from_numpy(trajectory["remainder_latent"]).to(device)
        actions = torch.from_numpy(trajectory["actions"]).long().to(device)
        if len(states) < 2:
            continue
        optimizer.zero_grad(set_to_none=True)
        state = states[0:1]
        remainder = remainders[0:1]
        memory = torch.zeros(1, model.memory_dim, device=device)
        total = torch.zeros((), device=device)
        for step in range(len(states) - 1):
            predicted_state, predicted_remainder, logits, memory = model(
                state, remainder, memory
            )
            target_state = states[step + 1 : step + 2]
            target_remainder = remainders[step + 1 : step + 2]
            target_action = actions[step + 1 : step + 2]
            total = total + (
                F.mse_loss(predicted_state, target_state)
                + F.mse_loss(predicted_remainder, target_remainder)
                + F.cross_entropy(logits, target_action)
            )
            use_prediction = bool(
                torch.rand((), generator=generator).item() < feedback_probability
            )
            state = predicted_state if use_prediction else target_state
            remainder = predicted_remainder if use_prediction else target_remainder
        loss = total / (len(states) - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate_recurrent(
    model: AutonomousRemainderGRU,
    trajectories: list[dict[str, Any]],
    representation: LinearRepresentation,
    *,
    device: torch.device,
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    model.eval()
    for trajectory in trajectories:
        maximum = min(max(horizons), len(trajectory["latent"]) - 1)
        if maximum < 1:
            continue
        state = torch.from_numpy(trajectory["latent"][0:1]).to(device)
        remainder = torch.from_numpy(trajectory["remainder_latent"][0:1]).to(device)
        predicted, _, action_logits = autonomous_remainder_rollout(
            model, state, remainder, steps=maximum
        )
        decoded = representation.decode(predicted[0].cpu().numpy())
        target = trajectory["states"][1 : maximum + 1]
        cosine = row_cosine(decoded, target)
        predicted_actions = action_logits[0].argmax(-1).cpu().numpy()
        target_actions = trajectory["actions"][1 : maximum + 1]
        for horizon in horizons:
            if horizon > maximum:
                continue
            rows.append(
                {
                    "example_id": trajectory["example_id"],
                    "family": trajectory["family"],
                    "split": trajectory["split"],
                    "horizon": horizon,
                    "decoded_cosine": float(cosine[horizon - 1]),
                    "trajectory_distance": float(np.mean(1 - cosine[:horizon])),
                    "semantic_action_accuracy": float(
                        np.mean(
                            predicted_actions[:horizon] == target_actions[:horizon]
                        )
                    ),
                    "finite": bool(np.isfinite(decoded[:horizon]).all()),
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
                "finite_fraction": float(group["finite"].mean()),
            }
        )
    return rows, summaries


def _references(context, *, dimension: int, seed: int) -> dict[str, Any]:
    frame = _load_trace_frame(context)
    arrays, raw_trajectories = _pairs(context, frame)
    representation = _selected_representation(context, dimension)
    arrays = _latent_arrays(arrays, representation)
    linear = fit_linear_current_remainder_reference(
        arrays["train_z"],
        arrays["train_r"],
        arrays["train_next_z"],
        remainder_dimension=int(
            context.config["compact_memory"]["remainder_pca_dimensions"][0]
        ),
    )
    linear_prediction = linear.predict(arrays["test_z"], arrays["test_r"])
    linear_result = {
        "reference_type": "linear_pca128_remainder_teacher_current_one_step",
        "teacher_current_only": True,
        "test": decoded_one_step_metrics(
            representation.decode(linear_prediction), arrays["test_y"]
        ),
    }
    device = torch.device(
        context.config["model"].get("device", 0)
        if torch.cuda.is_available()
        else "cpu"
    )
    nonlinear_result, nonlinear_model = _train_full_one_step(
        context, arrays, representation, seed=seed, device=device
    )
    train_remainder = arrays["train_r"].astype(np.float32)
    remainder_dimension = min(
        int(context.config["compact_memory"]["remainder_pca_dimensions"][-1]),
        len(train_remainder) - 1,
        train_remainder.shape[1],
    )
    remainder_pca = PCA(remainder_dimension, random_state=0).fit(train_remainder)
    trajectories = _trajectory_latents(
        raw_trajectories, representation, remainder_pca
    )
    train = [value for value in trajectories if value["split"] == "train"]
    validation = [
        value for value in trajectories if value["split"] == "validation"
    ]
    test = [value for value in trajectories if value["split"] == "test"]
    torch.manual_seed(seed)
    config = context.config["compact_memory"]
    recurrent = build_autonomous_remainder_reference(
        state_dim=representation.dimension,
        remainder_dim=remainder_dimension,
        memory_dim=512,
        action_count=len(ACTION_SURFACES),
        target=int(config["target_parameter_count"]),
        tolerance=float(config["parameter_tolerance"]),
    ).to(device)
    optimizer = torch.optim.AdamW(recurrent.parameters(), lr=2e-4)
    maximum_epochs = int(config["maximum_epochs"])
    patience = int(config["early_stopping_patience"])
    best = -float("inf")
    best_state = None
    stale = 0
    training = []
    for epoch in range(maximum_epochs):
        feedback = scheduled_feedback_probability(
            epoch,
            maximum_epochs,
            warmup_fraction=float(config["teacher_forcing_fraction"]),
            maximum_feedback=float(config["maximum_predicted_feedback"]),
        )
        loss = _train_recurrent_epoch(
            recurrent,
            train,
            optimizer,
            feedback_probability=feedback,
            device=device,
            seed=seed + epoch,
        )
        _, validation_summary = _evaluate_recurrent(
            recurrent,
            validation,
            representation,
            device=device,
            horizons=[8],
        )
        score = (
            validation_summary[0]["decoded_cosine_median"]
            if validation_summary
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
            best = float(score)
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in recurrent.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        recurrent.load_state_dict(best_state)
    rows, horizons = _evaluate_recurrent(
        recurrent,
        test,
        representation,
        device=device,
        horizons=[int(value) for value in config["horizons"]],
    )
    records = (
        context.processed_dir
        / f"compact_memory_remainder_reference_d{dimension}_s{seed}.parquet"
    )
    pd.DataFrame(rows).to_parquet(records, index=False)
    checkpoint_root = context.root / "artifacts/checkpoints/v3_1"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    nonlinear_checkpoint = checkpoint_root / f"full-remainder-d{dimension}-s{seed}.pt"
    recurrent_checkpoint = (
        checkpoint_root / f"autonomous-remainder-d{dimension}-s{seed}.pt"
    )
    torch.save(nonlinear_model.state_dict(), nonlinear_checkpoint)
    torch.save(recurrent.state_dict(), recurrent_checkpoint)
    return {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "run_id": context.run_id,
        "state_dimension": dimension,
        "seed": seed,
        "linear_current_one_step": linear_result,
        "nonlinear_full_current_one_step": nonlinear_result,
        "autonomous_pca512_recurrent": {
            "reference_type": "autonomous_pca512_remainder_recurrent",
            "reads_teacher_after_initial": False,
            "remainder_dimension": remainder_dimension,
            "memory_dimension": 512,
            "parameter_count": parameter_count(recurrent),
            "best_validation_horizon8_cosine": best,
            "epochs": len(training),
            "training": training,
            "test": horizons,
            "records": str(records.relative_to(context.root)),
            "records_sha256": sha256_file(records),
        },
        "checkpoints": {
            "nonlinear_full": str(nonlinear_checkpoint.relative_to(context.root)),
            "autonomous_recurrent": str(
                recurrent_checkpoint.relative_to(context.root)
            ),
        },
    }


def _fidelity(context) -> dict[str, Any]:
    trajectories = (
        context.processed_dir / "compact_memory_counterfactual_trajectories.parquet"
    )
    if not trajectories.is_file():
        return {
            "schema_version": 4,
            "protocol_version": MEMORY_PROTOCOL,
            "status": "UNAVAILABLE_NO_VALIDATED_TOKEN_TRAJECTORY",
            "reason": (
                "No teacher J-swap token trajectory passed validation; observational "
                "rollout must not be described as cognitive-dynamics reproduction."
            ),
        }
    frame = pd.read_parquet(trajectories)
    required = {
        "teacher_delta",
        "student_delta",
        "teacher_semantic_direction",
        "student_semantic_direction",
        "teacher_effect",
        "student_effect",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("counterfactual trajectory records lack fidelity fields")
    delta_cosine = [
        float(row_cosine(np.asarray([left]), np.asarray([right]))[0])
        for left, right in zip(
            frame["teacher_delta"], frame["student_delta"], strict=True
        )
    ]
    effect_ratio = np.abs(frame["student_effect"].to_numpy()) / np.maximum(
        np.abs(frame["teacher_effect"].to_numpy()), 1e-12
    )
    return {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "status": "COMPLETE",
        "n": len(frame),
        "trajectory_delta_cosine_median": float(np.median(delta_cosine)),
        "semantic_direction_agreement": float(
            np.mean(
                frame["teacher_semantic_direction"].to_numpy()
                == frame["student_semantic_direction"].to_numpy()
            )
        ),
        "decision_agreement": float(
            np.mean(np.sign(frame["teacher_effect"]) == np.sign(frame["student_effect"]))
        ),
        "effect_ratio_median": float(np.median(effect_ratio)),
    }


def main() -> None:
    parser = standard_parser(
        "Run compact-memory remainder references", "configs/compact_memory_v3_1.yaml"
    )
    parser.add_argument("--stage", choices=("references", "fidelity"), default="references")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--controller-seed", type=int, default=20260828)
    args = parser.parse_args()
    context = initialize_context("compact-memory-references-v3-1", args)
    try:
        verify_freeze(context.root, kind="memory", config=context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        result = (
            _references(
                context,
                dimension=int(args.dimension),
                seed=int(args.controller_seed),
            )
            if args.stage == "references"
            else _fidelity(context)
        )
        output = context.raw_dir / context.run_id / f"{args.stage}.json"
        write_json_atomic(output, result)
        context.finish(
            "COMPLETED",
            stage=args.stage,
            result=str(output.relative_to(context.root)),
        )
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
