"""Frozen remainder-aware references for compact-memory protocol v3.2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from jclosure.compact_memory_references_v3_1 import (
    build_autonomous_remainder_reference,
    decoded_one_step_metrics,
    fit_linear_current_remainder_reference,
)
from jclosure.compact_memory_v3_1 import parameter_count, scheduled_feedback_probability
from jclosure.config import config_digest
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_references_v3_1 import (
    _evaluate_recurrent,
    _latent_arrays,
    _train_full_one_step,
    _train_recurrent_epoch,
    _trajectory_latents,
)
from jclosure.experiments.compact_memory_v3_1 import (
    ACTION_SURFACES,
    _load_representation,
)
from jclosure.experiments.compact_memory_v3_2 import _pairs
from jclosure.protocol_v3_2 import verify_memory_freeze
from jclosure.provenance import git_commit, sha256_file, write_json_atomic
from jclosure.runtime_v3_2 import MEMORY_PROTOCOL_V32

REFERENCE_PROTOCOL = "compact_memory_reference_exploratory_v3_2"
REFERENCE_FREEZE = Path("artifacts/compact_memory_reference_v3_2.freeze.json")
REFERENCE_SOURCES = (
    "configs/compact_memory_v3_2.yaml",
    "src/jclosure/compact_memory_references_v3_1.py",
    "src/jclosure/compact_memory_v3_1.py",
    "src/jclosure/experiments/compact_memory_references_v3_1.py",
    "src/jclosure/experiments/compact_memory_v3_2.py",
    "src/jclosure/experiments/compact_memory_references_v3_2.py",
)


def _normalized_digest(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    if isinstance(value.get("model"), dict):
        value["model"].pop("device", None)
    return config_digest(value)


def _freeze(context) -> dict[str, Any]:
    verify_memory_freeze(context.root, context.config)
    screen_path = context.processed_dir / "compact_memory_representation_screen_v3_2.json"
    trace_path = context.processed_dir / "compact_memory_trace_records_v3_2.parquet"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    selected = screen.get("overall_selected")
    if not screen.get("temporal_training_authorized") or not selected:
        raise RuntimeError("v3.2 representation screen did not authorize references")
    representation = context.root / str(selected["representation_path"])
    payload = {
        "schema_version": 5,
        "protocol_version": REFERENCE_PROTOCOL,
        "status": "FROZEN_FOR_EXECUTION",
        "freeze_created_from_commit": git_commit(context.root),
        "config_digest": _normalized_digest(context.config),
        "source_hashes": {
            value: sha256_file(context.root / value) for value in REFERENCE_SOURCES
        },
        "data_hashes": {
            str(screen_path.relative_to(context.root)): sha256_file(screen_path),
            str(trace_path.relative_to(context.root)): sha256_file(trace_path),
            str(representation.relative_to(context.root)): sha256_file(representation),
        },
        "selected_representation": selected,
    }
    write_json_atomic(context.root / REFERENCE_FREEZE, payload)
    return payload


def _verify_freeze(context) -> dict[str, Any]:
    verify_memory_freeze(context.root, context.config)
    payload = json.loads((context.root / REFERENCE_FREEZE).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != REFERENCE_PROTOCOL:
        raise RuntimeError("compact-memory reference freeze protocol mismatch")
    if payload.get("config_digest") != _normalized_digest(context.config):
        raise RuntimeError("compact-memory reference config differs from freeze")
    for section in ("source_hashes", "data_hashes"):
        for relative, expected in payload[section].items():
            path = context.root / relative
            observed = sha256_file(path) if path.is_file() else "MISSING"
            if observed != expected:
                raise RuntimeError(
                    f"compact-memory reference freeze mismatch: {relative}"
                )
    return payload


def _data(context, freeze: dict[str, Any]):
    frame = pd.read_parquet(
        context.processed_dir / "compact_memory_trace_records_v3_2.parquet"
    )
    arrays, trajectories = _pairs(context, frame)
    selected = freeze["selected_representation"]
    representation = _load_representation(
        context.root, str(selected["representation_path"])
    )
    return _latent_arrays(arrays, representation), trajectories, representation


def _references(context, *, seed: int) -> dict[str, Any]:
    freeze = _verify_freeze(context)
    arrays, raw_trajectories, representation = _data(context, freeze)
    config = context.config["compact_memory"]
    linear = fit_linear_current_remainder_reference(
        arrays["train_z"],
        arrays["train_r"],
        arrays["train_next_z"],
        remainder_dimension=int(config["remainder_pca_dimensions"][0]),
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
        int(config["remainder_pca_dimensions"][-1]),
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
    training: list[dict[str, Any]] = []
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
        / f"compact_memory_remainder_reference_v3_2_s{seed}.parquet"
    )
    pd.DataFrame(rows).to_parquet(records, index=False)
    checkpoint_root = context.root / "artifacts/checkpoints/v3_2"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    nonlinear_checkpoint = checkpoint_root / f"full-remainder-s{seed}.pt"
    recurrent_checkpoint = checkpoint_root / f"autonomous-remainder-s{seed}.pt"
    torch.save(nonlinear_model.state_dict(), nonlinear_checkpoint)
    torch.save(recurrent.state_dict(), recurrent_checkpoint)
    return {
        "schema_version": 5,
        "protocol_version": REFERENCE_PROTOCOL,
        "base_protocol_version": MEMORY_PROTOCOL_V32,
        "run_id": context.run_id,
        "state_dimension": representation.dimension,
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
            "autonomous_recurrent": str(recurrent_checkpoint.relative_to(context.root)),
        },
    }


def _fidelity(context) -> dict[str, Any]:
    _verify_freeze(context)
    trajectories = (
        context.processed_dir
        / "compact_memory_counterfactual_trajectories_v3_2.parquet"
    )
    if not trajectories.is_file():
        return {
            "schema_version": 5,
            "protocol_version": REFERENCE_PROTOCOL,
            "status": "UNAVAILABLE_NO_VALIDATED_TOKEN_TRAJECTORY",
            "reason": (
                "No validated teacher J-swap token trajectory exists; observational "
                "rollout is not causal-fidelity evidence."
            ),
        }
    raise NotImplementedError("validated v3.2 token counterfactuals need an explicit endpoint")


def main() -> None:
    parser = standard_parser(
        "Run compact-memory remainder references v3.2",
        "configs/compact_memory_v3_2.yaml",
    )
    parser.add_argument(
        "--stage", choices=("freeze", "references", "fidelity"), default="references"
    )
    parser.add_argument("--controller-seed", type=int, default=20260828)
    args = parser.parse_args()
    context = initialize_context("compact-memory-references-v3-2", args)
    try:
        if args.stage == "freeze":
            result = _freeze(context)
        elif args.stage == "references":
            result = _references(context, seed=int(args.controller_seed))
        else:
            result = _fidelity(context)
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
