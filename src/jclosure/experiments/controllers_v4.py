"""Parameter-matched Markov/history/GRU training with autonomous v4 rollout."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.compact_memory_v3_1 import (
    LinearRepresentation,
    autonomous_rollout,
    parameter_count,
    row_cosine,
    scheduled_feedback_probability,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_v3_2 import (
    _build_controller_v32,
    _train_epoch,
)
from jclosure.experiments.predictive_state_v4 import _domain_arrays, _trace_summary
from jclosure.experiments.traces_v4 import ACTION_SURFACES
from jclosure.predictive_state_v4 import (
    PredictiveBottleneck,
    decode_numpy,
    encode_numpy,
)
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.statistics import clustered_bootstrap_ci

PROTOCOL = "compact_memory_v4"


class RepresentationAdapter:
    def __init__(self, selected: dict[str, Any], root: Path, device: torch.device):
        self.selected = selected
        self.device = device
        self.linear: LinearRepresentation | None
        self.learned: PredictiveBottleneck | None
        path = root / selected["representation_path"]
        if sha256_file(path) != selected["representation_sha256"]:
            raise RuntimeError("selected representation hash mismatch")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                self.linear = LinearRepresentation(
                    str(payload["family"].item()),
                    int(payload["dimension"]),
                    payload["mean"],
                    payload["encoder"],
                    payload["decoder"],
                )
            self.learned = None
            self.dimension = self.linear.dimension
        else:
            payload = torch.load(path, map_location="cpu")
            self.learned = PredictiveBottleneck(
                int(payload["input_dim"]),
                int(payload["dimension"]),
                int(payload["action_count"]),
                nonlinear=bool(payload["nonlinear"]),
            ).to(device)
            self.learned.load_state_dict(payload["state_dict"])
            self.learned.eval()
            self.linear = None
            self.dimension = int(payload["dimension"])

    def encode(self, values: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            return self.linear.encode(values).astype(np.float32)
        assert self.learned is not None
        return encode_numpy(self.learned, values, self.device).astype(np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            return self.linear.decode(values).astype(np.float32)
        assert self.learned is not None
        return decode_numpy(self.learned, values, self.device).astype(np.float32)


def _screen(context: Any) -> dict[str, Any]:
    payload = json.loads(
        (context.processed_dir / "predictive_state_screen_v4.json").read_text(
            encoding="utf-8"
        )
    )
    if not payload.get("selected"):
        candidates = list(payload.get("pareto", ()))
        if not candidates:
            raise RuntimeError("no v4 predictive-state candidates are available")
        selected = sorted(
            candidates,
            key=lambda value: (
                -min(
                    float(value["semantic_retention"]),
                    float(value["causal_direction_retention"]),
                    float(value["causal_magnitude_retention"]),
                ),
                -float(value["future_prediction_cosine"]),
                int(value["state_dimension"]),
                str(value["representation"]),
            ),
        )[0]
        payload = {
            **payload,
            "selected": selected,
            "selection_mode": "exploratory_maximin_retention_fallback",
            "exploratory_memory_only": True,
            "compact_state_authorized": False,
        }
    else:
        payload = {**payload, "selection_mode": "fully_gated"}
    return payload


def _trajectories(
    context: Any, adapter: RepresentationAdapter, domain: str
) -> list[dict[str, Any]]:
    _, _, _, values = _domain_arrays(context, _trace_summary(context), domain)
    output = []
    for value in values:
        output.append(
            {
                **value,
                "latent": adapter.encode(value["states"]),
                "ground_truth_actions": value["actions"],
                "teacher_correct": True,
            }
        )
    return output


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    adapter: RepresentationAdapter,
    *,
    family: str,
    history: int,
    memory_dim: int,
    device: torch.device,
    horizons: list[int],
    divergence_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    rows = []
    for trajectory in trajectories:
        maximum = min(max(horizons), len(trajectory["latent"]) - 1)
        if maximum < 1:
            continue
        initial = torch.from_numpy(trajectory["latent"][:1]).float().to(device)
        predicted, action_logits = autonomous_rollout(
            model,
            initial,
            steps=maximum,
            family=family,
            history=history,
            memory_dim=memory_dim,
        )
        predicted_latent = predicted[0].detach().cpu().numpy()
        decoded = adapter.decode(predicted_latent)
        target = trajectory["states"][1 : maximum + 1]
        cosine = row_cosine(decoded, target)
        predicted_actions = action_logits[0].argmax(-1).detach().cpu().numpy()
        target_actions = trajectory["actions"][1 : maximum + 1]
        for horizon in horizons:
            if horizon > maximum:
                continue
            prefix_cosine = cosine[:horizon]
            below = np.flatnonzero(prefix_cosine < divergence_threshold)
            rows.append(
                {
                    "example_id": trajectory["example_id"],
                    "family": trajectory["family"],
                    "task_horizon": trajectory["horizon"],
                    "horizon": int(horizon),
                    "latent_state_similarity": float(prefix_cosine[-1]),
                    "trajectory_divergence": float(np.mean(1 - prefix_cosine)),
                    "teacher_action_fidelity": float(
                        np.mean(predicted_actions[:horizon] == target_actions[:horizon])
                    ),
                    "ground_truth_action_accuracy": float(
                        np.mean(predicted_actions[:horizon] == target_actions[:horizon])
                    ),
                    "final_task_accuracy": float(
                        predicted_actions[horizon - 1] == target_actions[horizon - 1]
                    ),
                    "full_action_trajectory_accuracy": float(
                        np.array_equal(
                            predicted_actions[:horizon], target_actions[:horizon]
                        )
                    ),
                    "time_to_divergence": int(below[0] + 1)
                    if len(below)
                    else int(horizon + 1),
                    "finite": bool(np.isfinite(predicted_latent[:horizon]).all()),
                    "latent_variance": float(np.var(predicted_latent[:horizon])),
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    if not frame.empty:
        for horizon, group in frame.groupby("horizon", sort=True):
            summaries.append(
                {
                    "horizon": int(horizon),
                    "n": int(len(group)),
                    "latent_state_similarity_median": float(
                        group["latent_state_similarity"].median()
                    ),
                    "trajectory_divergence_mean": float(
                        group["trajectory_divergence"].mean()
                    ),
                    "teacher_action_fidelity": float(
                        group["teacher_action_fidelity"].mean()
                    ),
                    "ground_truth_action_accuracy": float(
                        group["ground_truth_action_accuracy"].mean()
                    ),
                    "final_task_accuracy": float(group["final_task_accuracy"].mean()),
                    "full_action_trajectory_accuracy": float(
                        group["full_action_trajectory_accuracy"].mean()
                    ),
                    "time_to_divergence_median": float(
                        group["time_to_divergence"].median()
                    ),
                    "finite_rate": float(group["finite"].mean()),
                    "latent_variance_median": float(group["latent_variance"].median()),
                }
            )
    return rows, summaries


def _train(
    context: Any,
    *,
    family: str,
    history: int,
    memory_dim: int,
    controller_seed: int,
) -> dict[str, Any]:
    screen = _screen(context)
    selected = screen["selected"]
    device = torch.device(
        f"cuda:{int(context.config['model'].get('device', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    adapter = RepresentationAdapter(selected, context.root, device)
    train = _trajectories(context, adapter, "train")
    validation = _trajectories(context, adapter, "validation")
    test = _trajectories(context, adapter, "rollout_test")
    section = context.config["compact_state_v4"]
    torch.manual_seed(controller_seed)
    model = _build_controller_v32(
        family,
        state_dim=adapter.dimension,
        action_count=len(ACTION_SURFACES),
        target=int(section["controller_parameter_target"]),
        tolerance=float(section["parameter_tolerance"]),
        history=history,
        memory_dim=memory_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    maximum_epochs = int(section["maximum_epochs"])
    patience = int(section["early_stopping_patience"])
    best = -float("inf")
    best_state = None
    stale = 0
    training = []
    validation_rows: list[dict[str, Any]] = []
    validation_summary: list[dict[str, Any]] = []
    horizons = [int(value) for value in section["rollout_horizons"]]
    for epoch in range(maximum_epochs):
        feedback = scheduled_feedback_probability(
            epoch,
            maximum_epochs,
            warmup_fraction=float(section["teacher_forcing_fraction"]),
            maximum_feedback=float(section["maximum_predicted_feedback"]),
        )
        loss = _train_epoch(
            model,
            train,
            optimizer,
            family=family,
            history=history,
            memory_dim=memory_dim,
            feedback_probability=feedback,
            device=device,
            seed=controller_seed + epoch,
        )
        validation_rows, validation_summary = _evaluate(
            model,
            validation,
            adapter,
            family=family,
            history=history,
            memory_dim=memory_dim,
            device=device,
            horizons=horizons,
            divergence_threshold=float(section["divergence_cosine"]),
        )
        available = {
            row["horizon"]: row["latent_state_similarity_median"]
            for row in validation_summary
            if row["horizon"] <= 8
        }
        score = available[max(available)] if available else -float("inf")
        training.append(
            {
                "epoch": epoch,
                "loss": loss,
                "feedback_probability": feedback,
                "validation_horizon8_or_longest": score,
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
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("controller training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_rows, validation_summary = _evaluate(
        model,
        validation,
        adapter,
        family=family,
        history=history,
        memory_dim=memory_dim,
        device=device,
        horizons=horizons,
        divergence_threshold=float(section["divergence_cosine"]),
    )
    test_rows, test_summary = _evaluate(
        model,
        test,
        adapter,
        family=family,
        history=history,
        memory_dim=memory_dim,
        device=device,
        horizons=horizons,
        divergence_threshold=float(section["divergence_cosine"]),
    )
    artifact_root = context.root / "artifacts/controllers/v4" / context.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_root / "controller.pt"
    torch.save(model.state_dict(), checkpoint)
    validation_path = context.raw_dir / context.run_id / "validation_rollout.parquet"
    test_path = context.raw_dir / context.run_id / "test_rollout.parquet"
    pd.DataFrame(validation_rows).to_parquet(
        validation_path, index=False, compression="zstd"
    )
    pd.DataFrame(test_rows).to_parquet(test_path, index=False, compression="zstd")
    return {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "family": family,
        "history": history,
        "memory_dimension": memory_dim,
        "state_dimension": adapter.dimension,
        "controller_seed": controller_seed,
        "parameter_count": parameter_count(model),
        "representation": selected,
        "representation_selection_mode": screen["selection_mode"],
        "compact_state_authorized": bool(screen.get("compact_state_authorized")),
        "epochs_ran": len(training),
        "training": training,
        "validation": validation_summary,
        "test": test_summary,
        "validation_records": str(validation_path.relative_to(context.root)),
        "test_records": str(test_path.relative_to(context.root)),
        "checkpoint": str(checkpoint.relative_to(context.root)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "teacher_correct_training_trajectories": len(train),
        "teacher_correct_validation_trajectories": len(validation),
        "teacher_correct_rollout_trajectories": len(test),
    }


def _aggregate(context: Any) -> dict[str, Any]:
    results = []
    for path in context.raw_dir.glob("controllers-v4-*/controller_result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("protocol_version") == PROTOCOL:
            results.append(value)
    if not results:
        raise RuntimeError("no completed v4 controller results")
    expected = []
    seeds = [
        int(value) for value in context.config["compact_state_v4"]["controller_seeds"]
    ]
    for seed in seeds:
        expected.append(("markov", 1, 0, seed))
        expected.extend(
            ("history", int(history), 0, seed)
            for history in context.config["compact_state_v4"]["histories"]
        )
        expected.extend(
            ("gru", 1, int(memory), seed)
            for memory in context.config["compact_state_v4"]["memory_dimensions"]
        )
    by_key = {
        (
            value["family"],
            int(value["history"]),
            int(value["memory_dimension"]),
            int(value["controller_seed"]),
        ): value
        for value in results
    }
    missing = sorted(set(expected) - set(by_key))
    # Baseline selection uses validation only, separately for every seed.
    baseline: dict[int, tuple[str, int, int, int]] = {}
    for seed in seeds:
        candidates = [
            (key, value)
            for key, value in by_key.items()
            if key[3] == seed and key[0] in {"markov", "history"}
        ]
        if not candidates:
            continue
        baseline[seed] = max(
            candidates,
            key=lambda item: next(
                (
                    row["latent_state_similarity_median"]
                    for row in item[1]["validation"]
                    if row["horizon"] == 8
                ),
                -float("inf"),
            ),
        )[0]
    comparisons = []
    for memory in (
        int(value) for value in context.config["compact_state_v4"]["memory_dimensions"]
    ):
        seed_directions = []
        pooled = []
        for seed in seeds:
            gru_key = ("gru", 1, memory, seed)
            if seed not in baseline or gru_key not in by_key:
                continue
            left = pd.read_parquet(
                context.root / by_key[baseline[seed]]["test_records"]
            )
            right = pd.read_parquet(context.root / by_key[gru_key]["test_records"])
            left = left[left["horizon"] == 8]
            right = right[right["horizon"] == 8]
            paired = left.merge(
                right,
                on=["example_id", "horizon"],
                suffixes=("_baseline", "_gru"),
            )
            paired["cosine_gain"] = (
                paired["latent_state_similarity_gru"]
                - paired["latent_state_similarity_baseline"]
            )
            paired["trajectory_reduction"] = (
                paired["trajectory_divergence_baseline"]
                - paired["trajectory_divergence_gru"]
            ) / paired["trajectory_divergence_baseline"].clip(lower=1e-12)
            paired["action_change"] = (
                paired["ground_truth_action_accuracy_gru"]
                - paired["ground_truth_action_accuracy_baseline"]
            )
            paired["seed"] = seed
            seed_directions.append(float(paired["cosine_gain"].mean()) > 0)
            pooled.append(paired)
        if not pooled:
            continue
        frame = pd.concat(pooled, ignore_index=True)
        frame["cluster"] = frame["seed"].astype(str) + ":" + frame["example_id"]
        ci = clustered_bootstrap_ci(
            frame,
            cluster_col="cluster",
            value_col="cosine_gain",
            n_resamples=10000,
            seed=int(context.config["reproducibility"]["bootstrap_seed"]),
        )
        utility = bool(
            ci.lower > 0
            and ci.estimate >= 0.02
            and float(frame["trajectory_reduction"].mean()) >= 0.20
            and float(frame["action_change"].mean()) >= -0.02
            and len(seed_directions) == len(seeds)
            and all(seed_directions)
        )
        comparisons.append(
            {
                "memory_dimension": memory,
                "cosine_gain": asdict(ci),
                "trajectory_distance_reduction": float(
                    frame["trajectory_reduction"].mean()
                ),
                "semantic_action_change": float(frame["action_change"].mean()),
                "all_three_seed_directions_positive": bool(
                    len(seed_directions) == len(seeds) and all(seed_directions)
                ),
                "memory_utility_gate_passed": utility,
            }
        )
    minimum = next(
        (
            value["memory_dimension"]
            for value in comparisons
            if value["memory_utility_gate_passed"]
        ),
        None,
    )
    output = {
        "schema_version": 6,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "completed_models": len(by_key),
        "expected_models": len(expected),
        "missing_models": missing,
        "validation_selected_baselines": {
            str(seed): list(key) for seed, key in baseline.items()
        },
        "memory_comparisons": comparisons,
        "minimum_beneficial_memory_dimension": minimum,
        "memory_utility_supported": minimum is not None,
    }
    path = context.processed_dir / "compact_memory_v4.json"
    write_json_atomic(path, output)
    return output


def main() -> None:
    parser = standard_parser(
        "Train and aggregate protocol-v4 compact-memory controllers",
        "configs/predictive_v4.yaml",
    )
    parser.add_argument("--stage", choices=("train", "aggregate"), required=True)
    parser.add_argument("--family", choices=("markov", "history", "gru"))
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--memory-dim", type=int, default=0)
    parser.add_argument("--controller-seed", type=int, default=20260828)
    args = parser.parse_args()
    context = initialize_context("controllers-v4", args)
    try:
        verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "train":
            if args.family is None:
                raise ValueError("--family is required for controller training")
            result = _train(
                context,
                family=args.family,
                history=int(args.history),
                memory_dim=int(args.memory_dim),
                controller_seed=int(args.controller_seed),
            )
            output = context.raw_dir / context.run_id / "controller_result.json"
            write_json_atomic(output, result)
            context.finish(
                "COMPLETED_MODEL", result=str(output.relative_to(context.root))
            )
            return
        result = _aggregate(context)
        context.finish(
            "COMPLETED_AGGREGATE",
            completed_models=result["completed_models"],
            missing_models=len(result["missing_models"]),
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
