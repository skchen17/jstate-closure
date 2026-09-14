"""Strong one-step and autonomous full-state references for protocol v4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jclosure.compact_memory_v3_1 import autonomous_rollout, row_cosine
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compact_memory_v3_2 import (
    _build_controller_v32,
    _train_epoch,
)
from jclosure.experiments.controllers_v4 import (
    RepresentationAdapter,
    _screen,
)
from jclosure.experiments.predictive_state_v4 import _domain_arrays, _trace_summary
from jclosure.experiments.traces_v4 import ACTION_SURFACES
from jclosure.protocol_v4 import verify_program_freeze
from jclosure.provenance import write_json_atomic

PROTOCOL = "full_state_reference_v4"


def _transition_arrays(
    context: Any, domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    _, _, _, trajectories = _domain_arrays(context, _trace_summary(context), domain)
    x_j, y_j, x_h, actions = [], [], [], []
    for value in trajectories:
        x_j.append(value["states"][:-1])
        y_j.append(value["states"][1:])
        x_h.append(value["full_states"][:-1])
        actions.append(value["actions"][1:])
    return (
        np.concatenate(x_j),
        np.concatenate(y_j),
        np.concatenate(x_h),
        np.concatenate(actions),
        trajectories,
    )


def _cosine_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    cosine = row_cosine(prediction, target)
    return {
        "mean": float(np.mean(cosine)),
        "median": float(np.median(cosine)),
        "p10": float(np.quantile(cosine, 0.10)),
    }


def _nonlinear_one_step(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    *,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a nonlinear reference without exposing any held-out targets."""

    torch.manual_seed(seed)
    width = 1024
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], width),
        torch.nn.GELU(),
        torch.nn.Linear(width, train_y.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    indices = np.arange(len(train_x))
    best = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    validation_input = torch.from_numpy(validation_x).float().to(device)
    validation_target = torch.from_numpy(validation_y).float().to(device)
    for epoch in range(50):
        generator.shuffle(indices)
        model.train()
        losses = []
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            inputs = torch.from_numpy(train_x[selected]).float().to(device)
            targets = torch.from_numpy(train_y[selected]).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = (1 - torch.nn.functional.cosine_similarity(prediction, targets, dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_input)
            score = float(
                torch.nn.functional.cosine_similarity(
                    validation_prediction, validation_target, dim=-1
                )
                .mean()
                .cpu()
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_cosine": score,
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
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("nonlinear one-step reference produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_prediction = model(validation_input).cpu().numpy()
        test_prediction = model(
            torch.from_numpy(test_x).float().to(device)
        ).cpu().numpy()
    return validation_prediction, test_prediction, history


def _one_step(
    context: Any, adapter: RepresentationAdapter
) -> tuple[dict[str, Any], Ridge, PCA, np.ndarray]:
    train_j, train_y, train_h, _, _ = _transition_arrays(context, "train")
    validation_j, validation_y, validation_h, _, _ = _transition_arrays(
        context, "validation"
    )
    test_j, test_y, test_h, _, _ = _transition_arrays(context, "rollout_test")
    train_z = adapter.encode(train_j)
    validation_z = adapter.encode(validation_j)
    test_z = adapter.encode(test_j)
    target_train_z = adapter.encode(train_y)
    hidden_from_z = Ridge(alpha=1.0, solver="lsqr").fit(train_z, train_h)
    train_hidden_remainder = train_h - hidden_from_z.predict(train_z)
    validation_hidden_remainder = validation_h - hidden_from_z.predict(validation_z)
    test_hidden_remainder = test_h - hidden_from_z.predict(test_z)
    components = min(512, len(train_h) - 1, train_h.shape[1])
    pca = PCA(components, svd_solver="randomized", random_state=0).fit(
        train_hidden_remainder
    )
    remainder_scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8)).astype(
        np.float32
    )
    train_r = (pca.transform(train_hidden_remainder) / remainder_scale).astype(
        np.float32
    )
    validation_r = (
        pca.transform(validation_hidden_remainder) / remainder_scale
    ).astype(np.float32)
    test_r = (pca.transform(test_hidden_remainder) / remainder_scale).astype(np.float32)
    inputs = {
        "compact_markov": (train_z, validation_z, test_z),
        "full_measured_j": (train_j, validation_j, test_j),
        "remainder_pca128": (train_r[:, :128], validation_r[:, :128], test_r[:, :128]),
        "remainder_pca512": (train_r, validation_r, test_r),
        "compact_plus_remainder_pca128": (
            np.concatenate((train_z, train_r[:, :128]), axis=1),
            np.concatenate((validation_z, validation_r[:, :128]), axis=1),
            np.concatenate((test_z, test_r[:, :128]), axis=1),
        ),
        "compact_plus_remainder_pca512": (
            np.concatenate((train_z, train_r), axis=1),
            np.concatenate((validation_z, validation_r), axis=1),
            np.concatenate((test_z, test_r), axis=1),
        ),
    }
    records = []
    predictions: dict[str, np.ndarray] = {}
    for name, (train_x, validation_x, test_x) in inputs.items():
        model = Ridge(alpha=1.0, solver="lsqr").fit(train_x, target_train_z)
        validation_prediction = adapter.decode(model.predict(validation_x))
        test_prediction = adapter.decode(model.predict(test_x))
        predictions[name] = test_prediction
        records.append(
            {
                "reference": name,
                "kind": "teacher_current_one_step",
                "input_dimension": int(train_x.shape[1]),
                "validation_decoded_j_cosine": _cosine_summary(
                    validation_prediction, validation_y
                ),
                "test_decoded_j_cosine": _cosine_summary(test_prediction, test_y),
            }
        )
    compact = next(value for value in records if value["reference"] == "compact_markov")
    combined = next(
        value
        for value in records
        if value["reference"] == "compact_plus_remainder_pca512"
    )
    combined_test = combined["test_decoded_j_cosine"]
    compact_test = compact["test_decoded_j_cosine"]
    assert isinstance(combined_test, dict)
    assert isinstance(compact_test, dict)
    information_gain = float(combined_test["mean"]) - float(compact_test["mean"])
    nonlinear_validation, nonlinear_test, nonlinear_history = _nonlinear_one_step(
        np.concatenate((train_z, train_r), axis=1),
        target_train_z,
        np.concatenate((validation_z, validation_r), axis=1),
        adapter.encode(validation_y),
        np.concatenate((test_z, test_r), axis=1),
        device=adapter.device,
        seed=int(context.seed),
    )
    nonlinear_validation_decoded = adapter.decode(nonlinear_validation)
    nonlinear_test_decoded = adapter.decode(nonlinear_test)
    nonlinear_record = {
        "reference": "nonlinear_compact_plus_remainder_pca512",
        "kind": "teacher_current_one_step",
        "input_dimension": int(train_z.shape[1] + train_r.shape[1]),
        "validation_decoded_j_cosine": _cosine_summary(
            nonlinear_validation_decoded, validation_y
        ),
        "test_decoded_j_cosine": _cosine_summary(nonlinear_test_decoded, test_y),
        "training": nonlinear_history,
    }
    records.append(nonlinear_record)
    nonlinear_test_summary = nonlinear_record["test_decoded_j_cosine"]
    assert isinstance(nonlinear_test_summary, dict)
    nonlinear_gain = float(nonlinear_test_summary["mean"]) - float(
        compact_test["mean"]
    )
    return (
        {
            "records": records,
            "compact_to_combined_gain": information_gain,
            "compact_to_nonlinear_combined_gain": nonlinear_gain,
            "interpretation": (
                "extra_hidden_information_detected"
                if max(information_gain, nonlinear_gain) >= 0.01
                else "no_material_extra_information_detected_by_tested_references"
            ),
        },
        hidden_from_z,
        pca,
        remainder_scale,
    )


def _with_remainder_trajectories(
    context: Any,
    adapter: RepresentationAdapter,
    hidden_from_z: Ridge,
    pca: PCA,
    remainder_scale: np.ndarray,
    domain: str,
) -> list[dict[str, Any]]:
    _, _, _, _, values = _transition_arrays(context, domain)
    output = []
    for value in values:
        latent = adapter.encode(value["states"])
        hidden_remainder = value["full_states"] - hidden_from_z.predict(latent)
        remainder = (pca.transform(hidden_remainder) / remainder_scale).astype(np.float32)
        output.append(
            {
                **value,
                "latent": np.concatenate((latent, remainder), axis=1),
                "ground_truth_actions": value["actions"],
                "teacher_correct": True,
            }
        )
    return output


@torch.no_grad()
def _evaluate_reference(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    adapter: RepresentationAdapter,
    *,
    device: torch.device,
    memory_dim: int,
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for value in trajectories:
        maximum = min(max(horizons), len(value["latent"]) - 1)
        initial = torch.from_numpy(value["latent"][:1]).float().to(device)
        predicted, actions = autonomous_rollout(
            model,
            initial,
            steps=maximum,
            family="gru",
            memory_dim=memory_dim,
        )
        predicted_np = predicted[0].cpu().numpy()
        decoded = adapter.decode(predicted_np[:, : adapter.dimension])
        target = value["states"][1 : maximum + 1]
        cosine = row_cosine(decoded, target)
        action_ids = actions[0].argmax(-1).cpu().numpy()
        expected = value["actions"][1 : maximum + 1]
        for horizon in horizons:
            if horizon > maximum:
                continue
            rows.append(
                {
                    "example_id": value["example_id"],
                    "family": value["family"],
                    "horizon": horizon,
                    "decoded_j_cosine": float(cosine[horizon - 1]),
                    "trajectory_distance": float(np.mean(1 - cosine[:horizon])),
                    "ground_truth_action_accuracy": float(
                        np.mean(action_ids[:horizon] == expected[:horizon])
                    ),
                    "final_task_accuracy": float(
                        action_ids[horizon - 1] == expected[horizon - 1]
                    ),
                    "finite": bool(np.isfinite(predicted_np[:horizon]).all()),
                }
            )
    return rows


def _autonomous(
    context: Any,
    adapter: RepresentationAdapter,
    hidden_from_z: Ridge,
    pca: PCA,
    remainder_scale: np.ndarray,
) -> dict[str, Any]:
    train = _with_remainder_trajectories(
        context, adapter, hidden_from_z, pca, remainder_scale, "train"
    )
    validation = _with_remainder_trajectories(
        context, adapter, hidden_from_z, pca, remainder_scale, "validation"
    )
    test = _with_remainder_trajectories(
        context, adapter, hidden_from_z, pca, remainder_scale, "rollout_test"
    )
    section = context.config["compact_state_v4"]
    device = adapter.device
    results = []
    for seed in (int(value) for value in section["controller_seeds"]):
        torch.manual_seed(seed)
        state_dim = adapter.dimension + pca.n_components_
        model = _build_controller_v32(
            "gru",
            state_dim=state_dim,
            action_count=len(ACTION_SURFACES),
            target=int(section["controller_parameter_target"]),
            tolerance=float(section["parameter_tolerance"]),
            history=1,
            memory_dim=256,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        best = -float("inf")
        best_state = None
        stale = 0
        history = []
        for epoch in range(int(section["maximum_epochs"])):
            feedback = min(
                float(section["maximum_predicted_feedback"]),
                max(
                    0.0,
                    (epoch / max(1, int(section["maximum_epochs"]) - 1) - 0.2)
                    / 0.8
                    * float(section["maximum_predicted_feedback"]),
                ),
            )
            loss = _train_epoch(
                model,
                train,
                optimizer,
                family="gru",
                history=1,
                memory_dim=256,
                feedback_probability=feedback,
                device=device,
                seed=seed + epoch,
            )
            validation_rows = _evaluate_reference(
                model,
                validation,
                adapter,
                device=device,
                memory_dim=256,
                horizons=[8],
            )
            score = float(
                np.median([value["decoded_j_cosine"] for value in validation_rows])
            )
            history.append({"epoch": epoch, "loss": loss, "validation_h8": score})
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
            raise RuntimeError("autonomous full-state reference failed to train")
        model.load_state_dict(best_state)
        rows = _evaluate_reference(
            model,
            test,
            adapter,
            device=device,
            memory_dim=256,
            horizons=[int(value) for value in section["rollout_horizons"]],
        )
        frame = pd.DataFrame(rows)
        summaries = []
        for horizon, group in frame.groupby("horizon", sort=True):
            summaries.append(
                {
                    "horizon": int(horizon),
                    "n": int(len(group)),
                    "decoded_j_cosine_median": float(
                        group["decoded_j_cosine"].median()
                    ),
                    "trajectory_distance_mean": float(
                        group["trajectory_distance"].mean()
                    ),
                    "ground_truth_action_accuracy": float(
                        group["ground_truth_action_accuracy"].mean()
                    ),
                    "final_task_accuracy": float(group["final_task_accuracy"].mean()),
                    "finite_rate": float(group["finite"].mean()),
                }
            )
        results.append(
            {"seed": seed, "training": history, "test": summaries, "rows": rows}
        )
    return {"state_dimension": adapter.dimension + pca.n_components_, "seeds": results}


def main() -> None:
    parser = standard_parser(
        "Run v4 full/remainder-aware references", "configs/predictive_v4.yaml"
    )
    args = parser.parse_args()
    context = initialize_context("references-v4", args)
    try:
        verify_program_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        screen = _screen(context)
        device = torch.device(
            f"cuda:{int(context.config['model'].get('device', 0))}"
            if torch.cuda.is_available()
            else "cpu"
        )
        adapter = RepresentationAdapter(screen["selected"], context.root, device)
        one_step, hidden_from_z, pca, remainder_scale = _one_step(context, adapter)
        autonomous = _autonomous(
            context, adapter, hidden_from_z, pca, remainder_scale
        )
        payload = {
            "schema_version": 6,
            "protocol_version": PROTOCOL,
            "run_id": context.run_id,
            "one_step": one_step,
            "autonomous_recurrent": autonomous,
            "reference_semantics": {
                "one_step": "teacher-current reference; not an autonomous oracle",
                "autonomous": "reads (Z0,H0) once and thereafter feeds back predicted Z/H",
            },
        }
        output = context.processed_dir / "full_state_references_v4.json"
        write_json_atomic(output, payload)
        context.finish("COMPLETED", summary=str(output.relative_to(context.root)))
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
