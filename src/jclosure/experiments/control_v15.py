"""Development-only quantization-aware finite closed loop and reachability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.config import load_config
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.actuation_v15 import apply, components, transfer_metrics
from jclosure.experiments.closed_loop_v14 import (
    _metrics, _raw_teacher_delta, _selection, _vector, _weighted,
)
from jclosure.experiments.curvature_v14 import _combined_rows
from jclosure.experiments.decoded_causal_v10 import apply_decoded_state
from jclosure.experiments.operator_v15 import _realization
from jclosure.experiments.persistent_channels_v7 import (
    _prefill, _teacher_forced_trajectory, _teacher_tokens,
)
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")
OBJECTIVES = {"h1": [1], "h1_h2_h4": [1, 2, 4], "h1_h2_h4_h8": [1, 2, 4, 8]}
EPSILON = 1.0
INITIAL_PROBES = 64
MAX_CONTROL_RANK = 8
TRUST_ACCEPT_RATIO = 0.1
TRUST_SHRINK_RATIO = 0.25
TRUST_EXPAND_RATIO = 0.75
MIN_TRUST = 0.05
MAX_TRUST = 1.0
REACHABILITY_OUTSIDE_FLOOR = 0.2


def _teacher_trajectory(root: Path, base_id: str, task: Any, bundle: Any, dense_map: Any, config: dict[str, Any], recurrent: list[int], attention: list[int], measured: list[int]) -> tuple[Any, list[int], dict[str, Any], dict[str, Any], dict[str, Any]]:
    v8 = config["persistent_state_v8"]
    clean = _prefill(bundle, task.prompt, measured_layers=measured, dense_map=dense_map, intervention_layer=int(v8["intervention"]["layer"]), candidate=None)
    tokens = _teacher_tokens(bundle, clean, count=8, measured_layers=measured, dense_map=dense_map)
    clean_trajectory = _teacher_forced_trajectory(bundle, clean["cache"], tokens, prompt_length=clean["prompt_length"], measured_layers=measured, dense_map=dense_map)
    raw_delta = _raw_teacher_delta(root, base_id)
    teacher_cache = apply_decoded_state(clean["cache"], raw_delta, recurrent_layers=recurrent, attention_layers=attention, prompt_length=clean["prompt_length"])
    teacher = _teacher_forced_trajectory(bundle, teacher_cache, tokens, prompt_length=clean["prompt_length"], measured_layers=measured, dense_map=dense_map)
    return clean, tokens, clean_trajectory, teacher, teacher_cache


def _trajectory(bundle: Any, cache: Any, tokens: list[int], prompt_length: int, measured: list[int], dense_map: Any) -> dict[str, Any]:
    return _teacher_forced_trajectory(bundle, cache, tokens, prompt_length=prompt_length, measured_layers=measured, dense_map=dense_map)


def _projection_floor(matrix: np.ndarray, residual: np.ndarray) -> tuple[int, float]:
    if matrix.size == 0 or float(np.linalg.norm(matrix)) == 0:
        return 0, 1.0
    basis, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tol = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int((singular > tol).sum())
    projected = basis[:, :rank] @ (basis[:, :rank].T @ residual)
    floor = float(np.linalg.norm(residual - projected) / max(float(np.linalg.norm(residual)), 1e-20))
    return rank, floor


def _channel_realization(cache: Any, edited: Any, direction: dict[str, torch.Tensor], recurrent: list[int], attention: list[int]) -> dict[str, dict[str, float | None]]:
    totals = {name: [0.0, 0.0] for name in ("recurrent", "conv", "kv")}
    for channel, layer, name, old, probe in components(cache, direction, recurrent, attention):
        actual = getattr(edited.layers[layer], name)
        if channel in ("keys", "values"):
            actual = actual[..., : old.shape[-2], :]
        item = transfer_metrics(old, probe, actual, 1.0)
        key = "kv" if channel in ("keys", "values") else channel
        totals[key][0] += float(item["requested_norm"]) ** 2
        totals[key][1] += float(item["realized_norm"]) ** 2
    return {name: {"requested_norm": float(np.sqrt(a)), "realized_norm": float(np.sqrt(b)), "gain": float(np.sqrt(b / a)) if a > 0 else None} for name, (a, b) in totals.items()}


def run(root: Path) -> dict[str, Any]:
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_control.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "control", ["src/jclosure/experiments/control_v15.py", "results/v15/processed/actuator_basis_v15.json", "results/v15/processed/finite_response_linearity_v15.json", "artifacts/quantization_aware_actuation_v15_splits.freeze.json"], {"role": "development_validation", "independent_confirmatory": False, "initial_probes": INITIAL_PROBES, "epsilon": EPSILON, "basis_rank_rule": "min(8, local realized-cost-weighted r95)", "objective_horizons": OBJECTIVES, "trust_accept_ratio": TRUST_ACCEPT_RATIO, "trust_shrink_ratio": TRUST_SHRINK_RATIO, "trust_expand_ratio": TRUST_EXPAND_RATIO, "trust_min": MIN_TRUST, "trust_max": MAX_TRUST, "reachable_projection_floor_threshold": REACHABILITY_OUTSIDE_FLOOR, "basis_reestimated_from_finite_response_at_each_step": True, "raw_teacher_state_used_only_for_teacher_response_label": True, "numerical_gate_failed_so_exploratory_only": True})
    else:
        stage = json.loads(stage_path.read_text())
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    bundle = _load_model(config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    context = type("EncoderContext", (), {"root": root, "config": config})()
    _, _, dense_map = _load_encoder_memory_efficient(context, bundle)
    data = geometry._load_features(root)
    panel = geometry._panel_ids(data, "validation", int(base["config"]["development"]["per_family"]))
    directions = torch.load(root / geometry.DIRECTIONS, map_location="cpu", weights_only=False)
    metadata = geometry._pair_metadata(root)
    from jclosure.datasets_v8 import load_tasks
    tasks = {task.example_id: task for _, task in load_tasks(root / geometry.SELECTION_PATH)}
    v8 = config["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    floors = json.loads((root / "results/v14/processed/numerical_snr_summary_v14.json").read_text())["quantization_floors"]
    atlas = geometry._atlas(root)
    with np.load(root / atlas[0]["matrix_path"], allow_pickle=False) as payload:
        j_indices = payload["selected_j"].astype(int)
    rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    reach_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for case_index, base_id in enumerate(panel):
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean, tokens, clean_trajectory, teacher, _ = _teacher_trajectory(root, base_id, task, bundle, dense_map, config, recurrent, attention, measured)
        logit_indices = _selection(clean_trajectory, task, bundle)
        initial_outputs: dict[str, list[np.ndarray]] = {key: [] for key in OBJECTIVES}
        realized_cost: list[float] = []
        for direction_index in range(INITIAL_PROBES):
            direction = {name: directions[name][direction_index] for name in ("recurrent", "conv", "kv")}
            plus_cache = apply(clean["cache"], EPSILON, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
            minus_cache = apply(clean["cache"], -EPSILON, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
            plus_trajectory = _trajectory(bundle, plus_cache, tokens, clean["prompt_length"], measured, dense_map)
            minus_trajectory = _trajectory(bundle, minus_cache, tokens, clean["prompt_length"], measured, dense_map)
            realized = _realization(clean["cache"], plus_cache, direction, recurrent, attention, EPSILON)
            realized_cost.append(float(realized["realized_norm"]))
            for objective, horizons in OBJECTIVES.items():
                plus_vector, _ = _vector(plus_trajectory, horizons, j_indices, logit_indices, main_layer)
                minus_vector, _ = _vector(minus_trajectory, horizons, j_indices, logit_indices, main_layer)
                initial_outputs[objective].append((plus_vector - minus_vector) / (2 * EPSILON))
            del plus_cache, minus_cache
        for objective, horizons in OBJECTIVES.items():
            teacher_vector, blocks = _vector(teacher, horizons, j_indices, logit_indices, main_layer)
            current_vector, _ = _vector(clean_trajectory, horizons, j_indices, logit_indices, main_layer)
            initial_residual = teacher_vector - current_vector
            weight = _weighted(initial_residual, blocks, floors)
            initial_r = float(np.linalg.norm(initial_residual))
            initial_wr = float(np.linalg.norm(weight * initial_residual))
            response = np.column_stack(initial_outputs[objective])
            cost = np.asarray(realized_cost, dtype=np.float64)
            cost_floor = max(float(np.median(cost[cost > 0])) * 0.01, 1e-12) if np.any(cost > 0) else 1e-12
            inv_cost = 1.0 / np.maximum(cost, cost_floor)
            weighted = (weight[:, None] * response) * inv_cost[None, :]
            u_initial, singular, vh = np.linalg.svd(weighted, full_matrices=False)
            energy = singular**2
            r95 = int(np.searchsorted(np.cumsum(energy) / max(float(energy.sum()), 1e-20), 0.95) + 1)
            rank = min(MAX_CONTROL_RANK, r95)
            rank_rows.extend({"role": "development_validation", "base_trial_id": base_id, "objective": objective, "rank": r, "projection_floor": _projection_floor(u_initial[:, :min(r, len(singular))], weight * initial_residual)[1], "r95": r95} for r in sorted(set([*base["config"]["development"]["ranks"], r95])))
            coefficients = (vh[:rank] * inv_cost[None, :]).astype(np.float32)
            basis_rows = _combined_rows({name: directions[name][:INITIAL_PROBES] for name in ("recurrent", "conv", "kv")}, coefficients)
            cache = clean["cache"]
            trajectory = clean_trajectory
            trust_radius = float(base["config"]["development"]["trust_radius"])
            local_matrices = []
            for step in range(int(base["config"]["development"]["maximum_steps"])):
                current_vector, _ = _vector(trajectory, horizons, j_indices, logit_indices, main_layer)
                residual = teacher_vector - current_vector
                current_r = float(np.linalg.norm(residual))
                current_wr = float(np.linalg.norm(weight * residual))
                columns = []
                basis_realization = []
                for direction in basis_rows:
                    plus_cache = apply(cache, EPSILON, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
                    minus_cache = apply(cache, -EPSILON, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
                    plus_trajectory = _trajectory(bundle, plus_cache, tokens, clean["prompt_length"], measured, dense_map)
                    minus_trajectory = _trajectory(bundle, minus_cache, tokens, clean["prompt_length"], measured, dense_map)
                    plus_vector, _ = _vector(plus_trajectory, horizons, j_indices, logit_indices, main_layer)
                    minus_vector, _ = _vector(minus_trajectory, horizons, j_indices, logit_indices, main_layer)
                    columns.append((plus_vector - minus_vector) / (2 * EPSILON))
                    basis_realization.append(_realization(cache, plus_cache, direction, recurrent, attention, EPSILON))
                    del plus_cache, minus_cache
                local = np.column_stack(columns)
                local_matrices.append(weight[:, None] * local)
                weighted_residual = weight * residual
                weighted_local = weight[:, None] * local
                ridge = float(base["config"]["development"]["ridge"])
                control = np.linalg.solve(weighted_local.T @ weighted_local + ridge * np.eye(rank), weighted_local.T @ weighted_residual)
                control_norm = float(np.linalg.norm(control))
                if control_norm > trust_radius:
                    control *= trust_radius / control_norm
                predicted_wr = float(np.linalg.norm(weighted_residual - weighted_local @ control))
                accepted = False
                chosen_scale = 0.0
                chosen_realization = None
                chosen_channels = None
                candidate_cache = cache
                candidate_trajectory = trajectory
                actual_wr = current_wr
                actual_r = current_r
                trust_ratio = None
                for scale in base["config"]["development"]["backtracking"]:
                    joint = {name: sum(float(control[i] * scale) * basis_rows[i][name] for i in range(rank)) for name in ("recurrent", "conv", "kv")}
                    test_cache = apply(cache, 1.0, joint, recurrent, attention, "native_fp32_add_bf16_writeback")
                    test_trajectory = _trajectory(bundle, test_cache, tokens, clean["prompt_length"], measured, dense_map)
                    test_vector, _ = _vector(test_trajectory, horizons, j_indices, logit_indices, main_layer)
                    test_wr = float(np.linalg.norm(weight * (teacher_vector - test_vector)))
                    predicted_scaled = float(np.linalg.norm(weighted_residual - weighted_local @ (scale * control)))
                    predicted_gain = current_wr - predicted_scaled
                    ratio = (current_wr - test_wr) / predicted_gain if predicted_gain > 1e-12 else -float("inf")
                    if test_wr < current_wr - 1e-8 and ratio >= TRUST_ACCEPT_RATIO:
                        accepted = True
                        chosen_scale = float(scale)
                        chosen_realization = _realization(cache, test_cache, joint, recurrent, attention, 1.0)
                        chosen_channels = _channel_realization(cache, test_cache, joint, recurrent, attention)
                        candidate_cache = test_cache
                        candidate_trajectory = test_trajectory
                        actual_wr = test_wr
                        actual_r = float(np.linalg.norm(teacher_vector - test_vector))
                        trust_ratio = float(ratio)
                        break
                rows.append({
                    "role": "development_validation", "freeze_digest": stage["freeze_digest"],
                    "base_trial_id": base_id, "family": pair["family"], "method": f"actuator_calibrated_{objective}",
                    "step": step + 1, "rank": rank, "r95_initial": r95,
                    "accepted": accepted, "backtracking_scale": chosen_scale,
                    "predicted_improvement": current_wr - predicted_wr,
                    "actual_improvement": current_wr - actual_wr,
                    "requested_control_norm": float(np.linalg.norm(control) * chosen_scale),
                    "realized_state_control_norm": chosen_realization["realized_norm"] if chosen_realization else 0.0,
                    "quantization_loss": 1 - chosen_realization["gain"] if chosen_realization and chosen_realization["gain"] is not None else None,
                    "realized_state_cosine": chosen_realization["cosine"] if chosen_realization else None,
                    "channel_realization_json": json.dumps(chosen_channels, sort_keys=True) if chosen_channels else None,
                    "basis_median_realization_gain": float(np.median([x["gain"] for x in basis_realization if x["gain"] is not None])) if any(x["gain"] is not None for x in basis_realization) else None,
                    "target_residual": actual_r,
                    "raw_residual_ratio": actual_r / max(initial_r, 1e-20),
                    "weighted_residual_ratio": actual_wr / max(initial_wr, 1e-20),
                    "trust_ratio": trust_ratio, "trust_radius": trust_radius,
                })
                if accepted:
                    cache = candidate_cache
                    trajectory = candidate_trajectory
                if trust_ratio is None or trust_ratio < TRUST_SHRINK_RATIO:
                    trust_radius = max(MIN_TRUST, 0.5 * trust_radius)
                elif trust_ratio > TRUST_EXPAND_RATIO:
                    trust_radius = min(MAX_TRUST, 1.5 * trust_radius)
                if not accepted:
                    break
            one_rank, one_floor = _projection_floor(local_matrices[0], weight * initial_residual)
            multi_rank, multi_floor = _projection_floor(np.concatenate(local_matrices, axis=1), weight * initial_residual)
            reach_rows.append({"role": "development_validation", "freeze_digest": stage["freeze_digest"], "base_trial_id": base_id, "objective": objective, "one_step_reachable_rank": one_rank, "multi_step_reachable_rank": multi_rank, "one_step_projection_floor": one_floor, "multi_step_projection_floor": multi_floor, "target_outside_tested_actuator_reachable_space": multi_floor > REACHABILITY_OUTSIDE_FLOOR, "transport_method": "same frozen target coordinates; empirical local response columns, no full transition matrix"})
            fidelity_rows.extend(_metrics(clean_trajectory, teacher, trajectory, task, bundle, main_layer, base_id, pair["family"], f"actuator_calibrated_{objective}", sum(row["base_trial_id"] == base_id and row["method"] == f"actuator_calibrated_{objective}" for row in rows)))
        (root / OUT).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(root / OUT / "control_development_v15.parquet", index=False, compression="zstd")
        pd.DataFrame(fidelity_rows).to_parquet(root / OUT / "control_fidelity_v15.parquet", index=False, compression="zstd")
        pd.DataFrame(reach_rows).to_parquet(root / OUT / "finite_controllability_v15.parquet", index=False, compression="zstd")
        pd.DataFrame(rank_rows).to_parquet(root / OUT / "control_rank_candidates_v15.parquet", index=False, compression="zstd")
        write_json_atomic(root / OUT / "control_progress_v15.json", {"completed_cases": case_index + 1, "total_cases": len(panel)})
    steps = pd.DataFrame(rows)
    fidelity = pd.DataFrame(fidelity_rows)
    reach = pd.DataFrame(reach_rows)
    summary = {
        "role": "development_validation", "freeze_digest": stage["freeze_digest"],
        "independent_confirmatory": False, "panel_count": len(panel),
        "residual_curves": steps.groupby(["method", "step"]).agg(raw_ratio=("raw_residual_ratio", "median"), weighted_ratio=("weighted_residual_ratio", "median"), acceptance=("accepted", "mean")).reset_index().to_dict("records"),
        "fidelity": fidelity.groupby(["method", "horizon"]).agg(direction=("direction_cosine", "mean"), magnitude=("magnitude_ratio", "mean"), output=("output_direction_cosine", "mean"), semantic_legacy=("semantic_delta_agreement", "mean"), semantic_continuous=("semantic_vector_cosine", "mean"), sign=("task_decision_sign_agreement", "mean")).reset_index().to_dict("records"),
        "reachability": reach.groupby("objective").agg(one_step_rank=("one_step_reachable_rank", "median"), multi_step_rank=("multi_step_reachable_rank", "median"), one_step_floor=("one_step_projection_floor", "median"), multi_step_floor=("multi_step_projection_floor", "median"), outside_fraction=("target_outside_tested_actuator_reachable_space", "mean")).reset_index().to_dict("records"),
        "records": {name: {"path": str(OUT / filename), "sha256": sha256_file(root / OUT / filename)} for name, filename in (("steps", "control_development_v15.parquet"), ("fidelity", "control_fidelity_v15.parquet"), ("reachability", "finite_controllability_v15.parquet"), ("ranks", "control_rank_candidates_v15.parquet"))},
    }
    write_json_atomic(root / OUT / "finite_causal_control_v15.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), sort_keys=True)[:3000])
