"""Exploratory finite-response closed-loop steering in a shared V13 probe space.

The frozen V14 numerical audit did not validate an infinitesimal JVP, so this
stage uses actual central finite responses, not the unvalidated JVP, and is
strictly developmental until an independent finalist can be frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.config import load_config
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.causal_v12 import _continuous_semantic, _effect_metrics
from jclosure.experiments.curvature_v14 import _combined_rows
from jclosure.experiments.decoded_causal_v10 import apply_decoded_state
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import _semantic_ids
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v14/processed")
RANK = 9
EPSILON = 1.0
TRUST_RADIUS = 0.5
RIDGE = 0.01
MAX_STEPS = 4
BACKTRACK = [1.0, 0.5, 0.25]
OBJECTIVES = {"h1": [1], "h1_h2_h4_h8": [1, 2, 4, 8]}


def _raw_teacher_delta(
    root: Path, base_id: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = json.loads(
        (root / "results/v13/processed/causal_capture_validation_v13.json").read_text()
    )
    shard = next(
        item for item in manifest["state_shards"] if base_id in item["base_trial_ids"]
    )
    path = root / shard["path"]
    if sha256_file(path) != shard["sha256"]:
        raise RuntimeError("frozen V13 teacher-state shard hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    case = next(row for row in payload["rows"] if row["base_trial_id"] == base_id)
    clean, pert = case["clean"], case["perturbed"]
    kv = torch.stack(
        [
            torch.stack(
                [
                    pert["kv"][layer][name] - clean["kv"][layer][name]
                    for name in ("keys", "values")
                ]
            )
            for layer in ("27", "31")
        ]
    )
    return (
        pert["recurrent"] - clean["recurrent"],
        pert["conv"] - clean["conv"],
        kv,
    )


def _selection(clean: dict[str, Any], task: Any, bundle: Any) -> list[int]:
    semantic = []
    for horizon in (1, 2, 4, 8):
        action = task.semantic_actions[min(horizon - 1, len(task.semantic_actions) - 1)]
        semantic.extend(_semantic_ids(bundle.tokenizer, action))
    clean_logits = clean["logits"]
    top = np.argsort(-clean_logits[0])[:32].tolist()
    return list(dict.fromkeys([*semantic, *top]))[:32]


def _vector(
    trajectory: dict[str, Any],
    horizons: list[int],
    j_indices: np.ndarray,
    logit_indices: list[int],
    main_layer: int,
) -> tuple[np.ndarray, dict[str, slice]]:
    parts = []
    blocks: dict[str, slice] = {}
    for horizon in horizons:
        index = horizon - 1
        logits = trajectory["logits"][index].astype(np.float64)
        semantic = logits[logit_indices] - np.logaddexp.reduce(logits)
        values = {
            "j": trajectory["j"][main_layer][index][j_indices].astype(np.float64),
            "logits": logits[logit_indices],
            "semantic_continuous": semantic,
            "workspace": np.concatenate(
                [
                    trajectory["hidden"][layer][index][:32].astype(np.float64)
                    for layer in (23, 26, 30)
                ]
            ),
        }
        for target, value in values.items():
            start = sum(len(part) for part in parts)
            parts.append(value)
            blocks[f"{target}_h{horizon}"] = slice(start, start + len(value))
    return np.concatenate(parts), blocks


def _weighted(
    residual: np.ndarray, blocks: dict[str, slice], floors: dict[str, float]
) -> np.ndarray:
    weight = np.empty_like(residual)
    for name, region in blocks.items():
        target = name.split("_h")[0]
        size = region.stop - region.start
        norm = max(float(np.linalg.norm(residual[region])), float(floors[target]))
        weight[region] = 1.0 / (norm * np.sqrt(size))
    return weight


def _metrics(
    clean: dict[str, Any],
    teacher: dict[str, Any],
    candidate: dict[str, Any],
    task: Any,
    bundle: Any,
    main_layer: int,
    base_id: str,
    family: str,
    method: str,
    step: int,
) -> list[dict[str, Any]]:
    rows = []
    for horizon in (1, 2, 4, 8):
        semantic_index = min(horizon - 1, len(task.semantic_actions) - 1)
        metrics = _effect_metrics(
            clean,
            teacher,
            candidate,
            main_layer=main_layer,
            horizon=horizon,
            semantic_ids=_semantic_ids(
                bundle.tokenizer, task.semantic_actions[semantic_index]
            ),
        )
        teacher_effect = (
            teacher["j"][main_layer][horizon - 1] - clean["j"][main_layer][horizon - 1]
        )
        effect = (
            candidate["j"][main_layer][horizon - 1]
            - clean["j"][main_layer][horizon - 1]
        )
        rows.append(
            {
                "base_trial_id": base_id,
                "family": family,
                "method": method,
                "step": step,
                "horizon": horizon,
                **metrics,
                **_continuous_semantic(teacher_effect, effect, 10),
            }
        )
    return rows


def run(root: Path) -> dict[str, Any]:
    verify_base(root)
    freeze_path = (
        root / "artifacts/finite_causal_control_v14_closed_loop_development.freeze.json"
    )
    if not freeze_path.exists():
        frozen = freeze_stage(
            root,
            "closed_loop_development",
            [
                "src/jclosure/experiments/closed_loop_v14.py",
                "results/v14/processed/numerical_snr_summary_v14.json",
                "results/v13/processed/moving_tangent_oracle_development_v13.parquet",
                "artifacts/causal_bank_v13_capture.freeze.json",
            ],
            {
                "source_split": "V13 validation",
                "independent_confirmation": False,
                "rank": RANK,
                "finite_response_epsilon": EPSILON,
                "trust_radius": TRUST_RADIUS,
                "ridge": RIDGE,
                "maximum_steps": MAX_STEPS,
                "backtracking": BACKTRACK,
                "objectives": OBJECTIVES,
                "raw_teacher_state_used_only_for_target_effect": True,
                "method_is_exploratory_because_jvp_gate_failed": True,
            },
        )
    else:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    data = geometry._load_features(root)
    directions = torch.load(
        root / geometry.DIRECTIONS, map_location="cpu", weights_only=False
    )
    coefficients, _ = geometry._coefficients(data, directions)
    lookup = {value: i for i, value in enumerate(data["base_trial_id"].astype(str))}
    atlas = geometry._atlas(root)
    anchor_scores = coefficients[[lookup[row["base_trial_id"]] for row in atlas]]
    metadata = geometry._pair_metadata(root)
    tasks = {
        task.example_id: task for _, task in load_tasks(root / geometry.SELECTION_PATH)
    }
    panel = geometry._panel_ids(data, "validation", 2)
    bundle = _load_model(config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    context = type("EncoderContext", (), {"root": root, "config": config})()
    _, _, dense_map = _load_encoder_memory_efficient(context, bundle)
    v8 = config["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    floors = json.loads((root / OUT / "numerical_snr_summary_v14.json").read_text())[
        "quantization_floors"
    ]
    device = next(bundle.hf_model.parameters()).device
    basis_cache: dict[int, list[dict[str, torch.Tensor]]] = {}
    rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    for case_index, base_id in enumerate(panel):
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle, clean, count=8, measured_layers=measured, dense_map=dense_map
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        raw_delta = _raw_teacher_delta(root, base_id)
        teacher_cache = apply_decoded_state(
            clean["cache"],
            raw_delta,
            recurrent_layers=recurrent,
            attention_layers=attention,
            prompt_length=clean["prompt_length"],
        )
        teacher = _teacher_forced_trajectory(
            bundle,
            teacher_cache,
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        with np.load(root / atlas[0]["matrix_path"], allow_pickle=False) as payload:
            j_indices = payload["selected_j"].astype(int)
        logit_indices = _selection(clean_trajectory, task, bundle)
        initial_nearest = int(np.argmin(np.linalg.norm(anchor_scores, axis=1)))
        for method, horizons in OBJECTIVES.items():
            cache = clean["cache"]
            coefficient = np.zeros(512, dtype=np.float64)
            trajectory = clean_trajectory
            initial_teacher, blocks = _vector(
                teacher, horizons, j_indices, logit_indices, main_layer
            )
            initial_current, _ = _vector(
                trajectory, horizons, j_indices, logit_indices, main_layer
            )
            initial_residual = initial_teacher - initial_current
            weight = _weighted(initial_residual, blocks, floors)
            r0 = float(np.linalg.norm(initial_residual))
            wr0 = float(np.linalg.norm(weight * initial_residual))
            for step in range(MAX_STEPS):
                current_vector, _ = _vector(
                    trajectory, horizons, j_indices, logit_indices, main_layer
                )
                residual = initial_teacher - current_vector
                current_r = float(np.linalg.norm(residual))
                current_wr = float(np.linalg.norm(weight * residual))
                nearest = int(
                    np.argmin(np.linalg.norm(anchor_scores - coefficient[None], axis=1))
                )
                if nearest not in basis_cache:
                    basis_cache[nearest] = _combined_rows(
                        directions, atlas[nearest]["vh"][:RANK]
                    )
                basis = atlas[nearest]["vh"][:RANK].astype(np.float64)
                basis_rows = basis_cache[nearest]
                columns = []
                for direction in basis_rows:
                    output = []
                    for sign in (-1, 1):
                        edited = _apply_direction(
                            cache,
                            torch.tensor(
                                sign * EPSILON, dtype=torch.float32, device=device
                            ),
                            direction,
                            recurrent,
                            attention,
                        )
                        result = _teacher_forced_trajectory(
                            bundle,
                            edited,
                            tokens,
                            prompt_length=clean["prompt_length"],
                            measured_layers=measured,
                            dense_map=dense_map,
                        )
                        output.append(
                            _vector(
                                result, horizons, j_indices, logit_indices, main_layer
                            )[0]
                        )
                    columns.append((output[1] - output[0]) / (2 * EPSILON))
                jacobian = np.stack(columns, axis=1)
                weighted_jacobian = weight[:, None] * jacobian
                weighted_residual = weight * residual
                control = np.linalg.solve(
                    weighted_jacobian.T @ weighted_jacobian + RIDGE * np.eye(RANK),
                    weighted_jacobian.T @ weighted_residual,
                )
                control_norm = float(np.linalg.norm(control))
                if control_norm > TRUST_RADIUS:
                    control *= TRUST_RADIUS / control_norm
                predicted_wr = float(
                    np.linalg.norm(weighted_residual - weighted_jacobian @ control)
                )
                accepted = False
                actual_wr = current_wr
                actual_r = current_r
                chosen_scale = 0.0
                candidate_trajectory = trajectory
                candidate_cache = cache
                for scale in BACKTRACK:
                    joint = {
                        name: sum(
                            float(control[i] * scale) * basis_rows[i][name]
                            for i in range(RANK)
                        )
                        for name in ("recurrent", "conv", "kv")
                    }
                    test_cache = _apply_direction(
                        cache,
                        torch.tensor(1.0, dtype=torch.float32, device=device),
                        joint,
                        recurrent,
                        attention,
                    )
                    test = _teacher_forced_trajectory(
                        bundle,
                        test_cache,
                        tokens,
                        prompt_length=clean["prompt_length"],
                        measured_layers=measured,
                        dense_map=dense_map,
                    )
                    value, _ = _vector(
                        test, horizons, j_indices, logit_indices, main_layer
                    )
                    test_wr = float(np.linalg.norm(weight * (initial_teacher - value)))
                    if test_wr < current_wr - 1e-8:
                        accepted = True
                        actual_wr = test_wr
                        actual_r = float(np.linalg.norm(initial_teacher - value))
                        chosen_scale = scale
                        candidate_cache = test_cache
                        candidate_trajectory = test
                        break
                rows.append(
                    {
                        "source_freeze_digest": frozen["freeze_digest"],
                        "base_trial_id": base_id,
                        "family": pair["family"],
                        "method": f"closed_loop_finite_response_{method}",
                        "step": step + 1,
                        "rank": RANK,
                        "nearest_atlas_index": nearest,
                        "control_norm": float(np.linalg.norm(control)),
                        "accepted": accepted,
                        "backtracking_scale": chosen_scale,
                        "predicted_weighted_residual": predicted_wr,
                        "actual_weighted_residual": actual_wr,
                        "weighted_residual_ratio": actual_wr / max(wr0, 1e-20),
                        "raw_residual_ratio": actual_r / max(r0, 1e-20),
                        "trust_ratio": (current_wr - actual_wr)
                        / max(current_wr - predicted_wr, 1e-20),
                        "teacher_effect_norm": r0,
                        "tangent_rotation_from_initial_degrees": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        np.linalg.svd(
                                            atlas[nearest]["vh"][:RANK]
                                            @ atlas[initial_nearest]["vh"][:RANK].T,
                                            compute_uv=False,
                                        ),
                                        -1,
                                        1,
                                    )
                                )
                            ).mean()
                        ),
                    }
                )
                if accepted:
                    cache = candidate_cache
                    trajectory = candidate_trajectory
                    coefficient += chosen_scale * control @ basis
                if not accepted:
                    break
            fidelity_rows.extend(
                _metrics(
                    clean_trajectory,
                    teacher,
                    trajectory,
                    task,
                    bundle,
                    main_layer,
                    base_id,
                    pair["family"],
                    f"closed_loop_finite_response_{method}",
                    len(
                        [
                            row
                            for row in rows
                            if row["base_trial_id"] == base_id
                            and row["method"] == f"closed_loop_finite_response_{method}"
                        ]
                    ),
                )
            )
        write_json_atomic(
            root / OUT / "closed_loop_progress_v14.json",
            {
                "completed_cases": case_index + 1,
                "total_cases": len(panel),
            },
        )
    path = root / OUT / "closed_loop_development_v14.parquet"
    fidelity_path = root / OUT / "closed_loop_fidelity_development_v14.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    pd.DataFrame(fidelity_rows).to_parquet(
        fidelity_path, index=False, compression="zstd"
    )
    frame = pd.DataFrame(rows)
    fidelity = pd.DataFrame(fidelity_rows)
    summary = {
        "protocol_version": "finite_causal_control_v14_closed_loop_development",
        "source_freeze_digest": frozen["freeze_digest"],
        "independent_confirmatory": False,
        "panel_count": len(panel),
        "residual_curve": frame.groupby(["method", "step"])
        .agg(
            raw_ratio=("raw_residual_ratio", "median"),
            weighted_ratio=("weighted_residual_ratio", "median"),
            acceptance=("accepted", "mean"),
        )
        .reset_index()
        .to_dict("records"),
        "fidelity": fidelity.groupby(["method", "horizon"])
        .agg(
            direction=("direction_cosine", "mean"),
            magnitude=("magnitude_ratio", "mean"),
            output=("output_direction_cosine", "mean"),
            semantic_legacy=("semantic_delta_agreement", "mean"),
            semantic_continuous=("semantic_vector_cosine", "mean"),
            sign=("task_decision_sign_agreement", "mean"),
        )
        .reset_index()
        .to_dict("records"),
        "records": {
            "steps": {"path": str(path.relative_to(root)), "sha256": sha256_file(path)},
            "fidelity": {
                "path": str(fidelity_path.relative_to(root)),
                "sha256": sha256_file(fidelity_path),
            },
        },
    }
    write_json_atomic(root / OUT / "closed_loop_development_v14.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), sort_keys=True)[:2000])
