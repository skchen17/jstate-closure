"""Metric-corrected input/output geometry from the frozen V21 paired panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.protocol_v22 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/input_geometry_v22.py"
OUT = Path("results/v22/processed")
SCRATCH = Path("/data/CSK/J-space-project/v22-action-manifold-work")
V21_RAW = Path("/data/CSK/J-space-project/v21-action-geometry-work/paired_geometry_forward_ad")


def _ranks(singular: np.ndarray) -> tuple[int, int, int, float]:
    energy = singular * singular
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-30)
    values = [int(np.searchsorted(cumulative, level) + 1) for level in (0.90, 0.95, 0.99)]
    stable = float(energy.sum() / max(float(singular[0] ** 2), 1e-30))
    return values[0], values[1], values[2], stable


def _angle(left: np.ndarray, right: np.ndarray) -> dict:
    if left.size == 0 or right.size == 0:
        return {"median_degrees": 90.0, "max_degrees": 90.0, "grassmann": float("nan"), "overlap": 0.0}
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular))
    return {
        "median_degrees": float(np.median(angles)),
        "max_degrees": float(np.max(angles)),
        "grassmann": float(np.linalg.norm(np.radians(angles))),
        "overlap": float(np.mean(singular * singular)),
    }


def _operator(matrix: np.ndarray, whitener: np.ndarray) -> dict:
    corrected = matrix @ whitener
    u, singular, vh = np.linalg.svd(corrected, full_matrices=False)
    r90, r95, r99, stable = _ranks(singular)
    return {
        "u": u,
        "v": vh.T,
        "singular": singular,
        "r90": r90,
        "r95": r95,
        "r99": r99,
        "stable_rank": stable,
    }


def prepare(root: Path) -> dict:
    verify(root)
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    scales = json.loads((root / "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json").read_text())
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    indices = [int(value) for value in roles["jvp_probe_direction_indices"]]
    reliable = [value for value in indices if scales["selected_alpha_by_direction"][str(value)] is not None]
    x = np.stack([
        np.asarray(directions["score_directions"][value], dtype=np.float64)
        * float(scales["selected_alpha_by_direction"][str(value)])
        for value in reliable
    ])
    del directions
    gram = x @ x.T
    eigen, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigen)[::-1]
    eigen, vectors = eigen[order], vectors[:, order]
    relative_floor = float(verify(root)["config"]["geometry"]["gram_relative_eigenvalue_floor"])
    keep = eigen > max(float(eigen[0]) * relative_floor, 1e-12)
    retained = eigen[keep]
    q = vectors[:, keep]
    whitener = q / np.sqrt(retained)[None, :]
    scratch = root / SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch / "metric_corrected_action_basis_v22.npz",
                        probe_indices=np.asarray(reliable, dtype=np.int32), action_matrix=x.astype(np.float32),
                        gram=gram.astype(np.float32), gram_eigenvalues=eigen.astype(np.float64),
                        whitener=whitener.astype(np.float64))
    detail = {
        "probe_count_input": len(indices),
        "probe_count_reliable": len(reliable),
        "discarded_probe_indices": sorted(set(indices) - set(reliable)),
        "raw_action_dimension": int(x.shape[1]),
        "gram_rank": int(keep.sum()),
        "gram_condition_number_retained": float(retained[0] / retained[-1]),
        "regularization": {"method": "rank_truncated_eigendecomposition", "relative_eigenvalue_floor": relative_floor},
        "discarded_near_null_modes": int((~keep).sum()),
        "scratch_sha256": sha256_file(scratch / "metric_corrected_action_basis_v22.npz"),
        "historical_final_six_opened": False,
    }
    frozen = stage_freeze(root, "input_metric", [SOURCE,
        "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
        "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json",
        "artifacts/causal/v13/probe_directions_v13.pt"], detail)
    return {"freeze_digest": frozen["freeze_digest"], **detail}


def run(root: Path) -> dict:
    metric = verify_stage(root, "input_metric")
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    with np.load(root / SCRATCH / "metric_corrected_action_basis_v22.npz") as source:
        all_indices = np.asarray(source["probe_indices"], dtype=int)
        whitener = np.asarray(source["whitener"], dtype=np.float64)
    full_order = [int(value) for value in roles["jvp_probe_direction_indices"]]
    columns = [full_order.index(int(value)) for value in all_indices]
    rows = []
    for role, items in (("development", roles["jvp_development"]), ("validation", roles["jvp_validation"])):
        for item in items:
            path = root / V21_RAW / role / f"paired_{item['base_trial_id']}.npz"
            with np.load(path) as source:
                operators = {}
                for state in ("P0", "Pq"):
                    for kind in ("JVP", "finite"):
                        operators[(state, kind)] = _operator(np.asarray(source[f"{state}_{kind}"][:, columns], dtype=np.float64), whitener)
            for kind in ("JVP", "finite"):
                p0, pq = operators[("P0", kind)], operators[("Pq", kind)]
                output_angle = _angle(p0["u"][:, :p0["r95"]], pq["u"][:, :pq["r95"]])
                input_angle = _angle(p0["v"][:, :p0["r95"]], pq["v"][:, :pq["r95"]])
                spectrum_error = float(np.linalg.norm(p0["singular"] - pq["singular"]) / max(np.linalg.norm(p0["singular"]), 1e-12))
                rows.append({
                    "role": role, "base_trial_id": item["base_trial_id"], "family": item["family"],
                    "q_name": item["q_name"], "kind": kind,
                    "P0_r90": p0["r90"], "P0_r95": p0["r95"], "P0_r99": p0["r99"],
                    "Pq_r90": pq["r90"], "Pq_r95": pq["r95"], "Pq_r99": pq["r99"],
                    "P0_stable_rank": p0["stable_rank"], "Pq_stable_rank": pq["stable_rank"],
                    "output_angle_median_degrees": output_angle["median_degrees"],
                    "output_angle_max_degrees": output_angle["max_degrees"],
                    "output_grassmann_distance": output_angle["grassmann"],
                    "output_subspace_overlap": output_angle["overlap"],
                    "input_angle_median_degrees": input_angle["median_degrees"],
                    "input_angle_max_degrees": input_angle["max_degrees"],
                    "input_grassmann_distance": input_angle["grassmann"],
                    "input_subspace_overlap": input_angle["overlap"],
                    "spectrum_relative_change": spectrum_error,
                })
            for state in ("P0", "Pq"):
                jvp, finite = operators[(state, "JVP")], operators[(state, "finite")]
                input_cross = _angle(jvp["v"][:, :jvp["r95"]], finite["v"][:, :finite["r95"]])
                output_cross = _angle(jvp["u"][:, :jvp["r95"]], finite["u"][:, :finite["r95"]])
                rows.append({
                    "role": role, "base_trial_id": item["base_trial_id"], "family": item["family"],
                    "q_name": item["q_name"], "kind": f"JVP_finite_{state}",
                    "input_subspace_overlap": input_cross["overlap"],
                    "input_angle_median_degrees": input_cross["median_degrees"],
                    "output_subspace_overlap": output_cross["overlap"],
                    "output_angle_median_degrees": output_cross["median_degrees"],
                })
    frame = pd.DataFrame(rows)
    target = root / OUT / "input_output_geometry_rows_v22.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    summary = {"input_metric": {key: metric[key] for key in (
        "raw_action_dimension", "gram_rank", "gram_condition_number_retained", "regularization", "discarded_near_null_modes")},
        "roles": {}, "historical_final_six_opened": False}
    for role in ("development", "validation"):
        summary["roles"][role] = {}
        for kind in ("JVP", "finite"):
            group = frame[(frame.role == role) & (frame.kind == kind)]
            summary["roles"][role][kind] = {
                "input_r90_r95_r99_median": [float(group.P0_r90.median()), float(group.P0_r95.median()), float(group.P0_r99.median())],
                "output_r90_r95_r99_median": [float(group.P0_r90.median()), float(group.P0_r95.median()), float(group.P0_r99.median())],
                "input_P0_Pq_angle_median_degrees": float(group.input_angle_median_degrees.median()),
                "output_P0_Pq_angle_median_degrees": float(group.output_angle_median_degrees.median()),
                "input_grassmann_median": float(group.input_grassmann_distance.median()),
                "output_grassmann_median": float(group.output_grassmann_distance.median()),
                "input_overlap_median": float(group.input_subspace_overlap.median()),
                "output_overlap_median": float(group.output_subspace_overlap.median()),
                "stable_rank_median": float(group.P0_stable_rank.median()),
            }
        for state in ("P0", "Pq"):
            cross = frame[(frame.role == role) & (frame.kind == f"JVP_finite_{state}")]
            summary["roles"][role][f"JVP_finite_{state}"] = {
                "input_subspace_overlap_median": float(cross.input_subspace_overlap.median()),
                "input_angle_median_degrees": float(cross.input_angle_median_degrees.median()),
                "output_subspace_overlap_median": float(cross.output_subspace_overlap.median()),
            }
    validation = summary["roles"]["validation"]
    input_rotation = max(validation["JVP"]["input_P0_Pq_angle_median_degrees"], validation["finite"]["input_P0_Pq_angle_median_degrees"])
    output_rotation = min(validation["JVP"]["output_P0_Pq_angle_median_degrees"], validation["finite"]["output_P0_Pq_angle_median_degrees"])
    summary["classification"] = {
        "V22_G1_STABLE_INPUT_CAUSAL_SUBSPACE": bool(input_rotation < 10.0 and output_rotation >= 15.0),
        "V22_G2_STATE_DEPENDENT_INPUT_CAUSAL_SUBSPACE": bool(input_rotation >= 15.0),
        "V22_G3_DIFFERENTIAL_FINITE_INPUT_DIVERGENCE": bool(validation["JVP_finite_P0"]["input_subspace_overlap_median"] < 0.5),
        "state_conditioned_chart_eligible": bool(input_rotation >= 15.0),
        "input_and_output_rank_wording": "Input-sensitive rank is computed after raw-action Gram whitening; output r95 alone is not an action-space dimension claim.",
    }
    summary["rows_sha256"] = sha256_file(target)
    write_json_atomic(root / OUT / "input_side_causal_geometry_v22.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd())
    print(json.dumps(result, indent=2))
