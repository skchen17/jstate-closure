"""Prospective local gain/rotation controls for the V37 native operator law."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/operator_controls_v37.py"
RANK = 16
DESIGN = {
    "schema_version": 1,
    "training_role": "development",
    "evaluation_role": "validation",
    "models": ["Q", "F"],
    "stages": ["raw", "mixer"],
    "positions": ["P1", "P2", "P3", "P4", "B23", "B34"],
    "operator_targets": ["B", "C"],
    "state_targets": ["B", "C"],
    "scalar_gain_scope": "one scalar per model, stage, position and target operator",
    "rotation_scope": "one fixed rank-16 orthogonal map in a development-only source subspace per model, stage, position and target operator",
    "native_baseline": "V37 frozen factor prediction, never fit to validation",
    "anchor": "predicted state response at operator A; no observed validation counterfactual used as anchor",
    "metric_unit": "state; six probes and positions are repeated measurements",
    "rank": RANK,
    "development_fit_only": True,
    "validation_refit_allowed": False,
}


def design(root: Path):
    verify_stage(root, "design")
    path = root / OUT / "operator_controls_design_v37.json"
    write_json_atomic(path, DESIGN)
    seal = stage_freeze(root, "operator_controls_design", [SOURCE, str(path.relative_to(root))],
                        {"design_sha256": sha256_file(path), "response_observed": False})
    return {"freeze_digest": seal["freeze_digest"], **DESIGN}


def _load(root: Path, key: str, role: str, kind: str):
    if kind == "observed":
        name = f"local_observed_vectors_{key}_{role}_v37.npz"
    else:
        name = f"local_prediction_vectors_{key}_{role}_v37.npz"
    z = np.load(root / OUT / name)
    return {stage: z[stage] for stage in ("raw", "mixer")}, z["state_ids"], z["positions"]


def _pairs(arr: np.ndarray, operator: int):
    source = np.concatenate([arr[:, 1, 0] - arr[:, 0, 0],
                             arr[:, 2, 0] - arr[:, 0, 0]], axis=0)
    target = np.concatenate([arr[:, 1, operator] - arr[:, 0, operator],
                             arr[:, 2, operator] - arr[:, 0, operator]], axis=0)
    return source.astype(np.float64), target.astype(np.float64)


def fit(root: Path, key: str):
    verify_stage(root, "operator_controls_design")
    verify_stage(root, f"local_observation_{key}_development")
    observed, _, positions = _load(root, key, "development", "observed")
    coeff, rows = {}, []
    for stage in DESIGN["stages"]:
        for position in DESIGN["positions"]:
            arr = observed[stage][positions == position]
            for target, oi in (("B", 1), ("C", 2)):
                x, y = _pairs(arr, oi)
                gain = float(np.sum(x*y) / max(np.sum(x*x), 1e-12))
                basis = randomized_svd(x, n_components=RANK, n_iter=3,
                                       random_state=370137)[2]
                xp, yp = x @ basis.T, y @ basis.T
                u, _, vt = np.linalg.svd(xp.T @ yp, full_matrices=False)
                rotation = u @ vt
                rotated = x + (xp @ rotation - xp) @ basis
                rotated_gain = float(np.sum(rotated*y) /
                                     max(np.sum(rotated*rotated), 1e-12))
                prefix = f"{stage}_{position}_{target}"
                coeff[prefix + "_gain"] = np.asarray(gain, np.float32)
                coeff[prefix + "_basis"] = basis.astype(np.float32)
                coeff[prefix + "_rotation"] = rotation.astype(np.float32)
                coeff[prefix + "_rotated_gain"] = np.asarray(rotated_gain, np.float32)
                rows.append({"model": key, "stage": stage, "position": position,
                             "operator": target, "development_pairs": len(x),
                             "scalar_gain": gain, "rotated_gain": rotated_gain,
                             "rank": RANK})
    coeff_path = root / OUT / f"operator_controls_fit_{key}_v37.npz"
    table_path = root / OUT / f"operator_controls_fit_{key}_v37.parquet"
    np.savez_compressed(coeff_path, **coeff)
    pd.DataFrame(rows).to_parquet(table_path, index=False, compression="zstd")
    summary = {"model": key, "role": "development_fit", "rows": len(rows),
               "validation_seen": False, "files_sha256": {
                   "coefficients": sha256_file(coeff_path), "rows": sha256_file(table_path)}}
    path = root / OUT / f"operator_controls_fit_{key}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"operator_controls_fit_{key}",
                        [SOURCE, str(path.relative_to(root)), str(coeff_path.relative_to(root)),
                         str(table_path.relative_to(root)),
                         "artifacts/computational_origin_v37_operator_controls_design.freeze.json"],
                        {"model": key, "summary_sha256": sha256_file(path),
                         "validation_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **summary}


def _cos(x, y):
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y)/den) if den > 1e-9 else None


def evaluate(root: Path, key: str):
    verify_stage(root, f"operator_controls_fit_{key}")
    verify_stage(root, f"local_observation_{key}_validation")
    frozen, ids, positions = _load(root, key, "validation", "predicted")
    observed, oid, opositions = _load(root, key, "validation", "observed")
    if not np.array_equal(ids, oid) or not np.array_equal(positions, opositions):
        raise RuntimeError("V37 control vector row alignment drift")
    coeff = np.load(root / OUT / f"operator_controls_fit_{key}_v37.npz")
    rows = []
    for stage in DESIGN["stages"]:
        for i, (sid, position) in enumerate(zip(ids, positions, strict=True)):
            p, o = frozen[stage][i].astype(np.float64), observed[stage][i].astype(np.float64)
            for target, oi in (("B", 1), ("C", 2)):
                prefix = f"{stage}_{position}_{target}"
                basis = coeff[prefix + "_basis"].astype(np.float64)
                rotation = coeff[prefix + "_rotation"].astype(np.float64)
                for state, si in (("B", 1), ("C", 2)):
                    x = p[si, 0] - p[0, 0]
                    truth = o[si, oi] - o[0, oi]
                    native = p[si, oi] - p[0, oi]
                    gain = float(coeff[prefix + "_gain"]) * x
                    xp = x @ basis.T
                    rotated = x + (xp @ rotation - xp) @ basis
                    rotated *= float(coeff[prefix + "_rotated_gain"])
                    for method, pred in (("native", native), ("gain", gain),
                                         ("fixed_rotation", rotated)):
                        rows.append({"state_id": sid, "model": key,
                                     "stage": stage, "position": position,
                                     "operator": target, "state": state,
                                     "method": method, "cosine": _cos(pred, truth),
                                     "relative_magnitude_error": abs(np.linalg.norm(pred)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-9),
                                     "relative_vector_error": float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-9)),
                                     "truth_norm": float(np.linalg.norm(truth)),
                                     "prediction_norm": float(np.linalg.norm(pred))})
    frame = pd.DataFrame(rows)
    design_record = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    families = {r["base_trial_id"]: r["family"] for r in design_record["validation"]}
    frame["family"] = frame.state_id.map(families)
    table = root / OUT / f"operator_controls_validation_{key}_v37.parquet"
    frame.to_parquet(table, index=False, compression="zstd")
    by_method = {}
    for (stage, method), part in frame.groupby(["stage", "method"]):
        state = part.groupby("state_id", as_index=False).agg(
            cosine=("cosine", "median"),
            relative_magnitude_error=("relative_magnitude_error", "median"),
            relative_vector_error=("relative_vector_error", "median"))
        by_method[f"{stage}_{method}"] = {"states": len(state),
            "median_cosine": float(state.cosine.median()),
            "median_relative_magnitude_error": float(state.relative_magnitude_error.median()),
            "median_relative_vector_error": float(state.relative_vector_error.median())}
    summary = {"model": key, "role": "validation", "rows": len(frame),
               "by_method": by_method, "fit_frozen_before_validation_observation": True,
               "table_sha256": sha256_file(table)}
    path = root / OUT / f"operator_controls_validation_{key}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"operator_controls_validation_{key}",
                        [SOURCE, str(path.relative_to(root)), str(table.relative_to(root)),
                         f"artifacts/computational_origin_v37_operator_controls_fit_{key}.freeze.json"],
                        {"model": key, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("design", "fit", "evaluate"))
    p.add_argument("model", nargs="?", choices=("Q", "F"))
    a = p.parse_args()
    if a.command != "design" and a.model is None:
        p.error("model is required for fit/evaluate")
    print(json.dumps({"design": design, "fit": fit, "evaluate": evaluate}[a.command](Path.cwd(), *( [a.model] if a.model else [])), indent=2))
