"""Frozen paired differential–finite rank, alignment, and rotation adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments.paired_geometry_v21 import _matrix_path
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/geometry_analysis_v21.py"
ROWS = bank.OUT / "paired_geometry_analysis_v21.parquet"
SUMMARY = bank.OUT / "paired_geometry_analysis_v21.json"


def prepare(root: Path) -> dict:
    design = verify_stage(root, "paired_geometry_design")
    amendment = verify_stage(root, "paired_geometry_unreliable_probe_amendment_1")
    return stage_freeze(root, "paired_geometry_analysis_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_paired_geometry_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_unreliable_probe_amendment_1.freeze.json"],
                        {"probe_sha256": design["probe_sha256"],
                         "reliability_qualified_probe_count": amendment["reliable_probe_count"],
                         "excluded_finite_probe_direction_indices": [amendment["unreliable_direction_index"]],
                         "spectral_energy_rank_thresholds": [0.90, 0.95, 0.99],
                         "subspace_rank": "minimum_P0_Pq_or_JVP_finite_r95; dimension_adapted",
                         "rotation_magnitude": "median_principal_angle_degrees_of_dominant_left_response_subspaces",
                         "shared_geometry_support_thresholds": {"median_r95_max": 16,
                                                                "median_P0_Pq_angle_min_degrees": 15,
                                                                "Spearman_rotation_correlation_min": 0.30,
                                                                "median_JVP_finite_overlap_min": 0.50,
                                                                "minimum_positive_family_correlation_count": 4},
                         "bootstrap_unit": "base_trial_id", "bootstrap_replicates": 2000,
                         "bootstrap_seed": 2021,
                         "final_six_action_responses_opened": False})


def _spectrum(matrix: np.ndarray) -> dict:
    u, singular, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-20)
    probability = energy / max(float(energy.sum()), 1e-20)
    effective = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    return {"u": u, "singular": singular, "vh": vh,
            "r90": int(np.searchsorted(cumulative, 0.90) + 1),
            "r95": int(np.searchsorted(cumulative, 0.95) + 1),
            "r99": int(np.searchsorted(cumulative, 0.99) + 1),
            "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-20)),
            "effective_rank": effective, "frobenius_norm": float(np.sqrt(energy.sum()))}


def _angles(a: dict, b: dict) -> dict:
    rank = min(a["r95"], b["r95"])
    spectrum = np.linalg.svd(a["u"][:, :rank].T @ b["u"][:, :rank], compute_uv=False)
    degrees = np.degrees(np.arccos(np.clip(spectrum, -1, 1)))
    return {"rank": rank, "principal_angle_median_degrees": float(np.median(degrees)),
            "principal_angle_mean_degrees": float(np.mean(degrees)),
            "principal_angle_max_degrees": float(np.max(degrees)),
            "subspace_overlap": float(np.mean(spectrum**2))}


def _spectral_js(a: dict, b: dict) -> float:
    aa = a["singular"]**2
    bb = b["singular"]**2
    aa /= max(float(aa.sum()), 1e-20)
    bb /= max(float(bb.sum()), 1e-20)
    m = (aa + bb) / 2
    def kl(x):
        ok = x > 0
        return float(np.sum(x[ok] * np.log(x[ok] / m[ok])))
    return (kl(aa) + kl(bb)) / 2


def _decompose(a: np.ndarray, b: np.ndarray, sa: dict, sb: dict) -> dict:
    norm_a = max(float(np.linalg.norm(a)), 1e-20)
    norm_b = max(float(np.linalg.norm(b)), 1e-20)
    pre = float(np.linalg.norm(b-a) / norm_b)
    nuclear = float(np.sum(np.linalg.svd(a.T.astype(np.float64) @ b.astype(np.float64), compute_uv=False)))
    post = float(np.sqrt(max(norm_a**2 + norm_b**2 - 2*nuclear, 0.0)) / norm_b)
    best_gain = nuclear / max(norm_a**2, 1e-20)
    post_gain = float(np.sqrt(max(norm_b**2 - nuclear**2 / max(norm_a**2, 1e-20), 0.0)) / norm_b)
    return {"pre_alignment_relative_error": pre, "post_orthogonal_Procrustes_relative_error": post,
            "post_orthogonal_and_gain_relative_error": post_gain,
            "optimal_Procrustes_gain": best_gain,
            "frobenius_gain_ratio": norm_b / norm_a,
            "singular_spectrum_JS_divergence": _spectral_js(sa, sb),
            "r95_difference": sb["r95"]-sa["r95"]}


def _column_cosine(a: np.ndarray, b: np.ndarray) -> float:
    top = np.sum(a*b, axis=0)
    bottom = np.maximum(np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0), 1e-20)
    return float(np.median(top/bottom))


def run(root: Path) -> dict:
    design = verify_stage(root, "paired_geometry_analysis_design")
    roles = verify_stage(root, "roles")
    records = []
    for role in ("development", "validation"):
        for number, item in enumerate(roles[f"jvp_{role}"], 1):
            path = _matrix_path(role, item["base_trial_id"])
            if not path.exists():
                raise RuntimeError(f"V21 paired matrix missing: {path}")
            with np.load(path, allow_pickle=False) as source:
                indices = source["probe_indices"].astype(int)
                keep = indices != design["excluded_finite_probe_direction_indices"][0]
                matrices = {f"{state}_{kind}": source[f"{state}_{kind}"][:, keep]
                            for state in ("P0", "Pq") for kind in ("JVP", "finite")}
                if str(source["probe_sha256"].item()) != design["probe_sha256"] or int(keep.sum()) != 63:
                    raise RuntimeError("V21 paired probe matrix hash/reliability mismatch")
            spectra = {name: _spectrum(matrix) for name, matrix in matrices.items()}
            row = {"role": role, **item, "matrix_sha256": sha256_file(path),
                   "qualified_probe_count": int(keep.sum())}
            for name, spectrum in spectra.items():
                for field in ("r90", "r95", "r99", "stable_rank", "effective_rank", "frobenius_norm"):
                    row[f"{name}_{field}"] = spectrum[field]
            for kind in ("JVP", "finite"):
                a, b = spectra[f"P0_{kind}"], spectra[f"Pq_{kind}"]
                row.update({f"{kind}_rotation_{key}": value for key,value in _angles(a,b).items()})
                row.update({f"{kind}_{key}": value for key,value in _decompose(matrices[f"P0_{kind}"], matrices[f"Pq_{kind}"], a,b).items()})
            for state in ("P0", "Pq"):
                j, f = matrices[f"{state}_JVP"], matrices[f"{state}_finite"]
                row.update({f"{state}_JVP_finite_{key}": value
                            for key,value in _angles(spectra[f"{state}_JVP"], spectra[f"{state}_finite"]).items()})
                row[f"{state}_JVP_finite_column_cosine_median"] = _column_cosine(j,f)
                scale = float(np.sum(j*f)/max(float(np.sum(j*j)), 1e-20))
                row[f"{state}_finite_minus_scaled_JVP_relative_l2"] = float(np.linalg.norm(f-scale*j)/max(float(np.linalg.norm(f)), 1e-20))
                row[f"{state}_finite_over_JVP_gain"] = float(np.linalg.norm(f)/max(float(np.linalg.norm(j)), 1e-20))
            records.append(row)
            print(f"V21 geometry {role} {number}/{len(roles[f'jvp_{role}'])}", flush=True)
    frame = pd.DataFrame(records)
    target = root / ROWS
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    summary = {"design_digest": design["freeze_digest"], "records": len(frame),
               "matrix_index_sha256": sha256_file(target), "by_role": {},
               "final_six_action_responses_opened": False}
    for role, group in frame.groupby("role"):
        x = group["JVP_rotation_principal_angle_median_degrees"].to_numpy(float)
        y = group["finite_rotation_principal_angle_median_degrees"].to_numpy(float)
        summary["by_role"][role] = {
            "base_states": len(group),
            "JVP_median_r90_r95_r99": [float(group[f"P0_JVP_r{x}"].median()) for x in (90,95,99)],
            "finite_median_r90_r95_r99": [float(group[f"P0_finite_r{x}"].median()) for x in (90,95,99)],
            "JVP_P0_Pq_median_principal_angle": float(np.median(x)),
            "finite_P0_Pq_median_principal_angle": float(np.median(y)),
            "rotation_Spearman_rho": float(spearmanr(x,y).statistic),
            "rotation_Pearson_r": float(pearsonr(x,y).statistic),
            "JVP_finite_P0_median_subspace_overlap": float(group["P0_JVP_finite_subspace_overlap"].median()),
            "JVP_finite_Pq_median_subspace_overlap": float(group["Pq_JVP_finite_subspace_overlap"].median()),
            "JVP_finite_P0_median_column_cosine": float(group["P0_JVP_finite_column_cosine_median"].median()),
            "finite_rotation_family_medians": group.groupby("family")["finite_rotation_principal_angle_median_degrees"].median().to_dict(),
            "JVP_rotation_family_medians": group.groupby("family")["JVP_rotation_principal_angle_median_degrees"].median().to_dict(),
            "family_rotation_Spearman_rho": {family: float(spearmanr(g["JVP_rotation_principal_angle_median_degrees"],
                                                                      g["finite_rotation_principal_angle_median_degrees"]).statistic)
                                              for family,g in group.groupby("family")},
            "finite_median_pre_alignment_error": float(group["finite_pre_alignment_relative_error"].median()),
            "finite_median_post_Procrustes_error": float(group["finite_post_orthogonal_Procrustes_relative_error"].median()),
            "finite_median_gain_ratio": float(group["finite_frobenius_gain_ratio"].median()),
            "finite_median_spectral_JS": float(group["finite_singular_spectrum_JS_divergence"].median()),
            "finite_median_r95_difference": float(group["finite_r95_difference"].median()),
            "finite_minus_scaled_JVP_P0_relative_l2_median": float(group["P0_finite_minus_scaled_JVP_relative_l2"].median())}
    val = summary["by_role"]["validation"]
    thresholds = design["shared_geometry_support_thresholds"]
    summary["shared_geometry_strong_support"] = bool(
        val["JVP_median_r90_r95_r99"][1] <= thresholds["median_r95_max"]
        and val["finite_median_r90_r95_r99"][1] <= thresholds["median_r95_max"]
        and val["JVP_P0_Pq_median_principal_angle"] >= thresholds["median_P0_Pq_angle_min_degrees"]
        and val["finite_P0_Pq_median_principal_angle"] >= thresholds["median_P0_Pq_angle_min_degrees"]
        and val["rotation_Spearman_rho"] >= thresholds["Spearman_rotation_correlation_min"]
        and val["JVP_finite_P0_median_subspace_overlap"] >= thresholds["median_JVP_finite_overlap_min"]
        and sum(x > 0 for x in val["family_rotation_Spearman_rho"].values()) >= thresholds["minimum_positive_family_correlation_count"])
    write_json_atomic(root / SUMMARY, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd())
    print(json.dumps(result, indent=2))
