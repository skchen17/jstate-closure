"""Finite response-operator rank, gain, rotation and manifold comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/operator_geometry_v20.py"
PAIRS = bank.OUT / "operator_rotation_pairs_v20.parquet"
STATES = bank.OUT / "operator_singular_spectra_v20.parquet"
SUMMARY = bank.OUT / "operator_geometry_v20.json"


def prepare(root: Path) -> dict:
    design = verify_stage(root, "operator_design")
    cfg = verify(root)["config"]["operator_analysis"]
    return stage_freeze(root, "geometry_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json",
                         "results/v13/processed/geometry_analysis_v13.json"],
                        {"positive_shared_actions": [x["coordinate_index"] for x in design["shared_measured_actions"]],
                         "response_target": "V16_normalized_stack_288",
                         "rank_fraction": cfg["geometry_spectrum_rank_threshold"],
                         "principal_angle_rank_cap": 8,
                         "procrustes_space": "output_288D_orthogonal_transport_via_18x18_nuclear_norm",
                         "manifold_dimension_method": "centered_response_fingerprint_Gram_spectrum_equal_base_state_panels",
                         "V13_JVP_relation_policy": "compare_aggregate_rank_only_without_fabricating_unmeasured_paired_JVP_subspaces",
                         "operator_responses_already_observed": 0})


def _load(root: Path, role: str) -> pd.DataFrame:
    paths = sorted((root / bank.OUT).glob(f"response_operator_{role}_*_v20.parquet"))
    if len(paths) != 5:
        raise RuntimeError(f"V20 geometry {role} requires five operator family partitions")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _r95(singular: np.ndarray, fraction: float) -> int:
    energy = singular * singular
    if float(energy.sum()) <= 1e-20:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), fraction) + 1)


def _spectrum_dimension(matrix: np.ndarray, fraction: float) -> dict:
    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(0)
    eigen = np.linalg.eigvalsh(centered @ centered.T)
    eigen = np.maximum(eigen[::-1], 0.0)
    if eigen.sum() <= 1e-20:
        return {"states": len(matrix), "r95": 0, "effective_rank": 0.0, "top_energy_fraction": None}
    p = eigen / eigen.sum()
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return {"states": len(matrix), "r95": int(np.searchsorted(np.cumsum(p), fraction) + 1),
            "effective_rank": float(np.exp(entropy)), "top_energy_fraction": float(p[0])}


def analyze(root: Path) -> dict:
    design = verify_stage(root, "geometry_design")
    operator = verify_stage(root, "operator_design")
    actions = design["positive_shared_actions"]
    fraction = float(design["rank_fraction"])
    state_rows = []
    pair_rows = []
    manifold = {}
    for role in ("operator_train", "operator_validation"):
        frame = _load(root, role)
        selected = frame[(frame.action_sign == 1) & frame.action_coordinate.isin(actions)]
        operators = {}
        for state_id, group in selected.groupby("operator_state_id", sort=True):
            lookup = group.set_index("action_coordinate")
            if len(lookup) != len(actions) or not set(actions) == set(lookup.index):
                raise RuntimeError(f"V20 operator geometry action grid incomplete: {state_id}")
            matrix = np.stack([np.asarray(lookup.loc[a].response_stack, dtype=np.float64) for a in actions])
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = _r95(singular, fraction)
            sample = group.iloc[0]
            operators[state_id] = (matrix, singular, vh, rank, sample)
            state_rows.append({"operator_state_id": state_id, "base_trial_id": sample.base_trial_id,
                               "family": sample.family, "role": role, "q_name": sample.q_name,
                               "q_channel": sample.q_channel, "state_distribution": sample.state_distribution,
                               "r95": rank, "frobenius_norm": float(np.linalg.norm(matrix)),
                               "singular_values": singular.tolist(),
                               "top_singular_value": float(singular[0]),
                               "spectrum_entropy_rank": float(np.exp(-np.sum((singular**2 / max(np.sum(singular**2), 1e-20)) *
                                                                            np.log(np.maximum(singular**2 / max(np.sum(singular**2), 1e-20), 1e-20)))))} )
        for base_id, group in selected.groupby("base_trial_id", sort=True):
            clean = operators[f"{base_id}::P0"]
            a, sa, va, ra, sample = clean
            for q in operator["q"]:
                b, sb, vb, rb, _ = operators[f"{base_id}::{q['name']}"]
                n = min(int(design["principal_angle_rank_cap"]), max(ra, 1), max(rb, 1))
                cosines = np.linalg.svd(va[:n] @ vb[:n].T, compute_uv=False)
                angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
                cross_sv = np.linalg.svd(a @ b.T, compute_uv=False)
                procrustes = math.sqrt(max(float(np.sum(a*a) + np.sum(b*b) - 2 * cross_sv.sum()), 0.0))
                gain = float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-8))
                pa = sa * sa / max(float(np.sum(sa*sa)), 1e-20)
                pb = sb * sb / max(float(np.sum(sb*sb)), 1e-20)
                midpoint = 0.5 * (pa + pb)
                js = 0.5 * (np.sum(pa * np.log(np.maximum(pa, 1e-20) / np.maximum(midpoint, 1e-20))) +
                            np.sum(pb * np.log(np.maximum(pb, 1e-20) / np.maximum(midpoint, 1e-20))))
                pair_rows.append({"base_trial_id": base_id, "family": sample.family, "role": role,
                                  "q_name": q["name"], "q_channel": q["channel"],
                                  "P0_r95": ra, "Pq_r95": rb, "rank_change": rb - ra,
                                  "gain_ratio_Pq_over_P0": gain,
                                  "mean_principal_angle_degrees": float(np.mean(angles)),
                                  "max_principal_angle_degrees": float(np.max(angles)),
                                  "procrustes_relative_residual": procrustes / max(float(np.linalg.norm(b)), 1e-8),
                                  "spectrum_Jensen_Shannon": float(js),
                                  "operator_modulation_relative": float(np.linalg.norm(b - a) /
                                                                        max(np.linalg.norm(a), np.linalg.norm(b), 1e-8))})
        role_manifold = {}
        for qname in ("P0", *(q["name"] for q in operator["q"])):
            ids = sorted(key for key in operators if key.endswith(f"::{qname}"))
            matrices = np.stack([operators[key][0].reshape(-1) for key in ids])
            role_manifold[qname] = _spectrum_dimension(matrices, fraction)
        all_ids = sorted(operators)
        role_manifold["pooled_natural_plus_counterfactual"] = _spectrum_dimension(
            np.stack([operators[key][0].reshape(-1) for key in all_ids]), fraction)
        manifold[role] = role_manifold
    states = pd.DataFrame(state_rows)
    pairs = pd.DataFrame(pair_rows)
    states.to_parquet(root / STATES, index=False, compression="zstd")
    pairs.to_parquet(root / PAIRS, index=False, compression="zstd")
    old = json.loads((root / "results/v13/processed/geometry_analysis_v13.json").read_text())
    by_channel = {}
    for key, group in pairs.groupby("q_channel", sort=True):
        by_channel[key] = {name: float(group[name].median()) for name in
                           ("rank_change", "gain_ratio_Pq_over_P0", "mean_principal_angle_degrees",
                            "procrustes_relative_residual", "spectrum_Jensen_Shannon", "operator_modulation_relative")}
    result = {"geometry_freeze_digest": design["freeze_digest"],
              "state_count": len(states), "pair_count": len(pairs), "manifold_dimensions": manifold,
              "median_natural_state_operator_r95": float(states[states.q_name == "P0"].r95.median()),
              "median_counterfactual_state_operator_r95": float(states[states.q_name != "P0"].r95.median()),
              "median_pair_principal_angle_degrees": float(pairs.mean_principal_angle_degrees.median()),
              "median_pair_gain_ratio": float(pairs.gain_ratio_Pq_over_P0.median()),
              "median_pair_rank_change": float(pairs.rank_change.median()),
              "median_pair_procrustes_relative_residual": float(pairs.procrustes_relative_residual.median()),
              "by_channel": by_channel,
              "V13_exact_JVP_historical_median_instantaneous_r95": old["median_instantaneous_r95"],
              "V13_exact_JVP_historical_median_cumulative_path_r95": old["median_cumulative_path_r95"],
              "paired_V20_state_JVP_subspace_similarity_measured": False,
              "paired_JVP_limitation": "No exact JVP was measured for the frozen V20 base IDs; historical V13 aggregate rank is a comparison, not a paired tangent-rotation test.",
              "state_spectra_path": str(STATES), "state_spectra_sha256": sha256_file(root / STATES),
              "pair_path": str(PAIRS), "pair_sha256": sha256_file(root / PAIRS)}
    write_json_atomic(root / SUMMARY, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze"))
    args = parser.parse_args()
    result = {"prepare": prepare, "analyze": analyze}[args.stage](Path.cwd())
    print(json.dumps({"stage": args.stage, "freeze_digest": result.get("freeze_digest", result.get("geometry_freeze_digest")),
                      "pair_count": result.get("pair_count")}, indent=2))
