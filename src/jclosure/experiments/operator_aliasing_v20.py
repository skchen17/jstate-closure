"""Natural-versus-same-J operator-coordinate aliasing, with train-only J predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.experiments import operator_model_v20 as model
from jclosure.protocol_v20 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/operator_aliasing_v20.py"
SUMMARY = bank.OUT / "operator_aliasing_v20.json"
PAIRS = bank.OUT / "operator_aliasing_pairs_v20.parquet"


def prepare(root: Path) -> dict:
    return stage_freeze(root, "operator_aliasing_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_operator_analysis.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_splits.freeze.json"],
                        {"coordinate": "frozen_selected_oracle_k_from_train_positive_action_fingerprint",
                         "workspace_input": "frozen_selected_J_128_at_boundary_same_for_P0_and_Pq",
                         "natural_J_to_C_model": "ridge_alpha_1_standardized_train_natural_only",
                         "pair_distance": "Euclidean_standardized_oracle_coordinate_and_train_action_fingerprint",
                         "alias_threshold_coordinate_distance": 0.1,
                         "alias_threshold_response_fingerprint_relative": 0.1,
                         "validation_operator_response_rows_observed": 0,
                         "independent_V20_final_opened": False})


def _meta(root: Path, role: str, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_parquet(root / bank.OUT / f"operator_state_metadata_{role}_v20.parquet")
    indexed = frame.set_index("base_trial_id")
    return np.stack([np.asarray(indexed.loc[state.split("::")[0]].boundary_j_vector, dtype=np.float32)
                     for state in ids])


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    matrix = x.T @ x / len(x) + alpha * np.eye(x.shape[1])
    return np.linalg.solve(matrix, x.T @ y / len(x))


def analyze(root: Path) -> dict:
    freeze = verify_stage(root, "operator_aliasing_design")
    operator = verify_stage(root, "operator_design")
    chosen = json.loads((root / model.SUMMARY).read_text())
    k = int(chosen["selected_k_for_diagnostics"])
    train = model._load(root, "operator_train", operator)
    validation = model._load(root, "operator_validation", operator)
    pca = model._pca(model._fingerprints(train), model._fingerprints(validation))
    ctrain, cval, _ = model._coordinate(pca, k)
    jtrain = _meta(root, "operator_train", train["ids"])
    jval = _meta(root, "operator_validation", validation["ids"])
    natural_train = train["q_name"] == "P0"
    natural_val = validation["q_name"] == "P0"
    mean = jtrain[natural_train].mean(axis=0)
    scale = np.maximum(jtrain[natural_train].std(axis=0), 1e-6)
    xtrain = (jtrain[natural_train] - mean) / scale
    xval = (jval - mean) / scale
    weight = _ridge(xtrain.astype(np.float64), ctrain[natural_train].astype(np.float64), 1.0)
    prediction = (xval @ weight).astype(np.float32)
    natural_error = float(np.linalg.norm(cval[natural_val] - prediction[natural_val]) /
                          max(np.linalg.norm(cval[natural_val]), 1e-8))
    counterfactual_error = float(np.linalg.norm(cval[~natural_val] - prediction[~natural_val]) /
                                 max(np.linalg.norm(cval[~natural_val]), 1e-8))
    centered = cval[natural_val] - cval[natural_val].mean(axis=0)
    natural_r2 = 1 - float(np.sum((cval[natural_val] - prediction[natural_val]) ** 2) /
                           max(np.sum(centered ** 2), 1e-8))
    by_id = {state: i for i, state in enumerate(validation["ids"])}
    rows = []
    for base_id in sorted(set(validation["base_ids"])):
        p0_id = f"{base_id}::P0"
        p0_idx = by_id[p0_id]
        for q in operator["q"]:
            pq_id = f"{base_id}::{q['name']}"
            pq_idx = by_id[pq_id]
            j_distance = float(np.linalg.norm(jval[p0_idx] - jval[pq_idx]))
            coordinate_distance = float(np.linalg.norm(cval[p0_idx] - cval[pq_idx]))
            f0 = model._fingerprints(validation)[p0_idx]
            fq = model._fingerprints(validation)[pq_idx]
            response_relative = float(np.linalg.norm(fq - f0) /
                                      max(np.linalg.norm(fq), np.linalg.norm(f0), 1e-8))
            rows.append({"base_trial_id": base_id, "family": validation["family"][p0_idx],
                         "q_name": q["name"], "boundary_J_distance": j_distance,
                         "oracle_coordinate_distance": coordinate_distance,
                         "train_action_fingerprint_relative_difference": response_relative,
                         "natural_J_prediction_distance_between_pair": float(np.linalg.norm(prediction[p0_idx] - prediction[pq_idx])),
                         "workspace_state_aliasing": bool(j_distance == 0 and coordinate_distance > freeze["alias_threshold_coordinate_distance"]
                                                          and response_relative > freeze["alias_threshold_response_fingerprint_relative"])})
    pairs = pd.DataFrame(rows)
    pairs.to_parquet(root / PAIRS, index=False, compression="zstd")
    result = {"design_digest": freeze["freeze_digest"], "selected_k_for_diagnostics": k,
              "natural_validation_base_states": int(natural_val.sum()),
              "counterfactual_validation_operator_states": int((~natural_val).sum()),
              "natural_J_to_oracle_C_relative_l2": natural_error,
              "natural_J_to_oracle_C_R2": natural_r2,
              "counterfactual_J_to_oracle_C_relative_l2": counterfactual_error,
              "same_J_pair_count": len(pairs), "all_boundary_J_exact_equal": bool((pairs.boundary_J_distance == 0).all()),
              "median_same_J_oracle_coordinate_distance": float(pairs.oracle_coordinate_distance.median()),
              "median_same_J_fingerprint_relative_difference": float(pairs.train_action_fingerprint_relative_difference.median()),
              "workspace_state_aliasing_pair_fraction": float(pairs.workspace_state_aliasing.mean()),
              "workspace_state_aliasing_observed": bool(pairs.workspace_state_aliasing.any()),
              "interpretation_limit": "Operational same-J operator-coordinate aliasing, not a full POMDP or Markov-state proof.",
              "pair_path": str(PAIRS), "pair_sha256": sha256_file(root / PAIRS)}
    write_json_atomic(root / SUMMARY, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze"))
    args = parser.parse_args()
    result = {"prepare": prepare, "analyze": analyze}[args.stage](Path.cwd())
    print(json.dumps({"digest": result.get("freeze_digest", result.get("design_digest")),
                      "workspace_state_aliasing_observed": result.get("workspace_state_aliasing_observed")}, indent=2))
