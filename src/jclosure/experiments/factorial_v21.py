"""One-axis-at-a-time state/action/model ceilings with train-only model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import operator_model_v20 as v20_model
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/factorial_v21.py"
SUMMARY = bank.OUT / "bottleneck_factorial_v21.json"


def prepare(root: Path) -> dict:
    config = verify(root)["config"]
    roles = verify_stage(root, "roles")
    verify_stage(root, "action_representations")
    verify_stage(root, "state_representations")
    verify_stage(root, "s0_dimension_amendment_1")
    conditions = [("S1", "Z0"), ("S2", "Z0"), ("S2", "Z1"), ("S2", "Z2"),
                  ("S3", "Z0"), ("S3", "Z1"), ("S0", "Z1")]
    return stage_freeze(root, "factorial_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_action_representations.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_state_representations.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_s0_dimension_amendment_1.freeze.json"],
                        {"conditions": [{"S": s, "Z": z} for s, z in conditions],
                         "models": ["G0_additive", "G1_bilinear", "G2_quadratic_action", "G3_tensor_polynomial"],
                         "ridge_grid": config["models"]["ridge_grid"],
                         "internal_state_holdout_rule": "sha256(base_trial_id) modulo 5 equals 0; all q siblings remain together",
                         "internal_train_action_coordinates": roles["nested_train_action_coordinates"][:8],
                         "internal_heldout_action_coordinates": roles["nested_train_action_coordinates"][8:],
                         "model_selection_metric": "internal_heldout_state_and_direction_stack_relative_L2_only",
                         "final_fit_state_role": "V20 operator_train only",
                         "evaluation_state_role": "V20 operator_validation",
                         "heldout_action_response_in_fit_or_selection": False,
                         "final_six_action_responses_opened": False,
                         "G4_G5_pending_separate_neural_design": True,
                         "S4_Z3_pending_separate_raw_feature_design": True})


def _load_state() -> dict:
    with np.load(bank.SCRATCH / "state_representations_v21.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _state_kernel(train: np.ndarray, evaluation: np.ndarray, polynomial: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    x = train.astype(np.float64) - mean
    y = evaluation.astype(np.float64) - mean
    rms = max(float(np.mean(np.sum(x*x, axis=1))) ** 0.5, 1e-8)
    x /= rms
    y /= rms
    gram = 1.0 + x @ x.T
    cross = 1.0 + y @ x.T
    if polynomial:
        gram = gram * gram
        cross = cross * cross
    return gram, cross


def _action_gram(descriptor: dict, z: str) -> np.ndarray:
    if z == "Z0":
        matrix = np.asarray(descriptor["Z0_exact_V20_descriptor"], dtype=np.float64)
        median = max(float(np.median(np.linalg.norm(matrix[:12], axis=1))), 1e-8)
        matrix = matrix / median
        return matrix @ matrix.T
    if z == "Z1":
        return np.asarray(descriptor["Z1_exact_linear_kernel"]["exact_requested_Gram"], dtype=np.float64)
    if z == "Z2":
        blocks = descriptor["Z2_channel_exact_linear_kernels"]
        return sum(np.asarray(blocks[channel]["exact_requested_Gram"], dtype=np.float64)
                   for channel in ("REC", "Conv", "KV")) / 3.0
    raise RuntimeError(f"unknown V21 action representation: {z}")


def _action_kernel(gram: np.ndarray, fit: list[int], evaluation: list[int], signs: list[int], model: str) -> tuple[np.ndarray, np.ndarray]:
    base = gram[np.ix_(fit, fit)]
    cross = gram[np.ix_(fit, evaluation)] * np.asarray(signs)[None, :]
    if model == "G0_additive":
        return 1.0 + base, 1.0 + cross
    if model == "G1_bilinear":
        return 1.0 + base, 1.0 + cross
    if model in ("G2_quadratic_action", "G3_tensor_polynomial"):
        return (1.0 + base) ** 2, (1.0 + cross) ** 2
    raise RuntimeError(model)


def _fit_predict(ks: np.ndarray, kn: np.ndarray, ka: np.ndarray, qan: np.ndarray,
                 targets: np.ndarray, ridge: float, model: str,
                 spectral: tuple | None = None) -> np.ndarray:
    ns, na, d = targets.shape
    if model == "G0_additive":
        # State-only offset and action-only mean; no state×action interaction.
        action_mean = targets.mean(axis=0)
        left = np.linalg.solve(ka + ridge*np.eye(na), action_mean)
        action_prediction = qan.T @ left
        residual = targets - action_mean[None, :, :]
        state_mean = residual.mean(axis=1)
        weight = np.linalg.solve(ks + ridge*np.eye(ns), state_mean)
        state_prediction = kn @ weight
        return (state_prediction[:, None, :] + action_prediction[None, :, :]).astype(np.float32)
    eig_s, u_s, eig_a, u_a = spectral if spectral is not None else _spectrum(ks, ka)
    eig_s = np.maximum(eig_s, 0)
    eig_a = np.maximum(eig_a, 0)
    transformed = np.einsum("ni,nad,am->imd", u_s, targets, u_a, optimize=True)
    transformed /= (eig_s[:, None, None] * eig_a[None, :, None] + ridge)
    left = kn @ u_s
    right = u_a.T @ qan
    return np.einsum("in,nmd,mj->ijd", left, transformed, right, optimize=True).astype(np.float32)


def _spectrum(ks: np.ndarray, ka: np.ndarray) -> tuple:
    eig_s, u_s = np.linalg.eigh(ks)
    eig_a, u_a = np.linalg.eigh(ka)
    return eig_s, u_s, eig_a, u_a


def _metric(truth: np.ndarray, predicted: np.ndarray, families: np.ndarray, base_ids: np.ndarray) -> dict:
    return v20_model._metric(truth, predicted, families, base_ids)


def _select_indices(base_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    holdout = np.array([int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % 5 == 0 for x in base_ids], dtype=bool)
    if holdout.sum() < 20 or (~holdout).sum() < 300:
        raise RuntimeError("V21 internal state split unexpectedly small")
    return np.flatnonzero(~holdout), np.flatnonzero(holdout)


def run(root: Path) -> dict:
    design = verify_stage(root, "factorial_design")
    config = verify(root)["config"]
    gate_thresholds = {"heldout_gates": {
        "relative_l2_max": config["gates"]["relative_l2_max"],
        "cosine_min": config["gates"]["stack_median_cosine_min"],
        "norm_ratio_min": config["gates"]["norm_ratio_min"],
        "norm_ratio_max": config["gates"]["norm_ratio_max"],
        "family_relative_l2_max": config["gates"]["family_relative_l2_max"]}}
    descriptor = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    features = _load_state()
    v20_operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20_model._load(root, "operator_train", v20_operator)
    validation = v20_model._load(root, "operator_validation", v20_operator)
    if not np.array_equal(features["train_ids"], train["ids"]) or not np.array_equal(features["validation_ids"], validation["ids"]):
        raise RuntimeError("V21 state bank order drift")
    roles = verify_stage(root, "roles")
    order = [int(x["coordinate_index"]) for x in roles["train_actions"] + roles["validation_actions"]]
    nested = roles["nested_train_action_coordinates"]
    internal_fit = [order.index(x) for x in nested[:8]]
    internal_eval = [order.index(x) for x in nested[8:]]
    final_fit = list(range(12))
    val_actions = list(range(12, 18))
    inner_train, inner_holdout = _select_indices(train["base_ids"])
    outcomes = []
    for condition in design["conditions"]:
        s, z = condition["S"], condition["Z"]
        state_train = features[f"{s}_train"]
        state_val = features[f"{s}_validation"]
        action_gram = _action_gram(descriptor, z)
        for model in design["models"]:
            poly = model == "G3_tensor_polynomial"
            ks_inner, kn_inner = _state_kernel(state_train[inner_train], state_train[inner_holdout], poly)
            ka_inner, qan_inner = _action_kernel(action_gram, internal_fit, internal_eval,
                                                  [1] * len(internal_eval), model)
            y_inner = train["Y"][("train", 1)][inner_train][:, internal_fit, :]
            truth_inner = train["Y"][("train", 1)][inner_holdout][:, internal_eval, :]
            spectral_inner = None if model == "G0_additive" else _spectrum(ks_inner, ka_inner)
            tried = []
            for ridge in design["ridge_grid"]:
                prediction = _fit_predict(ks_inner, kn_inner, ka_inner, qan_inner,
                                          y_inner, float(ridge), model, spectral_inner)
                metric = _metric(truth_inner, prediction, train["family"][inner_holdout], train["base_ids"][inner_holdout])
                tried.append({"ridge": float(ridge), "internal_unseen_direction_stack_relative_l2": metric["stack_relative_l2"]})
            chosen = min(tried, key=lambda x: x["internal_unseen_direction_stack_relative_l2"])
            ks, kn = _state_kernel(state_train, state_val, poly)
            y = train["Y"][("train", 1)]
            metrics = {}
            tests = {
                "seen_direction_new_state": (final_fit, [1]*12, validation["Y"][("train", 1)]),
                "unseen_direction": (val_actions, [1]*6, validation["Y"][("validation", 1)]),
                "unseen_sign": (final_fit, [-1]*12, validation["Y"][("train", -1)]),
                "unseen_direction_and_sign": (val_actions, [-1]*6, validation["Y"][("validation", -1)]),
            }
            ka_final, _ = _action_kernel(action_gram, final_fit, final_fit, [1]*12, model)
            spectral_final = None if model == "G0_additive" else _spectrum(ks, ka_final)
            for test, (indices, signs, truth) in tests.items():
                ka, qan = _action_kernel(action_gram, final_fit, indices, signs, model)
                prediction = _fit_predict(ks, kn, ka, qan, y, chosen["ridge"], model, spectral_final)
                metrics[test] = _metric(truth, prediction, validation["family"], validation["base_ids"])
            direction = metrics["unseen_direction"]
            gate = (v20_model._gate(direction, gate_thresholds)
                    and v20_model._gate(metrics["unseen_sign"], gate_thresholds))
            outcomes.append({"S": s, "Z": z, "model": model, "selected_ridge": chosen["ridge"],
                             "internal_selection_curve": tried, "metrics": metrics,
                             "direction_and_sign_preliminary_gate": bool(gate),
                             "state_feature_dimension": int(state_train.shape[1]),
                             "action_kernel_rank": int(np.linalg.matrix_rank(action_gram[:12, :12]))})
            print(f"V21 factorial {s} {z} {model} unseen={direction['stack_relative_l2']:.4f}", flush=True)
    result = {"design_digest": design["freeze_digest"], "train_operator_states": len(train["ids"]),
              "validation_operator_states": len(validation["ids"]),
              "internal_fit_operator_states": len(inner_train), "internal_holdout_operator_states": len(inner_holdout),
              "internal_fit_action_count": len(internal_fit), "internal_holdout_action_count": len(internal_eval),
              "results": outcomes, "any_practical_preliminary_direction_sign_gate": any(x["direction_and_sign_preliminary_gate"] for x in outcomes),
              "Z3_S4_G4_G5_pending_separate_stages": True,
              "final_six_actions_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return {"conditions": len(design["conditions"]), "models_per_condition": len(design["models"]),
            "best_unseen_direction": min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in outcomes)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run}[args.stage](Path.cwd()), indent=2))
