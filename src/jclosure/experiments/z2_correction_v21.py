"""Append-only train-normalized architecture-kernel correction for V21 Z2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import operator_model_v20 as v20
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/z2_correction_v21.py"
SUMMARY = bank.OUT / "z2_channel_normalization_correction_v21.json"


def _corrected_gram(descriptor: dict) -> tuple[np.ndarray, dict]:
    global_median = float(descriptor["Z1_exact_linear_kernel"]["train_normalization_norm"])
    blocks = descriptor["Z2_channel_exact_linear_kernels"]
    parts, scales = [], {}
    for channel in ("REC", "Conv", "KV"):
        old = float(blocks[channel]["train_normalization_norm"])
        robust = max(old, global_median / 4.0)
        weight = (old / robust) ** 2
        parts.append(np.asarray(blocks[channel]["exact_requested_Gram"], dtype=np.float64) * weight)
        scales[channel] = {"original_train_median": old, "robust_train_only_floor": robust,
                           "old_gram_multiplier": weight}
    gram = sum(parts) / 3.0
    train_scale = max(float(np.median(np.diag(gram[:12, :12]))), 1e-12)
    return gram / train_scale, {"global_train_median": global_median, "channel_scales": scales,
                                "combined_train_diagonal_median": train_scale}


def prepare(root: Path) -> dict:
    verify_stage(root, "factorial_design")
    original = root / "results/v21/processed/bottleneck_factorial_v21.json"
    if not original.exists():
        raise RuntimeError("original V21 Z2 numerical failure record missing")
    descriptor = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    _, scales = _corrected_gram(descriptor)
    return stage_freeze(root, "z2_channel_normalization_amendment_1",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_factorial_design.freeze.json",
                         "results/v21/processed/bottleneck_factorial_v21.json",
                         "results/v21/processed/action_representation_v21.json"],
                        {"reason": "Original Z2 channel-wise median scaling magnified near-zero train KV energy into huge validation kernel values; retain original failed run, use train-only floor global Z1 median/4 and combined train-diagonal renormalization.",
                         "condition": "S2×Z2", "models": ["G0_additive", "G1_bilinear", "G2_quadratic_action", "G3_tensor_polynomial"],
                         "normalization": scales, "original_factorial_sha256": sha256_file(original),
                         "validation_Z2_corrected_response_metrics_observed_before_amendment": False,
                         "final_six_action_responses_opened": False})


def run(root: Path) -> dict:
    design = verify_stage(root, "z2_channel_normalization_amendment_1")
    original = verify_stage(root, "factorial_design")
    descriptor = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    gram, scales = _corrected_gram(descriptor)
    features = fact._load_state()
    op = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20._load(root, "operator_train", op)
    val = v20._load(root, "operator_validation", op)
    roles = verify_stage(root, "roles")
    order = [int(x["coordinate_index"]) for x in op["shared_measured_actions"]]
    nested = [order.index(int(x)) for x in roles["nested_train_action_coordinates"]]
    fit_actions, internal_fit, internal_test = list(range(12)), nested[:8], nested[8:]
    inner_train, inner_val = fact._select_indices(train["base_ids"])
    states, val_states = features["S2_train"], features["S2_validation"]
    cfg = verify(root)["config"]["gates"]
    gate = {"heldout_gates": {"relative_l2_max": cfg["relative_l2_max"],
                              "cosine_min": cfg["stack_median_cosine_min"],
                              "norm_ratio_min": cfg["norm_ratio_min"], "norm_ratio_max": cfg["norm_ratio_max"],
                              "family_relative_l2_max": cfg["family_relative_l2_max"]}}
    rows = []
    for model in original["models"]:
        poly = model == "G3_tensor_polynomial"
        ks, kn = fact._state_kernel(states[inner_train], states[inner_val], poly)
        ka, qan = fact._action_kernel(gram, internal_fit, internal_test, [1]*len(internal_test), model)
        spectral = None if model == "G0_additive" else fact._spectrum(ks, ka)
        target = train["Y"][("train", 1)][inner_train][:, internal_fit, :]
        truth = train["Y"][("train", 1)][inner_val][:, internal_test, :]
        curve = []
        for ridge in original["ridge_grid"]:
            prediction = fact._fit_predict(ks, kn, ka, qan, target, float(ridge), model, spectral)
            metric = fact._metric(truth, prediction, train["family"][inner_val], train["base_ids"][inner_val])
            curve.append({"ridge": float(ridge), "internal_l2": metric["stack_relative_l2"]})
        chosen = min(curve, key=lambda x: x["internal_l2"])
        ks, kn = fact._state_kernel(states, val_states, poly)
        ka, _ = fact._action_kernel(gram, fit_actions, fit_actions, [1]*12, model)
        spectral = None if model == "G0_additive" else fact._spectrum(ks, ka)
        metrics = {}
        for name, indices, signs, truth in (
            ("seen_direction_new_state", fit_actions, [1]*12, val["Y"][("train", 1)]),
            ("unseen_direction", list(range(12,18)), [1]*6, val["Y"][("validation", 1)]),
            ("unseen_sign", fit_actions, [-1]*12, val["Y"][("train", -1)]),
            ("unseen_direction_and_sign", list(range(12,18)), [-1]*6, val["Y"][("validation", -1)])):
            _, qan = fact._action_kernel(gram, fit_actions, indices, signs, model)
            prediction = fact._fit_predict(ks, kn, ka, qan, train["Y"][("train", 1)], chosen["ridge"], model, spectral)
            metrics[name] = fact._metric(truth, prediction, val["family"], val["base_ids"])
        rows.append({"S": "S2", "Z": "Z2_train_floor_corrected", "model": model,
                     "selected_ridge": chosen["ridge"], "internal_selection_curve": curve,
                     "metrics": metrics,
                     "direction_and_sign_preliminary_gate": bool(v20._gate(metrics["unseen_direction"], gate)
                                                                 and v20._gate(metrics["unseen_sign"], gate))})
        print(f"V21 corrected Z2 {model} unseen={metrics['unseen_direction']['stack_relative_l2']:.5f}", flush=True)
    result = {"amendment_digest": design["freeze_digest"], "normalization": scales,
              "original_Z2_failure_preserved": True, "results": rows,
              "final_six_action_responses_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return {"best_unseen_direction": min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd()), indent=2))
