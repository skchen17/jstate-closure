"""Frozen nested 4/8/12 action-data scaling with fixed S2 and Z1/G2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import operator_model_v20 as v20
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/action_scaling_v21.py"
SUMMARY = bank.OUT / "action_data_scaling_v21.json"


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    verify_stage(root, "factorial_design")
    return stage_freeze(root, "action_scaling_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_factorial_design.freeze.json"],
                        {"nested_train_action_coordinate_order": roles["nested_train_action_coordinates"],
                         "sizes": [4, 8, 12], "fixed_state_representation": "S2_full_12_train_action_fingerprint",
                         "fixed_action_representation": "Z1_full_requested_exact_linear_kernel",
                         "fixed_model": "G2_quadratic_action", "ridge_grid": [0.0001, 0.001, 0.01, 0.1, 1.0],
                         "internal_selection": "heldout_development_states_and_last_action_quarter; no_validation_action_response_in_selection",
                         "evaluation": "same_six_V20_validation_actions_and_200_operator_validation_states",
                         "state_context_kept_fixed_across_action_counts": True,
                         "final_six_action_responses_opened": False})


def run(root: Path) -> dict:
    design = verify_stage(root, "action_scaling_design")
    state = fact._load_state()
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20._load(root, "operator_train", operator)
    val = v20._load(root, "operator_validation", operator)
    action = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    gram = fact._action_gram(action, "Z1")
    order = [int(x["coordinate_index"]) for x in operator["shared_measured_actions"]]
    nested = [order.index(int(x)) for x in design["nested_train_action_coordinate_order"]]
    heldout = list(range(12, 18))
    inner_train, inner_val = fact._select_indices(train["base_ids"])
    features = state["S2_train"]
    knn, knv = fact._state_kernel(features[inner_train], features[inner_val])
    ktrain, kval = fact._state_kernel(features, state["S2_validation"])
    rows = []
    for size in design["sizes"]:
        fit = nested[:size]
        internal_fit_count = size - size // 4
        internal_fit, internal_test = fit[:internal_fit_count], fit[internal_fit_count:]
        ka_inner, qan_inner = fact._action_kernel(gram, internal_fit, internal_test,
                                                   [1] * len(internal_test), "G2_quadratic_action")
        spectral_inner = fact._spectrum(knn, ka_inner)
        target_inner = train["Y"][("train", 1)][inner_train][:, internal_fit, :]
        truth_inner = train["Y"][("train", 1)][inner_val][:, internal_test, :]
        selection = []
        for ridge in design["ridge_grid"]:
            prediction = fact._fit_predict(knn, knv, ka_inner, qan_inner, target_inner,
                                           float(ridge), "G2_quadratic_action", spectral_inner)
            metric = fact._metric(truth_inner, prediction, train["family"][inner_val], train["base_ids"][inner_val])
            selection.append({"ridge": float(ridge), "internal_l2": metric["stack_relative_l2"]})
        chosen = min(selection, key=lambda x: x["internal_l2"])
        ka, qan = fact._action_kernel(gram, fit, heldout, [1] * 6, "G2_quadratic_action")
        spectral = fact._spectrum(ktrain, ka)
        prediction = fact._fit_predict(ktrain, kval, ka, qan,
                                       train["Y"][("train", 1)][:, fit, :], chosen["ridge"],
                                       "G2_quadratic_action", spectral)
        metric = fact._metric(val["Y"][("validation", 1)], prediction, val["family"], val["base_ids"])
        rows.append({"train_action_count": size, "selected_ridge": chosen["ridge"],
                     "internal_curve": selection, "validation_unseen_direction": metric})
        print(f"V21 action scaling {size}: {metric['stack_relative_l2']:.5f}", flush=True)
    result = {"design_digest": design["freeze_digest"], "rows": rows,
              "largest_supported_frozen_train_action_count": 12,
              "state_context_fixed_full_S2_with_all_12_action_fingerprints": True,
              "final_six_action_responses_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    return {"curve": [(x["train_action_count"], x["validation_unseen_direction"]["stack_relative_l2"]) for x in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd()), indent=2))
