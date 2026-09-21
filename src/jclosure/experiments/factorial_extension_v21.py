"""Append-only S0/S1/S4 state-ceiling extensions under fixed practical actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import operator_model_v20 as v20
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/factorial_extension_v21.py"
SUMMARY = bank.OUT / "bottleneck_factorial_extension_v21.json"
CONDITIONS = [("S0", "Z0"), ("S1", "Z1"), ("S4", "Z0"), ("S4", "Z1")]


def prepare(root: Path) -> dict:
    old = verify_stage(root, "factorial_design")
    verify_stage(root, "raw_state_reference")
    return stage_freeze(root, "factorial_extension_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_factorial_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_raw_state_reference.freeze.json"],
                        {"conditions": [{"S":s,"Z":z} for s,z in CONDITIONS],
                         "models": old["models"], "ridge_grid": old["ridge_grid"],
                         "same_train_only_ridge_selection": True,
                         "same_V20_operator_validation_states_and_actions": True,
                         "S4_raw_Nystrom_reference_not_compact": True,
                         "final_six_action_responses_opened": False})


def run(root: Path) -> dict:
    design = verify_stage(root, "factorial_extension_design")
    state = fact._load_state()
    with np.load(bank.SCRATCH / "raw_state_reference_v21.npz", allow_pickle=False) as source:
        state["S4_train"] = source["S4_train"]
        state["S4_validation"] = source["S4_validation"]
        if not np.array_equal(state["train_ids"], source["train_ids"]) or not np.array_equal(state["validation_ids"], source["validation_ids"]):
            raise RuntimeError("V21 S4 state order differs from V20 operator bank")
    descriptor = json.loads((root / "results/v21/processed/action_representation_v21.json").read_text())
    op = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train = v20._load(root, "operator_train", op)
    val = v20._load(root, "operator_validation", op)
    roles = verify_stage(root, "roles")
    action_order = [int(x["coordinate_index"]) for x in op["shared_measured_actions"]]
    nested = [action_order.index(int(x)) for x in roles["nested_train_action_coordinates"]]
    internal_fit, internal_val = nested[:8], nested[8:]
    inner_train, inner_val = fact._select_indices(train["base_ids"])
    cfg = verify(root)["config"]["gates"]
    gate = {"heldout_gates": {"relative_l2_max": cfg["relative_l2_max"],
                              "cosine_min": cfg["stack_median_cosine_min"],
                              "norm_ratio_min": cfg["norm_ratio_min"],
                              "norm_ratio_max": cfg["norm_ratio_max"],
                              "family_relative_l2_max": cfg["family_relative_l2_max"]}}
    rows = []
    for condition in design["conditions"]:
        s,z = condition["S"],condition["Z"]
        x, xv = state[f"{s}_train"],state[f"{s}_validation"]
        gram = fact._action_gram(descriptor,z)
        for model in design["models"]:
            poly = model == "G3_tensor_polynomial"
            ks,kn = fact._state_kernel(x[inner_train],x[inner_val],poly)
            ka,qan = fact._action_kernel(gram,internal_fit,internal_val,[1]*4,model)
            spectrum = None if model == "G0_additive" else fact._spectrum(ks,ka)
            target = train["Y"][("train",1)][inner_train][:,internal_fit,:]
            truth = train["Y"][("train",1)][inner_val][:,internal_val,:]
            curve = []
            for ridge in design["ridge_grid"]:
                prediction = fact._fit_predict(ks,kn,ka,qan,target,float(ridge),model,spectrum)
                metric = fact._metric(truth,prediction,train["family"][inner_val],train["base_ids"][inner_val])
                curve.append({"ridge":float(ridge),"internal_l2":metric["stack_relative_l2"]})
            chosen = min(curve,key=lambda x:x["internal_l2"])
            ks,kn = fact._state_kernel(x,xv,poly)
            ka,_ = fact._action_kernel(gram,list(range(12)),list(range(12)),[1]*12,model)
            spectrum = None if model == "G0_additive" else fact._spectrum(ks,ka)
            metrics = {}
            for name,indices,signs,truth in (
                ("seen_direction_new_state",list(range(12)),[1]*12,val["Y"][("train",1)]),
                ("unseen_direction",list(range(12,18)),[1]*6,val["Y"][("validation",1)]),
                ("unseen_sign",list(range(12)),[-1]*12,val["Y"][("train",-1)]),
                ("unseen_direction_and_sign",list(range(12,18)),[-1]*6,val["Y"][("validation",-1)])):
                _,qan = fact._action_kernel(gram,list(range(12)),indices,signs,model)
                prediction = fact._fit_predict(ks,kn,ka,qan,train["Y"][("train",1)],chosen["ridge"],model,spectrum)
                metrics[name] = fact._metric(truth,prediction,val["family"],val["base_ids"])
            rows.append({"S":s,"Z":z,"model":model,"selected_ridge":chosen["ridge"],
                         "internal_selection_curve":curve,"metrics":metrics,
                         "direction_and_sign_preliminary_gate":bool(v20._gate(metrics["unseen_direction"],gate)
                                                                    and v20._gate(metrics["unseen_sign"],gate))})
            print(f"V21 factorial extension {s} {z} {model} unseen={metrics['unseen_direction']['stack_relative_l2']:.5f}",flush=True)
    result = {"design_digest":design["freeze_digest"],"results":rows,
              "final_six_action_responses_opened":False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,result)
    return {"conditions":len(design["conditions"]),"best_unseen_direction":min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
