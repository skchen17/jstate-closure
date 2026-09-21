"""S2/S4 × state-dependent Z3 realized-action decoder ceiling and coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import neural_models_v21 as neural
from jclosure.experiments import operator_model_v20 as v20
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/realized_factorial_v21.py"
SUMMARY = bank.OUT / "realized_action_factorial_v21.json"
COVERAGE = bank.OUT / "realized_action_coverage_v21.json"


def prepare(root: Path) -> dict:
    verify_stage(root, "raw_state_reference")
    verify_stage(root, "realized_action_representations")
    neural_design = verify_stage(root, "neural_model_design")
    return stage_freeze(root, "realized_action_factorial_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_raw_state_reference.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_realized_action_representations.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_neural_model_design.freeze.json"],
                        {"conditions": [{"S":"S2","Z":"Z3"},{"S":"S4","Z":"Z3"}],
                         "models": neural_design["models"],
                         "hyperparameters": {key:neural_design[key] for key in ("epochs","batch_size","learning_rate","weight_decay","seed")},
                         "Z3": "39D_realized_BF16_state_dependent_REC_Conv_KV_Nystrom_features",
                         "training": "600_train_states_x_12_positive_train_actions_only",
                         "evaluation": "same_200_validation_states_x_18_open_actions_x_both_signs",
                         "G0_G3_separable_kernel_ineligible_for_state_dependent_Z3": True,
                         "action_coverage": "same_state_realized_Z3_train_span_and_nearest_abs_cosine",
                         "final_six_action_responses_opened": False})


def _train(s: np.ndarray, z: np.ndarray, y: np.ndarray, model_name: str, design: dict, path: Path):
    torch.manual_seed(int(design["hyperparameters"]["seed"]))
    torch.set_num_threads(4)
    n,a,d = y.shape
    states = torch.from_numpy(np.repeat(s,a,axis=0).astype(np.float32))
    actions = torch.from_numpy(z.reshape(n*a,-1).astype(np.float32))
    targets = torch.from_numpy(y.reshape(n*a,d).astype(np.float32))
    model = neural.JointMLP(s.shape[1],z.shape[-1],d) if model_name=="G4_joint_MLP" else neural.LowRankHyper(s.shape[1],z.shape[-1],d)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(design["hyperparameters"]["learning_rate"]),
                                  weight_decay=float(design["hyperparameters"]["weight_decay"]))
    rng = np.random.default_rng(int(design["hyperparameters"]["seed"]))
    model.train()
    for epoch in range(int(design["hyperparameters"]["epochs"])):
        order = rng.permutation(n*a)
        for start in range(0,n*a,int(design["hyperparameters"]["batch_size"])):
            idx = order[start:start+int(design["hyperparameters"]["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(states[idx],actions[idx])-targets[idx])**2).mean()
            loss.backward()
            optimizer.step()
        if (epoch+1)%5==0:
            print(f"V21 Z3 {path.stem} epoch {epoch+1} train_last_mse {float(loss.item()):.5g}",flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path)
    return model.eval()


def _predict(model,s,z):
    n,a,dz=z.shape
    states=torch.from_numpy(np.repeat(s,a,axis=0).astype(np.float32))
    actions=torch.from_numpy(z.reshape(n*a,dz).astype(np.float32))
    out=[]
    with torch.no_grad():
        for start in range(0,n*a,256):
            out.append(model(states[start:start+256],actions[start:start+256]).numpy())
    return np.vstack(out).reshape(n,a,-1)


def _coverage(z: np.ndarray) -> dict:
    # Validation states only; geometry readback, no response labels.
    by_action=[]
    for action in range(12,18):
        cosines,residuals,norm_ratios=[] ,[],[]
        for state in z:
            train=state[:12,0,:].astype(np.float64)
            val=state[action,0,:].astype(np.float64)
            train_norm=np.linalg.norm(train,axis=1)
            val_norm=max(float(np.linalg.norm(val)),1e-12)
            cosine=train@val/np.maximum(train_norm*val_norm,1e-12)
            _, singular,vh=np.linalg.svd(train,full_matrices=False)
            rank=int((singular>max(float(singular[0])*1e-9,1e-10)).sum())
            basis=vh[:rank]
            residual=np.linalg.norm(val-val@basis.T@basis)/val_norm
            cosines.append(float(np.max(np.abs(cosine))))
            residuals.append(float(residual))
            norm_ratios.append(val_norm/max(float(np.median(train_norm)),1e-12))
        by_action.append({"validation_action_index_in_open_18":action,
                          "median_nearest_train_abs_cosine":float(np.median(cosines)),
                          "median_train_span_relative_residual":float(np.median(residuals)),
                          "median_norm_over_train_median":float(np.median(norm_ratios))})
    return {"by_validation_action":by_action,
            "median_nearest_train_abs_cosine":float(np.median([x["median_nearest_train_abs_cosine"] for x in by_action])),
            "median_train_span_relative_residual":float(np.median([x["median_train_span_relative_residual"] for x in by_action]))}


def run(root: Path) -> dict:
    design=verify_stage(root,"realized_action_factorial_design")
    with np.load(bank.SCRATCH/"realized_action_v21.npz",allow_pickle=False) as source:
        z_train,z_val=source["Z3_train"],source["Z3_validation"]
        zids_train,zids_val=source["train_ids"],source["validation_ids"]
    state=fact._load_state()
    with np.load(bank.SCRATCH/"raw_state_reference_v21.npz",allow_pickle=False) as source:
        state["S4_train"],state["S4_validation"]=source["S4_train"],source["S4_validation"]
    if not np.array_equal(state["train_ids"],zids_train) or not np.array_equal(state["validation_ids"],zids_val):
        raise RuntimeError("V21 realized-action state grid drift")
    z_norm=max(float(np.median(np.linalg.norm(z_train.reshape(-1,z_train.shape[-1]),axis=1))),1e-8)
    z_train=z_train/z_norm
    z_val=z_val/z_norm
    coverage=_coverage(z_val)
    coverage.update({"design_digest":design["freeze_digest"],"Z3_action_features_are_state_dependent":True,
                     "validation_response_labels_used":False,"final_six_action_responses_opened":False})
    write_json_atomic(root/COVERAGE,coverage)
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train=v20._load(root,"operator_train",op)
    val=v20._load(root,"operator_validation",op)
    cfg=verify(root)["config"]["gates"]
    gate={"heldout_gates":{"relative_l2_max":cfg["relative_l2_max"],
                            "cosine_min":cfg["stack_median_cosine_min"],
                            "norm_ratio_min":cfg["norm_ratio_min"],
                            "norm_ratio_max":cfg["norm_ratio_max"],
                            "family_relative_l2_max":cfg["family_relative_l2_max"]}}
    rows=[]
    for condition in design["conditions"]:
        sname=condition["S"]
        raw_train,raw_val=state[f"{sname}_train"],state[f"{sname}_validation"]
        mean=raw_train.mean(axis=0)
        scale=max(float(np.mean(np.sum((raw_train-mean)**2,axis=1)))**0.5,1e-8)
        s_train,s_val=(raw_train-mean)/scale,(raw_val-mean)/scale
        for name in design["models"]:
            path=neural.WEIGHTS/f"{sname}_Z3_{name}.pt"
            model=_train(s_train,z_train,train["Y"][("train",1)],name,design,path)
            metrics={}
            for test,indices,sign_index,truth in (
                ("seen_direction_new_state",list(range(12)),0,val["Y"][("train",1)]),
                ("unseen_direction",list(range(12,18)),0,val["Y"][("validation",1)]),
                ("unseen_sign",list(range(12)),1,val["Y"][("train",-1)]),
                ("unseen_direction_and_sign",list(range(12,18)),1,val["Y"][("validation",-1)])):
                predicted=_predict(model,s_val,z_val[:,indices,sign_index,:])
                metrics[test]=fact._metric(truth,predicted,val["family"],val["base_ids"])
            rows.append({"S":sname,"Z":"Z3","model":name,
                         "parameter_count":sum(p.numel() for p in model.parameters()),
                         "weights_scratch_path":str(path),"weights_sha256":sha256_file(path),
                         "metrics":metrics,
                         "direction_and_sign_preliminary_gate":bool(v20._gate(metrics["unseen_direction"],gate)
                                                                    and v20._gate(metrics["unseen_sign"],gate))})
            print(f"V21 realized factorial {sname} {name} unseen={metrics['unseen_direction']['stack_relative_l2']:.5f}",flush=True)
    result={"design_digest":design["freeze_digest"],"action_normalization_train_only":z_norm,
            "models":rows,"coverage_sha256":sha256_file(root/COVERAGE),
            "final_six_action_responses_opened":False}
    write_json_atomic(root/SUMMARY,result)
    return {"conditions":len(design["conditions"]),"best_unseen_direction":min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in rows)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
