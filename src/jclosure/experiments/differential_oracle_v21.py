"""Z4 exact-JVP same-action oracle ceiling for V20 finite responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import neural_models_v21 as neural
from jclosure.experiments import operator_model_v20 as v20
from jclosure.experiments import realized_factorial_v21 as realized
from jclosure.experiments.paired_geometry_v21 import _matrix_path
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="src/jclosure/experiments/differential_oracle_v21.py"
SUMMARY=bank.OUT/"differential_oracle_ceiling_v21.json"


def prepare(root:Path)->dict:
    roles=verify_stage(root,"roles")
    verify_stage(root,"paired_geometry_design")
    verify_stage(root,"raw_state_reference")
    neural_design=verify_stage(root,"neural_model_design")
    return stage_freeze(root,"differential_oracle_design",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_paired_geometry_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_raw_state_reference.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_neural_model_design.freeze.json"],
                        {"Z4":"exact_same_state_same_action_autograd_JVP_normalized_stack288",
                         "oracle_not_deployable":True,
                         "development_base_states":len(roles["jvp_development"]),
                         "validation_base_states":len(roles["jvp_validation"]),
                         "paired_operator_states_per_base":2,
                         "train_actions":12,"unseen_validation_actions":6,
                         "models":["direct_identity","train_scalar_calibrated","G4_joint_MLP","G5_rank8_hypernetwork"],
                         "state_conditions":["S2","S4"],
                         "hyperparameters":{key:neural_design[key] for key in ("epochs","batch_size","learning_rate","weight_decay","seed")},
                         "training":"only_50_development_bases_P0/Pq_x_12_train_actions_positive",
                         "evaluation":"25_disjoint_validation_bases_P0/Pq_x_6_unseen_actions_and_signed_tests",
                         "same_action_JVP_input_is_target_side_oracle_measurement":True,
                         "final_six_action_responses_opened":False})


def _panel(root:Path,role:str,items:list[dict],expected_indices:list[int])->tuple[np.ndarray,list[str]]:
    matrices=[]
    ids=[]
    for item in items:
        base_id=item["base_trial_id"]
        path=_matrix_path(role,base_id)
        if not path.exists():
            raise RuntimeError(f"V21 Z4 paired matrix missing: {path}")
        with np.load(path,allow_pickle=False) as source:
            if source["probe_indices"][:18].astype(int).tolist()!=expected_indices:
                raise RuntimeError("V21 Z4 first18 action columns differ from frozen V20 partition")
            matrices.extend([source["P0_JVP"][:,:18].T,source["Pq_JVP"][:,:18].T])
        ids.extend([f"{base_id}::P0",f"{base_id}::{item['q_name']}"])
    return np.stack(matrices).astype(np.float32),ids


def run(root:Path)->dict:
    design=verify_stage(root,"differential_oracle_design")
    roles=verify_stage(root,"roles")
    directions=[int(x["direction_index"]) for x in roles["train_actions"]+roles["validation_actions"]]
    ztrain,train_ids=_panel(root,"development",roles["jvp_development"],directions)
    zval,val_ids=_panel(root,"validation",roles["jvp_validation"],directions)
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train=v20._load(root,"operator_train",op)
    val=v20._load(root,"operator_validation",op)
    train_lookup={str(x):i for i,x in enumerate(train["ids"])}
    val_lookup={str(x):i for i,x in enumerate(val["ids"])}
    itrain=np.asarray([train_lookup[x] for x in train_ids])
    ival=np.asarray([val_lookup[x] for x in val_ids])
    ytrain=train["Y"][("train",1)][itrain]
    families=val["family"][ival]
    bases=val["base_ids"][ival]
    direct={}
    alpha=float(np.sum(ztrain[:,:12,:]*ytrain)/max(float(np.sum(ztrain[:,:12,:]**2)),1e-20))
    for name,multiplier in (("direct_identity",1.0),("train_scalar_calibrated",alpha)):
        metrics={}
        for test,observed,truth in (
            ("seen_direction_new_state",zval[:,:12,:],val["Y"][("train",1)][ival]),
            ("unseen_direction",zval[:,12:18,:],val["Y"][("validation",1)][ival]),
            ("unseen_sign",-zval[:,:12,:],val["Y"][("train",-1)][ival]),
            ("unseen_direction_and_sign",-zval[:,12:18,:],val["Y"][("validation",-1)][ival])):
            metrics[test]=fact._metric(truth,observed*multiplier,families,bases)
        direct[name]={"train_scalar":multiplier,"metrics":metrics}
    state=fact._load_state()
    with np.load(bank.SCRATCH/"raw_state_reference_v21.npz",allow_pickle=False) as source:
        state["S4_train"],state["S4_validation"]=source["S4_train"],source["S4_validation"]
    state_train_lookup={str(x):i for i,x in enumerate(state["train_ids"])}
    state_val_lookup={str(x):i for i,x in enumerate(state["validation_ids"])}
    z_norm=max(float(np.median(np.linalg.norm(ztrain[:,:12,:].reshape(-1,288),axis=1))),1e-8)
    ztrain=ztrain/z_norm
    zval=zval/z_norm
    rows=[]
    for sname in design["state_conditions"]:
        raw_train=state[f"{sname}_train"][[state_train_lookup[x] for x in train_ids]]
        raw_val=state[f"{sname}_validation"][[state_val_lookup[x] for x in val_ids]]
        mean=raw_train.mean(axis=0)
        scale=max(float(np.mean(np.sum((raw_train-mean)**2,axis=1)))**0.5,1e-8)
        s_train,s_val=(raw_train-mean)/scale,(raw_val-mean)/scale
        for model_name in ("G4_joint_MLP","G5_rank8_hypernetwork"):
            path=neural.WEIGHTS/f"{sname}_Z4_{model_name}.pt"
            model=realized._train(s_train,ztrain[:,:12,:],ytrain,model_name,design,path)
            metrics={}
            for test,indices,sign,truth in (
                ("seen_direction_new_state",list(range(12)),1,val["Y"][("train",1)][ival]),
                ("unseen_direction",list(range(12,18)),1,val["Y"][("validation",1)][ival]),
                ("unseen_sign",list(range(12)),-1,val["Y"][("train",-1)][ival]),
                ("unseen_direction_and_sign",list(range(12,18)),-1,val["Y"][("validation",-1)][ival])):
                prediction=realized._predict(model,s_val,zval[:,indices,:]*sign)
                metrics[test]=fact._metric(truth,prediction,families,bases)
            rows.append({"S":sname,"Z":"Z4_exact_JVP_oracle","model":model_name,
                         "parameter_count":sum(p.numel() for p in model.parameters()),
                         "weights_scratch_path":str(path),"weights_sha256":sha256_file(path),
                         "metrics":metrics})
            print(f"V21 Z4 oracle {sname} {model_name} unseen={metrics['unseen_direction']['stack_relative_l2']:.5f}",flush=True)
    result={"design_digest":design["freeze_digest"],"development_operator_states":len(train_ids),
            "validation_operator_states":len(val_ids),"JVP_train_norm_median":z_norm,
            "direct_oracle":direct,"state_conditioned_models":rows,
            "same_action_JVP_oracle_not_deployable":True,
            "final_six_action_responses_opened":False}
    target=root/SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,result)
    return {"direct_unseen_direction":direct["direct_identity"]["metrics"]["unseen_direction"]["stack_relative_l2"],
            "best_state_conditioned_unseen_direction":min(x["metrics"]["unseen_direction"]["stack_relative_l2"] for x in rows)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
