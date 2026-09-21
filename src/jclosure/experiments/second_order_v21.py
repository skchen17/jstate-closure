"""Z5 half-amplitude finite second-difference oracle, distinct from full response labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import neural_models_v21 as neural
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import operator_model_v20 as v20
from jclosure.experiments import realized_factorial_v21 as realized
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.operator_v15 import stack
from jclosure.experiments.paired_geometry_v21 import _matrix_path
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="src/jclosure/experiments/second_order_v21.py"
RAW=bank.SCRATCH/"second_order"
SUMMARY=bank.OUT/"finite_second_order_oracle_v21.json"


def prepare(root:Path)->dict:
    roles=verify_stage(root,"roles")
    verify_stage(root,"paired_geometry_design")
    neural_design=verify_stage(root,"neural_model_design")
    return stage_freeze(root,"second_order_design",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_neural_model_design.freeze.json"],
                        {"Z5":"exact_JVP_plus_half_amplitude_central_finite_second_difference",
                         "finite_second_difference_not_exact_Hessian":True,
                         "epsilon_relative_to_frozen_action_base_alpha":0.5,
                         "target_full_action_response_amplitude":1.0,
                         "train":"50_development_bases_P0/Pq_x_12_train_actions_x_both_half_signs",
                         "validation":"25_disjoint_validation_bases_P0/Pq_x_6_unseen_actions_x_both_half_signs",
                         "same_action_half_step_oracle_not_deployable":True,
                         "full_amplitude_label_never_used_as_Z5_feature":True,
                         "models":["direct_Taylor","train_scalar_pair","S2_G4_joint_MLP","S2_G5_rank8_hypernetwork"],
                         "hyperparameters":{key:neural_design[key] for key in ("epochs","batch_size","learning_rate","weight_decay","seed")},
                         "V20_reliability_gate":True,
                         "final_six_action_responses_opened":False})


def _path(role:str,base_id:str)->Path:
    return RAW/role/f"second_{base_id}.npz"


def measure(root:Path,role:str,limit:int|None=None)->dict:
    design=verify_stage(root,"second_order_design")
    if role not in ("development","validation"):
        raise RuntimeError("V21 Z5 role invalid")
    roles=verify_stage(root,"roles")
    items=roles[f"jvp_{role}"]
    if limit is not None:
        items=items[:limit]
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    split=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    actions=roles["train_actions"] if role=="development" else roles["validation_actions"]
    prompts=v20_bank.prompt_index(root)
    teacher=v20_response._teacher_map(root)
    q_lookup={x["name"]:x for x in op["q"]}
    bundle,dense,_,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
    jids=np.asarray(split["selected_j"],dtype=int)
    lids=np.asarray(split["selected_logits"],dtype=int)
    scales={key:float(value) for key,value in split["target_scales"].items()}
    completed=0
    for number,item in enumerate(items,1):
        base_id=item["base_trial_id"]
        target=_path(role,base_id)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            completed+=1
            continue
        clean=v19._prefill_history(bundle,str(prompts[base_id]["prompt"]),measured,dense,state_layer)
        token=int(teacher[base_id][0])
        p0=clean["cache"]
        q=q_lookup[item["q_name"]]
        qrow=v19._q_row(values["directions"],q,float(q["alpha"]))
        pq=v19.apply(p0,1.0,qrow,rec,att,"native_fp32_add_bf16_writeback")
        states=(("P0",p0),(item["q_name"],pq))
        Q=[]
        reliable=[]
        for name,state in states:
            _,base_endpoint=v19._trajectory(bundle,dense,state,None,[token],1,clean["prompt_length"],
                                            jids,lids,ws_layers,ws_count,max(measured))
            baseline=stack(base_endpoint[1],scales).astype(np.float32)
            columns=[]
            elig=[]
            for action in actions:
                row=v19._action_row(values["directions"],int(action["direction_index"]),
                                    float(action["base_alpha"])*float(design["epsilon_relative_to_frozen_action_base_alpha"]),1)
                responses=[]
                statuses=[]
                for sign in (1,-1):
                    edited=v19.apply(state,sign,row,rec,att,"native_fp32_add_bf16_writeback")
                    read=_readback(state,edited,{key:value*sign for key,value in row.items()},rec,att)
                    statuses.append(v20_response._action_ok(read,op["action_reliability"]))
                    _,endpoint=v19._trajectory(bundle,dense,edited,None,[token],1,clean["prompt_length"],
                                               jids,lids,ws_layers,ws_count,max(measured))
                    responses.append(stack(endpoint[1],scales).astype(np.float32)-baseline)
                h=float(design["epsilon_relative_to_frozen_action_base_alpha"])
                columns.append((responses[0]+responses[1])/(h*h))
                elig.append(all(statuses))
            Q.append(np.stack(columns))
            reliable.append(elig)
        temporary=target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,Q=np.stack(Q).astype(np.float32),
                            reliable=np.asarray(reliable,dtype=bool),
                            operator_state_ids=np.asarray([f"{base_id}::P0",f"{base_id}::{item['q_name']}"]),
                            action_direction_indices=np.asarray([x["direction_index"] for x in actions],dtype=np.int32),
                            base_trial_id=np.asarray(base_id),family=np.asarray(item["family"]))
        os.replace(temporary,target)
        completed+=1
        print(f"V21 Z5 measure {role} {number}/{len(items)}",flush=True)
    return {"role":role,"completed":completed,"requested":len(items)}


def _load_panel(root:Path,role:str,items:list[dict],action_offset:int,action_count:int):
    z4,z5,eligible,ids=[],[],[],[]
    hashes={}
    for item in items:
        base_id=item["base_trial_id"]
        second=_path(role,base_id)
        first=_matrix_path(role,base_id)
        if not second.exists() or not first.exists():
            raise RuntimeError(f"V21 Z5 missing half-step or exact-JVP matrix for {base_id}")
        hashes[base_id]={"half_second_difference_sha256":sha256_file(second),"JVP_sha256":sha256_file(first)}
        with np.load(second,allow_pickle=False) as source:
            q=source["Q"]
            reliable=source["reliable"]
        with np.load(first,allow_pickle=False) as source:
            d=np.stack((source["P0_JVP"][:,action_offset:action_offset+action_count].T,
                        source["Pq_JVP"][:,action_offset:action_offset+action_count].T))
        if q.shape!=d.shape:
            raise RuntimeError("V21 Z5 JVP/Q target shape mismatch")
        z4.extend(d)
        z5.extend(q)
        eligible.extend(reliable)
        ids.extend([f"{base_id}::P0",f"{base_id}::{item['q_name']}"])
    return np.asarray(z4),np.asarray(z5),np.asarray(eligible),ids,hashes


def evaluate(root:Path)->dict:
    design=verify_stage(root,"second_order_design")
    roles=verify_stage(root,"roles")
    dtrain,qtrain,oktrain,train_ids,train_hash=_load_panel(root,"development",roles["jvp_development"],0,12)
    dval,qval,okval,val_ids,val_hash=_load_panel(root,"validation",roles["jvp_validation"],12,6)
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train=v20._load(root,"operator_train",op)
    val=v20._load(root,"operator_validation",op)
    train_lookup={str(x):i for i,x in enumerate(train["ids"])}
    val_lookup={str(x):i for i,x in enumerate(val["ids"])}
    itrain=np.asarray([train_lookup[x] for x in train_ids])
    ival=np.asarray([val_lookup[x] for x in val_ids])
    ytrain=train["Y"][("train",1)][itrain]
    yval=val["Y"][("validation",1)][ival]
    ynegative=val["Y"][("validation",-1)][ival]
    family=val["family"][ival]
    bases=val["base_ids"][ival]
    ridge=1e-4
    features=np.stack((dtrain,qtrain),axis=-1).reshape(-1,2)
    targets=ytrain.reshape(-1)
    coefficients=np.linalg.solve(features.T@features+ridge*np.eye(2),features.T@targets)
    direct={}
    for name,alpha,beta in (("JVP_only",1.0,0.0),("direct_Taylor",1.0,0.5),
                            ("train_scalar_pair",float(coefficients[0]),float(coefficients[1]))):
        positive=alpha*dval+beta*qval
        negative=-alpha*dval+beta*qval
        direct[name]={"JVP_coefficient":alpha,"finite_second_difference_coefficient":beta,
                      "positive_unseen_direction":fact._metric(yval,positive,family,bases),
                      "negative_unseen_direction":fact._metric(ynegative,negative,family,bases),
                      "reliable_half_step_fraction":float(okval.mean())}
    state=fact._load_state()
    state_train_lookup={str(x):i for i,x in enumerate(state["train_ids"])}
    state_val_lookup={str(x):i for i,x in enumerate(state["validation_ids"])}
    raw_train=state["S2_train"][[state_train_lookup[x] for x in train_ids]]
    raw_val=state["S2_validation"][[state_val_lookup[x] for x in val_ids]]
    mean=raw_train.mean(axis=0)
    scale=max(float(np.mean(np.sum((raw_train-mean)**2,axis=1)))**0.5,1e-8)
    s_train,s_val=(raw_train-mean)/scale,(raw_val-mean)/scale
    ztrain=np.concatenate((dtrain,qtrain),axis=-1)
    zval=np.concatenate((dval,qval),axis=-1)
    norm=max(float(np.median(np.linalg.norm(ztrain.reshape(-1,576),axis=1))),1e-8)
    ztrain,zval=ztrain/norm,zval/norm
    learned=[]
    for name in ("G4_joint_MLP","G5_rank8_hypernetwork"):
        path=neural.WEIGHTS/f"S2_Z5_{name}.pt"
        model=realized._train(s_train,ztrain,ytrain,name,design,path)
        positive=realized._predict(model,s_val,zval)
        negative_z=np.concatenate((-zval[:,:,:288],zval[:,:,288:]),axis=-1)
        negative=realized._predict(model,s_val,negative_z)
        learned.append({"S":"S2","Z":"Z5_finite_second_order_oracle","model":name,
                        "parameter_count":sum(p.numel() for p in model.parameters()),
                        "weights_sha256":sha256_file(path),"weights_scratch_path":str(path),
                        "positive_unseen_direction":fact._metric(yval,positive,family,bases),
                        "negative_unseen_direction":fact._metric(ynegative,negative,family,bases)})
        print(f"V21 Z5 {name} positive_unseen={learned[-1]['positive_unseen_direction']['stack_relative_l2']:.5f}",flush=True)
    result={"design_digest":design["freeze_digest"],"development_operator_states":len(train_ids),
            "validation_operator_states":len(val_ids),"half_step_reliable_train_fraction":float(oktrain.mean()),
            "half_step_reliable_validation_fraction":float(okval.mean()),
            "source_matrix_hashes":{"development":train_hash,"validation":val_hash},
            "direct_oracle":direct,"S2_state_conditioned_models":learned,
            "finite_second_difference_not_exact_Hessian":True,
            "same_action_half_step_oracle_not_deployable":True,
            "final_six_action_responses_opened":False}
    target=root/SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,result)
    return {name:row["positive_unseen_direction"]["stack_relative_l2"] for name,row in direct.items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","measure","evaluate"))
    parser.add_argument("--role",choices=("development","validation"))
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.stage=="measure" and not args.role:
        parser.error("--role required")
    result={"prepare":lambda:prepare(Path.cwd()),
            "measure":lambda:measure(Path.cwd(),args.role,args.limit),
            "evaluate":lambda:evaluate(Path.cwd())}[args.stage]()
    print(json.dumps(result,indent=2))
