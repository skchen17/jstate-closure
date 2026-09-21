"""Frozen heldout amplitude/pair/dense finite-response tests for V21 best practical decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import factorial_v21 as fact
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import operator_model_v20 as v20_model
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.operator_v15 import stack
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="src/jclosure/experiments/composition_v21.py"
RAW=bank.SCRATCH/"composition_response_v21.parquet"
SUMMARY=bank.OUT/"scale_pair_dense_v21.json"


def _combos() -> list[dict]:
    rows=[]
    for index in (0,1,2,8):
        rows.append({"name":f"seen_scale_{index}","category":"unseen_amplitude_seen_direction",
                     "terms":[[index,0.5]]})
    for index in (12,13,14,15,16,17):
        rows.append({"name":f"unseen_scale_{index}","category":"unseen_amplitude_unseen_direction",
                     "terms":[[index,0.5]]})
    for a,b in ((12,13),(14,15),(16,17)):
        rows.append({"name":f"unseen_pair_{a}_{b}","category":"unseen_pair",
                     "terms":[[a,0.5],[b,0.5]]})
    rows.append({"name":"unseen_dense_6","category":"unseen_dense_combination",
                 "terms":[[i,0.25] for i in (12,13,14,15,16,17)]})
    return rows


def prepare(root:Path)->dict:
    roles=verify_stage(root,"roles")
    verify_stage(root,"factorial_design")
    result=json.loads((root/"results/v21/processed/bottleneck_factorial_v21.json").read_text())
    selected=[x for x in result["results"] if x["S"]=="S2" and x["Z"]=="Z1" and x["model"]=="G2_quadratic_action"]
    if len(selected)!=1:
        raise RuntimeError("V21 practical composition candidate missing")
    return stage_freeze(root,"composition_design",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_factorial_design.freeze.json",
                         "results/v21/processed/bottleneck_factorial_v21.json"],
                        {"selected_candidate":"S2×Z1×G2_quadratic_action",
                         "selected_ridge":selected[0]["selected_ridge"],
                         "selection_basis":"representative_near_best_full_requested_Z1_candidate_with_exact_linear_combination_rule; corrected_Z2_G2_development_L2_is_slightly_lower; no_new_composition_response_seen",
                         "validation_base_ids":[x["base_trial_id"] for x in roles["jvp_validation"]],
                         "paired_states_per_base":["P0","one_frozen_V20_Pq"],
                         "combinations":_combos(),
                         "calibrated_action_alpha_source":"frozen_V20_train_and_validation_base_alpha",
                         "action_reliability_gate":"V20_operator_design_action_reliability",
                         "response_target":"V20_h1_normalized_stack288_with_frozen_teacher",
                         "final_six_action_responses_opened":False})


def _action(values,actions,terms):
    result=None
    for index,coeff in terms:
        action=actions[int(index)]
        row=v19._action_row(values,int(action["direction_index"]),float(action["base_alpha"])*float(coeff),1)
        result=row if result is None else {name:result[name]+row[name] for name in row}
    return result


def measure(root:Path)->dict:
    design=verify_stage(root,"composition_design")
    if RAW.exists():
        raise RuntimeError("V21 composition response bank exists; append-only policy")
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    split=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    actions=op["shared_measured_actions"]
    prompts=v20_bank.prompt_index(root)
    teachers=v20_response._teacher_map(root)
    bundle,dense,_,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
    jids=np.asarray(split["selected_j"],dtype=int)
    lids=np.asarray(split["selected_logits"],dtype=int)
    scales={key:float(value) for key,value in split["target_scales"].items()}
    q_lookup={x["name"]:x for x in op["q"]}
    roles=verify_stage(root,"roles")
    by_id={x["base_trial_id"]:x for x in roles["jvp_validation"]}
    rows=[]
    for number,base_id in enumerate(design["validation_base_ids"],1):
        item=by_id[base_id]
        clean=v19._prefill_history(bundle,str(prompts[base_id]["prompt"]),measured,dense,state_layer)
        token=int(teachers[base_id][0])
        p0=clean["cache"]
        q=q_lookup[item["q_name"]]
        qrow=v19._q_row(values["directions"],q,float(q["alpha"]))
        pq=v19.apply(p0,1.0,qrow,rec,att,"native_fp32_add_bf16_writeback")
        qread=_readback(p0,pq,qrow,rec,att)
        qok=v19._reliable(qread,op["V19_q_reliability"])
        states=[("P0",p0,True),(item["q_name"],pq,bool(qok))]
        baseline={}
        for name,state,_ in states:
            _,endpoint=v19._trajectory(bundle,dense,state,None,[token],1,clean["prompt_length"],jids,lids,ws_layers,ws_count,max(measured))
            baseline[name]=stack(endpoint[1],scales).astype(np.float32)
        for combo in design["combinations"]:
            actionrow=_action(values["directions"],actions,combo["terms"])
            edited0=v19.apply(p0,1.0,actionrow,rec,att,"native_fp32_add_bf16_writeback")
            editedq=v19.apply(pq,1.0,actionrow,rec,att,"native_fp32_add_bf16_writeback")
            match=v19._matching(p0,edited0,pq,editedq,actionrow,rec,att)
            matchok=v20_response._matching_ok(match,op["V19_action_matching"])
            for name,state,qvalid,edited in (("P0",p0,True,edited0),(item["q_name"],pq,bool(qok),editedq)):
                read=_readback(state,edited,actionrow,rec,att)
                actionok=v20_response._action_ok(read,op["action_reliability"])
                _,endpoint=v19._trajectory(bundle,dense,edited,None,[token],1,clean["prompt_length"],jids,lids,ws_layers,ws_count,max(measured))
                response=stack(endpoint[1],scales).astype(np.float32)-baseline[name]
                rows.append({"base_trial_id":base_id,"family":item["family"],"q_name":name,
                             "operator_state_id":f"{base_id}::{name}","combination":combo["name"],
                             "category":combo["category"],"terms_json":json.dumps(combo["terms"]),
                             "q_reliable":qvalid,"action_reliable":bool(actionok),
                             "matched_vs_P0":bool(matchok),
                             "requested_action_norm":read["requested_state_norm"],
                             "realized_action_norm":read["realized_state_norm"],
                             "realized_action_cosine":read["realized_state_cosine"],
                             "realized_action_gain":read["realized_state_gain"],
                             "response_stack":response.tolist()})
        print(f"V21 composition measure {number}/{len(design['validation_base_ids'])}",flush=True)
    RAW.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(RAW,index=False,compression="zstd")
    return {"rows":len(rows),"raw_response_sha256":sha256_file(RAW),
            "reliable_and_matched_fraction":float(np.mean([x["q_reliable"] and x["action_reliable"] and x["matched_vs_P0"] for x in rows]))}


def evaluate(root:Path)->dict:
    design=verify_stage(root,"composition_design")
    if not RAW.exists():
        raise RuntimeError("V21 composition response bank missing")
    frame=pd.read_parquet(RAW)
    roles=verify_stage(root,"roles")
    state=fact._load_state()
    train_ids=state["train_ids"].astype(str)
    validation_ids=state["validation_ids"].astype(str)
    op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    train=v20_model._load(root,"operator_train",op)
    descriptor=json.loads((root/"results/v21/processed/action_representation_v21.json").read_text())
    gram=fact._action_gram(descriptor,"Z1")
    combos=design["combinations"]
    cross=np.stack([sum(float(coeff)*gram[:12,int(index)] for index,coeff in combo["terms"]) for combo in combos],axis=1)
    ka=(1+gram[:12,:12])**2
    qan=(1+cross)**2
    desired=[]
    families=[]
    base_ids=[]
    for base_id in design["validation_base_ids"]:
        role=next(x for x in roles["jvp_validation"] if x["base_trial_id"]==base_id)
        for qname in ("P0",role["q_name"]):
            desired.append(f"{base_id}::{qname}")
            families.append(role["family"])
            base_ids.append(base_id)
    lookup={value:i for i,value in enumerate(validation_ids)}
    xv=state["S2_validation"][[lookup[x] for x in desired]]
    ks,kn=fact._state_kernel(state["S2_train"],xv)
    spectrum=fact._spectrum(ks,ka)
    prediction=fact._fit_predict(ks,kn,ka,qan,train["Y"][("train",1)],float(design["selected_ridge"]),
                                 "G2_quadratic_action",spectrum)
    indexed=frame.set_index(["operator_state_id","combination"])
    truth=np.stack([[np.asarray(indexed.loc[(state_id,combo["name"])].response_stack,dtype=np.float32)
                     for combo in combos] for state_id in desired])
    reliability=np.asarray([[bool(indexed.loc[(state_id,combo["name"])].q_reliable
                                  and indexed.loc[(state_id,combo["name"])].action_reliable
                                  and indexed.loc[(state_id,combo["name"])].matched_vs_P0)
                             for combo in combos] for state_id in desired])
    metrics={}
    for category in sorted({x["category"] for x in combos}):
        idx=[i for i,x in enumerate(combos) if x["category"]==category]
        primary=fact._metric(truth[:,idx,:],prediction[:,idx,:],np.asarray(families),np.asarray(base_ids))
        qualified=reliability[:,idx].all(axis=1)
        eligible=fact._metric(truth[qualified][:,idx,:],prediction[qualified][:,idx,:],
                              np.asarray(families)[qualified],np.asarray(base_ids)[qualified]) if qualified.sum()>=10 else None
        metrics[category]={"all_rows":primary,"reliability_qualified_states":int(qualified.sum()),
                           "reliable_rows":eligible}
    result={"design_digest":design["freeze_digest"],"response_bank_sha256":sha256_file(RAW),
            "validation_base_states":len(design["validation_base_ids"]),"operator_states":len(desired),
            "metrics":metrics,"final_six_action_responses_opened":False}
    target=root/SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,result)
    return {key:{"all_L2":value["all_rows"]["stack_relative_l2"],
                 "qualified_states":value["reliability_qualified_states"]}
            for key,value in metrics.items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","measure","evaluate"))
    args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"measure":measure,"evaluate":evaluate}[args.stage](Path.cwd()),indent=2))
