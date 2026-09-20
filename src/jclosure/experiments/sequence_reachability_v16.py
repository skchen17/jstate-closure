"""Restricted, exhaustive 1/2/4-step teacher-forced V16 reachable sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.actuation_v15 import apply, load_context
from jclosure.experiments.action_bank_v16 import _verify_splits
from jclosure.experiments.numerics_v14 import _evaluate
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v16/processed")
SOURCE="src/jclosure/experiments/sequence_reachability_v16.py"


def run(root:Path)->dict[str,Any]:
    split=_verify_splits(root);verify(root)
    stage_path=root/"artifacts/nonlinear_finite_causal_action_v16_sequence_reachability.freeze.json"
    if not stage_path.exists():
        stage=stage_freeze(root,"sequence_reachability",[SOURCE,"artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json"],
                           {"role":"validation_diagnostic","states":"first frozen validation state per family",
                            "actions":"first two reliable positive single coordinates and their reliable negatives",
                            "depths":[1,2,4],"selection":"exhaustive over four primitives at each active step",
                            "common_target":"frozen V13 h4 teacher J on V16 selected J coordinates",
                            "continuation":"same teacher tokens, untransported raw action directions",
                            "teacher_raw_state_usage":"labels for post-hoc nearest search only; no response model/controller input"})
    else:
        stage=json.loads(stage_path.read_text())
        if sha256_file(root/SOURCE)!=stage["input_hashes"][SOURCE]:raise RuntimeError("sequence source changed")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle,dense_map,v13,metadata,values=load_context(root)
    v8=v13["persistent_state_v8"]
    measured=[int(x) for x in v8["intervention"]["measured_layers"]]
    rec=[int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att=[int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp=v13["causal_geometry_v13"]["jvp"]
    indices=[int(x) for x in split["direction_indices"]]
    data=geometry._load_features(root)
    lookup={str(x):i for i,x in enumerate(data["base_trial_id"].astype(str))}
    selected_j=np.asarray(split["selected_j"],dtype=int)
    states=[];seen=set()
    for item in split["validation"]:
        if item["family"] not in seen:
            states.append(item);seen.add(item["family"])
    rows=[]
    for item in states:
        base_id=item["base_trial_id"]
        bank=pd.read_parquet(root/"results/v16/raw"/f"bank_validation_{base_id}.parquet")
        singles=bank[(bank.design=="single")&(bank.reliability_status=="RELIABLE")]
        eligible=[]
        for index,group in singles.groupby("coordinate_index"):
            if set(group.sign.astype(int))=={-1,1}:
                eligible.append((int(index),float(group.calibration_alpha.iloc[0])))
        eligible.sort()
        if len(eligible)<2:
            rows.append({"base_trial_id":base_id,"family":item["family"],"status":"FEWER_THAN_TWO_SIGNED_RELIABLE_ACTIONS"})
            continue
        task=values["tasks"][str(metadata[base_id]["prompt_id"])]
        clean=_prefill(bundle,task.prompt,measured_layers=measured,dense_map=dense_map,
                       intervention_layer=int(v8["intervention"]["layer"]),candidate=None)
        cache=clean["cache"]
        tokens=_teacher_tokens(bundle,clean,count=4,measured_layers=measured,dense_map=dense_map)
        kwargs=dict(bundle=bundle,dense_map=dense_map,
                    selected_j=selected_j,selected_logits=np.asarray(split["selected_logits"],dtype=int),
                    workspace_layers=[int(x) for x in jvp["workspace_layers"]],
                    workspace_count=int(jvp["selected_workspace_count"]),main_layer=max(measured))
        source=clone_hybrid_cache(cache)
        for step in range(3):
            source=geometry._advance_cache(bundle,source,tokens[step],int(clean["prompt_length"])+step)
        baseline=_evaluate(cache=source,token=tokens[3],prompt_length=int(clean["prompt_length"])+3,**kwargs)
        target_index=lookup[base_id]
        teacher=(data["endpoint__perturbed_j_h4"][target_index,selected_j].astype(np.float64)
                 -data["endpoint__clean_j_h4"][target_index,selected_j].astype(np.float64))
        teacher_norm=float(np.linalg.norm(teacher))
        primitives=[]
        for coordinate,alpha in eligible[:2]:
            index=indices[coordinate]
            positive={name:values["directions"][name][index].float()*alpha for name in ("recurrent","conv","kv")}
            primitives.extend([(f"+{coordinate}",positive),(f"-{coordinate}",{name:-value for name,value in positive.items()})])
        for depth in (1,2,4):
            best=(float("inf"),None,None)
            evaluated=0
            def visit(state:Any,step:int,path:tuple[str,...])->None:
                nonlocal best,evaluated
                options=primitives if step<depth else [("0",None)]
                for label,action in options:
                    edited=apply(state,1.0,action,rec,att,"native_fp32_add_bf16_writeback") if action is not None else state
                    next_path=path+(label,) if action is not None else path
                    if step==3:
                        output=_evaluate(cache=edited,token=tokens[step],prompt_length=int(clean["prompt_length"])+step,**kwargs)
                        delta=output["j"]-baseline["j"]
                        distance=float(np.linalg.norm(delta-teacher)/max(teacher_norm,1e-20))
                        evaluated+=1
                        if distance<best[0]:
                            best=(distance,next_path,float(np.linalg.norm(delta)))
                    else:
                        advanced=geometry._advance_cache(bundle,edited,tokens[step],int(clean["prompt_length"])+step)
                        visit(advanced,step+1,next_path)
            visit(cache,0,())
            rows.append({"base_trial_id":base_id,"family":item["family"],"status":"MEASURED",
                         "depth":depth,"candidate_count":evaluated,"teacher_norm":teacher_norm,
                         "nearest_relative_residual":best[0],"nearest_sequence":list(best[1]) if best[1] else None,
                         "nearest_response_norm":best[2],"nearest_action_coordinate_norm":float(np.sqrt(depth)),
                         "freeze_digest":stage["freeze_digest"]})
            print(f"sequence {base_id} depth={depth} candidates={evaluated} residual={best[0]:.4f}",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);path=root/OUT/"sequence_reachability_v16.parquet"
    frame.to_parquet(path,index=False,compression="zstd")
    good=frame[frame.status=="MEASURED"]
    pooled=good.groupby("depth").agg(states=("base_trial_id","nunique"),candidate_count=("candidate_count","median"),nearest_relative_residual=("nearest_relative_residual","median")).reset_index()
    summary={"freeze_digest":stage["freeze_digest"],"records":str(path),"records_sha256":sha256_file(path),
             "states":int(good.base_trial_id.nunique()),"by_depth":pooled.to_dict("records"),
             "restricted_action_dictionary_size":4,"same_h4_teacher_target_for_all_depths":True,
             "untransported_direction_and_five_state_limit":True}
    write_json_atomic(root/OUT/"sequence_reachability_v16.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("run",),required=True);parser.parse_args()
    print(json.dumps(run(Path.cwd()),sort_keys=True))
