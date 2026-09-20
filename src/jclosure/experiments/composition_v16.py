"""V16 static BF16 and teacher-forced dynamic action-order diagnostics."""

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
from jclosure.experiments.numerics_v14 import _cache_difference, _evaluate
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.protocol_v16 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT=Path("results/v16/processed")
SOURCE="src/jclosure/experiments/composition_v16.py"
TARGETS=("j","logits","semantic_continuous","workspace")


def _first_per_family(split:dict[str,Any],role:str)->list[dict[str,str]]:
    seen=set();out=[]
    for item in split[role]:
        if item["family"] not in seen:
            out.append(item);seen.add(item["family"])
    return out


def _norms(left:dict[str,np.ndarray],right:dict[str,np.ndarray])->dict[str,float]:
    return {f"{name}_l2":float(np.linalg.norm(left[name]-right[name])) for name in TARGETS}


def run(root:Path)->dict[str,Any]:
    split=_verify_splits(root);verify(root)
    stage_path=root/"artifacts/nonlinear_finite_causal_action_v16_composition.freeze.json"
    if not stage_path.exists():
        stage=stage_freeze(root,"composition",[SOURCE,"artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json"],
                           {"role":"diagnostic_train_and_validation","state_selection":"first frozen state per family in each split",
                            "action_selection":"first two reliable positive single coordinates by frozen index",
                            "dynamic_second_action":"same frozen raw direction reused at next teacher-forced step; transport not estimated",
                            "noncommutative_threshold":"static UV-vs-VU J order norm >1e-6 and relative to larger first-order J effect >0.01",
                            "not_a_lie_bracket":True})
    else:
        stage=json.loads(stage_path.read_text())
        if sha256_file(root/SOURCE)!=stage["input_hashes"][SOURCE]:raise RuntimeError("composition source changed")
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
    records=[]
    for role in ("train","validation"):
        for item in _first_per_family(split,role):
            base_id=item["base_trial_id"]
            bank=pd.read_parquet(root/"results/v16/raw"/f"bank_{role}_{base_id}.parquet")
            singles=bank[(bank.design=="single")&(bank.sign==1)&(bank.reliability_status=="RELIABLE")].sort_values("coordinate_index")
            if len(singles)<2:
                records.append({"base_trial_id":base_id,"family":item["family"],"role":role,"status":"FEWER_THAN_TWO_RELIABLE_ACTIONS"})
                continue
            selected=singles.iloc[:2]
            action_rows=[]
            for _,record in selected.iterrows():
                index=indices[int(record.coordinate_index)]
                amplitude=float(record.calibration_alpha)
                action_rows.append({name:values["directions"][name][index].float()*amplitude for name in ("recurrent","conv","kv")})
            u,v=action_rows
            task=values["tasks"][str(metadata[base_id]["prompt_id"])]
            clean=_prefill(bundle,task.prompt,measured_layers=measured,dense_map=dense_map,
                           intervention_layer=int(v8["intervention"]["layer"]),candidate=None)
            cache=clean["cache"]
            tokens=_teacher_tokens(bundle,clean,count=2,measured_layers=measured,dense_map=dense_map)
            kwargs=dict(bundle=bundle,dense_map=dense_map,prompt_length=int(clean["prompt_length"]),
                        selected_j=np.asarray(split["selected_j"],dtype=int),
                        selected_logits=np.asarray(split["selected_logits"],dtype=int),
                        workspace_layers=[int(x) for x in jvp["workspace_layers"]],
                        workspace_count=int(jvp["selected_workspace_count"]),main_layer=max(measured))
            base0=_evaluate(cache=cache,token=tokens[0],**kwargs)
            cache_u=apply(cache,1.0,u,rec,att,"native_fp32_add_bf16_writeback")
            cache_v=apply(cache,1.0,v,rec,att,"native_fp32_add_bf16_writeback")
            resp_u=_evaluate(cache=cache_u,token=tokens[0],**kwargs)
            resp_v=_evaluate(cache=cache_v,token=tokens[0],**kwargs)
            uv=apply(cache_u,1.0,v,rec,att,"native_fp32_add_bf16_writeback")
            vu=apply(cache_v,1.0,u,rec,att,"native_fp32_add_bf16_writeback")
            resp_uv=_evaluate(cache=uv,token=tokens[0],**kwargs)
            resp_vu=_evaluate(cache=vu,token=tokens[0],**kwargs)
            static_state_l2,static_reference=_cache_difference(uv,vu)
            loop=apply(uv,-1.0,u,rec,att,"native_fp32_add_bf16_writeback")
            loop=apply(loop,-1.0,v,rec,att,"native_fp32_add_bf16_writeback")
            resp_loop=_evaluate(cache=loop,token=tokens[0],**kwargs)
            loop_state_l2,_=_cache_difference(cache,loop)
            dynamic_uv=geometry._advance_cache(bundle,cache_u,tokens[0],int(clean["prompt_length"]))
            dynamic_uv=apply(dynamic_uv,1.0,v,rec,att,"native_fp32_add_bf16_writeback")
            dynamic_vu=geometry._advance_cache(bundle,cache_v,tokens[0],int(clean["prompt_length"]))
            dynamic_vu=apply(dynamic_vu,1.0,u,rec,att,"native_fp32_add_bf16_writeback")
            kwargs_next=dict(kwargs);kwargs_next["prompt_length"]+=1
            resp_dynamic_uv=_evaluate(cache=dynamic_uv,token=tokens[1],**kwargs_next)
            resp_dynamic_vu=_evaluate(cache=dynamic_vu,token=tokens[1],**kwargs_next)
            j_first=max(float(np.linalg.norm(resp_u["j"]-base0["j"])),float(np.linalg.norm(resp_v["j"]-base0["j"])))
            j_loop=float(np.linalg.norm(resp_loop["j"]-base0["j"]))
            static_j_order=float(np.linalg.norm(resp_uv["j"]-resp_vu["j"]))
            records.append({"base_trial_id":base_id,"family":item["family"],"role":role,"status":"MEASURED",
                            "u_coordinate_index":int(selected.iloc[0].coordinate_index),
                            "v_coordinate_index":int(selected.iloc[1].coordinate_index),
                            "u_alpha":float(selected.iloc[0].calibration_alpha),"v_alpha":float(selected.iloc[1].calibration_alpha),
                            "static_state_order_l2":static_state_l2,"static_state_reference_norm":static_reference,
                            "loop_state_residual_l2":loop_state_l2,
                            "loop_j_residual_over_first_effect":j_loop/max(j_first,1e-20),
                            "noncommutative_order_threshold_pass":bool(static_j_order>1e-6 and static_j_order/max(j_first,1e-20)>0.01),
                            **{f"static_order_{name}":value for name,value in _norms(resp_uv,resp_vu).items()},
                            **{f"loop_{name}":value for name,value in _norms(resp_loop,base0).items()},
                            **{f"dynamic_order_{name}":value for name,value in _norms(resp_dynamic_uv,resp_dynamic_vu).items()},
                            "freeze_digest":stage["freeze_digest"]})
            print(f"composition {role} {base_id}",flush=True)
    frame=pd.DataFrame(records);OUT.mkdir(parents=True,exist_ok=True)
    path=root/OUT/"action_composition_v16.parquet";frame.to_parquet(path,index=False,compression="zstd")
    measured=frame[frame.status=="MEASURED"]
    summary={"freeze_digest":stage["freeze_digest"],"records":str(path),"records_sha256":sha256_file(path),
             "measured_states":len(measured),"static_order_j_l2_median":float(measured.static_order_j_l2.median()) if len(measured) else None,
             "dynamic_order_j_l2_median":float(measured.dynamic_order_j_l2.median()) if len(measured) else None,
             "loop_j_relative_median":float(measured.loop_j_residual_over_first_effect.median()) if len(measured) else None,
             "noncommutative_order_threshold_pass_count":int(measured.noncommutative_order_threshold_pass.sum()) if len(measured) else 0,
             "dynamic_transport_limitation":"second direction reused unchanged after teacher-forced step; no tangent transport inferred"}
    write_json_atomic(root/OUT/"action_composition_v16.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("run",),required=True);parser.parse_args()
    print(json.dumps(run(Path.cwd()),sort_keys=True))
