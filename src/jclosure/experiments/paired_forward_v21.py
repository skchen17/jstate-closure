"""Append-only full paired-panel rerun using exact forward-mode JVP on all 75 bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import differential_oracle_v21 as oracle
from jclosure.experiments import geometry_gate_amend_v21 as gate
from jclosure.experiments import geometry_analysis_v21 as geometry
from jclosure.experiments import paired_geometry_v21 as paired
from jclosure.experiments import second_order_v21 as second
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import write_json_atomic

SOURCE="src/jclosure/experiments/paired_forward_v21.py"
FAST_RAW=bank.SCRATCH/"paired_geometry_forward_ad"
COMPARISON=bank.OUT/"forward_ad_comparison_v21.json"
BENCHMARK=[
    {"direction_index":4,"backward_seconds":1.2485178131610155,"forward_seconds":0.2459486648440361,"relative_l2_difference":0.009731720201671124,"cosine":0.9999530911445618},
    {"direction_index":9,"backward_seconds":1.0287090856581926,"forward_seconds":0.14581991359591484,"relative_l2_difference":0.007447290234267712,"cosine":0.9999738335609436},
    {"direction_index":19,"backward_seconds":1.0256303437054157,"forward_seconds":0.14621706306934357,"relative_l2_difference":0.0067903767339885235,"cosine":0.9999774694442749},
    {"direction_index":1,"backward_seconds":1.040654230862856,"forward_seconds":0.14810736104846,"relative_l2_difference":0.006244627293199301,"cosine":0.9999811053276062},
    {"direction_index":6,"backward_seconds":1.0309810154139996,"forward_seconds":0.14918864332139492,"relative_l2_difference":0.008300036191940308,"cosine":0.999966561794281},
]


def prepare(root:Path)->dict:
    verify_stage(root,"paired_geometry_design")
    verify_stage(root,"paired_geometry_unreliable_probe_amendment_1")
    verify_stage(root,"paired_geometry_family_gate_amendment_2")
    if max(x["relative_l2_difference"] for x in BENCHMARK)>=0.02 or min(x["cosine"] for x in BENCHMARK)<0.9999:
        raise RuntimeError("V21 exact forward-AD numeric agreement below preregistered engineering tolerance")
    detail={"benchmarked_same_P0_state_probe_count":len(BENCHMARK),"rows":BENCHMARK,
            "old_autograd_functional_JVP_completed_development_bases":17,
            "old_matrices_retained_in_separate_scratch_directory":True,
            "final_panel_method":"torch.func.jvp_exact_forward_AD_on_all_50_development_plus_25_validation_bases",
            "old_and_new_matrices_never_mixed":True,
            "final_six_action_responses_opened":False}
    target=root/COMPARISON
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,detail)
    return stage_freeze(root,"paired_forward_ad_amendment_3",
                        [SOURCE,"scripts/benchmark_v21_forward_jvp.py",str(COMPARISON),
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_unreliable_probe_amendment_1.freeze.json"],
                        {"reason":"Exact forward-mode JVP is 5–7× faster on five frozen train probes and agrees with the original exact reverse-over-reverse JVP within 1% relative L2, cosine above 0.99995; rerun all 75 bases with one consistent method in a new append-only scratch directory.",
                         "numerical_comparison_max_relative_l2":max(x["relative_l2_difference"] for x in BENCHMARK),
                         "numerical_comparison_min_cosine":min(x["cosine"] for x in BENCHMARK),
                         "method":"torch.func.jvp_exact_forward_AD",
                         "old_completed_matrices_retained":17,
                         "old_new_mixing_prohibited":True,
                         "final_six_action_responses_opened":False})


def _fast_path(role:str,base_id:str)->Path:
    return FAST_RAW/role/f"paired_{base_id}.npz"


def _fast_jvp(bundle,dense,state,prompt_length,token,jids,lids,ws_layers,ws_count,main,row,rec,att,scales):
    device=next(bundle.hf_model.parameters()).device
    def target(epsilon):
        return paired._torch_stack(paired._target(bundle,dense,state,prompt_length,token,jids,lids,
                                                  ws_layers,ws_count,main,row,rec,att,epsilon),scales)
    epsilon=torch.zeros((),device=device,dtype=torch.float32)
    _,derivative=torch.func.jvp(target,(epsilon,),(torch.ones_like(epsilon),))
    return derivative.detach().cpu().numpy().astype(np.float32)


def _with_paired_patch(root:Path,role:str|None,limit:int|None,summary:bool):
    amend=verify_stage(root,"paired_forward_ad_amendment_3")
    unreliable=verify_stage(root,"paired_geometry_unreliable_probe_amendment_1")
    old_path,old_jvp,old_verify=paired._matrix_path,paired._one_jvp,paired.verify_stage
    def with_fallback(path:Path,stage:str):
        result=old_verify(path,stage)
        if stage=="paired_geometry_design":
            result=dict(result)
            result["probe_alpha_by_direction"]=dict(result["probe_alpha_by_direction"])
            result["probe_alpha_by_direction"][str(unreliable["unreliable_direction_index"])]=unreliable["fallback_alpha"]
        return result
    paired._matrix_path=_fast_path
    paired._one_jvp=_fast_jvp
    paired.verify_stage=with_fallback
    try:
        if summary:
            result=paired.summarize(root)
            target=root/paired.SUMMARY
            value=json.loads(target.read_text())
            value.update({"forward_ad_amendment_digest":amend["freeze_digest"],
                          "exact_autograd_method":"torch.func.jvp_forward_AD",
                          "old_reverse_over_reverse_matrices_excluded":True})
            write_json_atomic(target,value)
            return value
        return paired.run(root,role,limit)
    finally:
        paired._matrix_path,paired._one_jvp,paired.verify_stage=old_path,old_jvp,old_verify


def analyze(root:Path):
    verify_stage(root,"paired_forward_ad_amendment_3")
    old=geometry._matrix_path
    geometry._matrix_path=_fast_path
    try:
        result=gate.run(root)
        target=root/geometry.SUMMARY
        value=json.loads(target.read_text())
        value["exact_autograd_method"]="torch.func.jvp_forward_AD"
        value["old_reverse_over_reverse_matrices_excluded"]=True
        write_json_atomic(target,value)
        return result
    finally:
        geometry._matrix_path=old


def evaluate_oracle(root:Path):
    amend=verify_stage(root,"paired_forward_ad_amendment_3")
    old=oracle._matrix_path
    oracle._matrix_path=_fast_path
    try:
        result=oracle.run(root)
        target=root/oracle.SUMMARY
        value=json.loads(target.read_text())
        value["forward_ad_amendment_digest"]=amend["freeze_digest"]
        write_json_atomic(target,value)
        return result
    finally:
        oracle._matrix_path=old


def evaluate_second(root:Path):
    amend=verify_stage(root,"paired_forward_ad_amendment_3")
    old=second._matrix_path
    second._matrix_path=_fast_path
    try:
        result=second.evaluate(root)
        target=root/second.SUMMARY
        value=json.loads(target.read_text())
        value["forward_ad_amendment_digest"]=amend["freeze_digest"]
        write_json_atomic(target,value)
        return result
    finally:
        second._matrix_path=old


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run","summarize","analyze","oracle","second"))
    parser.add_argument("--role",choices=("development","validation"))
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.stage=="run" and not args.role:
        parser.error("--role required")
    root=Path.cwd()
    result={"prepare":lambda:prepare(root),
            "run":lambda:_with_paired_patch(root,args.role,args.limit,False),
            "summarize":lambda:_with_paired_patch(root,None,None,True),
            "analyze":lambda:analyze(root),
            "oracle":lambda:evaluate_oracle(root),
            "second":lambda:evaluate_second(root)}[args.stage]()
    print(json.dumps(result,indent=2))
