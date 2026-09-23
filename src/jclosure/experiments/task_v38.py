"""Gated Phase-B answer-token diagnostic; no semantic-fork claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import load, native_swap, step
from jclosure.protocol_v34 import verify as verify_v34
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/task_v38.py"
ROLES=("development","validation")
PRIMARY=("R000","R100","R010","R001","R110","R101","R011","R111")
EXTRA=("RECIPIENT","REC_ONLY","JOINT","DONOR","KV_ONLY")


def design(root:Path)->dict:
    verify_stage(root,"composition_development")
    high=verify_stage(root,"high_level_development")
    f=json.loads((root/OUT/"trajectory_analysis_F_development_v38.json").read_text())
    if not high["both_pass"] or not any(f["class_passes"].values()):
        raise RuntimeError("V38 Task Phase B not authorized")
    plan=json.loads((root/OUT/"execution_plan_v38.json").read_text())
    panel=json.loads((root/OUT/"panel_v38.json").read_text())
    specs=verify_v34(root)["config"]["models"]
    result={"task_ground_truth_source":"frozen_v38_generator_semantic_actions_first",
            "semantic_surface_type_S_T_forks_available":False,
            "strict_task_grounding_possible":False,
            "diagnostic_scope":"first answer token after one common newline probe; no task-changing fork",
            "roles":{},"models":{}}
    for role in ROLES:
        ids={sid for family_ids in plan["task_subsets"][role].values() for sid in family_ids}
        result["roles"][role]=[{"state_id":x["base_trial_id"],"family":x["family"],
                                "correct_first_action":str(x["semantic_actions"][0]),
                                "program_hash":x["program_hash"]}
                               for x in panel[role] if x["base_trial_id"] in ids]
        if len(result["roles"][role])!=len(ids):
            raise RuntimeError("V38 task subset mismatch")
    for key in ("Q","F"):
        tokenizer=AutoTokenizer.from_pretrained(specs[key]["local_path"],local_files_only=True,
                                                trust_remote_code=False)
        probe=tokenizer.encode("\n",add_special_tokens=False)
        if len(probe)!=1:
            raise RuntimeError(f"V38 newline probe not one token: {key}:{probe}")
        by_family={}
        for family in verify(root)["config"]["families"]:
            labels=sorted({str(x["semantic_actions"][0]) for role in ROLES
                           for x in panel[role] if x["family"]==family})
            mapping={}
            for label in labels:
                candidates=[tokenizer.encode(" "+label,add_special_tokens=False),
                            tokenizer.encode(label,add_special_tokens=False)]
                one=next((ids[0] for ids in candidates if len(ids)==1),None)
                if one is not None:
                    mapping[label]=int(one)
            if len(set(mapping.values()))!=len(mapping):
                raise RuntimeError(f"V38 task candidate token collision {key}:{family}")
            by_family[family]={"labels":mapping,"total_labels":len(labels)}
        result["models"][key]={"neutral_probe_token_id":int(probe[0]),
                                "candidate_labels_by_family":by_family,
                                "tokenizer_source":specs[key]["local_path"]}
    path=root/OUT/"task_design_v38.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"task_design",[SOURCE,str(path.relative_to(root)),
                      "artifacts/trajectory_composition_v38_design.freeze.json",
                      "artifacts/trajectory_composition_v38_composition_development.freeze.json"],
                      {"task_ground_truth_frozen":True,"Type_S_T_available":False,
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],"states_by_role":
            {role:len(result["roles"][role]) for role in ROLES},
            "single_token_candidate_coverage":
            {key:{family:len(obj["labels"]) for family,obj in result["models"][key]["candidate_labels_by_family"].items()}
             for key in ("Q","F")}}


@torch.no_grad()
def run(root:Path,key:str,role:str)->dict:
    if role not in ROLES:raise ValueError(role)
    verify_stage(root,"task_design")
    verify_stage(root,f"trajectory_analysis_{key}_{role}")
    task=json.loads((root/OUT/"task_design_v38.json").read_text())
    design=json.loads((root/OUT/f"design_{key}_v38.json").read_text())
    panel=json.loads((root/OUT/"panel_v38.json").read_text())
    prompts={x["base_trial_id"]:x["prompt"] for r in
             ("calibration","development","validation","independent_final") for x in panel[r]}
    items={x["base_trial_id"]:x for x in design[role]}
    groups=design["relative_depth_layers"]
    conditions=verify(root)["config"]["conditions"]
    model,tokenizer=load(root,key)
    rows=[]
    for n,selected in enumerate(task["roles"][role],1):
        sid=selected["state_id"]
        item=items[sid]
        family=item["family"]
        mapping=task["models"][key]["candidate_labels_by_family"][family]["labels"]
        correct=selected["correct_first_action"]
        if correct not in mapping or len(mapping)<2:
            rows.append({"state_id":sid,"model":key,"role":role,"family":family,
                         "status":"ONE_TOKEN_CANDIDATE_COVERAGE_FAIL"})
            continue
        caches=prepare(model,tokenizer,key,item,prompts[sid])
        branches={}
        for name in PRIMARY:
            chosen=conditions[name]
            if chosen:
                layers=[layer for group in chosen for layer in groups[group]]
                branches[name],_= _selective_rec(caches["conv"],caches["joint"],layers)
            else:branches[name]=caches["conv"]
        branches["RECIPIENT"]=caches["recipient"]
        branches["JOINT"]=caches["joint"]
        branches["DONOR"]=caches["donor"]
        branches["REC_ONLY"],_=native_swap(caches["recipient"],caches["donor"],["REC"],key)
        branches["KV_ONLY"],_=native_swap(caches["recipient"],caches["donor"],["KV"],key)
        token=int(task["models"][key]["neutral_probe_token_id"])
        scores={}
        for name,cache in branches.items():
            logits=step(model,cache,token,caches["length"]+1)["logits"]
            sampled={label:float(logits[tok].item()) for label,tok in mapping.items()}
            others=[value for label,value in sampled.items() if label!=correct]
            scores[name]={"margin":sampled[correct]-max(others),
                          "correct":max(sampled,key=sampled.get)==correct}
        row={"state_id":sid,"model":key,"role":role,"family":family,
             "status":"VALID_ONE_TOKEN_DIAGNOSTIC","correct_first_action":correct,
             "candidate_count":len(mapping),"neutral_probe_token_id":token,
             "no_Type_S_T_fork":True,"recipient_native_KV_primary":True}
        for name,data in scores.items():
            row[f"{name}_margin"]=data["margin"]
            row[f"{name}_correct"]=data["correct"]
        row["R111_minus_R000_margin"]=scores["R111"]["margin"]-scores["R000"]["margin"]
        rows.append(row)
        if n%5==0 or n==len(task["roles"][role]):
            print(f"V38 task diagnostic {key} {role} {n}/{len(task['roles'][role])}",flush=True)
    frame=pd.DataFrame(rows)
    table=root/OUT/f"task_diagnostic_{key}_{role}_v38.parquet"
    frame.to_parquet(table,index=False,compression="zstd")
    valid=frame[frame.status=="VALID_ONE_TOKEN_DIAGNOSTIC"]
    summary={"model":key,"role":role,"selected_states":len(frame),"valid_states":len(valid),
             "positive_margin_change_fraction":float((valid["R111_minus_R000_margin"]>0).mean()) if len(valid) else None,
             "median_margin_change":float(valid["R111_minus_R000_margin"].median()) if len(valid) else None,
             "families_with_valid_states":int(valid.family.nunique()),
             "exact_answer_accuracy":{name:float(valid[f"{name}_correct"].mean()) if len(valid) else None
                                      for name in (*PRIMARY,*EXTRA)},
             "Type_S_T_contrasts":False,"strict_task_grounding_established":False,
             "table_sha256":sha256_file(table)}
    path=root/OUT/f"task_diagnostic_{key}_{role}_v38.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"task_diagnostic_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),str(table.relative_to(root)),
                       "artifacts/trajectory_composition_v38_task_design.freeze.json"],
                      {"model":key,"role":role,"strict_task_grounding_established":False,
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("design","run"))
    p.add_argument("model",nargs="?",choices=("Q","F"));p.add_argument("role",nargs="?",choices=ROLES)
    a=p.parse_args();print(json.dumps(design(Path.cwd()) if a.action=="design" else run(Path.cwd(),a.model,a.role),indent=2))
