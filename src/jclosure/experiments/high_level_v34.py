"""Locked fresh-panel high-level REC-correction reconfirmation gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.analyze_v32 import bootstrap_median_lower
from jclosure.protocol_v34 import stage_freeze,verify,verify_stage
from jclosure.provenance import sha256_file,write_json_atomic

OUT=Path("results/v34/processed")
SOURCE="src/jclosure/experiments/high_level_v34.py"


def summarize(frame,cfg):
    gate=cfg["high_level_gate"];boot=cfg["bootstrap"]
    family={}
    for name,part in frame.groupby("family"):
        family[name]={"rows":len(part),"positive_fraction":float(part.improvement_positive.mean()),"median_reduction":float(part.relative_conv_error_reduction.median()),"median_alignment":float(part.residual_alignment_cosine.median())}
    positive=float(frame.improvement_positive.mean())
    reduction=float(frame.relative_conv_error_reduction.median())
    alignment=float(frame.residual_alignment_cosine.median())
    reduction_lb=bootstrap_median_lower(frame.relative_conv_error_reduction,boot["iterations"],boot["seed"],boot["lower_tail"])
    alignment_lb=bootstrap_median_lower(frame.residual_alignment_cosine,boot["iterations"],boot["seed"],boot["lower_tail"])
    family_pass=sum(x["positive_fraction"]>=gate["family_positive_fraction_min"] and x["median_reduction"]>=gate["family_median_reduction_min"] and x["median_alignment"]>=gate["family_median_alignment_min"] for x in family.values())
    passed=positive>=gate["positive_fraction_min"] and reduction>=gate["median_reduction_min"] and alignment>=gate["median_alignment_min"] and reduction_lb>gate["bootstrap_lower_reduction_gt"] and alignment_lb>=gate["bootstrap_lower_alignment_min"] and family_pass>=gate["families_required"]
    return {"rows":len(frame),"positive_fraction":positive,"median_reduction":reduction,"median_alignment":alignment,"reduction_bootstrap_lower":reduction_lb,"alignment_bootstrap_lower":alignment_lb,"families_passing":family_pass,"family":family,"pass":bool(passed)}


def run(root:Path,role:str):
    if role not in ("development","validation","independent_final"):raise ValueError(role)
    if role=="validation":verify_stage(root,"high_level_development")
    if role=="independent_final":verify_stage(root,"final_opening")
    cfg=verify(root)["config"]
    result={"role":role,"models":{},"thresholds_retuned":False,"same_semantic_state_ids":True}
    for key in ("Q","F"):
        verify_stage(root,f"factorial_{key}_{role}")
        frame=pd.read_parquet(root/OUT/f"factorial_{key}_{role}_v34.parquet")
        result["models"][key]=summarize(frame,cfg)
    result["both_pass"]=all(x["pass"] for x in result["models"].values())
    result["formal_mediation_authorized"]=bool(result["both_pass"] if role=="development" else result["both_pass"] and json.loads((root/OUT/"high_level_development_v34.json").read_text())["both_pass"])
    path=root/OUT/f"high_level_{role}_v34.json"
    write_json_atomic(path,result)
    stage=stage_freeze(root,f"high_level_{role}",[SOURCE,str(path.relative_to(root)),*(f"artifacts/functional_mediation_v34_factorial_{key}_{role}.freeze.json" for key in ("Q","F"))],{"role":role,"summary_sha256":sha256_file(path),"both_pass":result["both_pass"],"formal_mediation_authorized":result["formal_mediation_authorized"]})
    return {"freeze_digest":stage["freeze_digest"],**result}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("role",choices=("development","validation","independent_final"))
    args=parser.parse_args()
    print(json.dumps(run(Path.cwd(),args.role),indent=2))
