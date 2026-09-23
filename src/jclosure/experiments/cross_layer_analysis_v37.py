"""State-unit cross-layer operator-conditioning gate and scope audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/cross_layer_analysis_v37.py"


def _gate(frame,t):
    if frame.empty:return {"states":0,"pass":False}
    valid=frame.dropna(subset=["mixer_change_cosine"])
    cosine=float(valid.mixer_change_cosine.median()) if len(valid) else None
    ordering=float(frame.future_order_accuracy.median())
    factor=float(frame.factor_change_fraction.median())
    hidden=float(frame.hidden_change_norm.median())
    passed=bool(cosine is not None and cosine>=t["prospective_cosine_min"]
                and ordering>=t["ordering_accuracy_min"] and factor>0 and hidden>1e-6)
    return {"states":len(frame),"eligible_cosine":len(valid),
            "median_mixer_change_cosine":cosine,
            "median_future_order_accuracy":ordering,
            "median_factor_change_fraction":factor,
            "median_hidden_change_norm":hidden,"pass":passed}


def model(root,key,role,cfg):
    local=pd.read_parquet(root/OUT/f"cross_layer_local_{key}_{role}_v37.parquet")
    site=pd.read_parquet(root/OUT/f"cross_layer_state_{key}_{role}_v37.parquet")
    state_local=local.groupby(["state_id","family"],as_index=False).agg(
        mixer_change_cosine=("mixer_change_cosine","median"),
        factor_change_fraction=("factor_changed","mean"),
        hidden_change_norm=("hidden_change_norm","median"),
        mixer_change_norm=("mixer_change_norm","median"))
    state_future=site.groupby("state_id",as_index=False).agg(
        future_order_accuracy=("future_order_correct","mean"),
        future_change_norm=("future_effect_change_norm","median"))
    state=state_local.merge(state_future,on="state_id",validate="one_to_one")
    if len(state)!={"development":80,"validation":40}[role]:
        raise RuntimeError(f"Incomplete V37 cross-layer state table {key}:{role}")
    path=root/OUT/f"cross_layer_state_analysis_{key}_{role}_v37.parquet"
    state.to_parquet(path,index=False,compression="zstd")
    threshold=cfg["cross_layer_gate"]
    whole=_gate(state,threshold)
    families={family:_gate(part,threshold) for family,part in state.groupby("family")}
    passed=bool(whole["pass"] and sum(x["pass"] for x in families.values())>=threshold["families_required"])
    return {"states":len(state),"whole":whole,"families":families,"cross_layer_gate_pass":passed,
            "state_table_sha256":sha256_file(path),
            "only_Q2_upstream_outputs_patched":True,
            "Q3_Q4_and_downstream_native":True,
            "direct_vs_indirect_Q2_not_identified_by_this_test":True}


def run(root:Path,role:str):
    if role not in ("development","validation"):raise ValueError(role)
    if role=="validation":verify_stage(root,"cross_layer_analysis_development")
    cfg=verify(root)["config"]
    models={}
    for key in cfg["models"]:
        verify_stage(root,f"cross_layer_observation_{key}_{role}")
        models[key]=model(root,key,role,cfg)
    result={"role":role,"models":models,
            "both_models_cross_layer_pass":all(m["cross_layer_gate_pass"] for m in models.values()),
            "thresholds_retuned":False,"state_independent_unit":True,
            "six_probes_and_four_sites_repeated_within_state":True,
            "mechanism_level":"cross-layer local operator prediction plus natural downstream order",
            "independent_final_seen":False}
    path=root/OUT/f"cross_layer_analysis_{role}_v37.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,f"cross_layer_analysis_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[f"artifacts/computational_origin_v37_cross_layer_observation_{key}_{role}.freeze.json"
                         for key in cfg["models"]],
                       *[f"results/v37/processed/cross_layer_state_analysis_{key}_{role}_v37.parquet"
                         for key in cfg["models"]]],
                      {"role":role,"both_pass":result["both_models_cross_layer_pass"],
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.role),indent=2))
