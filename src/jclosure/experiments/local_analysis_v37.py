"""State-unit V37 local prediction gate; downstream effects stay separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/local_analysis_v37.py"


def _ci(values,seed,iterations=2000):
    x=np.asarray(values,np.float64)
    x=x[np.isfinite(x)]
    if not len(x):return [None,None]
    rng=np.random.default_rng(seed)
    boots=np.median(x[rng.integers(0,len(x),size=(iterations,len(x)))],axis=1)
    return [float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]


def _gate(frame,t):
    if frame.empty:return {"states":0,"pass":False}
    values={"cosine":float(frame.raw_cosine.median()),
            "rank":float(frame.raw_rank.median()),
            "ordering":float(frame.raw_ordering.median()),
            "mixer_cosine":float(frame.mixer_cosine.median()),
            "mixer_rank":float(frame.mixer_rank.median()),
            "mixer_ordering":float(frame.mixer_ordering.median())}
    passed=(values["cosine"]>=t["local_effect_cosine_min"] and
            values["rank"]>=t["relative_magnitude_rank_min"] and
            values["ordering"]>=t["pairwise_ordering_min"] and
            values["mixer_cosine"]>=t["local_effect_cosine_min"] and
            values["mixer_rank"]>=t["relative_magnitude_rank_min"] and
            values["mixer_ordering"]>=t["pairwise_ordering_min"])
    return {"states":len(frame),**values,"pass":bool(passed)}


def model(root,key,role,cfg):
    local=pd.read_parquet(root/OUT/f"local_observation_{key}_{role}_v37.parquet")
    causal=pd.read_parquet(root/OUT/f"causal_observation_{key}_{role}_v37.parquet")
    state=local.groupby(["state_id","family"],as_index=False).agg(
        raw_cosine=("raw_effect_cosine","median"),
        raw_rank=("raw_magnitude_rank","median"),
        raw_ordering=("raw_pairwise_order_accuracy","median"),
        mixer_cosine=("mixer_effect_cosine","median"),
        mixer_rank=("mixer_magnitude_rank","median"),
        mixer_ordering=("mixer_pairwise_order_accuracy","median"),
        fixed_interaction=("mixer_interaction_norm","median"),
        natural_interaction=("natural_interaction_norm","median"),
        normalization_gain=("normalization_gain","median"),
        max_raw_error=("max_raw_error","max"),
        max_mixer_error=("max_mixer_error","max"))
    future=causal.groupby("state_id",as_index=False).agg(
        local_fraction=("local_benefit_fraction","median"),
        local_direction=("local_direction_cosine","median"),
        future_interaction=("future_interaction_norm","median"))
    state=state.merge(future,on="state_id",validate="one_to_one")
    state["fixed_to_natural_ratio"]=state.fixed_interaction/state.natural_interaction.clip(lower=1e-6)
    if len(state)!={"development":80,"validation":40}[role]:
        raise RuntimeError(f"V37 incomplete local state table {key}:{role}")
    path=root/OUT/f"local_state_analysis_{key}_{role}_v37.parquet"
    state.to_parquet(path,index=False,compression="zstd")
    gate=cfg["operator_gate"]
    whole=_gate(state,gate)
    families={name:_gate(part,gate) for name,part in state.groupby("family")}
    passed=bool(whole["pass"] and sum(x["pass"] for x in families.values())>=gate["families_required"])
    seed=int(cfg["bootstrap"]["seed"])+(0 if key=="Q" else 1)+(0 if role=="development" else 100)
    return {"states":len(state),"whole":whole,"families":families,
            "prediction_gate_pass":passed,
            "median_local_benefit_fraction":float(state.local_fraction.median()),
            "median_fixed_to_natural_ratio":float(state.fixed_to_natural_ratio.median()),
            "local_fraction_ci_state":_ci(state.local_fraction,seed),
            "raw_cosine_ci_state":_ci(state.raw_cosine,seed+1),
            "state_table_sha256":sha256_file(path),
            "local_tensor_law_not_equated_to_global_mechanism":True}


def run(root:Path,role:str):
    if role not in ("development","validation"):raise ValueError(role)
    if role=="validation":verify_stage(root,"local_analysis_development")
    cfg=verify(root)["config"]
    models={}
    for key in cfg["models"]:
        verify_stage(root,f"local_observation_{key}_{role}")
        models[key]=model(root,key,role,cfg)
    result={"role":role,"models":models,"both_models_local_prediction_pass":all(x["prediction_gate_pass"] for x in models.values()),
            "six_probes_and_six_positions_repeated_within_state":True,
            "state_independent_unit":True,"thresholds_retuned":False,
            "independent_final_seen":False}
    path=root/OUT/f"local_analysis_{role}_v37.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,f"local_analysis_{role}",
                      [SOURCE,str(path.relative_to(root)),
                       *[f"artifacts/computational_origin_v37_local_observation_{key}_{role}.freeze.json"
                         for key in cfg["models"]],
                       *[f"results/v37/processed/local_state_analysis_{key}_{role}_v37.parquet"
                         for key in cfg["models"]]],
                      {"role":role,"both_pass":result["both_models_local_prediction_pass"],
                       "summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.role),indent=2))
