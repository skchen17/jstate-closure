"""Descriptive, same-prefix natural A/B/C state×operator compatibility audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v37/processed")
SOURCE="src/jclosure/experiments/compatibility_v37.py"


def run(root:Path,key:str,role:str):
    if role not in ("development","validation"):raise ValueError(role)
    verify_stage(root,f"local_observation_{key}_{role}")
    path=root/OUT/f"local_observed_vectors_{key}_{role}_v37.npz"
    z=np.load(path)
    frame=pd.read_parquet(root/OUT/f"local_observation_{key}_{role}_v37.parquet")
    if len(frame)!=len(z["mixer"]):raise RuntimeError("V37 compatibility row drift")
    rows=[]
    for i,matrix in enumerate(z["mixer"]):
        x=matrix.astype(np.float64)
        b=np.linalg.norm(x[1]-x[0],axis=1)
        c=np.linalg.norm(x[2]-x[0],axis=1)
        rows.append({"state_id":frame.iloc[i].state_id,"model":key,"role":role,
                     "family":frame.iloc[i].family,"position":frame.iloc[i].position,
                     "probe_index":int(frame.iloc[i].probe_index),
                     "B_state_effect_operator_A":float(b[0]),
                     "B_state_effect_operator_B":float(b[1]),
                     "B_state_effect_operator_C":float(b[2]),
                     "C_state_effect_operator_A":float(c[0]),
                     "C_state_effect_operator_B":float(c[1]),
                     "C_state_effect_operator_C":float(c[2]),
                     "B_matched_largest":bool(b[1]>b[0] and b[1]>b[2]),
                     "C_matched_largest":bool(c[2]>c[0] and c[2]>c[1]),
                     "primary_same_prefix_natural_mismatch":True,
                     "descriptive_only":True})
    result=pd.DataFrame(rows)
    table=root/OUT/f"compatibility_{key}_{role}_v37.parquet"
    result.to_parquet(table,index=False,compression="zstd")
    state=result.groupby(["state_id","family"],as_index=False).agg(
        B_matched_fraction=("B_matched_largest","mean"),
        C_matched_fraction=("C_matched_largest","mean"))
    family={name:{"states":len(part),
                  "median_B_matched_fraction":float(part.B_matched_fraction.median()),
                  "median_C_matched_fraction":float(part.C_matched_fraction.median())}
            for name,part in state.groupby("family")}
    summary={"model":key,"role":role,"states":len(state),"rows":len(result),
             "median_B_matched_fraction":float(state.B_matched_fraction.median()),
             "median_C_matched_fraction":float(state.C_matched_fraction.median()),
             "families":family,"causal_future_matching_claim":False,
             "same_prefix_natural_mismatch_primary":True,"table_sha256":sha256_file(table)}
    summary_path=root/OUT/f"compatibility_{key}_{role}_v37.json"
    write_json_atomic(summary_path,summary)
    seal=stage_freeze(root,f"compatibility_{key}_{role}",
                      [SOURCE,str(summary_path.relative_to(root)),str(table.relative_to(root)),
                       f"artifacts/computational_origin_v37_local_observation_{key}_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(summary_path)})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
