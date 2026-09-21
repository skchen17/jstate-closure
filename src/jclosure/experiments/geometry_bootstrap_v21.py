"""Family-stratified base-state bootstrap for paired V21 geometry summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jclosure.experiments import bank_v21 as bank
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE="src/jclosure/experiments/geometry_bootstrap_v21.py"
SUMMARY=bank.OUT/"paired_geometry_bootstrap_v21.json"


def prepare(root:Path)->dict:
    design=verify_stage(root,"paired_geometry_analysis_design")
    verify_stage(root,"paired_geometry_family_gate_amendment_2")
    return stage_freeze(root,"paired_geometry_bootstrap_design",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_paired_geometry_analysis_design.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_paired_geometry_family_gate_amendment_2.freeze.json"],
                        {"unit":"base_trial_id","stratification":"five_task_families_within_role",
                         "replicates":design["bootstrap_replicates"],"seed":design["bootstrap_seed"],
                         "confidence_interval":"percentile_2.5_97.5",
                         "final_six_action_responses_opened":False})


def _stat(frame:pd.DataFrame)->dict:
    x=frame["JVP_rotation_principal_angle_median_degrees"].to_numpy(float)
    y=frame["finite_rotation_principal_angle_median_degrees"].to_numpy(float)
    return {"rotation_Spearman_rho":float(spearmanr(x,y).statistic),
            "JVP_median_r95":float(frame["P0_JVP_r95"].median()),
            "finite_median_r95":float(frame["P0_finite_r95"].median()),
            "JVP_median_rotation_degrees":float(np.median(x)),
            "finite_median_rotation_degrees":float(np.median(y)),
            "P0_JVP_finite_overlap_median":float(frame["P0_JVP_finite_subspace_overlap"].median()),
            "finite_pre_alignment_error_median":float(frame["finite_pre_alignment_relative_error"].median()),
            "finite_post_Procrustes_error_median":float(frame["finite_post_orthogonal_Procrustes_relative_error"].median())}


def run(root:Path)->dict:
    design=verify_stage(root,"paired_geometry_bootstrap_design")
    source=root/"results/v21/processed/paired_geometry_analysis_v21.parquet"
    frame=pd.read_parquet(source)
    output={"design_digest":design["freeze_digest"],"source_sha256":sha256_file(source),
            "by_role":{},"final_six_action_responses_opened":False}
    rng=np.random.default_rng(int(design["seed"]))
    for role,group in frame.groupby("role"):
        families={name:part.reset_index(drop=True) for name,part in group.groupby("family")}
        point=_stat(group)
        samples={key:[] for key in point}
        for _ in range(int(design["replicates"])):
            draw=pd.concat([part.iloc[rng.integers(0,len(part),size=len(part))]
                            for part in families.values()],ignore_index=True)
            values=_stat(draw)
            for key,value in values.items():
                if np.isfinite(value):
                    samples[key].append(value)
        output["by_role"][role]={"base_states":len(group),"point":point,
                                  "CI95":{key:[float(np.quantile(value,0.025)),float(np.quantile(value,0.975))]
                                          if value else [None,None] for key,value in samples.items()},
                                  "valid_bootstrap_replicates":{key:len(value) for key,value in samples.items()}}
    target=root/SUMMARY
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(target,output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
