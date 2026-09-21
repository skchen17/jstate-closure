"""Pre-analysis V21 geometry gate correction to the base protocol's all-family rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.experiments import geometry_analysis_v21 as geometry
from jclosure.experiments.geometry_analysis_amend_v21 import corrected_decompose
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage

SOURCE="src/jclosure/experiments/geometry_gate_amend_v21.py"


def prepare(root:Path)->dict:
    verify_stage(root,"paired_geometry_Procrustes_amendment_1")
    config=verify(root)["config"]
    required=config["gates"]["geometry_min_family_positive_fraction"]
    if required!=1.0:
        raise RuntimeError("V21 original all-five-family geometry rule drift")
    return stage_freeze(root,"paired_geometry_family_gate_amendment_2",
                        [SOURCE,"artifacts/action_coordinate_geometry_v21_paired_geometry_Procrustes_amendment_1.freeze.json",
                         "configs/action_coordinate_geometry_v21.yaml"],
                        {"reason":"Derived analysis design accidentally used four of five positive family correlations, weaker than base protocol's 1.0 positive-family fraction; restore all five before any geometry spectrum analysis.",
                         "required_positive_family_correlation_count":5,
                         "prior_derived_count":4,
                         "geometry_analysis_rows_observed_before_amendment":0,
                         "raw_JVP_finite_matrices_unchanged":True,
                         "final_six_action_responses_opened":False})


def run(root:Path)->dict:
    amend=verify_stage(root,"paired_geometry_family_gate_amendment_2")
    original_verify=geometry.verify_stage
    original_decompose=geometry._decompose
    def stricter(path:Path,stage:str):
        result=original_verify(path,stage)
        if stage=="paired_geometry_analysis_design":
            result=dict(result)
            result["shared_geometry_support_thresholds"]=dict(result["shared_geometry_support_thresholds"])
            result["shared_geometry_support_thresholds"]["minimum_positive_family_correlation_count"]=amend["required_positive_family_correlation_count"]
        return result
    geometry.verify_stage=stricter
    geometry._decompose=corrected_decompose
    try:
        return geometry.run(root)
    finally:
        geometry.verify_stage=original_verify
        geometry._decompose=original_decompose


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run"))
    args=parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage=="prepare" else run(Path.cwd()),indent=2))
