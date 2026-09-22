"""Confirmatory five-stage mediation plan, frozen before formal stage responses."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v34 import stage_freeze,verify,verify_stage
from jclosure.provenance import write_json_atomic

OUT=Path("results/v34/processed")
SOURCE="src/jclosure/experiments/mediation_plan_v34.py"


def run(root:Path):
    verify_stage(root,"design")
    for model in ("Q","F"):verify_stage(root,f"interface_{model}")
    cfg=verify(root)["config"]
    panel=json.loads((root/OUT/"panel_v34.json").read_text())
    plan={"models":["Q","F"],"formal_roles":["development","validation"],"all_states_in_each_role":True,"stage_order":cfg["functional_stages"]["order"],"stage_scope":"all 24 recurrent mixers simultaneously in each model","interventions":["JOINT_MINUS_REC_EFFECT_AT_S","CONV_PLUS_REC_EFFECT_AT_S"],"recipient_native_KV":True,"same_probe_token_and_position":True,"mediation_denominator":"error_conv-error_joint; never full donor effect","nonpositive_benefit":"retain row with explicit NONPOSITIVE_BENEFIT; undefined REM/REST does not count as stage success","baseline_replay":"bitwise calibration replay required; formal branch signatures must match frozen factorial within 1e-5","gate":cfg["stage_gate"],"family_gate":"family median REM, REST and reverse correction cosine each exceed threshold in at least 4/5 families per role and model","residual_integration":"same exact mixer contribution enters residual without intervening operation in both tested implementations; run calibration equivalence then report ALIASED_NOT_SEPARATELY_IDENTIFIABLE, never count as an independent site","stages_to_execute_formally":["TRUE_UPDATE","TRANSFORMED_CONTROL","POSTCONV_INPUT","RECURRENT_READ"],"stage5_calibration_alias_verified":True,"pipeline_sets_if_no_single_stage":cfg["pipeline_sets"],"context_subset":{role:{family:[x["base_trial_id"] for x in panel[role] if x["family"]==family][:2] for family in cfg["families"]} for role in ("development","validation")},"KV_subset":{role:{family:[x["base_trial_id"] for x in panel[role] if x["family"]==family][:1] for family in cfg["families"]} for role in ("development","validation")},"depth_groups":"relative quarters of recurrent layers; only after stage qualification","no_validation_site_selection":True,"no_historical_final_reopening":True,"formal_mediation_responses_observed_before_freeze":False}
    path=root/OUT/"mediation_plan_v34.json"
    write_json_atomic(path,plan)
    stage=stage_freeze(root,"mediation_plan",[SOURCE,str(path.relative_to(root)),"artifacts/functional_mediation_v34_interface_Q.freeze.json","artifacts/functional_mediation_v34_interface_F.freeze.json"],{"plan_hash":hd(plan),"formal_mediation_responses_observed_before_freeze":False})
    return {"freeze_digest":stage["freeze_digest"],"formal_stages":plan["stages_to_execute_formally"],"context_states_per_model":20}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
