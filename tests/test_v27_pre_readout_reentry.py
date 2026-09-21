import json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v27 import verify,verify_stage
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/v27/processed"
def load(name):return json.loads((OUT/name).read_text())
def test_v27_protocol_and_sequential_freezes():
 assert verify(ROOT)["freeze_digest"]
 for name in ("design","boundary_audit","silent_bank","candidate_expansion","silent_bank_expanded","timing_diagnostic","routing_authorization","final_opening","adjudication"):assert verify_stage(ROOT,name)["freeze_digest"]
def test_v27_roles_disjoint_balanced_and_prospective():
 d=load("design_v27.json");groups=[]
 for role,n in (("calibration",25),("development",50),("validation",25),("independent_final",25)):
  rows=d[role];assert len(rows)==n and len({x["base_trial_id"] for x in rows})==n;assert all(sum(x["family"]==f for x in rows)==n//5 for f in d["families"]);groups.append({x["base_trial_id"] for x in rows})
 assert all(not groups[i]&groups[j] for i in range(4) for j in range(i+1,4));assert d["responses_observed_before_freeze"]==0
def test_v27_boundary_is_genuinely_pre_readout():
 a=load("pre_readout_boundary_audit_v27.json");assert a["J_downstream"] and a["logits_downstream"] and a["fields_writable"] and a["current_output_can_change_in_principle"] and not a["structurally_guaranteed_invariance"]
 assert all(v["p99"]==0 for v in a["replay_floor"].values())
def test_v27_h0_only_banks_are_empty_under_frozen_silence_gates():
 a=load("pre_readout_silent_bank_v27.json");b=load("pre_readout_silent_bank_expanded_v27.json")
 assert a["rows"]==975 and a["silent_rows"]==a["distal_rows"]==0;assert b["rows"]==3600 and b["silent_rows"]==b["distal_rows"]==0
 assert a["future_responses_observed_before_freeze"]==b["future_responses_observed_before_freeze"]==0
def test_v27_timing_diagnostic_does_not_rescue_primary_gate():
 t=load("pre_vs_post_timing_v27.json");assert t["diagnostic_only"] and t["not_used_to_rescue_V27A"] and t["median_pre_h0_q"]>1 and t["median_post_h0_q"]==0 and t["median_pre_h1_q"]>.25 and t["median_post_h1_q"]>.25
def test_v27_formal_outcome_and_routing_stop():
 a=load("v27_adjudication.json");assert not any(a[f"V27_{x}"] for x in "ABCDE") and a["V27_F"] and not a["V27_G"] and not a["V27_H"]
 assert not a["DETAILED_ROUTING_AUTHORIZED"] and a["routing_status"]=="NOT_RUN_NOT_AUTHORIZED" and not a["v27_independent_final_opened"]
 assert a["H2_REMAINS"] and not a["H3_AUTHORIZED"] and not a["DYNAMIC_STATE_SEARCH_AUTHORIZED"] and not a["AUTONOMOUS_CONTROLLER_AUTHORIZED"]
 for name in ("next_token_layer_trace_v27.parquet","first_visibility_profile_v27.parquet","native_component_restoration_v27.parquet","native_component_transplant_v27.parquet","channel_routing_factorial_v27.parquet","strict_writeback_audit_v27.parquet"):assert pd.read_parquet(OUT/name).empty
