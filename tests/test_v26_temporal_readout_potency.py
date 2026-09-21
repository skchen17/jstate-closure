import json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v26 import verify, verify_stage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/v26/processed"
def load(name):return json.loads((OUT/name).read_text())

def test_v26_protocol_and_frozen_stages():
 assert verify(ROOT)["freeze_digest"]
 for name in ("design","numerical_floor","current_distal_bank","early_rollout","extension_opening","extension_rollout","devval_analysis","final_opening","final_rollout","adjudication","diagnostics"):
  assert verify_stage(ROOT,name)["freeze_digest"]

def test_v26_roles_are_balanced_disjoint_and_final_was_prospectively_frozen():
 d=load("design_v26.json");groups=[]
 for role,n in (("development",100),("validation",50),("independent_final",50)):
  rows=d[role];ids={x["base_trial_id"] for x in rows};assert len(rows)==len(ids)==n
  assert {f:sum(x["family"]==f for x in rows) for f in d["families"]}=={f:n//5 for f in d["families"]};groups.append(ids)
 assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 assert d["responses_observed_before_freeze"]==0 and d["finalist"]["horizon"]==2

def test_v26_numerical_floor_and_h0_only_bank():
 floor=load("current_readout_numerical_floor_v26.json");assert floor["all_targets_classifiable"] and all(x["p99"]==0 for x in floor["targets"].values())
 bank=load("current_distal_bank_v26.json");assert bank["states"]==150 and bank["rows"]==1170 and bank["strict_current_silent_rows"]==1152
 assert bank["selection_information"]=="h0_only" and bank["future_responses_observed_before_freeze"]==0

def test_v26_staged_horizon_opening_and_primary_gates():
 early=load("early_temporal_analysis_v26.json");assert early["extension_opened"]
 for role in ("development","validation"):
  assert early["gates"][role]["1"]["pass"] and early["gates"][role]["2"]["pass"]
 full=load("v26_devval_analysis.json");assert full["final_opened"] and full["V26_A_DEVVAL"] and full["V26_B_DEVVAL"] and full["V26_C_DEVVAL"]

def test_v26_temporal_geometry_and_emergence():
 geometry=load("future_potent_subspaces_v26.json");assert geometry["U0_rank"]==0 and geometry["future_requires_outside_U0"]
 assert geometry["ROTATING_FUTURE_POTENT_GEOMETRY"] and not geometry["FIXED_FUTURE_POTENT_GEOMETRY"]
 emergence=pd.read_parquet(OUT/"emergence_times_v26.parquet");assert len(emergence)==1050 and set(emergence.emergence_time)=={"1"}

def test_v26_independent_final_and_formal_outcomes():
 a=load("v26_adjudication.json");assert a["final_confirmation"]["pass"] and a["final_confirmation"]["rows"]==50
 assert a["V26_A"] and a["V26_B"] and a["V26_C"] and a["V26_D"]
 assert not a["V26_E"] and not a["V26_F"] and not a["V26_G"] and not a["V26_H"] and not a["V26_I"]
 assert a["CURRENT_READOUT_CAUSAL_SUFFICIENCY_REJECTED_UNDER_TESTED_PANEL"]
 assert a["H2_REMAINS"] and not a["H3_AUTHORIZED"] and not a["DYNAMIC_STATE_SEARCH_AUTHORIZED"] and a["CAUSAL_ROUTING_V27_AUTHORIZED"]
 assert not a["historical_final_opened"] and a["v26_independent_final_opened"] and not a["future_based_bank_selection"]

def test_v26_secondary_diagnostics_do_not_change_formal_outcomes():
 d=load("v26_secondary_diagnostics.json");assert not d["formal_outcomes_changed"]
 assert d["future_directions_outside_U0"] and d["sign_and_scale"]["smooth_scaling_supported"] is False
