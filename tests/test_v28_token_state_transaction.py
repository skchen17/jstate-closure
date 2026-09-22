import json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v28 import verify,verify_stage
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/v28/processed"
def load(n):return json.loads((OUT/n).read_text())
def test_v28_protocol_sequential_stages_and_historical_seal():
 b=verify(ROOT);assert b["freeze_digest"] and not b["historical_final_opened"]
 for name in ("design","execution_plan","boundary_audit","endpoint_amendment","development_interception","development_analysis","validation_interception","validation_analysis","final_opening","final_response","transplant_development","transplant_development_analysis","transplant_validation","transplant_validation_analysis","layerwise_development","layerwise_validation","horizons_development","horizons_validation","wrong_write_pairs","wrong_write_development","wrong_write_validation","adjudication"):
  assert verify_stage(ROOT,name)["freeze_digest"]
 if (ROOT/"artifacts/token_state_transaction_v28_final.freeze.json").exists():assert verify_stage(ROOT,"final")["freeze_digest"]
def test_v28_disjoint_balanced_roles_and_endpoint_amendment():
 d=load("design_v28.json");groups=[]
 for role,n in (("calibration",25),("development",75),("validation",50),("independent_final",50)):
  r=d[role];assert len(r)==n and len({x["base_trial_id"] for x in r})==n;assert all(sum(x["family"]==f for x in r)==n//5 for f in d["families"]);groups.append({x["base_trial_id"] for x in r})
 assert all(not groups[i]&groups[j] for i in range(4) for j in range(i+1,4));a=load("endpoint_amendment_v28.json");assert a["primary_readout_j_layer"]==30 and a["formal_development_states"]==74 and a["pilot_future_response_observed_before_amendment"]
def test_v28_native_boundary_and_controls():
 a=load("transaction_boundary_audit_v28.json");assert a["all_same_rewrite_exact"] and a["max_replay_current_q"]==a["max_same_rewrite_h1_q"]==0;assert a["REC_Conv_old_state_restore_valid"] and not a["KV_old_state_restore_valid"]
 assert a["min_read_current_q"]>.25 and len(a["recurrent_layers"])==6 and len(a["attention_layers"])==2
def test_v28_read_write_dissociation_and_dose():
 for role,n in (("development",74),("validation",50)):
  f=pd.read_parquet(OUT/f"write_interception_{role}_v28.parquet");assert len(f)==5*n and f.writeback_pass.all() and (f.write_current_h0_q==0).all()
  x=f[f.condition=="REC+Conv"];assert x.write_h1_q.median()>.25 and x.read_current_h0_q.median()>.25 and x.write_j_h1_q.median()>.25
  a=load(f"{role}_analysis_v28.json");assert a["gates"]["REC+Conv"]["pass"] and a["dose_monotonic_median"]
def test_v28_transplant_reciprocity_and_partial_limit():
 for role in ("development","validation"):
  a=load(f"transplant_{role}_analysis_v28.json");assert a["profiles"]["REC+Conv+KV_last_slot_previous_copy"]["reciprocal_pass"]
  assert not a["profiles"]["REC+Conv"]["reciprocal_pass"]
def test_v28_final_and_formal_outcomes():
 f=load("final_confirmation_v28.json");assert f["final_opened"] and f["rows"]==50 and f["all_writeback_pass"] and f["max_current_q"]==0 and f["potent_fraction"]==1
 a=load("v28_adjudication.json");assert a["V28_A"] and a["V28_B"] and a["V28_C"] and not any(a[f"V28_{x}"] for x in "DEFGH")
 assert a["strict_writeback_all_pass"] and a["current_all_preserved"] and a["tokens_all_identical"] and a["H2_REMAINS"] and not a["H3_AUTHORIZED"]
