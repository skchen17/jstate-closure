import json
from pathlib import Path
import numpy as np
from jclosure.protocol_v24 import verify,verify_stage
ROOT=Path(__file__).resolve().parents[1]
def test_v24_protocol_and_stages():
 assert verify(ROOT)["freeze_digest"]
 for name in ("design","layer_response_design","natural_transition_amendment","layer_response_summary_amendment","candidate_selection","mediation_protocol","mediation_results_amendment"):assert verify_stage(ROOT,name)["freeze_digest"]
def test_v24_roles_actions_and_final_are_disjoint():
 d=json.loads((ROOT/"results/v24/processed/bottleneck_design_v24.json").read_text()); dev={x["base_trial_id"] for x in d["development_states"]}; val={x["base_trial_id"] for x in d["validation_states"]}; final={x["base_trial_id"] for x in d["independent_final_states"]}
 assert len(dev)==50 and len(val)==25 and len(final)==50 and not(dev&val or dev&final or val&final)
 assert len(d["train_action_ids"])==256 and len(d["heldout_action_ids"])==32 and not set(d["train_action_ids"])&set(d["heldout_action_ids"])
def test_v24_layer_measurement_shapes():
 d=json.loads((ROOT/"results/v24/processed/bottleneck_design_v24.json").read_text()); p=Path("/data/CSK/J-space-project/v24-output-bottleneck-work/finite_development")/f"layer_{d['development_states'][0]['base_trial_id']}.npz"
 with np.load(p) as z:
  assert z["P0_residual"].shape==(256,32,256); assert z["Pq_vocab"].shape==(256,512); assert z["P0_t0"].shape==(256,288)
def test_v24_rank_and_coverage_outputs():
 v=json.loads((ROOT/"results/v24/processed/output_bottleneck_validation_v24.json").read_text()); assert set(v["rank_curves"])=={"residual","T0","vocab_random","broad_T4"}; assert {"B_GLOBAL","B_FAMILY","B_LOCAL_J","B_LOCAL_P","B_STATE_ORACLE"}<=set(v["coverage"])
def test_v24_mediation_branches_and_writeback():
 m=json.loads((ROOT/"results/v24/processed/causal_bottleneck_mediation_v24.json").read_text()); assert {"B_ONLY","PERP_ONLY","FULL_MINUS_B","restoration","transplant","regeneration"}<=set(m); assert m["writeback_audit"]["min_cosine"]>.90
def test_v24_authorization_and_final_sealing():
 a=json.loads((ROOT/"results/v24/processed/v24_adjudication.json").read_text()); assert not a["H3_AUTHORIZED"] and not a["DYNAMIC_STATE_SEARCH_AUTHORIZED"]; assert not a["historical_final_opened"] and not a["v24_independent_final_opened"]
