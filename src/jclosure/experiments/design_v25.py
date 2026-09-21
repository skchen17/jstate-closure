"""Freeze V25 audit, interaction, convergence, action, projection, and final roles."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from jclosure.protocol_v25 import verify,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/design_v25.py"; OUT=Path("results/v25/processed")
def _digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _hash_ids(x):return hashlib.sha256("\n".join(sorted(i["base_trial_id"] for i in x)).encode()).hexdigest()

def prepare(root:Path)->dict:
 cfg=verify(root)["config"]; v24=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text()); v20=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
 audit=v24["validation_states"]; basis=v24["mediation_basis_states"]; validation=v24["mediation_validation_states"]; actions=v24["mediation_action_ids"]
 used={x["base_trial_id"] for role in ("development_states","validation_states","independent_final_states") for x in v24[role]}
 final=[]
 for family in sorted({x["family"] for x in v20["operator_train"]}):
  candidates=[x for x in v20["operator_train"] if x["family"]==family and x["base_trial_id"] not in used]
  candidates=sorted(candidates,key=lambda x:hashlib.sha256(f"V25-FINAL:{x['base_trial_id']}".encode()).hexdigest())
  final.extend({"base_trial_id":x["base_trial_id"],"family":family,"role":"v25_independent_final"} for x in candidates[:10])
 if len(final)!=50 or {x["base_trial_id"] for x in final}&used:raise RuntimeError("V25 disjoint final unavailable")
 rng=np.random.default_rng(250026); late_idx=rng.permutation(2560)[:1024]; late_sign=rng.choice([-1,1],1024)
 OUT.mkdir(parents=True,exist_ok=True); projection=root/OUT/"late_hidden_projections_v25.npz"
 np.savez_compressed(projection,indices=late_idx.astype(np.int32),sign=late_sign.astype(np.int8),dimensions=np.asarray(cfg["late_projection_dimensions"],np.int16))
 action_specs={x:v24["action_specs"][x] for x in set(v24["train_action_ids"]+v24["heldout_action_ids"])}
 payload={"rank_audit_states":audit,"component_basis_states":basis,"interaction_validation_states":validation,
  "background_source_states":[next(x for x in validation if x["family"]==f) for f in sorted({x["family"] for x in validation})],
  "independent_final_states":final,"rank_audit_action_ids":v24["heldout_action_ids"][:32],"interaction_action_ids":actions,
  "nonlinear_action_pairs":[actions[i:i+2] for i in range(0,len(actions),2)],"action_specs":action_specs,
  "action_alphas":v24["action_alphas"],"action_channel_tags":v24["action_channel_tags"],"layers":cfg["layers"],
  "mapping_layers":cfg["mapping_layers"],"realization_dimensions":cfg["realization_dimensions"],
  "state_hashes":{"rank_audit":_hash_ids(audit),"component_basis":_hash_ids(basis),"interaction_validation":_hash_ids(validation),"independent_final":_hash_ids(final)},
  "action_hashes":{"rank_audit":_digest(v24["heldout_action_ids"][:32]),"interaction":_digest(actions)},
  "late_projection_sha256":sha256_file(projection),"late_projection_rule":"response-independent nested signed random direct coordinates",
  "historical_final_opened":False,"v24_independent_final_opened":False,"v25_independent_final_opened":False,"responses_observed_before_freeze":0}
 target=root/OUT/"design_v25.json";write_json_atomic(target,payload)
 frozen=stage_freeze(root,"design",[SOURCE,"results/v24/processed/bottleneck_design_v24.json","artifacts/compact_causal_response_operator_v20_splits.freeze.json",str(projection),str(target)],
  {k:payload[k] for k in ("state_hashes","action_hashes","late_projection_sha256","late_projection_rule","historical_final_opened","v24_independent_final_opened","v25_independent_final_opened")})
 return {"freeze_digest":frozen["freeze_digest"],**payload["state_hashes"],**payload["action_hashes"]}

if __name__=="__main__":print(json.dumps(prepare(Path.cwd()),indent=2))
