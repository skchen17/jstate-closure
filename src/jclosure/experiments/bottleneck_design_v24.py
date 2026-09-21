"""Freeze V24 states, actions, layers, random projections, and unopened final roles."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np, torch
from jclosure.experiments import action_pool_v22 as a22
from jclosure.protocol_v24 import verify,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/bottleneck_design_v24.py"; OUT=Path("results/v24/processed")
def _digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _hash_ids(x): return hashlib.sha256("\n".join(sorted(i["base_trial_id"] for i in x)).encode()).hexdigest()

def prepare(root:Path)->dict:
 cfg=verify(root)["config"]; roles=json.loads((root/"artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
 v23=json.loads((root/"results/v23/processed/probe_selection_v23.json").read_text())
 v22sel=json.loads((root/"results/v22/processed/action_selection_v22.json").read_text())
 v20=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
 v19=json.loads((root/"artifacts/counterfactual_workspace_v19_splits.freeze.json").read_text())
 train_states=roles["jvp_development"]; val_states=roles["jvp_validation"]
 if len(train_states)!=50 or len(val_states)!=25: raise RuntimeError("V24 expected frozen V21 50/25 panel")
 specs={x["action_id"]:x for x in v23["action_specs"]}; ids=v23["train_action_ids"]
 directions=torch.load(root/"artifacts/causal/v13/probe_directions_v13.pt",map_location="cpu",weights_only=False)
 score=np.asarray(directions["score_directions"],np.float64); del directions
 tags={}
 for action_id in ids:
  row=a22._score(specs[action_id],score).reshape(3,-1); n=np.linalg.norm(row,axis=1); share=n/max(n.sum(),1e-12)
  tags[action_id]=["REC","Conv","KV"][int(np.argmax(share))] if share.max()>=.55 else "joint"
 bins={name:[] for name in ("REC","Conv","KV","joint")}
 for action_id in sorted(ids,key=lambda x:hashlib.sha256(f"V24-ACTION:{x}".encode()).hexdigest()): bins[tags[action_id]].append(action_id)
 ordered=[]
 while len(ordered)<len(ids):
  for name in bins:
   if bins[name]: ordered.append(bins[name].pop(0))
 train_actions=ordered[:256]
 alpha={x:float(v23["action_alphas"][x]) for x in train_actions}
 held=list(v23["heldout_validation_action_ids"]); alpha.update({x:float(v22sel["action_alphas"][x]) for x in held})
 pool=json.loads((root/"results/v22/processed/action_pool_v22.json").read_text())
 all_specs={x["action_id"]:x for x in pool["candidate_pool"]}; all_specs.update(specs)
 used={x["base_trial_id"] for x in train_states+val_states}
 final=[]
 for family in sorted({x["family"] for x in v20["operator_train"]}):
  candidates=[x for x in v20["operator_train"] if x["family"]==family and x["base_trial_id"] not in used]
  candidates=sorted(candidates,key=lambda x:hashlib.sha256(f"V24-FINAL:{x['base_trial_id']}".encode()).hexdigest())
  final.extend({"base_trial_id":x["base_trial_id"],"family":family,"role":"v24_independent_final_for_new_broad_layer_estimand"} for x in candidates[:10])
 if len(final)!=50: raise RuntimeError("V24 final state panel unavailable")
 by_dev={}; by_val={}
 for x in train_states: by_dev.setdefault(x["family"],[]).append(x)
 for x in val_states: by_val.setdefault(x["family"],[]).append(x)
 mediation_dev=[x for f in sorted(by_dev) for x in by_dev[f][:2]]
 mediation_val=[x for f in sorted(by_val) for x in by_val[f][:2]]
 natural=[]
 for family in sorted({x["family"] for x in v19["train"]}):
  natural.extend([x for x in v19["train"] if x["family"]==family and x.get("horizon_panel")][:2])
 if len(natural)!=10: raise RuntimeError("V24 natural transition panel unavailable")
 rng=np.random.default_rng(int(cfg["targets"]["projection_seed"]))
 residual_idx=rng.permutation(2560)[:int(cfg["targets"]["residual_random_coordinate_dimension"])]
 arch_idx=rng.permutation(2560)[:int(cfg["targets"]["architecture_random_coordinate_dimension"])]
 vocab_idx=rng.permutation(248320)[:int(cfg["targets"]["vocabulary_random_coordinate_dimension"])]
 residual_sign=rng.choice([-1,1],len(residual_idx)); arch_sign=rng.choice([-1,1],len(arch_idx)); vocab_sign=rng.choice([-1,1],len(vocab_idx))
 OUT.mkdir(parents=True,exist_ok=True); projection=root/OUT/"target_projections_v24.npz"
 np.savez_compressed(projection,residual_indices=residual_idx.astype(np.int32),residual_sign=residual_sign.astype(np.int8),
  architecture_indices=arch_idx.astype(np.int32),architecture_sign=arch_sign.astype(np.int8),
  vocabulary_indices=vocab_idx.astype(np.int32),vocabulary_sign=vocab_sign.astype(np.int8),
  layers=np.asarray(cfg["layers"],np.int16),architecture_layers=np.asarray(cfg["architecture_layers"],np.int16))
 payload={"development_states":train_states,"validation_states":val_states,"jvp_states":mediation_dev,
  "mediation_basis_states":mediation_dev,"mediation_validation_states":mediation_val,"natural_transition_states":natural,
  "independent_final_states":final,"persistent_states":["P0","Pq"],"train_action_ids":train_actions,"heldout_action_ids":held,
  "mediation_action_ids":held[:8],"action_alphas":alpha,"action_specs":{x:all_specs[x] for x in train_actions+held},
  "action_channel_tags":{x:tags.get(x,"heldout_mixed") for x in train_actions+held},"layers":cfg["layers"],"architecture_layers":cfg["architecture_layers"],
  "state_hashes":{"development":_hash_ids(train_states),"validation":_hash_ids(val_states),"mediation_basis":_hash_ids(mediation_dev),
                  "mediation_validation":_hash_ids(mediation_val),"natural_transition":_hash_ids(natural),"independent_final":_hash_ids(final)},
  "action_hashes":{"train":_digest(train_actions),"heldout":_digest(held),"mediation":_digest(held[:8])},
  "target_projection_sha256":sha256_file(projection),"target_projection_rule":"train-frozen seeded signed random coordinate projections; rows are orthonormal coordinate selectors and are response-label independent",
  "historical_final_opened":False,"v24_independent_final_opened":False,"responses_observed_before_freeze":0}
 target=root/OUT/"bottleneck_design_v24.json"; write_json_atomic(target,payload)
 frozen=stage_freeze(root,"design",[SOURCE,"artifacts/action_coordinate_geometry_v21_roles.freeze.json",
  "results/v23/processed/probe_selection_v23.json","results/v22/processed/action_selection_v22.json",
  "artifacts/compact_causal_response_operator_v20_splits.freeze.json","artifacts/counterfactual_workspace_v19_splits.freeze.json",str(projection),str(target)],
  {k:payload[k] for k in ("state_hashes","action_hashes","target_projection_sha256","target_projection_rule","historical_final_opened","v24_independent_final_opened")})
 return {"freeze_digest":frozen["freeze_digest"],"development":50,"validation":25,"final":50,**payload["state_hashes"],**payload["action_hashes"]}

if __name__=="__main__": print(json.dumps(prepare(Path.cwd()),indent=2))
