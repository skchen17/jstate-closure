"""Freeze V26 state roles, targets, action constructions, and single finalist."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from jclosure.protocol_v26 import verify,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/design_v26.py";OUT=Path("results/v26/processed")
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def hash_ids(x):return hashlib.sha256("\n".join(sorted(i["base_trial_id"] for i in x)).encode()).hexdigest()
def metadata(root,role,family):
 p=root/f"results/v18/processed/crossed_state_{role}_{family}_v18.parquet"
 return pd.read_parquet(p)

def prepare(root:Path)->dict:
 cfg=verify(root)["config"];families=["boolean_logic","modular_arithmetic","short_graph_traversal","simple_state_transition","variable_binding"]
 v24=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text());v25=json.loads((root/"results/v25/processed/design_v25.json").read_text());excluded={x["base_trial_id"] for x in v24["independent_final_states"]+v25["independent_final_states"]}
 roles={"development":[],"validation":[],"independent_final":[]};seed=cfg["seed"]
 for family in families:
  train=metadata(root,"train",family);train=train[train.horizon_panel].to_dict("records");val=metadata(root,"validation",family);val=val[val.horizon_panel].to_dict("records")
  train=[x for x in train if x["base_trial_id"] not in excluded];val=[x for x in val if x["base_trial_id"] not in excluded]
  train.sort(key=lambda x:hashlib.sha256(f"{seed}:train:{x['base_trial_id']}".encode()).hexdigest());val.sort(key=lambda x:hashlib.sha256(f"{seed}:validation:{x['base_trial_id']}".encode()).hexdigest())
  nd=cfg["state_panels"]["development_per_family"];nv=cfg["state_panels"]["validation_per_family"];nf=cfg["state_panels"]["independent_final_per_family"]
  if len(train)<nd+nf or len(val)<nv:raise RuntimeError(f"insufficient V26 states: {family}")
  roles["development"] += [{"base_trial_id":x["base_trial_id"],"family":family,"source_role":"train","role":"development"} for x in train[:nd]]
  roles["independent_final"] += [{"base_trial_id":x["base_trial_id"],"family":family,"source_role":"train","role":"independent_final"} for x in train[nd:nd+nf]]
  roles["validation"] += [{"base_trial_id":x["base_trial_id"],"family":family,"source_role":"validation","role":"validation"} for x in val[:nv]]
 ids=[x["base_trial_id"] for rows in roles.values() for x in rows]
 if len(ids)!=len(set(ids)):raise RuntimeError("V26 role overlap")
 old=json.loads((root/"artifacts/counterfactual_workspace_v19_splits.freeze.json").read_text());projections=np.load(root/"results/v24/processed/target_projections_v24.npz")
 target={"selected_j":old["selected_j"],"selected_logits":old["selected_logits"],"broad_vocabulary_indices":projections["vocabulary_indices"].astype(int).tolist(),"broad_vocabulary_sign":projections["vocabulary_sign"].astype(int).tolist(),"late_residual_indices":projections["residual_indices"].astype(int).tolist(),"late_residual_sign":projections["residual_sign"].astype(int).tolist(),"workspace_layers":[23,26,30],"workspace_per_layer":32,"normalization_scales":old["target_scales"]};projections.close()
 specs={x:v24["action_specs"][x] for x in [*cfg["actuation"]["component_action_ids"].values(),cfg["actuation"]["reference_action_id"],cfg["actuation"]["random_control_action_id"]]};alphas={x:v24["action_alphas"][x] for x in specs}
 constructions=[]
 for channel in cfg["actuation"]["channels"]:
  components=list(cfg["actuation"]["component_action_ids"]) if channel=="joint" else channel.split("+")
  constructions.append({"candidate_id":channel.replace("+","_").lower(),"channel":channel,"components":[cfg["actuation"]["component_action_ids"][x] for x in components],"sign":1,"scale":1.0,"kind":"primary"})
 for sign,scale in cfg["actuation"]["scale_sign_grid"]:
  constructions.append({"candidate_id":f"joint_scale_{str(scale).replace('.','p')}_{'pos' if sign>0 else 'neg'}","channel":"joint","components":list(cfg["actuation"]["component_action_ids"].values()),"sign":int(sign),"scale":float(scale),"kind":"scaling_diagnostic"})
 payload={**roles,"families":families,"state_hashes":{k:hash_ids(v) for k,v in roles.items()},"target_bundle":target,"target_hash":digest(target),"action_specs":specs,"action_alphas":alphas,"candidate_constructions":constructions,"candidate_hash":digest(constructions),"reference_action_id":cfg["actuation"]["reference_action_id"],"finalist":cfg["finalist"],"final_opening_hash":digest(cfg["finalist"]),"responses_observed_before_freeze":0,"historical_final_opened":False,"v26_independent_final_opened":False}
 OUT.mkdir(parents=True,exist_ok=True);path=root/OUT/"design_v26.json";write_json_atomic(path,payload)
 frozen=stage_freeze(root,"design",[SOURCE,"results/v24/processed/bottleneck_design_v24.json","results/v25/processed/design_v25.json","artifacts/counterfactual_workspace_v19_splits.freeze.json","results/v24/processed/target_projections_v24.npz",str(path.relative_to(root))],{"state_hashes":payload["state_hashes"],"target_hash":payload["target_hash"],"candidate_hash":payload["candidate_hash"],"final_opening_hash":payload["final_opening_hash"],"historical_final_opened":False,"v26_independent_final_opened":False})
 return {"freeze_digest":frozen["freeze_digest"],**payload["state_hashes"],"target_hash":payload["target_hash"],"candidate_hash":payload["candidate_hash"]}
if __name__=="__main__":print(json.dumps(prepare(Path.cwd()),indent=2))
