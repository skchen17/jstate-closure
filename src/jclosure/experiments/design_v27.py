"""Freeze V27 boundaries, disjoint roles, candidates, targets, estimands and gates."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v27 import verify,stage_freeze
from jclosure.provenance import write_json_atomic
SOURCE="src/jclosure/experiments/design_v27.py";OUT=Path("results/v27/processed")
def hd(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def ids(x):return hashlib.sha256("\n".join(sorted(i["base_trial_id"] for i in x)).encode()).hexdigest()
def prepare(root:Path):
 cfg=verify(root)["config"];families=["boolean_logic","modular_arithmetic","short_graph_traversal","simple_state_transition","variable_binding"]
 excluded=set()
 for name,groups in (("results/v24/processed/bottleneck_design_v24.json",("independent_final_states",)),("results/v25/processed/design_v25.json",("independent_final_states",)),("results/v26/processed/design_v26.json",("development","validation","independent_final"))):
  old=json.loads((root/name).read_text())
  for group in groups:excluded|={x["base_trial_id"] for x in old[group]}
 roles={k:[] for k in ("calibration","development","validation","independent_final")}
 for family in families:
  train=pd.read_parquet(root/f"results/v18/processed/crossed_state_train_{family}_v18.parquet");train=train[train.horizon_panel & ~train.base_trial_id.isin(excluded)].to_dict("records")
  val=pd.read_parquet(root/f"results/v18/processed/crossed_state_validation_{family}_v18.parquet");val=val[val.horizon_panel & ~val.base_trial_id.isin(excluded)].to_dict("records")
  train.sort(key=lambda x:hashlib.sha256(f"{cfg['seed']}:train:{x['base_trial_id']}".encode()).hexdigest());val.sort(key=lambda x:hashlib.sha256(f"{cfg['seed']}:validation:{x['base_trial_id']}".encode()).hexdigest())
  nc=cfg["states"]["calibration_per_family"];nd=cfg["states"]["development_per_family"];nf=cfg["states"]["independent_final_per_family"];nv=cfg["states"]["validation_per_family"]
  if len(train)<nc+nd+nf or len(val)<nv:raise RuntimeError(f"insufficient V27 states: {family}")
  for role,pool in (("calibration",train[:nc]),("development",train[nc:nc+nd]),("independent_final",train[nc+nd:nc+nd+nf]),("validation",val[:nv])):
   source="validation" if role=="validation" else "train";roles[role]+=[{"base_trial_id":x["base_trial_id"],"family":family,"source_role":source,"role":role} for x in pool]
 allids=[x["base_trial_id"] for rows in roles.values() for x in rows]
 if len(allids)!=len(set(allids)):raise RuntimeError("V27 role overlap")
 v26=json.loads((root/"results/v26/processed/design_v26.json").read_text());target=v26["target_bundle"]
 constructions=v26["candidate_constructions"]
 payload={**roles,"families":families,"state_hashes":{k:ids(v) for k,v in roles.items()},"boundary":cfg["boundary"],"boundary_hash":hd(cfg["boundary"]),"target_bundle":target,"target_hash":hd(target),"action_specs":v26["action_specs"],"action_alphas":v26["action_alphas"],"candidate_constructions":constructions,"candidate_hash":hd(constructions),"silence_rules":cfg["silence"],"silence_hash":hd(cfg["silence"]),"routing_rules":cfg["routing"],"routing_hash":hd(cfg["routing"]),"final_opening_rule":{"single_candidate_chosen_from_development_h1_after_bank_freeze":True,"requires_development_and_validation_V27A":True},"final_opening_hash":hd({"single_candidate_chosen_from_development_h1_after_bank_freeze":True,"requires_development_and_validation_V27A":True}),"responses_observed_before_freeze":0,"DETAILED_ROUTING_AUTHORIZED":False,"historical_final_opened":False}
 OUT.mkdir(parents=True,exist_ok=True);path=root/OUT/"design_v27.json";write_json_atomic(path,payload)
 frozen=stage_freeze(root,"design",[SOURCE,"results/v26/processed/design_v26.json",str(path.relative_to(root))],{"state_hashes":payload["state_hashes"],"boundary_hash":payload["boundary_hash"],"target_hash":payload["target_hash"],"candidate_hash":payload["candidate_hash"],"silence_hash":payload["silence_hash"],"routing_hash":payload["routing_hash"],"final_opening_hash":payload["final_opening_hash"],"responses_observed_before_freeze":0,"DETAILED_ROUTING_AUTHORIZED":False})
 return {"freeze_digest":frozen["freeze_digest"],"role_counts":{k:len(v) for k,v in roles.items()},"state_hashes":payload["state_hashes"]}
if __name__=="__main__":print(json.dumps(prepare(Path.cwd()),indent=2))
