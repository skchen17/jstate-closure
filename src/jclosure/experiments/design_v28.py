"""Prospectively freeze disjoint V28 roles, native write rules and gates."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v28 import verify,stage_freeze
from jclosure.provenance import write_json_atomic
OUT=Path("results/v28/processed");SOURCE="src/jclosure/experiments/design_v28.py"
def hd(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def ids(x):return hashlib.sha256("\n".join(sorted(r["base_trial_id"] for r in x)).encode()).hexdigest()
def prepare(root):
 cfg=verify(root)["config"];families=["boolean_logic","modular_arithmetic","short_graph_traversal","simple_state_transition","variable_binding"];excluded=set()
 for path,groups in (("results/v24/processed/bottleneck_design_v24.json",("independent_final_states",)),("results/v25/processed/design_v25.json",("independent_final_states",)),("results/v26/processed/design_v26.json",("development","validation","independent_final")),("results/v27/processed/design_v27.json",("calibration","development","validation","independent_final"))):
  old=json.loads((root/path).read_text())
  for group in groups:excluded|={x["base_trial_id"] for x in old[group]}
 roles={k:[] for k in ("calibration","development","validation","independent_final")}
 for fam in families:
  for source in ("train","validation"):
   f=pd.read_parquet(root/f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet");pool=f[~f.base_trial_id.isin(excluded)].to_dict("records");pool.sort(key=lambda x:hashlib.sha256(f"{cfg['seed']}:{source}:{x['base_trial_id']}".encode()).hexdigest())
   spec=(("calibration",5),("development",15),("independent_final",10)) if source=="train" else (("validation",10),)
   offset=0
   for role,count in spec:
    if len(pool)<offset+count:raise RuntimeError(f"insufficient {fam}/{source}")
    roles[role]+=[{"base_trial_id":x["base_trial_id"],"family":fam,"source_role":source,"role":role,"token_class":"reasoning_relevant" if x["horizon_panel"] else "ordinary_or_reasoning_mixed"} for x in pool[offset:offset+count]];offset+=count
 allids=[x["base_trial_id"] for rows in roles.values() for x in rows]
 if len(allids)!=len(set(allids)):raise RuntimeError("role overlap")
 v27=json.loads((root/"results/v27/processed/design_v27.json").read_text());payload={**roles,"families":families,"state_hashes":{k:ids(v) for k,v in roles.items()},"boundary":cfg["boundary"],"boundary_hash":hd(cfg["boundary"]),"target_bundle":v27["target_bundle"],"target_hash":v27["target_hash"],"action_specs":v27["action_specs"],"action_alphas":v27["action_alphas"],"channel_rules":cfg["channels"],"channel_hash":hd(cfg["channels"]),"intervention_rules":{"REC":"exact incoming recurrent state","Conv":"exact incoming convolution state","KV":"previous KV slot copied into newly appended slot; no sequence shortening","same_state_rewrite":"exact outgoing native tensors","transplant":"same-length native outgoing fields"},"intervention_hash":hd({"channels":cfg["channels"],"boundary":cfg["boundary"],"scaling":cfg["scaling_alphas"]}),"gates":{"current_individual_max":cfg["current_individual_max"],"current_aggregate_max":cfg["current_aggregate_max"],"future_potency_threshold":cfg["future_potency_threshold"],"future_median_min":cfg["future_median_min"],"future_fraction_min":cfg["future_fraction_min"],"bootstrap_lower_min":cfg["bootstrap_lower_min"],"families_required":cfg["families_required"],"transplant_cosine_min":cfg["transplant_cosine_min"],"transplant_magnitude_min":cfg["transplant_magnitude_min"]},"gate_hash":hd(cfg),"final_opening_hash":hd(cfg["final_rule"]),"future_responses_before_freeze":0,"historical_final_opened":False}
 OUT.mkdir(parents=True,exist_ok=True);p=root/OUT/"design_v28.json";write_json_atomic(p,payload);fr=stage_freeze(root,"design",[SOURCE,"results/v27/processed/design_v27.json",str(p.relative_to(root))],{"state_hashes":payload["state_hashes"],"boundary_hash":payload["boundary_hash"],"channel_hash":payload["channel_hash"],"intervention_hash":payload["intervention_hash"],"gate_hash":payload["gate_hash"],"final_opening_hash":payload["final_opening_hash"],"future_responses_before_freeze":0});return {"freeze_digest":fr["freeze_digest"],"role_counts":{k:len(v) for k,v in roles.items()}}
if __name__=="__main__":print(json.dumps(prepare(Path.cwd()),indent=2))
