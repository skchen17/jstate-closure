"""Diagnostic same-family shuffled natural outgoing-state control."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.experiments.transaction_v28 import OUT,design,context,native_pair,teacher_tensor,intercept,aggregate,qnorm,token
from jclosure.experiments.pre_readout_v27 import step,scales,delta,vector,cosine,metadata
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/wrong_write_v28.py";CHANNELS="REC+Conv+KV_last_slot_previous_copy"
def prepare(root):
 verify_stage(root,"transplant_validation_analysis");d=design(root);excluded=json.loads((root/OUT/"endpoint_amendment_v28.json").read_text())["pilot_excluded_base_trial_id"];pairs={}
 for role in ("development","validation"):
  chosen=[]
  for fam in d["families"]:
   pool=[x for x in d[role] if x["family"]==fam and x["base_trial_id"]!=excluded]
   for recipient in pool[:2]:
    n=metadata(root,recipient)["prompt_length"];donor=next((x for x in pool[2:] if x["base_trial_id"]!=recipient["base_trial_id"] and metadata(root,x)["prompt_length"]==n),None)
    if donor:chosen.append({"recipient":recipient,"donor":donor,"same_family":True,"matched_recorded_prompt_length":int(n)})
  pairs[role]=chosen
 x={"pairs":pairs,"pair_hash":hashlib.sha256(json.dumps(pairs,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"selection":"state IDs and prompt length only; no V28 response used","diagnostic_only":True,"independent_final_opened":False};p=root/OUT/"wrong_write_pairs_v28.json";write_json_atomic(p,x);fr=stage_freeze(root,"wrong_write_pairs",[SOURCE,str(p.relative_to(root)),"artifacts/token_state_transaction_v28_transplant_validation_analysis.freeze.json"],{"pair_hash":x["pair_hash"],"pair_counts":{k:len(v) for k,v in pairs.items()},"diagnostic_only":True});return {"freeze_digest":fr["freeze_digest"],"pair_counts":{k:len(v) for k,v in pairs.items()}}
def run(root,role):
 verify_stage(root,"wrong_write_pairs");pairs=json.loads((root/OUT/"wrong_write_pairs_v28.json").read_text())["pairs"][role];d=design(root);bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);rows=[]
 for i,pair in enumerate(pairs,1):
  a=pair["recipient"];b=pair["donor"];ma,pa,last,n,ao=native_pair(bundle,dense,a,root,d,state_layer);mb,pb,lastb,nb,bo=native_pair(bundle,dense,b,root,d,state_layer)
  if n!=nb:raise RuntimeError("actual prompt length mismatch")
  teacher=teacher_tensor(ma);af=step(bundle,dense,ao["cache"],teacher,n+1,d,state_layer);shuffled,wb=intercept(pa,ao["cache"],CHANNELS,rec,att,donor=bo["cache"]);sf=step(bundle,dense,shuffled,teacher,n+1,d,state_layer);z=qnorm(sf["targets"],af["targets"],s);rows.append({"recipient_id":a["base_trial_id"],"donor_id":b["base_trial_id"],"family":a["family"],"role":role,"h1_q":aggregate(z),"j_h1_q":z["j"],"writeback_pass":wb["writeback_pass"],"current_preserved":True,"continuation_token_identical":True,"recipient_token_id":int(last[0,0]),"donor_token_id":int(lastb[0,0]),"next_token_id":token(ma)});print(f"V28 shuffled {role} {i}/{len(pairs)}",flush=True)
 f=pd.DataFrame(rows);fp=root/OUT/f"wrong_write_{role}_v28.parquet";f.to_parquet(fp,index=False,compression="zstd");summary={"role":role,"rows":len(f),"median_h1_q":float(f.h1_q.median()) if len(f) else None,"all_writeback_pass":bool(f.writeback_pass.all()) if len(f) else None,"diagnostic_only":True,"response_sha256":sha256_file(fp)};jp=root/OUT/f"wrong_write_{role}_v28.json";write_json_atomic(jp,summary);fr=stage_freeze(root,f"wrong_write_{role}",[SOURCE,str(fp.relative_to(root)),str(jp.relative_to(root)),"artifacts/token_state_transaction_v28_wrong_write_pairs.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("prepare","development","validation"));a=p.parse_args();print(json.dumps(prepare(Path.cwd()) if a.command=="prepare" else run(Path.cwd(),a.command),indent=2))
