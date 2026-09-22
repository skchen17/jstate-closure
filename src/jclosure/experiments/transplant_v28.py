"""Same-prompt natural outgoing-state transplant, both directions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.transaction_v28 import OUT,design,context,native_pair,teacher_tensor,incoming_action,intercept,aggregate,qnorm,token
from jclosure.experiments.pre_readout_v27 import step,scales,delta,vector,cosine
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/transplant_v28.py"
CONDITIONS=("REC","Conv","KV_last_slot_previous_copy","REC+Conv","REC+Conv+KV_last_slot_previous_copy")
def run(root,role,pilot=False):
 if not pilot:verify_stage(root,"final_opening" if role=="development" else "transplant_development_analysis")
 d=design(root);plan=json.loads((root/OUT/"execution_plan_v28.json").read_text());bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);items=[]
 exclude=json.loads((root/OUT/"endpoint_amendment_v28.json").read_text())["pilot_excluded_base_trial_id"]
 if pilot:items=[next(x for x in d["development"] if x["base_trial_id"]==exclude)]
 else:
  for fam in d["families"]:items += [x for x in d[role] if x["family"]==fam and x["base_trial_id"]!=exclude][:plan["transplant_states_per_family"]]
 rows=[];moves=[];donors=[]
 for i,item in enumerate(items,1):
  m,p,last,n,clean0=native_pair(bundle,dense,item,root,d,state_layer);teacher=teacher_tensor(m);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer);qprefix=v19.apply(p,1.0,incoming_action(values,d),rec,att,"native_fp32_add_bf16_writeback");donor0=step(bundle,dense,qprefix,last,n,d,state_layer);donor1=step(bundle,dense,donor0["cache"],teacher,n+1,d,state_layer);dv=vector(delta(donor1["targets"],clean1["targets"]));donor_q=aggregate(qnorm(donor1["targets"],clean1["targets"],s))
  for c in CONDITIONS:
   for direction,recipient,source,base_future,target_future,goal in (("A_from_B",clean0,donor0,clean1,donor1,dv),("B_from_A",donor0,clean0,donor1,clean1,-dv)):
    transplanted,wb=intercept(p,recipient["cache"],c,rec,att,donor=source["cache"]);out=step(bundle,dense,transplanted,teacher,n+1,d,state_layer);movement=vector(delta(out["targets"],base_future["targets"]));effect=float(np.linalg.norm(movement));goalnorm=float(np.linalg.norm(goal));cos=cosine(movement,goal);mag=effect/max(goalnorm,1e-12);relative=float(np.linalg.norm(vector(delta(out["targets"],target_future["targets"])))/max(goalnorm,1e-12));rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"condition":c,"direction":direction,"donor_h1_q":donor_q,"movement_h1_q":aggregate(qnorm(out["targets"],base_future["targets"],s)),"donor_target_cosine":cos,"magnitude_ratio":mag,"relative_l2_to_donor":relative,"writeback_pass":wb["writeback_pass"],"current_recipient_preserved":True,"continuation_token_identical":True,"token_id":int(last[0,0]),"next_token_id":token(m),"recipient_outgoing_hashes":json.dumps({k:wb["channel_hashes"][k] for k in ("REC","Conv","KV")},sort_keys=True),"vector_index":len(moves)});moves.append(movement);donors.append(goal)
  print(f"V28 transplant {role} {i}/{len(items)}",flush=True)
 if pilot:return {"rows":rows[:4]}
 f=pd.DataFrame(rows);fp=root/OUT/f"natural_write_transplant_{role}_v28.parquet";vp=root/OUT/f"natural_write_transplant_vectors_{role}_v28.npz";f.to_parquet(fp,index=False,compression="zstd");np.savez_compressed(vp,movement=np.stack(moves).astype(np.float32),donor_target=np.stack(donors).astype(np.float32),base_trial_id=f.base_trial_id.to_numpy(str),condition=f.condition.to_numpy(str),direction=f.direction.to_numpy(str))
 summary={"role":role,"states":len(items),"rows":len(f),"all_writeback_pass":bool(f.writeback_pass.all()),"same_prompt_pairing":True,"reciprocal":True,"response_sha256":sha256_file(fp),"vectors_sha256":sha256_file(vp),"historical_final_opened":False};jp=root/OUT/f"natural_write_transplant_{role}_v28.json";write_json_atomic(jp,summary);prereq="artifacts/token_state_transaction_v28_final_opening.freeze.json" if role=="development" else "artifacts/token_state_transaction_v28_transplant_development_analysis.freeze.json";fr=stage_freeze(root,f"transplant_{role}",[SOURCE,str(fp.relative_to(root)),str(vp.relative_to(root)),str(jp.relative_to(root)),prereq],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("role",choices=("pilot","development","validation"));a=p.parse_args();print(json.dumps(run(Path.cwd(),"development" if a.role=="pilot" else a.role,a.role=="pilot"),indent=2))
