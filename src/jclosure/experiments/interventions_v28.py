"""Native outgoing-write blocks, read contrasts, and prospective dose panels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.transaction_v28 import OUT,SOURCE as TXSOURCE,design,context,native_pair,teacher_tensor,incoming_action,intercept,qnorm,aggregate,current_aggregate,channel_hash,write_stats,token
from jclosure.experiments.pre_readout_v27 import TARGETS,PRIMARY,step,scales,delta,vector,norms,normalized,cosine
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/interventions_v28.py"
def requested_for(values,d,c):
 x=incoming_action(values,d);channels=set(c.split("+"));channels={"KV" if z=="KV_last_slot_previous_copy" else z for z in channels}
 return {k:v if ("REC" if k=="recurrent" else "Conv" if k=="conv" else "KV") in channels else torch.zeros_like(v) for k,v in x.items()}
def endpoint_amendment(root):
 verify_stage(root,"boundary_audit");d=design(root);pilot_id=d["development"][0]["base_trial_id"]
 x={"primary_readout_j_layer":30,"state_proxy_j_layer":23,"reason":"pilot showed next-token layer-23 state proxy is upstream of REC/Conv writes; V26 h1 readout-J uses layer 30","pilot_excluded_base_trial_id":pilot_id,"formal_development_states":74,"pilot_future_response_observed_before_amendment":True,"thresholds_unchanged":True,"validation_responses_observed":False,"independent_final_opened":False};p=root/OUT/"endpoint_amendment_v28.json";write_json_atomic(p,x);fr=stage_freeze(root,"endpoint_amendment",[SOURCE,str(p.relative_to(root)),"artifacts/token_state_transaction_v28_boundary_audit.freeze.json"],x);return {"freeze_digest":fr["freeze_digest"],**x}
def run(root,role,pilot=False):
 if not pilot:verify_stage(root,"endpoint_amendment")
 cfg=verify(root)["config"];plan=json.loads((root/OUT/"execution_plan_v28.json").read_text());d=design(root);bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);rows=[];wvec=[];rvec=[];dose=[];replay=[]
 items=d[role][:1] if pilot else d[role]
 if not pilot and role=="development":
  excluded=json.loads((root/OUT/"endpoint_amendment_v28.json").read_text())["pilot_excluded_base_trial_id"]
  items=[x for x in items if x["base_trial_id"]!=excluded]
 for i,item in enumerate(items,1):
  m,p,last,n,clean0=native_pair(bundle,dense,item,root,d,state_layer);next_token=teacher_tensor(m);clean1=step(bundle,dense,clean0["cache"],next_token,n+1,d,state_layer);again=step(bundle,dense,clean0["cache"],next_token,n+1,d,state_layer);replay.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"h1_replay_q":aggregate(qnorm(again["targets"],clean1["targets"],s)),"token_id":int(last[0,0]),"next_token_id":token(m)})
  for c in plan["main_conditions"]:
   blocked,wb=intercept(p,clean0["cache"],c,rec,att);write1=step(bundle,dense,blocked,next_token,n+1,d,state_layer);wv=vector(delta(write1["targets"],clean1["targets"]));wz=qnorm(write1["targets"],clean1["targets"],s)
   qprefix=v19.apply(p,1.0,requested_for(values,d,c),rec,att,"native_fp32_add_bf16_writeback");read0=step(bundle,dense,qprefix,last,n,d,state_layer);read1=step(bundle,dense,read0["cache"],next_token,n+1,d,state_layer);rv=vector(delta(read1["targets"],clean1["targets"]));r0=qnorm(read0["targets"],clean0["targets"],s);r1=qnorm(read1["targets"],clean1["targets"],s)
   rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"token_class":item["token_class"],"condition":c,"token_id":int(last[0,0]),"next_token_id":token(m),"write_current_h0_q":0.0,"read_current_h0_q":current_aggregate(r0),"write_h1_q":aggregate(wz),"read_h1_q":aggregate(r1),"write_j_h1_q":wz["j"],"write_logits_h1_q":wz["logits"],"write_semantic_h1_q":wz["semantic"],"write_workspace_h1_q":wz["workspace"],"read_j_h0_q":r0["j"],"read_logits_h0_q":r0["logits"],"read_workspace_h0_q":r0["workspace"],"writeback_pass":wb["writeback_pass"],"requested_exact":wb["requested_exact"],"untouched_exact":wb["untouched_exact"],"touched_fields":wb["touched_fields"],"write_channel_hashes":json.dumps(wb["channel_hashes"],sort_keys=True),"current_targets_identical_by_capture":True,"continuation_token_identical":True,"write_h1_vector_index":len(wvec)});wvec.append(wv);rvec.append(rv)
  fam_items=[x for x in d[role] if x["family"]==item["family"]]
  if item in fam_items[:plan["dose_states_per_family"]]:
   for alpha in plan["dose_alphas"]:
    ccache,wb=intercept(p,clean0["cache"],"REC+Conv",rec,att,alpha=float(alpha));x=step(bundle,dense,ccache,next_token,n+1,d,state_layer);dv=vector(delta(x["targets"],clean1["targets"]));z=qnorm(x["targets"],clean1["targets"],s);dose.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"alpha":alpha,"h1_q":aggregate(z),"j_h1_q":z["j"],"writeback_pass":wb["writeback_pass"],"delta_norm":float(np.linalg.norm(dv))})
  print(f"V28 {role} {i}/{len(items)}",flush=True)
 if pilot:return {"pilot_rows":rows,"pilot_replay":replay,"pilot_dose":dose}
 f=pd.DataFrame(rows);rp=pd.DataFrame(replay);dp=pd.DataFrame(dose);fp=root/OUT/f"write_interception_{role}_v28.parquet";vp=root/OUT/f"read_write_vectors_{role}_v28.npz";rep=root/OUT/f"clean_replay_{role}_v28.parquet";dop=root/OUT/f"write_dose_{role}_v28.parquet";f.to_parquet(fp,index=False,compression="zstd");rp.to_parquet(rep,index=False,compression="zstd");dp.to_parquet(dop,index=False,compression="zstd");np.savez_compressed(vp,write_vectors=np.stack(wvec).astype(np.float32),read_vectors=np.stack(rvec).astype(np.float32),base_trial_id=f.base_trial_id.to_numpy(str),condition=f.condition.to_numpy(str),family=f.family.to_numpy(str))
 summary={"role":role,"states":len(items),"rows":len(f),"dose_rows":len(dp),"all_writeback_pass":bool(f.writeback_pass.all() and dp.writeback_pass.all()),"max_write_current_q":float(f.write_current_h0_q.max()),"max_clean_replay_h1_q":float(rp.h1_replay_q.max()),"primary_REC_Conv_median_h1_q":float(f[f.condition=="REC+Conv"].write_h1_q.median()),"read_REC_Conv_median_current_q":float(f[f.condition=="REC+Conv"].read_current_h0_q.median()),"response_sha256":sha256_file(fp),"vector_sha256":sha256_file(vp),"historical_final_opened":False,"independent_final_opened":role=="independent_final"};jp=root/OUT/f"write_interception_{role}_v28.json";write_json_atomic(jp,summary);prereq="artifacts/token_state_transaction_v28_endpoint_amendment.freeze.json" if role=="development" else "artifacts/token_state_transaction_v28_development_analysis.freeze.json" if role=="validation" else "artifacts/token_state_transaction_v28_final_opening.freeze.json";fr=stage_freeze(root,f"{role}_interception",[SOURCE,str(fp.relative_to(root)),str(vp.relative_to(root)),str(rep.relative_to(root)),str(dop.relative_to(root)),str(jp.relative_to(root)),prereq],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("role",choices=("amend","pilot","development","validation","independent_final"));a=p.parse_args();print(json.dumps(endpoint_amendment(Path.cwd()) if a.role=="amend" else run(Path.cwd(),"development" if a.role=="pilot" else a.role,a.role=="pilot"),indent=2))
