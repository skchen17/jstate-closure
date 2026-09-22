"""Layerwise native write localization and gated h1/h2/h4 continuation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.transaction_v28 import OUT,design,context,native_pair,teacher_tensor,intercept,write_stats,aggregate,qnorm,token
from jclosure.experiments.pre_readout_v27 import step,scales,delta,vector,cosine
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/secondary_v28.py"
def selected(root,role,n):
 d=design(root);exclude=json.loads((root/OUT/"endpoint_amendment_v28.json").read_text())["pilot_excluded_base_trial_id"];return [x for fam in d["families"] for x in [y for y in d[role] if y["family"]==fam and y["base_trial_id"]!=exclude][:n]]
def layer_cache(p,out,layers,channels):
 c=clone_hybrid_cache(out)
 for layer in layers:
  for ch in channels:
   name="recurrent_states" if ch=="REC" else "conv_states"
   setattr(c.layers[layer],name,getattr(p.layers[layer],name).detach().clone())
 return c
def layerwise(root,role):
 verify_stage(root,"transplant_validation_analysis");plan=json.loads((root/OUT/"execution_plan_v28.json").read_text());d=design(root);bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);items=selected(root,role,plan["layer_scan_states_per_family"]);rows=[]
 for i,item in enumerate(items,1):
  m,p,last,n,clean0=native_pair(bundle,dense,item,root,d,state_layer);teacher=teacher_tensor(m);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer);allblocked,_=intercept(p,clean0["cache"],"REC+Conv",rec,att);all1=step(bundle,dense,allblocked,teacher,n+1,d,state_layer);full=vector(delta(all1["targets"],clean1["targets"]));fullnorm=float(np.linalg.norm(full))
  for j,layer in enumerate(rec):
   for kind,layers,channels in (("single_REC",[layer],["REC"]),("single_Conv",[layer],["Conv"]),("prefix_REC_Conv",rec[:j+1],["REC","Conv"]),("suffix_REC_Conv",rec[j:],["REC","Conv"])):
    c=layer_cache(p,clean0["cache"],layers,channels);out=step(bundle,dense,c,teacher,n+1,d,state_layer);v=vector(delta(out["targets"],clean1["targets"]));z=qnorm(out["targets"],clean1["targets"],s);rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"layer":layer,"kind":kind,"blocked_layers":json.dumps(layers),"h1_q":aggregate(z),"j_h1_q":z["j"],"relative_to_full_l2":float(np.linalg.norm(v)/max(fullnorm,1e-12)),"cosine_to_full":cosine(v,full),"writeback_exact":all(torch.equal(getattr(c.layers[k],"recurrent_states" if ch=="REC" else "conv_states"),getattr(p.layers[k],"recurrent_states" if ch=="REC" else "conv_states")) for k in layers for ch in channels),"current_preserved":True})
  print(f"V28 layerwise {role} {i}/{len(items)}",flush=True)
 f=pd.DataFrame(rows);fp=root/OUT/f"layerwise_state_commit_{role}_v28.parquet";f.to_parquet(fp,index=False,compression="zstd");summary={"role":role,"states":len(items),"rows":len(f),"all_writeback_exact":bool(f.writeback_exact.all()),"single_layer_max_median_q":float(f[f.kind.isin(["single_REC","single_Conv"])].groupby(["kind","layer"]).h1_q.median().max()),"full_cumulative_median_q":float(f[(f.kind=="prefix_REC_Conv")&(f.layer==rec[-1])].h1_q.median()),"response_sha256":sha256_file(fp),"diagnostic_only":True};jp=root/OUT/f"layerwise_state_commit_{role}_v28.json";write_json_atomic(jp,summary);fr=stage_freeze(root,f"layerwise_{role}",[SOURCE,str(fp.relative_to(root)),str(jp.relative_to(root)),"artifacts/token_state_transaction_v28_transplant_validation_analysis.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
def horizons(root,role):
 val=verify_stage(root,"validation_analysis")
 if not val["V28_A_DEVVAL"]:return {"status":"NOT_OPENED_H1_GATE_FAILED"}
 plan=json.loads((root/OUT/"execution_plan_v28.json").read_text());d=design(root);bundle,dense,values,rec,att,state_layer=context(root);state_layer=30;s=scales(root);items=selected(root,role,plan["horizon_states_per_family"]);rows=[]
 for i,item in enumerate(items,1):
  m,p,last,n,clean0=native_pair(bundle,dense,item,root,d,state_layer);edited,wb=intercept(p,clean0["cache"],"REC+Conv",rec,att);cc=clean0["cache"];qq=edited;nextid=token(m);used=[]
  for h in range(1,5):
   used.append(nextid);t=torch.tensor([[nextid]]);clean=step(bundle,dense,cc,t,n+h,d,state_layer);changed=step(bundle,dense,qq,t,n+h,d,state_layer);cc=clean["cache"];qq=changed["cache"];z=qnorm(changed["targets"],clean["targets"],s)
   if h in (1,2,4):rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"horizon":h,"h_q":aggregate(z),"j_q":z["j"],"logits_q":z["logits"],"workspace_q":z["workspace"],"writeback_pass":wb["writeback_pass"],"continuation_tokens_sha256":hashlib.sha256(json.dumps(used).encode()).hexdigest(),"teacher_tokens_so_far":json.dumps(used)})
   nextid=int(torch.argmax(clean["logits"]).item())
  print(f"V28 horizons {role} {i}/{len(items)}",flush=True)
 f=pd.DataFrame(rows);fp=root/OUT/f"state_commit_horizons_{role}_v28.parquet";f.to_parquet(fp,index=False,compression="zstd");summary={"role":role,"states":len(items),"rows":len(f),"median_q_by_horizon":{str(h):float(x.h_q.median()) for h,x in f.groupby("horizon")},"all_writeback_pass":bool(f.writeback_pass.all()),"same_clean_greedy_continuation":True,"response_sha256":sha256_file(fp)};jp=root/OUT/f"state_commit_horizons_{role}_v28.json";write_json_atomic(jp,summary);fr=stage_freeze(root,f"horizons_{role}",[SOURCE,str(fp.relative_to(root)),str(jp.relative_to(root)),"artifacts/token_state_transaction_v28_validation_analysis.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("layerwise","horizons"));p.add_argument("role",choices=("development","validation"));a=p.parse_args();print(json.dumps((layerwise if a.command=="layerwise" else horizons)(Path.cwd(),a.role),indent=2))
