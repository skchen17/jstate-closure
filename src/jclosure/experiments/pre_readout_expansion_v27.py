"""Prospective V27 h0-only expansion with historical V19 same-J directions."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.pre_readout_v27 import OUT,TARGETS,PRIMARY,design,metadata,context,prefix,step,delta,norms,scales,normalized,vector,cosine
from jclosure.protocol_v27 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/pre_readout_expansion_v27.py"
def candidates(root):
 old=json.loads((root/"artifacts/counterfactual_workspace_v19_interventions.freeze.json").read_text());out=[]
 for q in old["q"]:
  if q["calibration_status"]!="RELIABLE":continue
  for sign in (-1,1):
   for multiplier in (.10,.25,.50,1.0):out.append({"candidate_id":f"v19_{q['name']}_{'p' if sign>0 else 'n'}_{str(multiplier).replace('.','p')}","source":"V19_same_J","name":q["name"],"channel":q["channel"],"direction_index":q["direction_index"],"base_alpha":q["alpha"],"sign":sign,"multiplier":multiplier})
 return out
def freeze_candidates(root):
 verify_stage(root,"silent_bank");cs=candidates(root);p=root/OUT/"candidate_expansion_v27.json";value={"candidates":cs,"candidate_count":len(cs),"candidate_hash":hashlib.sha256(json.dumps(cs,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"source_freeze":"artifacts/counterfactual_workspace_v19_interventions.freeze.json","h1_responses_observed_before_freeze":0,"reason":"complete predeclared V27 source A after initial architecture-resolved source B/C bank was empty"};write_json_atomic(p,value);fr=stage_freeze(root,"candidate_expansion",[SOURCE,str(p.relative_to(root)),"artifacts/counterfactual_workspace_v19_interventions.freeze.json","artifacts/pre_readout_reentry_v27_silent_bank.freeze.json"],value);return {"freeze_digest":fr["freeze_digest"],**value}
def row(values,c):
 return v19._q_row(values["directions"],c,float(c["base_alpha"])*float(c["multiplier"]),int(c["sign"]))
def bank(root):
 st=verify_stage(root,"candidate_expansion");cfg=verify(root)["config"];d=design(root);bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[]
 for role in ("development","validation"):
  for i,item in enumerate(d[role],1):
   m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean=step(bundle,dense,p,last,n,d,state_layer)
   for c in st["candidates"]:
    requested=row(values,c);edited=v19.apply(p,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");rb=_readback(p,edited,requested,rec,att);reliable=bool(rb["realized_state_cosine"] is not None and rb["realized_state_cosine"]>=cfg["actuation"]["reliability_cosine_min"] and rb["realized_state_gain"] is not None and cfg["actuation"]["reliability_gain_min"]<=rb["realized_state_gain"]<=cfg["actuation"]["reliability_gain_max"]);q=step(bundle,dense,edited,last,n,d,state_layer);raw=norms(delta(q["targets"],clean["targets"]));z=normalized(raw,s);agg=float(np.mean([z[k] for k in PRIMARY]));silent=reliable and agg<=cfg["silence"]["aggregate_max"] and all(z[k]<=cfg["silence"]["individual_max"] for k in PRIMARY);distal=reliable and agg<=cfg["silence"]["distal_aggregate_max"]
    ident=hashlib.sha256(json.dumps({"base_trial_id":item["base_trial_id"],"candidate":c,"boundary":"pre_readout"},sort_keys=True,separators=(",",":")).encode()).hexdigest();rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"candidate_id":c["candidate_id"],"channel":c["channel"],"source":c["source"],"direction_index":c["direction_index"],"base_alpha":c["base_alpha"],"sign":c["sign"],"multiplier":c["multiplier"],"perturbation_hash":ident,**{f"{k}_h0_norm":raw[k] for k in TARGETS},**{f"{k}_h0_q":z[k] for k in TARGETS},"aggregate_h0_q":agg,"PRE_READOUT_DISTAL":distal,"PRE_READOUT_SILENT":silent,"reliable":reliable,"realized_state_cosine":rb["realized_state_cosine"],"realized_state_gain":rb["realized_state_gain"]})
   print(f"V27 expanded bank {role} {i}/{len(d[role])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_silent_bank_expanded_v27.parquet";f.to_parquet(path,index=False,compression="zstd");summary={"rows":len(f),"states":int(f.base_trial_id.nunique()),"candidate_count":st["candidate_count"],"reliable_rows":int(f.reliable.sum()),"silent_rows":int(f.PRE_READOUT_SILENT.sum()),"distal_rows":int(f.PRE_READOUT_DISTAL.sum()),"by_role":{r:{"rows":int(len(x)),"silent":int(x.PRE_READOUT_SILENT.sum()),"distal":int(x.PRE_READOUT_DISTAL.sum())} for r,x in f.groupby("role")},"PRE_READOUT_SILENT_BANK_HASH":sha256_file(path),"selection_information":"h0_only","future_responses_observed_before_freeze":0,"historical_final_opened":False};j=root/OUT/"pre_readout_silent_bank_expanded_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"silent_bank_expanded",[SOURCE,str(path.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_candidate_expansion.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
def future(root):
 st=verify_stage(root,"candidate_expansion");verify_stage(root,"silent_bank_expanded");d=design(root);bankf=pd.read_parquet(root/OUT/"pre_readout_silent_bank_expanded_v27.parquet");bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[];vecs=[];postvecs=[]
 cmap={x["candidate_id"]:x for x in st["candidates"]}
 for role in ("development","validation"):
  for i,item in enumerate(d[role],1):
   accepted=bankf[(bankf.role==role)&(bankf.base_trial_id==item["base_trial_id"])&(bankf.PRE_READOUT_SILENT)]
   if accepted.empty:continue
   m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean0=step(bundle,dense,p,last,n,d,state_layer);teacher=torch.tensor([[int(m["teacher_tokens_h8_or_h1"][0])]]);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer)
   for br in accepted.itertuples():
    c=cmap[br.candidate_id];requested=row(values,c);qprefix=v19.apply(p,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");q0=step(bundle,dense,qprefix,last,n,d,state_layer);q1=step(bundle,dense,q0["cache"],teacher,n+1,d,state_layer);post0=v19.apply(clean0["cache"],1.0,requested,rec,att,"native_fp32_add_bf16_writeback");post1=step(bundle,dense,post0,teacher,n+1,d,state_layer);dv=delta(q1["targets"],clean1["targets"]);pv=delta(post1["targets"],clean1["targets"]);z=normalized(norms(dv),s);q=float(np.mean(list(z.values())));v=vector(dv);w=vector(pv);rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"candidate_id":c["candidate_id"],"channel":c["channel"],"aggregate_h0_q":float(br.aggregate_h0_q),**{f"{k}_h1_norm":norms(dv)[k] for k in TARGETS},**{f"{k}_h1_q":z[k] for k in TARGETS},"h1_q":q,"future_potent":q>=.25,"post_h1_q":float(np.mean(list(normalized(norms(pv),s).values()))),"pre_post_cosine":cosine(v,w),"pre_post_magnitude_ratio":float(np.linalg.norm(v)/max(np.linalg.norm(w),1e-12))});vecs.append(v);postvecs.append(w)
   print(f"V27 expanded h1 {role} {i}/{len(d[role])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_h1_expanded_v27.parquet";f.to_parquet(path,index=False,compression="zstd");vp=root/OUT/"pre_readout_h1_expanded_vectors_v27.npz";np.savez_compressed(vp,vectors=np.stack(vecs).astype(np.float32) if vecs else np.zeros((0,1),np.float32),post_vectors=np.stack(postvecs).astype(np.float32) if postvecs else np.zeros((0,1),np.float32),base_trial_id=f.base_trial_id.to_numpy(str) if len(f) else np.array([],str),candidate_id=f.candidate_id.to_numpy(str) if len(f) else np.array([],str),role=f.role.to_numpy(str) if len(f) else np.array([],str),family=f.family.to_numpy(str) if len(f) else np.array([],str));summary={"rows":len(f),"states":int(f.base_trial_id.nunique()) if len(f) else 0,"horizon":1,"teacher_forcing":"identical frozen clean continuation token","future_based_bank_selection":False,"bank_hash":sha256_file(root/OUT/"pre_readout_silent_bank_expanded_v27.parquet"),"response_sha256":sha256_file(path),"vectors_sha256":sha256_file(vp)};j=root/OUT/"pre_readout_h1_expanded_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"h1_response_expanded",[SOURCE,str(path.relative_to(root)),str(vp.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_silent_bank_expanded.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("freeze","bank","future"));a=p.parse_args();print(json.dumps({"freeze":freeze_candidates,"bank":bank,"future":future}[a.command](Path.cwd()),indent=2))
