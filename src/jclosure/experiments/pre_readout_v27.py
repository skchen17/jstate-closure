"""V27A true pre-readout bank, h1 response, timing comparison and final confirmation."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.protocol_v27 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
from jclosure.recorder import ActivationRecorder
from jclosure.runtime_v3_1 import encode_direct_prompt

SOURCE="src/jclosure/experiments/pre_readout_v27.py";OUT=Path("results/v27/processed")
TARGETS=("j","logits","semantic","workspace","broad_vocabulary","late_residual")
PRIMARY=("j","logits","semantic","workspace","late_residual")
def design(root):return json.loads((root/OUT/"design_v27.json").read_text())
def metadata(root,item):
 p=root/f"results/v18/processed/crossed_state_{item['source_role']}_{item['family']}_v18.parquet";f=pd.read_parquet(p,filters=[[('base_trial_id','==',item['base_trial_id'])]])
 if len(f)!=1:raise RuntimeError(f"metadata missing {item['base_trial_id']}")
 return f.iloc[0].to_dict()
def context(root):
 bundle,dense,metadata0,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
 return bundle,dense,values,rec,att,state_layer
def sum_rows(*rows):return {k:sum((r[k] for r in rows),torch.zeros_like(rows[0][k])) for k in rows[0]}
def scale_row(row,scale):return {k:v*float(scale) for k,v in row.items()}
def construction(values,d,c):
 rows=[actions._row(values["directions"],d["action_specs"][aid],float(d["action_alphas"][aid]),1) for aid in c["components"]]
 return scale_row(sum_rows(*rows),float(c["sign"])*float(c["scale"]))
def single(values,d,aid):return actions._row(values["directions"],d["action_specs"][aid],float(d["action_alphas"][aid]),1)
def targets(bundle,dense,recorder,logits,d,state_layer):
 t=d["target_bundle"];dev=logits.device;sj=torch.as_tensor(t["selected_j"],device=dev);sl=torch.as_tensor(t["selected_logits"],device=dev);bv=torch.as_tensor(t["broad_vocabulary_indices"],device=dev);bs=torch.as_tensor(t["broad_vocabulary_sign"],device=dev,dtype=torch.float32);ri=torch.as_tensor(t["late_residual_indices"],device=dev);rs=torch.as_tensor(t["late_residual_sign"],device=dev,dtype=torch.float32)
 hidden=recorder.activations[state_layer][0,-1].float();j=dense.dense_state(hidden,state_layer)[sj];workspace=torch.cat([recorder.activations[x][0,-1].float()[:t["workspace_per_layer"]] for x in t["workspace_layers"]]);residual=recorder.activations[30][0,-1].float()[ri]*rs
 return {"j":j.cpu().numpy().astype(np.float32),"logits":logits[sl].cpu().numpy().astype(np.float32),"semantic":torch.log_softmax(logits,dim=-1)[sl].cpu().numpy().astype(np.float32),"workspace":workspace.cpu().numpy().astype(np.float32),"broad_vocabulary":(logits[bv]*bs).cpu().numpy().astype(np.float32),"late_residual":residual.cpu().numpy().astype(np.float32)}
@torch.no_grad()
def prefix(bundle,prompt):
 ids=encode_direct_prompt(bundle,prompt)
 if ids.shape[1]<2:raise RuntimeError("prompt too short for pre-readout split")
 out=bundle.hf_model(input_ids=ids[:,:-1],use_cache=True)
 return clone_hybrid_cache(out.past_key_values),ids[:,-1:],int(ids.shape[1])
@torch.no_grad()
def step(bundle,dense,cache,token,total_length,d,state_layer,layers=None):
 dev=next(bundle.hf_model.parameters()).device;at=sorted(set(layers or (d["target_bundle"]["workspace_layers"]+[state_layer,30])))
 kwargs={"input_ids":token.to(dev),"attention_mask":torch.ones((1,total_length),device=dev,dtype=torch.long),"use_cache":True}
 with ActivationRecorder(bundle.layers,at=at,clone=True,detach=True) as r:o=bundle.hf_model(past_key_values=clone_hybrid_cache(cache),**kwargs)
 return {"cache":clone_hybrid_cache(o.past_key_values),"logits":o.logits[0,-1].float(),"targets":targets(bundle,dense,r,o.logits[0,-1].float(),d,state_layer),"activations":{x:r.activations[x][0,-1].float().cpu().numpy().astype(np.float32) for x in at}}
def delta(a,b):return {k:(a[k]-b[k]).astype(np.float32) for k in TARGETS}
def vector(x):return np.concatenate([x[k] for k in TARGETS]).astype(np.float32)
def norms(x):return {k:float(np.linalg.norm(x[k])) for k in TARGETS}
def scales(root):return json.loads((root/"results/v26/processed/v26_devval_analysis.json").read_text())["reference_scales"]["1"]
def normalized(n,s):return {k:n[k]/max(float(s[k]),1e-12) for k in TARGETS}
def cosine(a,b):
 na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));return float(np.dot(a,b)/(na*nb)) if na*nb else None
def phash(item,c):return hashlib.sha256(json.dumps({"base_trial_id":item["base_trial_id"],"candidate":c,"boundary":"pre_readout"},sort_keys=True,separators=(",",":")).encode()).hexdigest()

def boundary(root:Path):
 verify_stage(root,"design");cfg=verify(root)["config"];d=design(root);bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[]
 ref=single(values,d,cfg["actuation"]["reference_action_id"])
 for i,item in enumerate(d["calibration"],1):
  m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));a=step(bundle,dense,p,last,n,d,state_layer);b=step(bundle,dense,p,last,n,d,state_layer);edited=v19.apply(p,1.0,ref,rec,att,"native_fp32_add_bf16_writeback");q=step(bundle,dense,edited,last,n,d,state_layer)
  replay=norms(delta(a["targets"],b["targets"]));influence=normalized(norms(delta(q["targets"],a["targets"])),s);rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],**{f"{k}_replay_norm":replay[k] for k in TARGETS},**{f"{k}_reference_normalized":influence[k] for k in TARGETS}});print(f"V27 boundary {i}/{len(d['calibration'])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_boundary_audit_v27.parquet";f.to_parquet(path,index=False,compression="zstd")
 summary={"boundary":d["boundary"],"states":len(f),"persistent_fields":{"REC":"recurrent_states","Conv":"conv_states","KV":["keys","values"]},"remaining_computation":"entire final prompt-token forward, all 32 blocks, state/workspace J, final norm, unembedding and logits","J_downstream":True,"logits_downstream":True,"persistent_update_already_happened_for_prefix":True,"fields_writable":True,"current_output_can_change_in_principle":bool(max(float(f[f'{k}_reference_normalized'].median()) for k in PRIMARY)>0.05),"replay_floor":{k:{"p99":float(f[f'{k}_replay_norm'].quantile(.99)),"normalized_safety_threshold":max(float(f[f'{k}_replay_norm'].quantile(.99))*cfg["boundary"]["replay_safety_factor"]/max(s[k],1e-12),1e-12)} for k in TARGETS},"reference_current_effect_medians":{k:float(f[f'{k}_reference_normalized'].median()) for k in TARGETS},"structurally_guaranteed_invariance":False}
 j=root/OUT/"pre_readout_boundary_audit_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"boundary_audit",[SOURCE,str(path.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_design.freeze.json"],{"boundary_hash":d["boundary_hash"],"audit_sha256":sha256_file(path),**summary});return {"freeze_digest":fr["freeze_digest"],**summary}

def bank(root:Path):
 audit=verify_stage(root,"boundary_audit");cfg=verify(root)["config"];d=design(root);bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[]
 floors={k:audit["replay_floor"][k]["normalized_safety_threshold"] for k in TARGETS}
 for role in ("development","validation"):
  for i,item in enumerate(d[role],1):
   m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean=step(bundle,dense,p,last,n,d,state_layer)
   for c in d["candidate_constructions"]:
    requested=construction(values,d,c);edited=v19.apply(p,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");rb=_readback(p,edited,requested,rec,att);reliable=bool(rb["realized_state_cosine"] is not None and rb["realized_state_cosine"]>=cfg["actuation"]["reliability_cosine_min"] and rb["realized_state_gain"] is not None and cfg["actuation"]["reliability_gain_min"]<=rb["realized_state_gain"]<=cfg["actuation"]["reliability_gain_max"]);q=step(bundle,dense,edited,last,n,d,state_layer);raw=norms(delta(q["targets"],clean["targets"]));z=normalized(raw,s);agg=float(np.mean([z[k] for k in PRIMARY]));silent=reliable and agg<=cfg["silence"]["aggregate_max"] and all(z[k]<=max(cfg["silence"]["individual_max"],floors[k]) for k in PRIMARY);distal=reliable and agg<=cfg["silence"]["distal_aggregate_max"]
    rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"candidate_id":c["candidate_id"],"channel":c["channel"],"kind":c["kind"],"sign":c["sign"],"scale":c["scale"],"perturbation_hash":phash(item,c),**{f"{k}_h0_norm":raw[k] for k in TARGETS},**{f"{k}_h0_q":z[k] for k in TARGETS},"aggregate_h0_q":agg,"PRE_READOUT_DISTAL":distal,"PRE_READOUT_SILENT":silent,"reliable":reliable,"realized_state_cosine":rb["realized_state_cosine"],"realized_state_gain":rb["realized_state_gain"]})
   print(f"V27 bank {role} {i}/{len(d[role])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_silent_bank_v27.parquet";f.to_parquet(path,index=False,compression="zstd");summary={"rows":len(f),"states":int(f.base_trial_id.nunique()),"reliable_rows":int(f.reliable.sum()),"silent_rows":int(f.PRE_READOUT_SILENT.sum()),"distal_rows":int(f.PRE_READOUT_DISTAL.sum()),"by_role":{r:{"rows":int(len(x)),"silent":int(x.PRE_READOUT_SILENT.sum()),"distal":int(x.PRE_READOUT_DISTAL.sum())} for r,x in f.groupby("role")},"PRE_READOUT_SILENT_BANK_HASH":sha256_file(path),"selection_information":"h0_only","future_responses_observed_before_freeze":0,"thresholds":cfg["silence"],"historical_final_opened":False}
 j=root/OUT/"pre_readout_silent_bank_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"silent_bank",[SOURCE,str(path.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_boundary_audit.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}

def future(root:Path):
 verify_stage(root,"silent_bank");d=design(root);bankf=pd.read_parquet(root/OUT/"pre_readout_silent_bank_v27.parquet");bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);rows=[];vecs=[];postvecs=[]
 for role in ("development","validation"):
  for i,item in enumerate(d[role],1):
   accepted=bankf[(bankf.role==role)&(bankf.base_trial_id==item["base_trial_id"])&(bankf.PRE_READOUT_SILENT)]
   if accepted.empty:continue
   m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean0=step(bundle,dense,p,last,n,d,state_layer);teacher=torch.tensor([[int(m["teacher_tokens_h8_or_h1"][0])]],device=last.device);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer)
   for br in accepted.itertuples():
    c=next(x for x in d["candidate_constructions"] if x["candidate_id"]==br.candidate_id);requested=construction(values,d,c);qprefix=v19.apply(p,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");q0=step(bundle,dense,qprefix,last,n,d,state_layer);q1=step(bundle,dense,q0["cache"],teacher,n+1,d,state_layer);post0=v19.apply(clean0["cache"],1.0,requested,rec,att,"native_fp32_add_bf16_writeback");post1=step(bundle,dense,post0,teacher,n+1,d,state_layer);dv=delta(q1["targets"],clean1["targets"]);pv=delta(post1["targets"],clean1["targets"]);raw=norms(dv);z=normalized(raw,s);q=float(np.mean([z[k] for k in TARGETS]));v=vector(dv);w=vector(pv)
    rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"candidate_id":c["candidate_id"],"channel":c["channel"],"kind":c["kind"],"aggregate_h0_q":float(br.aggregate_h0_q),**{f"{k}_h1_norm":raw[k] for k in TARGETS},**{f"{k}_h1_q":z[k] for k in TARGETS},"h1_q":q,"future_potent":q>=.25,"post_h1_q":float(np.mean(list(normalized(norms(pv),s).values()))),"pre_post_cosine":cosine(v,w),"pre_post_magnitude_ratio":float(np.linalg.norm(v)/max(np.linalg.norm(w),1e-12))});vecs.append(v);postvecs.append(w)
   print(f"V27 h1 {role} {i}/{len(d[role])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_h1_v27.parquet";f.to_parquet(path,index=False,compression="zstd");vp=root/OUT/"pre_readout_h1_vectors_v27.npz";np.savez_compressed(vp,vectors=np.stack(vecs).astype(np.float32) if vecs else np.zeros((0,1),np.float32),post_vectors=np.stack(postvecs).astype(np.float32) if postvecs else np.zeros((0,1),np.float32),base_trial_id=f.base_trial_id.to_numpy(str),candidate_id=f.candidate_id.to_numpy(str),role=f.role.to_numpy(str),family=f.family.to_numpy(str))
 summary={"rows":len(f),"states":int(f.base_trial_id.nunique()) if len(f) else 0,"horizon":1,"teacher_forcing":"identical frozen clean continuation token","future_based_bank_selection":False,"bank_hash":sha256_file(root/OUT/"pre_readout_silent_bank_v27.parquet"),"response_sha256":sha256_file(path),"vectors_sha256":sha256_file(vp)};j=root/OUT/"pre_readout_h1_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"h1_response",[SOURCE,str(path.relative_to(root)),str(vp.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_silent_bank.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}

def final(root:Path):
 op=verify_stage(root,"final_opening")
 if not op["final_opened"]:return {"final_opened":False,"reason":"V27A dev/validation gate failed"}
 d=design(root);bundle,dense,values,rec,att,state_layer=context(root);s=scales(root);c=next(x for x in d["candidate_constructions"] if x["candidate_id"]==op["candidate_id"]);rows=[];vecs=[]
 for i,item in enumerate(d["independent_final"],1):
  m=metadata(root,item);p,last,n=prefix(bundle,str(m["prompt"]));clean0=step(bundle,dense,p,last,n,d,state_layer);requested=construction(values,d,c);qprefix=v19.apply(p,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");q0=step(bundle,dense,qprefix,last,n,d,state_layer);h0=normalized(norms(delta(q0["targets"],clean0["targets"])),s);agg0=float(np.mean([h0[k] for k in PRIMARY]));silent=agg0<=verify(root)["config"]["silence"]["aggregate_max"] and all(h0[k]<=verify(root)["config"]["silence"]["individual_max"] for k in PRIMARY);teacher=torch.tensor([[int(m["teacher_tokens_h8_or_h1"][0])]],device=last.device);clean1=step(bundle,dense,clean0["cache"],teacher,n+1,d,state_layer);q1=step(bundle,dense,q0["cache"],teacher,n+1,d,state_layer);dv=delta(q1["targets"],clean1["targets"]);z=normalized(norms(dv),s);q=float(np.mean(list(z.values())));rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"candidate_id":c["candidate_id"],"aggregate_h0_q":agg0,"PRE_READOUT_SILENT":silent,"h1_q":q,"future_potent":silent and q>=.25,**{f"{k}_h0_q":h0[k] for k in TARGETS},**{f"{k}_h1_q":z[k] for k in TARGETS}});vecs.append(vector(dv));print(f"V27 final {i}/{len(d['independent_final'])}",flush=True)
 f=pd.DataFrame(rows);path=root/OUT/"pre_readout_final_v27.parquet";f.to_parquet(path,index=False,compression="zstd");vp=root/OUT/"pre_readout_final_vectors_v27.npz";np.savez_compressed(vp,vectors=np.stack(vecs).astype(np.float32),base_trial_id=f.base_trial_id.to_numpy(str),family=f.family.to_numpy(str));summary={"final_opened":True,"candidate_id":c["candidate_id"],"rows":len(f),"silent_rows":int(f.PRE_READOUT_SILENT.sum()),"potent_rows":int(f.future_potent.sum()),"response_sha256":sha256_file(path),"vectors_sha256":sha256_file(vp),"historical_final_opened":False,"v27_independent_final_opened":True};j=root/OUT/"pre_readout_final_v27.json";write_json_atomic(j,summary);fr=stage_freeze(root,"final_response",[SOURCE,str(path.relative_to(root)),str(vp.relative_to(root)),str(j.relative_to(root)),"artifacts/pre_readout_reentry_v27_final_opening.freeze.json"],summary);return {"freeze_digest":fr["freeze_digest"],**summary}

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("boundary","bank","future","final"));a=p.parse_args();fn={"boundary":boundary,"bank":bank,"future":future,"final":final}[a.command];print(json.dumps(fn(Path.cwd()),indent=2))
