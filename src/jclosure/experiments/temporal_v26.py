"""Prospective h0 bank construction and teacher-forced V26 temporal rollouts."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.protocol_v26 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
from jclosure.recorder import ActivationRecorder
from jclosure.runtime_v3_1 import encode_direct_prompt

SOURCE="src/jclosure/experiments/temporal_v26.py";OUT=Path("results/v26/processed")
TARGETS=("j","logits","semantic","workspace","broad_vocabulary","late_residual")

def _design(root):return json.loads((root/OUT/"design_v26.json").read_text())
def _metadata(root,item):
 p=root/f"results/v18/processed/crossed_state_{item['source_role']}_{item['family']}_v18.parquet";frame=pd.read_parquet(p,filters=[[('base_trial_id','==',item['base_trial_id'])]])
 if len(frame)!=1:raise RuntimeError(f"metadata missing {item['base_trial_id']}")
 return frame.iloc[0].to_dict()
def _context(root):
 bundle,dense,metadata,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
 return bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count
def _sum_rows(*rows):return {k:sum((r[k] for r in rows),torch.zeros_like(rows[0][k])) for k in rows[0]}
def _scale_row(row,scale):return {k:v*float(scale) for k,v in row.items()}
def _construction_row(values,d,construction):
 rows=[actions._row(values["directions"],d["action_specs"][aid],float(d["action_alphas"][aid]),1) for aid in construction["components"]]
 return _scale_row(_sum_rows(*rows),float(construction["sign"])*float(construction["scale"]))
def _single_action_row(values,d,aid,scale=1.0):return _scale_row(actions._row(values["directions"],d["action_specs"][aid],float(d["action_alphas"][aid]),1),scale)

def _targets(bundle,dense,recorder,logits,d,state_layer):
 target=d["target_bundle"];device=logits.device;main=30
 sj=torch.as_tensor(target["selected_j"],device=device);sl=torch.as_tensor(target["selected_logits"],device=device);bv=torch.as_tensor(target["broad_vocabulary_indices"],device=device);brs=torch.as_tensor(target["broad_vocabulary_sign"],device=device,dtype=torch.float32);ri=torch.as_tensor(target["late_residual_indices"],device=device);rs=torch.as_tensor(target["late_residual_sign"],device=device,dtype=torch.float32)
 hidden=recorder.activations[state_layer][0,-1].float();j=dense.dense_state(hidden,state_layer)[sj]
 workspace=torch.cat([recorder.activations[layer][0,-1].float()[:target["workspace_per_layer"]] for layer in target["workspace_layers"]])
 residual=recorder.activations[main][0,-1].float()[ri]*rs
 return {"j":j.cpu().numpy().astype(np.float32),"logits":logits[sl].cpu().numpy().astype(np.float32),"semantic":torch.log_softmax(logits,dim=-1)[sl].cpu().numpy().astype(np.float32),"workspace":workspace.cpu().numpy().astype(np.float32),"broad_vocabulary":(logits[bv]*brs).cpu().numpy().astype(np.float32),"late_residual":residual.cpu().numpy().astype(np.float32)}

@torch.no_grad()
def _prefill(bundle,dense,prompt,measured,d,state_layer):
 ids=encode_direct_prompt(bundle,prompt);layers=sorted(set(d["target_bundle"]["workspace_layers"]+[state_layer,30]))
 with ActivationRecorder(bundle.layers,at=layers,clone=True,detach=True) as recorder:output=bundle.hf_model(input_ids=ids,use_cache=True)
 return {"cache":clone_hybrid_cache(output.past_key_values),"logits":output.logits[0,-1].float(),"prompt_length":int(ids.shape[1]),"targets":_targets(bundle,dense,recorder,output.logits[0,-1].float(),d,state_layer)}

def _channel_delta(a,b,rec,att):
 out={"REC":0.0,"Conv":0.0,"KV":0.0}
 for layer in rec:
  out["REC"]+=float(torch.sum((a.layers[layer].recurrent_states.float()-b.layers[layer].recurrent_states.float())**2))
  out["Conv"]+=float(torch.sum((a.layers[layer].conv_states.float()-b.layers[layer].conv_states.float())**2))
 for layer in att:
  for name in ("keys","values"):
   x=getattr(a.layers[layer],name).float();y=getattr(b.layers[layer],name).float();n=min(x.shape[-2],y.shape[-2]);out["KV"]+=float(torch.sum((x[...,:n,:]-y[...,:n,:])**2))
 return {k:math.sqrt(max(v,0.0)) for k,v in out.items()}

@torch.no_grad()
def _paired(bundle,dense,base_cache,edited_cache,tokens,count,prompt_length,d,state_layer,rec,att,wanted):
 device=next(bundle.hf_model.parameters()).device;clean=clone_hybrid_cache(base_cache);edited=clone_hybrid_cache(edited_cache);layers=sorted(set(d["target_bundle"]["workspace_layers"]+[state_layer,30]));result={}
 for step in range(count):
  token=int(tokens[step]);kwargs={"input_ids":torch.tensor([[token]],device=device),"attention_mask":torch.ones((1,prompt_length+step+1),device=device,dtype=torch.long),"use_cache":True}
  with ActivationRecorder(bundle.layers,at=layers,clone=False,detach=True) as rc:oc=bundle.hf_model(past_key_values=clean,**kwargs)
  clean=oc.past_key_values
  with ActivationRecorder(bundle.layers,at=layers,clone=False,detach=True) as re:oe=bundle.hf_model(past_key_values=edited,**kwargs)
  edited=oe.past_key_values;h=step+1
  if h not in wanted:continue
  tc=_targets(bundle,dense,rc,oc.logits[0,-1].float(),d,30);te=_targets(bundle,dense,re,oe.logits[0,-1].float(),d,30);delta={k:(te[k]-tc[k]).astype(np.float32) for k in TARGETS};result[h]={"delta":delta,"channels":_channel_delta(clean,edited,rec,att)}
 return result

def _vector(delta):return np.concatenate([delta[k] for k in TARGETS]).astype(np.float32)
def _row_metrics(base,family,role,candidate,channel,kind,h,delta,channels):
 row={"base_trial_id":base,"family":family,"role":role,"candidate_id":candidate,"channel":channel,"control_type":kind,"horizon":int(h),**{f"{k}_norm":float(np.linalg.norm(delta[k])) for k in TARGETS},**{f"persistent_{k}_norm":v for k,v in channels.items()}}
 row["aggregate_raw_norm"]=float(np.linalg.norm(_vector(delta)));return row

def floor(root:Path)->dict:
 verify_stage(root,"design");cfg=verify(root)["config"];d=_design(root);bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root);rows=[]
 chosen=[]
 for family in d["families"]:chosen += [x for x in d["development"] if x["family"]==family][:cfg["state_panels"]["numerical_floor_per_family"]]
 for i,item in enumerate(chosen,1):
  m=_metadata(root,item);a=_prefill(bundle,dense,str(m["prompt"]),measured,d,state_layer);b=_prefill(bundle,dense,str(m["prompt"]),measured,d,state_layer)
  row={"base_trial_id":item["base_trial_id"],"family":item["family"]}
  for k in TARGETS:row[f"{k}_replay_norm"]=float(np.linalg.norm(a["targets"][k]-b["targets"][k]))
  rows.append(row);print(f"V26 floor {i}/{len(chosen)}",flush=True)
 frame=pd.DataFrame(rows);path=root/OUT/"current_readout_numerical_floor_v26.parquet";frame.to_parquet(path,index=False,compression="zstd")
 summary={"state_count":len(frame),"targets":{k:{"median":float(frame[f'{k}_replay_norm'].median()),"p95":float(frame[f'{k}_replay_norm'].quantile(.95)),"p99":float(frame[f'{k}_replay_norm'].quantile(.99)),"safety_threshold":float(max(frame[f'{k}_replay_norm'].quantile(.99)*cfg["current_tiers"]["numerical_floor_safety_factor"],1e-12))} for k in TARGETS},"all_targets_classifiable":True,"historical_final_opened":False,"v26_independent_final_opened":False}
 target=root/OUT/"current_readout_numerical_floor_v26.json";write_json_atomic(target,summary);frozen=stage_freeze(root,"numerical_floor",[SOURCE,str(path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_design.freeze.json"],{"floor_sha256":sha256_file(path),**summary});return {"freeze_digest":frozen["freeze_digest"],**summary}

def bank(root:Path)->dict:
 verify_stage(root,"numerical_floor");cfg=verify(root)["config"];d=_design(root);bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root);rows=[]
 for role in ("development","validation"):
  family_seen={f:0 for f in d["families"]}
  for number,item in enumerate(d[role],1):
   m=_metadata(root,item);pref=_prefill(bundle,dense,str(m["prompt"]),measured,d,state_layer);p0=pref["cache"];family_seen[item["family"]]+=1
   for c in d["candidate_constructions"]:
    if c["kind"]=="scaling_diagnostic" and family_seen[item["family"]]>cfg["actuation"]["scaling_subset_per_family"]:continue
    requested=_construction_row(values,d,c);edited=v19.apply(p0,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");rb=_readback(p0,edited,requested,rec,att);reliable=bool(rb["realized_state_cosine"] is not None and rb["realized_state_cosine"]>=cfg["actuation"]["reliability_cosine_min"] and rb["realized_state_gain"] is not None and cfg["actuation"]["reliability_gain_min"]<=rb["realized_state_gain"]<=cfg["actuation"]["reliability_gain_max"])
    identity={"base_trial_id":item["base_trial_id"],"candidate":c};phash=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"role":role,"candidate_id":c["candidate_id"],"channel":c["channel"],"kind":c["kind"],"components":json.dumps(c["components"]),"sign":c["sign"],"scale":c["scale"],"perturbation_hash":phash,"j_effect_h0":0.0,"logits_effect_h0":0.0,"semantic_effect_h0":0.0,"workspace_effect_h0":0.0,"broad_vocabulary_effect_h0":0.0,"late_residual_effect_h0":0.0,"aggregate_current_effect":0.0,"tier_same_j":True,"tier_current_readout_distal":reliable,"tier_strict_current_silent":reliable,"reliable":reliable,"requested_state_norm":rb["requested_state_norm"],"realized_state_norm":rb["realized_state_norm"],"realized_state_cosine":rb["realized_state_cosine"],"realized_state_gain":rb["realized_state_gain"],"channel_survival":json.dumps(rb["channel_survival"],sort_keys=True)})
   print(f"V26 bank {role} {number}/{len(d[role])}",flush=True)
 frame=pd.DataFrame(rows);path=root/OUT/"current_distal_bank_v26.parquet";frame.to_parquet(path,index=False,compression="zstd");summary={"rows":len(frame),"states":int(frame.base_trial_id.nunique()),"reliable_rows":int(frame.reliable.sum()),"strict_current_silent_rows":int(frame.tier_strict_current_silent.sum()),"by_role":frame.groupby("role").size().to_dict(),"by_channel":frame.groupby("channel").size().to_dict(),"CURRENT_DISTAL_BANK_HASH":sha256_file(path),"selection_information":"h0_only","future_responses_observed_before_freeze":0,"historical_final_opened":False,"v26_independent_final_opened":False};target=root/OUT/"current_distal_bank_v26.json";write_json_atomic(target,summary);frozen=stage_freeze(root,"current_distal_bank",[SOURCE,str(path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_numerical_floor.freeze.json"],summary);return {"freeze_digest":frozen["freeze_digest"],**summary}

def rollout(root:Path,phase:str)->dict:
 verify_stage(root,"current_distal_bank");cfg=verify(root)["config"];d=_design(root);bankf=pd.read_parquet(root/OUT/"current_distal_bank_v26.parquet");bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 if phase=="early":roles=("development","validation");horizons=cfg["horizons"]["early"];stage="early_rollout";prerequisite="artifacts/temporal_readout_potency_v26_current_distal_bank.freeze.json"
 elif phase=="extension":
  opening=verify_stage(root,"extension_opening")
  if not opening["extension_opened"]:raise RuntimeError("V26 h4/h8 extension not authorized")
  roles=("development","validation");horizons=cfg["horizons"]["extension"];stage="extension_rollout";prerequisite="artifacts/temporal_readout_potency_v26_extension_opening.freeze.json"
 elif phase=="final":
  opening=verify_stage(root,"final_opening")
  if not opening["final_opened"]:raise RuntimeError("V26 independent final not authorized")
  roles=("independent_final",);horizons=[cfg["finalist"]["horizon"]];stage="final_rollout";prerequisite="artifacts/temporal_readout_potency_v26_final_opening.freeze.json"
 else:raise ValueError(phase)
 rows=[];vectors=[];maxh=max(horizons)
 for role in roles:
  items=d[role]
  family_seen={f:0 for f in d["families"]}
  for number,item in enumerate(items,1):
   family_seen[item["family"]]+=1
   m=_metadata(root,item);tokens=[int(x) for x in m["teacher_tokens_h8_or_h1"]][:maxh]
   if len(tokens)<maxh:raise RuntimeError(f"short teacher continuation {item['base_trial_id']}")
   pref=_prefill(bundle,dense,str(m["prompt"]),measured,d,state_layer);p0=pref["cache"]
   if role=="independent_final":c=next(x for x in d["candidate_constructions"] if x["candidate_id"]==cfg["finalist"]["candidate_id"]);candidate_rows=[c]
   else:
    sub=bankf[(bankf.role==role)&(bankf.base_trial_id==item["base_trial_id"])&bankf.reliable]
    candidate_rows=[{"candidate_id":x.candidate_id,"channel":x.channel,"kind":x.kind,"components":json.loads(x.components),"sign":int(x.sign),"scale":float(x.scale)} for x in sub.itertuples()]
   controls=[]
   controls.append(("ordinary_reference","reference","ordinary_reference",_single_action_row(values,d,d["reference_action_id"])))
   controls.append(("same_norm_random","random","same_norm_random",_single_action_row(values,d,cfg["actuation"]["random_control_action_id"])))
   if family_seen[item["family"]]<=1:controls.append(("numerical_scale","numerical","numerical_scale",_single_action_row(values,d,d["reference_action_id"],1e-4)))
   branches=[]
   for c in candidate_rows:branches.append((c["candidate_id"],c["channel"],c.get("kind","primary"),_construction_row(values,d,c)))
   branches+=controls
   for candidate,channel,kind,requested in branches:
    edited=v19.apply(p0,1.0,requested,rec,att,"native_fp32_add_bf16_writeback");out=_paired(bundle,dense,p0,edited,tokens,maxh,pref["prompt_length"],d,state_layer,rec,att,set(horizons))
    for h in horizons:
     rows.append(_row_metrics(item["base_trial_id"],item["family"],role,candidate,channel,kind,h,out[h]["delta"],out[h]["channels"]));vectors.append(_vector(out[h]["delta"]))
   print(f"V26 {phase} {role} {number}/{len(items)}",flush=True)
 frame=pd.DataFrame(rows);path=root/OUT/f"temporal_rollout_{phase}_v26.parquet";frame.to_parquet(path,index=False,compression="zstd");vpath=root/OUT/f"temporal_vectors_{phase}_v26.npz";np.savez_compressed(vpath,vectors=np.stack(vectors).astype(np.float32),base_trial_id=frame.base_trial_id.to_numpy(str),candidate_id=frame.candidate_id.to_numpy(str),role=frame.role.to_numpy(str),family=frame.family.to_numpy(str),horizon=frame.horizon.to_numpy(np.int16),control_type=frame.control_type.to_numpy(str))
 summary={"phase":phase,"roles":list(roles),"horizons":horizons,"rows":len(frame),"states":int(frame.base_trial_id.nunique()),"rollout_sha256":sha256_file(path),"vectors_sha256":sha256_file(vpath),"teacher_forcing":"identical frozen clean-greedy continuation tokens","future_based_bank_selection":False,"historical_final_opened":False,"v26_independent_final_opened":role=="independent_final"};target=root/OUT/f"temporal_rollout_{phase}_v26.json";write_json_atomic(target,summary);frozen=stage_freeze(root,stage,[SOURCE,str(path.relative_to(root)),str(vpath.relative_to(root)),str(target.relative_to(root)),prerequisite],summary);return {"freeze_digest":frozen["freeze_digest"],**summary}

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("floor","bank","early","extension","final"));a=p.parse_args();root=Path.cwd();result=floor(root) if a.command=="floor" else bank(root) if a.command=="bank" else rollout(root,a.command);print(json.dumps(result,indent=2))
