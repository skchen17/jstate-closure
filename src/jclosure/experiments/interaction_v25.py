"""V25 distributed B×C interaction, cancellation, background, and reconstruction experiments."""
from __future__ import annotations
import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.mediation_v24 import _context,_full_capture as _v24_full_capture,_edited_capture as _v24_edited_capture
from jclosure.experiments.layer_response_v24 import _projectors
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/interaction_v25.py";OUT=Path("results/v25/processed");SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def _full_capture(root,bundle,dense,cache,*args,**kwargs):return _v24_full_capture(root,bundle,dense,clone_hybrid_cache(cache),*args,**kwargs)
def _edited_capture(root,bundle,dense,cache,*args,**kwargs):return _v24_edited_capture(root,bundle,dense,clone_hybrid_cache(cache),*args,**kwargs)
def _contrast(a,b):return {k:a[k]-b[k] for k in ("residual","arch_attn","arch_mlp","vocab","t0")}
def _targets(x,scales):
 t=np.asarray(x["t0"],np.float64);res=np.asarray(x["residual"][-1],np.float64);voc=np.asarray(x["vocab"],np.float64)
 return {"J":t[:128],"logits":t[128:160],"semantic":t[160:192],"workspace":t[192:],"T0":t,"late_residual":res,"random_vocab":voc,"broad":np.concatenate([res/scales["residual"][-1],t/scales["t0"],voc/scales["vocab"]])}
def _m(target,b,c):
 i=target-b-c;add=b+c;tn=np.linalg.norm(target);bn=np.linalg.norm(b);cn=np.linalg.norm(c);an=bn+cn
 return {"interaction_ratio":float(np.linalg.norm(i)/max(tn,1e-12)),"interaction_cosine":float(np.dot(i,target)/max(np.linalg.norm(i)*tn,1e-12)),"additive_relative_l2":float(np.linalg.norm(add-target)/max(tn,1e-12)),"additive_cosine":float(np.dot(add,target)/max(np.linalg.norm(add)*tn,1e-12)),"cancellation_index":float((an-tn)/max(an,1e-12)),"component_cosine":float(np.dot(b,c)/max(bn*cn,1e-12)),"B_projection_on_full":float(np.dot(b,target)/max(tn*tn,1e-12)),"C_projection_on_full":float(np.dot(c,target)/max(tn*tn,1e-12))}
def _fidelity(target,pred):
 tn=np.linalg.norm(target);pn=np.linalg.norm(pred);return {"relative_l2":float(np.linalg.norm(pred-target)/max(tn,1e-12)),"cosine":float(np.dot(pred,target)/max(pn*tn,1e-12)),"norm_ratio":float(pn/max(tn,1e-12))}
def _sum_rows(*rows):return {k:sum((r[k] for r in rows),torch.zeros_like(rows[0][k])) for k in rows[0]}

@torch.no_grad()
def run(root:Path,limit:int|None=None)->dict:
 basis_stage=verify_stage(root,"component_basis");cfg=verify(root)["config"];d=json.loads((root/OUT/"design_v25.json").read_text());v24=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text());candidate=json.loads((root/"results/v24/processed/bottleneck_candidate_selection_v24.json").read_text());layer=int(candidate["diagnostic_layer"]);norm_scales=candidate["normalization_scales"]
 design,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root);device=next(bundle.hf_model.parameters()).device;proj=_projectors(root,device);jids=torch.as_tensor(split["selected_j"],device=device);lids=torch.as_tensor(split["selected_logits"],device=device);tscales={x:float(v) for x,v in split["target_scales"].items()}
 with np.load(root/OUT/"distributed_component_bases_v25.npz") as z:bases={k:np.asarray(z[k],np.float32) for k in z.files}
 items=d["interaction_validation_states"][:limit];families=sorted({x["family"] for x in d["interaction_validation_states"]});rows=[];multi=[];recon=[];audits=[];full_raw=[];full_broad=[];interaction_broad=[];clean_j={};source_B={};source_effect={};clean_replay_abs=[];clean_replay_rel=[]
 for number,item in enumerate(items,1):
  base=item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);q=qmap[item["q_name"]];pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback");clean={}
  for state_name,state in (("P0",p0),("Pq",pq)):
   clean[state_name]=_full_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer)
  replay=_full_capture(root,bundle,dense,p0,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);replay_delta=_targets(_contrast(replay,clean["P0"]),norm_scales)["broad"];replay_ref=_targets(clean["P0"],norm_scales)["broad"];clean_replay_abs.append(float(np.max(np.abs(replay_delta))));clean_replay_rel.append(float(np.linalg.norm(replay_delta)/max(np.linalg.norm(replay_ref),1e-12)))
  clean_j[base]=clean["P0"]["t0"][:128].copy()
  for state_name,state in (("P0",p0),("Pq",pq)):
   for ai,action_id in enumerate(d["interaction_action_ids"]):
    action_row=actions._row(values["directions"],d["action_specs"][action_id],float(d["action_alphas"][action_id]),1);edited=v19.apply(state,1.0,action_row,rec,att,"native_fp32_add_bf16_writeback");full=_full_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);delta=full["raw_hidden"]-clean[state_name]["raw_hidden"]
    cb=bases["causal_basis"][:,:24];db=cb@(cb.T@delta);dc=delta-db
    bval=_edited_capture(root,bundle,dense,state,db,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);cval=_edited_capture(root,bundle,dense,state,dc,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
    effects={"full":_targets(_contrast(full,clean[state_name]),norm_scales),"B":_targets(_contrast(bval,clean[state_name]),norm_scales),"C":_targets(_contrast(cval,clean[state_name]),norm_scales)}
    row={"base_trial_id":base,"family":item["family"],"state":state_name,"action_id":action_id,"success":True}
    for target in effects["full"]:row.update({f"{target}_{k}":v for k,v in _m(effects["full"][target],effects["B"][target],effects["C"][target]).items()})
    for control,bkey in (("random","random_basis"),("variance","variance_basis")):
     bb=bases[bkey][:,:24];dv=bb@(bb.T@delta);val=_edited_capture(root,bundle,dense,state,dv,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);row.update({f"CONTROL_{control}_{k}":v for k,v in _fidelity(effects["full"]["broad"],_targets(_contrast(val,clean[state_name]),norm_scales)["broad"]).items()})
    signval=_edited_capture(root,bundle,dense,state,db,-1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);row.update({f"CONTROL_signflip_{k}":v for k,v in _fidelity(effects["full"]["broad"],_targets(_contrast(signval,clean[state_name]),norm_scales)["broad"]).items()})
    wrong=families[(families.index(item["family"])+1)%len(families)];wb=bases[f"family_{wrong}"][:,:24];wd=wb@(wb.T@delta);wrongval=_edited_capture(root,bundle,dense,state,wd,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);row.update({f"CONTROL_wrong_family_{k}":v for k,v in _fidelity(effects["full"]["broad"],_targets(_contrast(wrongval,clean[state_name]),norm_scales)["broad"]).items()});rows.append(row)
    full_raw.append(delta.astype(np.float32));full_broad.append(effects["full"]["broad"].astype(np.float32));interaction_broad.append((effects["full"]["broad"]-effects["B"]["broad"]-effects["C"]["broad"]).astype(np.float32))
    if number<=5 and state_name=="P0" and ai==0:source_B[base]=db.astype(np.float32);source_effect[base]=effects["B"]["broad"].astype(np.float32)
    if number<=5 and ai<4:
     b1=db;b2basis=bases["causal_basis"][:,24:48];b2=b2basis@(b2basis.T@delta);b3=delta-b1-b2;parts={"B1":b1,"B2":b2,"B3":b3,"B1_B2":b1+b2,"B1_B3":b1+b3,"B2_B3":b2+b3}
     values_out={}
     for name,vec in parts.items():
      val=_edited_capture(root,bundle,dense,state,vec,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);values_out[name]=_targets(_contrast(val,clean[state_name]),norm_scales)["broad"]
     triple=effects["full"]["broad"]-values_out["B1_B2"]-values_out["B1_B3"]-values_out["B2_B3"]+values_out["B1"]+values_out["B2"]+values_out["B3"]
     multi.append({"base_trial_id":base,"family":item["family"],"state":state_name,"action_id":action_id,"triple_interaction_ratio":float(np.linalg.norm(triple)/max(np.linalg.norm(effects["full"]["broad"]),1e-12)),**{f"{name}_retained_ratio":float(np.linalg.norm(x)/max(np.linalg.norm(effects["full"]["broad"]),1e-12)) for name,x in values_out.items()}})
  # Same-J distributed reconstruction from P0 background.
  qdelta=clean["Pq"]["raw_hidden"]-clean["P0"]["raw_hidden"];te=_targets(_contrast(clean["Pq"],clean["P0"]),norm_scales)["broad"]
  for method,key in (("causal","causal_basis"),("variance","variance_basis"),("random","random_basis")):
   for dim in cfg["realization_dimensions"]:
    bb=bases[key][:,:dim];dv=bb@(bb.T@qdelta);val=_edited_capture(root,bundle,dense,p0,dv,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);recon.append({"base_trial_id":base,"family":item["family"],"method":method,"dimension":dim,**_fidelity(te,_targets(_contrast(val,clean["P0"]),norm_scales)["broad"])})
  recon.append({"base_trial_id":base,"family":item["family"],"method":"full","dimension":2560,**_fidelity(te,te)})
  print(f"V25 interaction {number}/{len(items)}",flush=True)
 frame=pd.DataFrame(rows);branch_path=root/OUT/"distributed_interaction_branches_v25.parquet";frame.to_parquet(branch_path,index=False,compression="zstd");multif=pd.DataFrame(multi);multi_path=root/OUT/"three_component_interactions_v25.parquet";multif.to_parquet(multi_path,index=False,compression="zstd");reconf=pd.DataFrame(recon);recon_path=root/OUT/"same_j_reconstruction_v25.parquet";reconf.to_parquet(recon_path,index=False,compression="zstd")
 vectors_path=root/OUT/"interaction_vectors_v25.npz";np.savez_compressed(vectors_path,full_raw=np.stack(full_raw),full_broad=np.stack(full_broad),interaction_broad=np.stack(interaction_broad))

 # Background transfer: native, same-J, nearest-J, same-family, across-family.
 background=[];item_map={x["base_trial_id"]:x for x in items};sources=[x for x in items if x["base_trial_id"] in source_B]
 for source in sources:
  sid=source["base_trial_id"];others=[x for x in items if x["base_trial_id"]!=sid];nearest=min(others,key=lambda x:np.linalg.norm(clean_j[sid]-clean_j[x["base_trial_id"]]));samefam=next((x for x in others if x["family"]==source["family"]),nearest);different=next(x for x in others if x["family"]!=source["family"])
  targets=[("native",source,"P0"),("same_J",source,"Pq"),("nearest_J",nearest,"P0"),("same_family",samefam,"P0"),("different_family",different,"P0")]
  for category,target_item,state_name in targets:
   base=target_item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);state=p0
   if state_name=="Pq":q=qmap[target_item["q_name"]];state=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
   baseline=_full_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);val=_edited_capture(root,bundle,dense,state,source_B[sid],1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits);effect=_targets(_contrast(val,baseline),norm_scales)["broad"]
   background.append({"source_base_trial_id":sid,"source_family":source["family"],"target_base_trial_id":base,"target_family":target_item["family"],"background":category,**_fidelity(source_effect[sid],effect)})
 bg=pd.DataFrame(background);bgpath=root/OUT/"background_transfer_v25.parquet";bg.to_parquet(bgpath,index=False,compression="zstd")

 # Nonlinear pair interaction by layer and finite action order.
 nonlinear=[];order_rows=[]
 for item in items[:5]:
  base=item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);baseline=_full_capture(root,bundle,dense,p0,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);z=np.load(root/SCRATCH/"finite_validation"/f"layer_{base}.npz")
  for pair in d["nonlinear_action_pairs"]:
   r1=actions._row(values["directions"],d["action_specs"][pair[0]],float(d["action_alphas"][pair[0]]),1);r2=actions._row(values["directions"],d["action_specs"][pair[1]],float(d["action_alphas"][pair[1]]),1);combined=v19.apply(p0,1.0,_sum_rows(r1,r2),rec,att,"native_fp32_add_bf16_writeback");cv=_full_capture(root,bundle,dense,combined,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);idx1=256+v24["heldout_action_ids"].index(pair[0]);idx2=256+v24["heldout_action_ids"].index(pair[1])
   for li,l in enumerate(cfg["layers"]):
    full=np.asarray(cv["residual"][li]-baseline["residual"][li],np.float64);a=np.asarray(z["P0_residual"][idx1,li],np.float64);b=np.asarray(z["P0_residual"][idx2,li],np.float64);inter=full-a-b;nonlinear.append({"base_trial_id":base,"family":item["family"],"pair":"+".join(pair),"layer":l,"interaction_ratio":float(np.linalg.norm(inter)/max(np.linalg.norm(full),1e-12)),"cancellation_index":float((np.linalg.norm(a)+np.linalg.norm(b)-np.linalg.norm(full))/max(np.linalg.norm(a)+np.linalg.norm(b),1e-12))})
   ab=v19.apply(v19.apply(p0,1.0,r1,rec,att,"native_fp32_add_bf16_writeback"),1.0,r2,rec,att,"native_fp32_add_bf16_writeback");ba=v19.apply(v19.apply(p0,1.0,r2,rec,att,"native_fp32_add_bf16_writeback"),1.0,r1,rec,att,"native_fp32_add_bf16_writeback");av=_full_capture(root,bundle,dense,ab,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);bv=_full_capture(root,bundle,dense,ba,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);ae=_targets(_contrast(av,baseline),norm_scales)["broad"];be=_targets(_contrast(bv,baseline),norm_scales)["broad"];order_rows.append({"base_trial_id":base,"family":item["family"],"pair":"+".join(pair),"order_relative_difference":float(np.linalg.norm(ae-be)/max((np.linalg.norm(ae)+np.linalg.norm(be))/2,1e-12))})
  z.close()
 non=pd.DataFrame(nonlinear);nonpath=root/OUT/"layerwise_nonlinear_interaction_v25.parquet";non.to_parquet(nonpath,index=False,compression="zstd");order=pd.DataFrame(order_rows);orderpath=root/OUT/"intervention_order_v25.parquet";order.to_parquet(orderpath,index=False,compression="zstd")

 # REC/Conv/KV factorial panel on one state per family.
 channel_rows=[];train_ids=v24["train_action_ids"];tags=v24["action_channel_tags"];chosen={name:next(x for x in train_ids if tags[x]==name) for name in ("REC","Conv","KV")}
 for item in items[:5]:
  base=item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);baseline=_full_capture(root,bundle,dense,p0,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);arows={name:actions._row(values["directions"],v24["action_specs"][aid],float(v24["action_alphas"][aid]),1) for name,aid in chosen.items()};effects={}
  for name,names in (("REC",["REC"]),("Conv",["Conv"]),("KV",["KV"]),("REC_Conv",["REC","Conv"]),("REC_KV",["REC","KV"]),("Conv_KV",["Conv","KV"]),("ALL",["REC","Conv","KV"])):
   edited=v19.apply(p0,1.0,_sum_rows(*(arows[n] for n in names)),rec,att,"native_fp32_add_bf16_writeback");val=_full_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);effects[name]=_targets(_contrast(val,baseline),norm_scales)["broad"]
  for pair,a,b in (("RECxConv","REC","Conv"),("RECxKV","REC","KV"),("ConvxKV","Conv","KV")):
   joint=effects[a+"_"+b] if a+"_"+b in effects else effects[b+"_"+a];inter=joint-effects[a]-effects[b];channel_rows.append({"base_trial_id":base,"family":item["family"],"interaction":pair,"interaction_ratio":float(np.linalg.norm(inter)/max(np.linalg.norm(joint),1e-12))})
  triple=effects["ALL"]-effects["REC_Conv"]-effects["REC_KV"]-effects["Conv_KV"]+effects["REC"]+effects["Conv"]+effects["KV"];channel_rows.append({"base_trial_id":base,"family":item["family"],"interaction":"RECxConvxKV","interaction_ratio":float(np.linalg.norm(triple)/max(np.linalg.norm(effects["ALL"]),1e-12))})
 ch=pd.DataFrame(channel_rows);chpath=root/OUT/"channel_factorial_interaction_v25.parquet";ch.to_parquet(chpath,index=False,compression="zstd")

 audit=pd.DataFrame(audits);auditpath=root/OUT/"writeback_audit_v25.parquet";audit.to_parquet(auditpath,index=False,compression="zstd")
 expected=len(items)*2*len(d["interaction_action_ids"]);values=frame.broad_interaction_ratio.to_numpy();rng=np.random.default_rng(250030);boots=[float(np.median(rng.choice(values,len(values),replace=True))) for _ in range(1000)];lower=float(np.quantile(boots,.025));family_med=frame.groupby("family").broad_interaction_ratio.median().to_dict();interaction_pass=bool(np.median(values)>=cfg["interaction_gate"]["median_ratio_min"] and lower>cfg["interaction_gate"]["bootstrap_lower_min"] and min(family_med.values())>=cfg["interaction_gate"]["median_ratio_min"])
 cancellation_pass=bool(frame.broad_cancellation_index.median()>=cfg["cancellation_gate"]["median_index_min"] and np.median((np.linalg.norm(np.stack(full_broad),axis=1)+np.linalg.norm(np.stack(interaction_broad),axis=1))/np.maximum(np.linalg.norm(np.stack(full_broad),axis=1),1e-12))>=1)
 native=bg[bg.background=="native"].set_index("source_base_trial_id");non_native=bg[bg.background!="native"];background_drop=float(non_native.groupby("source_base_trial_id").cosine.median().rsub(1).median());background_cv=float(non_native.groupby("source_base_trial_id").norm_ratio.std().median());background_pass=bool(background_drop>=cfg["background_gate"]["transfer_fidelity_drop_min"] or background_cv>=cfg["background_gate"]["conditional_effect_cv_min"])
 summary={"candidate_layer":layer,"candidate_k":24,"expected_branch_rows":expected,"observed_branch_rows":len(frame),"failed_branch_rows":expected-len(frame),"interaction":{"broad_median_ratio":float(frame.broad_interaction_ratio.median()),"bootstrap_2_5_lower":lower,"family_medians":family_med,"additive_relative_l2_median":float(frame.broad_additive_relative_l2.median()),"interaction_cosine_median":float(frame.broad_interaction_cosine.median()),"STRONG_DISTRIBUTED_NONLINEAR_INTERACTION":interaction_pass},
  "cancellation":{"index_median":float(frame.broad_cancellation_index.median()),"component_cosine_median":float(frame.broad_component_cosine.median()),"CAUSAL_CANCELLATION_SUPPORTED":cancellation_pass},"background":{"non_native_cosine_median":float(non_native.cosine.median()),"fidelity_drop_median":background_drop,"conditional_norm_ratio_std_median":background_cv,"BACKGROUND_CONDITIONED_CAUSAL_EFFECT":background_pass},
  "reconstruction":{"causal_by_dimension":{str(dim):{"relative_l2":float(reconf[(reconf.method=="causal")&(reconf.dimension==dim)].relative_l2.median()),"cosine":float(reconf[(reconf.method=="causal")&(reconf.dimension==dim)].cosine.median()),"norm_ratio":float(reconf[(reconf.method=="causal")&(reconf.dimension==dim)].norm_ratio.median())} for dim in cfg["realization_dimensions"]}},
  "clean_replay_max_abs":float(max(clean_replay_abs)),"clean_replay_relative_l2_max":float(max(clean_replay_rel)),"writeback_min_cosine":float(audit.cosine.min()),"branch_sha256":sha256_file(branch_path),"vectors_sha256":sha256_file(vectors_path),"background_sha256":sha256_file(bgpath),"reconstruction_sha256":sha256_file(recon_path),"nonlinear_sha256":sha256_file(nonpath),"channel_sha256":sha256_file(chpath),"historical_final_opened":False,"v25_independent_final_opened":False,"component_basis_digest":basis_stage["freeze_digest"]}
 target=root/OUT/"distributed_interaction_summary_v25.json";write_json_atomic(target,summary)
 if limit is None:
  frozen=stage_freeze(root,"interaction",[SOURCE,"artifacts/distributed_causal_interaction_v25_component_basis.freeze.json",str(branch_path.relative_to(root)),str(multi_path.relative_to(root)),str(recon_path.relative_to(root)),str(vectors_path.relative_to(root)),str(bgpath.relative_to(root)),str(nonpath.relative_to(root)),str(orderpath.relative_to(root)),str(chpath.relative_to(root)),str(auditpath.relative_to(root)),str(target.relative_to(root))],summary);summary={"freeze_digest":frozen["freeze_digest"],**summary}
 return summary

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--limit",type=int);a=p.parse_args();print(json.dumps(run(Path.cwd(),a.limit),indent=2))
