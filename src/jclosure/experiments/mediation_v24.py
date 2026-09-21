"""Raw-hidden bottleneck basis, causal branch, restoration, and regeneration diagnostics."""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.layer_response_v24 import _capture_torch,_projectors
from jclosure.protocol_v24 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
from jclosure.recorder import ActivationRecorder,ResidualEditor

SOURCE="src/jclosure/experiments/mediation_v24.py"; OUT=Path("results/v24/processed"); SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def _basis(x,k): _,_,vh=np.linalg.svd(np.asarray(x,np.float64),full_matrices=False); return vh[:k].T
def _context(root):
 d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); split=json.loads((root/"artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text()); op=json.loads((root/"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
 prompts=v20_bank.prompt_index(root); teachers=v20_response._teacher_map(root); q={x["name"]:x for x in op["q"]}; bundle,dense,_,values,rec,att,measured,state_layer,ws_layers,ws_count=v19._setup(root)
 return d,split,prompts,teachers,q,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count
def _full_capture(root,bundle,dense,cache,token,prompt_length,jids,lids,scales,ws_layers,ws_count,main,layers,arch_layers,proj,candidate_layer):
 with ActivationRecorder(bundle.layers,at=[candidate_layer],clone=True,detach=True) as raw:
  value,_=_capture_torch(bundle,dense,cache,token,prompt_length,jids,lids,scales,ws_layers,ws_count,main,layers,arch_layers,proj)
 out={k:v.detach().cpu().numpy().astype(np.float32) for k,v in value.items()}; out["raw_hidden"]=raw.activations[candidate_layer][0,-1].float().cpu().numpy(); return out
def _effect(x,scales):
 return np.concatenate([x["t0"]/scales["t0"],x["vocab"]/scales["vocab"],x["residual"][-1]/scales["residual"][-1]])
def _contrast(a,b): return {k:a[k]-b[k] for k in ("residual","arch_attn","arch_mlp","vocab","t0")}
def _metric(target,pred):
 tn=np.linalg.norm(target); pn=np.linalg.norm(pred); return {"relative_l2":float(np.linalg.norm(pred-target)/max(tn,1e-12)),"cosine":float(np.dot(target,pred)/max(tn*pn,1e-12)),"norm_ratio":float(pn/max(tn,1e-12)),"retained_ratio":float(pn/max(tn,1e-12))}

@torch.no_grad()
def build_basis(root:Path)->dict:
 candidate=verify_stage(root,"candidate_selection"); cfg=verify(root)["config"]; c=json.loads((root/OUT/"bottleneck_candidate_selection_v24.json").read_text()); d,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 layer=int(c["diagnostic_layer"]); k=int(c["diagnostic_k"]); device=next(bundle.hf_model.parameters()).device; proj=_projectors(root,device); jids=torch.as_tensor(split["selected_j"],device=device); lids=torch.as_tensor(split["selected_logits"],device=device); scales={x:float(v) for x,v in split["target_scales"].items()}
 rows=[]; clean=[]; family=defaultdict(list); coords_j=[]
 for number,item in enumerate(d["mediation_basis_states"],1):
  base=item["base_trial_id"]; token=int(teachers[base][0]); pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer); p0=clone_hybrid_cache(pref["cache"]); q=qmap[item["q_name"]]; pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  for state in (p0,pq):
   baseline=_full_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer); clean.append(baseline["raw_hidden"])
   for action_id in d["train_action_ids"][:32]:
    edited=v19.apply(state,1.0,actions._row(values["directions"],d["action_specs"][action_id],float(d["action_alphas"][action_id]),1),rec,att,"native_fp32_add_bf16_writeback")
    full=_full_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,scales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer); delta=full["raw_hidden"]-baseline["raw_hidden"]; rows.append(delta); family[item["family"]].append(delta); coords_j.append((delta,(full["t0"]-baseline["t0"])[:128]))
  print(f"V24 mediation basis {number}/10",flush=True)
 rows=np.stack(rows); clean=np.stack(clean); global_basis=_basis(rows,k).astype(np.float32); rng=np.random.default_rng(240027)
 random_basis=np.linalg.qr(rng.standard_normal((2560,k)))[0].astype(np.float32); variance_basis=_basis(clean-clean.mean(0),k).astype(np.float32)
 weights=rng.standard_normal((k,len(rows))); random_causal=np.linalg.qr(rows.T@weights.T)[0][:,:k].astype(np.float32); family_basis={f:_basis(np.stack(x),k).astype(np.float32) for f,x in family.items()}
 coord=np.stack([global_basis.T@x[0] for x in coords_j]); jeff=np.stack([x[1] for x in coords_j]); ridge=1e-4*np.trace(coord.T@coord)/max(k,1); beta=np.linalg.solve(coord.T@coord+ridge*np.eye(k),coord.T@jeff); pred=coord@beta
 relation={"J_from_B_train_relative_l2":float(np.linalg.norm(pred-jeff)/max(np.linalg.norm(jeff),1e-12)),"B_dimension":k,"J_dimension":128}
 path=root/OUT/"raw_hidden_bottleneck_basis_v24.npz"; np.savez_compressed(path,global_basis=global_basis,random_basis=random_basis,variance_basis=variance_basis,random_causal_basis=random_causal,
  **{f"family_{f}":x for f,x in family_basis.items()})
 result={"candidate_layer":layer,"candidate_k":k,"basis_train_states":20,"basis_train_actions":32,"basis_response_rows":len(rows),"basis_sha256":sha256_file(path),"workspace_relation_train":relation,
 "branch_protocol":{"CLEAN":"clean persistent cache","FULL":"finite heldout action cache","B_ONLY":"clean cache plus projected full-minus-clean hidden delta",
 "PERP_ONLY":"clean cache plus orthogonal residual","FULL_MINUS_B":"full cache minus projected hidden delta","RESTORE_Q_TO_0":"Pq cache minus projected Pq-minus-P0 delta","TRANSPLANT_0_TO_Q":"P0 cache plus projected Pq-minus-P0 delta"},
 "thresholds":cfg["causal_gate"],"validation_states":d["mediation_validation_states"],"heldout_actions":d["mediation_action_ids"],
 "historical_final_opened":False,"v24_independent_final_opened":False,"candidate_selection_digest":candidate["freeze_digest"]}
 target=root/OUT/"mediation_protocol_v24.json"; write_json_atomic(target,result)
 frozen=stage_freeze(root,"mediation_protocol",[SOURCE,"artifacts/causal_output_bottleneck_v24_candidate_selection.freeze.json",str(path),str(target)],result)
 return {"freeze_digest":frozen["freeze_digest"],**result}

def _edited_capture(root,bundle,dense,cache,vector,sign,token,prompt_length,jids,lids,scales,ws_layers,ws_count,main,layers,arch_layers,proj,layer,audit):
 requested=torch.as_tensor(vector,device=next(bundle.hf_model.parameters()).device,dtype=torch.float32)*float(sign)
 def transform(current,index):
  before=current[0,-1].float(); updated=(before+requested).to(current.dtype); realized=updated.float()-before
  audit.append({"requested_norm":float(requested.norm().item()),"realized_norm":float(realized.norm().item()),"cosine":float(torch.dot(requested,realized).item()/max(requested.norm().item()*realized.norm().item(),1e-12)),"gain":float(realized.norm().item()/max(requested.norm().item(),1e-12))})
  answer=current.clone(); answer[0,-1]=updated; return answer
 with ResidualEditor(bundle.layers,{layer:transform}): return _full_capture(root,bundle,dense,cache,token,prompt_length,jids,lids,scales,ws_layers,ws_count,main,layers,arch_layers,proj,layer)

@torch.no_grad()
def run(root:Path)->dict:
 protocol=verify_stage(root,"mediation_protocol"); cfg=verify(root)["config"]; c=json.loads((root/OUT/"bottleneck_candidate_selection_v24.json").read_text()); d,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root)
 layer=int(protocol["candidate_layer"]); scales_norm=c["normalization_scales"]; device=next(bundle.hf_model.parameters()).device; proj=_projectors(root,device); jids=torch.as_tensor(split["selected_j"],device=device); lids=torch.as_tensor(split["selected_logits"],device=device); tscales={x:float(v) for x,v in split["target_scales"].items()}
 with np.load(root/OUT/"raw_hidden_bottleneck_basis_v24.npz") as b: bases={name:np.asarray(b[name],np.float32) for name in b.files}
 rows=[]; restore=[]; audits=[]; regeneration=[]; families=sorted({x["family"] for x in d["mediation_validation_states"]})
 for number,item in enumerate(d["mediation_validation_states"],1):
  base=item["base_trial_id"]; token=int(teachers[base][0]); pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer); p0=clone_hybrid_cache(pref["cache"]); q=qmap[item["q_name"]]; pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  clean_state={}
  for state_name,state in (("P0",p0),("Pq",pq)):
   clean_state[state_name]=_full_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer)
   for action_id in d["mediation_action_ids"]:
    edited=v19.apply(state,1.0,actions._row(values["directions"],d["action_specs"][action_id],float(d["action_alphas"][action_id]),1),rec,att,"native_fp32_add_bf16_writeback")
    full=_full_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer); delta=full["raw_hidden"]-clean_state[state_name]["raw_hidden"]
    basis=bases["global_basis"]; db=basis@(basis.T@delta); perp=delta-db
    b_only=_edited_capture(root,bundle,dense,state,db,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
    p_only=_edited_capture(root,bundle,dense,state,perp,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
    f_minus=_edited_capture(root,bundle,dense,edited,db,-1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
    full_eff=_effect(_contrast(full,clean_state[state_name]),scales_norm)
    branch={"B_ONLY":b_only,"PERP_ONLY":p_only,"FULL_MINUS_B":f_minus}
    row={"base_trial_id":base,"family":item["family"],"state":state_name,"action_id":action_id}
    for name,val in branch.items():
     metric=_metric(full_eff,_effect(_contrast(val,clean_state[state_name]),scales_norm)); row.update({f"{name}_{k}":v for k,v in metric.items()})
    for control in ("random_basis","variance_basis","random_causal_basis"):
     cb=bases[control]; cd=cb@(cb.T@delta); val=_edited_capture(root,bundle,dense,state,cd,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits); metric=_metric(full_eff,_effect(_contrast(val,clean_state[state_name]),scales_norm)); row.update({f"CONTROL_{control}_{k}":v for k,v in metric.items()})
    wrong=families[(families.index(item["family"])+1)%len(families)]; wb=bases[f"family_{wrong}"]; wd=wb@(wb.T@delta); val=_edited_capture(root,bundle,dense,state,wd,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits); metric=_metric(full_eff,_effect(_contrast(val,clean_state[state_name]),scales_norm)); row.update({f"CONTROL_shuffled_{k}":v for k,v in metric.items()}); rows.append(row)
    # Regeneration: how much PERP_ONLY downstream response enters train response basis.
    with np.load(root/OUT/"projected_bottleneck_bases_v24.npz") as pb:
     for later,basis_proj in zip(pb["layers"],pb["layer_bases"],strict=True):
      if int(later)>layer:
       e=np.asarray(p_only["residual"][int(later)]-clean_state[state_name]["residual"][int(later)],np.float64); regeneration.append({"base_trial_id":base,"state":state_name,"action_id":action_id,"layer":int(later),"projected_ratio":float(np.linalg.norm(e@basis_proj@basis_proj.T)/max(np.linalg.norm(e),1e-12))})
  # Same-J restoration/transplant once per base.
  delta=clean_state["Pq"]["raw_hidden"]-clean_state["P0"]["raw_hidden"]; basis=bases["global_basis"]; db=basis@(basis.T@delta)
  qrest=_edited_capture(root,bundle,dense,pq,db,-1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
  transplant=_edited_capture(root,bundle,dense,p0,db,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audits)
  y0=_effect(clean_state["P0"],scales_norm); yq=_effect(clean_state["Pq"],scales_norm); te=yq-y0
  qr=_metric(te,_effect(qrest,scales_norm)-y0); tr=_metric(te,_effect(transplant,scales_norm)-y0)
  restore.append({"base_trial_id":base,"family":item["family"],"restoration_relative_to_P0":float(np.linalg.norm(_effect(qrest,scales_norm)-y0)/max(np.linalg.norm(te),1e-12)),
  "intervention_mediated_fraction":float(1-np.linalg.norm(_effect(qrest,scales_norm)-y0)/max(np.linalg.norm(te),1e-12)),"transplant_relative_l2_to_TE":tr["relative_l2"],"transplant_cosine_to_TE":tr["cosine"],"transplant_norm_ratio":tr["norm_ratio"]})
  print(f"V24 mediation validation {number}/10",flush=True)
 frame=pd.DataFrame(rows); path=root/OUT/"mediation_branches_v24.parquet"; frame.to_parquet(path,index=False,compression="zstd"); rframe=pd.DataFrame(restore); rpath=root/OUT/"restoration_transplant_v24.parquet"; rframe.to_parquet(rpath,index=False,compression="zstd"); gframe=pd.DataFrame(regeneration); gpath=root/OUT/"bottleneck_regeneration_v24.parquet"; gframe.to_parquet(gpath,index=False,compression="zstd")
 def med(col): return float(frame[col].median())
 rng=np.random.default_rng(240028); values=frame["PERP_ONLY_retained_ratio"].to_numpy(); boots=[float(np.median(rng.choice(values,len(values),replace=True))) for _ in range(1000)]; upper=float(np.quantile(boots,.975))
 summary={"candidate_layer":layer,"candidate_k":int(protocol["candidate_k"]),"branch_rows":len(frame),
 "B_ONLY":{"relative_l2":med("B_ONLY_relative_l2"),"cosine":med("B_ONLY_cosine"),"norm_ratio":med("B_ONLY_norm_ratio")},
 "PERP_ONLY":{"retained_ratio":med("PERP_ONLY_retained_ratio"),"bootstrap_97_5_upper":upper},"FULL_MINUS_B":{"retained_ratio":med("FULL_MINUS_B_retained_ratio")},
 "controls":{name:{"relative_l2":med(f"CONTROL_{name}_relative_l2"),"cosine":med(f"CONTROL_{name}_cosine")} for name in ("random_basis","variance_basis","random_causal_basis","shuffled")},
 "restoration":{"relative_to_P0_median":float(rframe.restoration_relative_to_P0.median()),"intervention_mediated_fraction_median":float(rframe.intervention_mediated_fraction.median())},
 "transplant":{"relative_l2_median":float(rframe.transplant_relative_l2_to_TE.median()),"cosine_median":float(rframe.transplant_cosine_to_TE.median()),"norm_ratio_median":float(rframe.transplant_norm_ratio.median())},
 "regeneration":{"later_projected_ratio_median":float(gframe.projected_ratio.median()),"by_layer":{str(int(l)):float(x.projected_ratio.median()) for l,x in gframe.groupby("layer")}},
 "writeback_audit":{"count":len(audits),"min_cosine":float(min(x["cosine"] for x in audits)),"median_gain":float(np.median([x["gain"] for x in audits]))},
 "branch_parquet_sha256":sha256_file(path),"restoration_parquet_sha256":sha256_file(rpath),"regeneration_parquet_sha256":sha256_file(gpath),"historical_final_opened":False,"v24_independent_final_opened":False}
 gate=cfg["causal_gate"]; summary["CAUSAL_MEDIATION_GATE_PASS"]=bool(summary["B_ONLY"]["relative_l2"]<=gate["relative_l2_max"] and summary["B_ONLY"]["cosine"]>=gate["cosine_min"] and gate["magnitude_ratio_min"]<=summary["B_ONLY"]["norm_ratio"]<=gate["magnitude_ratio_max"] and summary["PERP_ONLY"]["retained_ratio"]<=gate["perp_retained_ratio_max"] and upper<=gate["bootstrap_upper_margin"] and summary["FULL_MINUS_B"]["retained_ratio"]<=gate["perp_retained_ratio_max"] and summary["restoration"]["intervention_mediated_fraction_median"]>=gate["mediation_fraction_min"])
 summary["horizon_status"]="H1_FAILED_NO_H2_H4_H8_OPENING" if not summary["CAUSAL_MEDIATION_GATE_PASS"] else "H1_PASSED_H2_H4_H8_REQUIRED"
 write_json_atomic(root/OUT/"causal_bottleneck_mediation_v24.json",summary); return summary

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("stage",choices=("basis","run")); a=p.parse_args(); ans=build_basis(Path.cwd()) if a.stage=="basis" else run(Path.cwd()); print(json.dumps(ans,indent=2))
