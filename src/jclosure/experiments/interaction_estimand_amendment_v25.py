"""Append-only correction: Y11 is clean hidden plus jointly written B+C."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.mediation_v24 import _context,_full_capture as _base_capture,_edited_capture as _base_edit
from jclosure.experiments.layer_response_v24 import _projectors
from jclosure.experiments.interaction_v25 import _targets,_contrast,_m,_fidelity
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/interaction_estimand_amendment_v25.py";OUT=Path("results/v25/processed")
def _capture(root,bundle,dense,cache,*args):return _base_capture(root,bundle,dense,clone_hybrid_cache(cache),*args)
def _edit(root,bundle,dense,cache,*args):return _base_edit(root,bundle,dense,clone_hybrid_cache(cache),*args)

@torch.no_grad()
def run(root:Path)->dict:
 verify_stage(root,"interaction");cfg=verify(root)["config"];d=json.loads((root/OUT/"design_v25.json").read_text());candidate=json.loads((root/"results/v24/processed/bottleneck_candidate_selection_v24.json").read_text());layer=int(candidate["diagnostic_layer"]);norm_scales=candidate["normalization_scales"]
 design,split,prompts,teachers,qmap,bundle,dense,values,rec,att,measured,state_layer,ws_layers,ws_count=_context(root);device=next(bundle.hf_model.parameters()).device;proj=_projectors(root,device);jids=torch.as_tensor(split["selected_j"],device=device);lids=torch.as_tensor(split["selected_logits"],device=device);tscales={x:float(v) for x,v in split["target_scales"].items()}
 with np.load(root/OUT/"distributed_component_bases_v25.npz") as z:basis=np.asarray(z["causal_basis"][:,:24],np.float32)
 rows=[];audit_rows=[];correct_interactions=[]
 for number,item in enumerate(d["interaction_validation_states"],1):
  base=item["base_trial_id"];token=int(teachers[base][0]);pref=v19._prefill_history(bundle,str(prompts[base]["prompt"]),measured,dense,state_layer);p0=clone_hybrid_cache(pref["cache"]);q=qmap[item["q_name"]];pq=v19.apply(p0,1.0,v19._q_row(values["directions"],q,float(q["alpha"])),rec,att,"native_fp32_add_bf16_writeback")
  for state_name,state in (("P0",p0),("Pq",pq)):
   clean=_capture(root,bundle,dense,state,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer)
   for action_id in d["interaction_action_ids"]:
    row_action=actions._row(values["directions"],d["action_specs"][action_id],float(d["action_alphas"][action_id]),1);edited=v19.apply(state,1.0,row_action,rec,att,"native_fp32_add_bf16_writeback");persistent_full=_capture(root,bundle,dense,edited,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer);delta=persistent_full["raw_hidden"]-clean["raw_hidden"];db=basis@(basis.T@delta);dc=delta-db
    branch={}
    for name,vec in (("Y10_B",db),("Y01_C",dc),("Y11_BC",delta)):
     audit=[];branch[name]=_edit(root,bundle,dense,state,vec,1,token,pref["prompt_length"],jids,lids,tscales,ws_layers,ws_count,max(measured),cfg["layers"],cfg["architecture_layers"],proj,layer,audit);audit_rows.append({"base_trial_id":base,"family":item["family"],"state":state_name,"action_id":action_id,"branch":name,**audit[-1]})
    effects={name:_targets(_contrast(value,clean),norm_scales) for name,value in branch.items()};original=_targets(_contrast(persistent_full,clean),norm_scales);row={"base_trial_id":base,"family":item["family"],"state":state_name,"action_id":action_id}
    for target in effects["Y11_BC"]:
     row.update({f"{target}_{k}":v for k,v in _m(effects["Y11_BC"][target],effects["Y10_B"][target],effects["Y01_C"][target]).items()});row.update({f"{target}_Y11_vs_persistent_{k}":v for k,v in _fidelity(original[target],effects["Y11_BC"][target]).items()})
    rows.append(row);correct_interactions.append((effects["Y11_BC"]["broad"]-effects["Y10_B"]["broad"]-effects["Y01_C"]["broad"]).astype(np.float32))
  print(f"V25 corrected interaction estimand {number}/10",flush=True)
 frame=pd.DataFrame(rows);path=root/OUT/"distributed_interaction_corrected_v25.parquet";frame.to_parquet(path,index=False,compression="zstd");audit=pd.DataFrame(audit_rows);auditpath=root/OUT/"corrected_interaction_writeback_audit_v25.parquet";audit.to_parquet(auditpath,index=False,compression="zstd");vectors=root/OUT/"corrected_interaction_vectors_v25.npz";np.savez_compressed(vectors,interaction_broad=np.stack(correct_interactions))
 values_i=frame.broad_interaction_ratio.to_numpy();rng=np.random.default_rng(250031);lower=float(np.quantile([np.median(rng.choice(values_i,len(values_i),replace=True)) for _ in range(1000)],.025));family=frame.groupby("family").broad_interaction_ratio.median().to_dict();primary=audit[audit.branch.isin(["Y10_B","Y01_C","Y11_BC"])]
 replay_l2=float(frame.broad_Y11_vs_persistent_relative_l2.median());writeback_ok=bool(primary.groupby("branch").cosine.median().min()>=.90 and replay_l2<=.10)
 interaction_pass=bool(np.median(values_i)>=cfg["interaction_gate"]["median_ratio_min"] and lower>cfg["interaction_gate"]["bootstrap_lower_min"] and min(family.values())>=cfg["interaction_gate"]["median_ratio_min"] and writeback_ok)
 cancellation_pass=bool(frame.broad_cancellation_index.median()>=cfg["cancellation_gate"]["median_index_min"])
 result={"estimand":"Y00=clean; Y10=clean+B; Y01=clean+C; Y11=clean+(B+C)","rows":len(frame),"broad_interaction_ratio_median":float(frame.broad_interaction_ratio.median()),"bootstrap_2_5_lower":lower,"family_medians":family,"additive_relative_l2_median":float(frame.broad_additive_relative_l2.median()),"interaction_cosine_median":float(frame.broad_interaction_cosine.median()),"cancellation_index_median":float(frame.broad_cancellation_index.median()),"component_cosine_median":float(frame.broad_component_cosine.median()),"Y11_vs_persistent_relative_l2_median":replay_l2,"Y11_vs_persistent_cosine_median":float(frame.broad_Y11_vs_persistent_cosine.median()),"writeback":{"branch_median_cosine":primary.groupby("branch").cosine.median().to_dict(),"branch_min_cosine":primary.groupby("branch").cosine.min().to_dict(),"branch_median_gain":primary.groupby("branch").gain.median().to_dict(),"numerical_gate_pass":writeback_ok},"STRONG_DISTRIBUTED_NONLINEAR_INTERACTION":interaction_pass,"CAUSAL_CANCELLATION_SUPPORTED":cancellation_pass,"original_interaction_records_overwritten":False,"corrected_branch_sha256":sha256_file(path),"corrected_audit_sha256":sha256_file(auditpath),"corrected_vectors_sha256":sha256_file(vectors),"historical_final_opened":False,"v25_independent_final_opened":False}
 target=root/OUT/"interaction_estimand_amendment_v25.json";write_json_atomic(target,result);frozen=stage_freeze(root,"interaction_estimand_amendment",[SOURCE,"artifacts/distributed_causal_interaction_v25_interaction.freeze.json",str(path.relative_to(root)),str(auditpath.relative_to(root)),str(vectors.relative_to(root)),str(target.relative_to(root))],result);return {"freeze_digest":frozen["freeze_digest"],**result}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
