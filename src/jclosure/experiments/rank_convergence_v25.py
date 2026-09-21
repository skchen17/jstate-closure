"""V25 rank-construction audit, convergence, mapping, and contraction analyses."""
from __future__ import annotations
import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from jclosure.protocol_v25 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/rank_convergence_v25.py";OUT=Path("results/v25/processed");SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
def _sv(y):
 s=np.linalg.svd(np.asarray(y,np.float64),compute_uv=False);e=s*s;p=e/max(e.sum(),1e-30);c=np.cumsum(p)
 return {"r90":int(np.searchsorted(c,.9)+1),"r95":int(np.searchsorted(c,.95)+1),"r99":int(np.searchsorted(c,.99)+1),"stable_rank":float(e.sum()/max(e[0],1e-30)),"entropy_rank":float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),"top1_fraction":float(p[0]),"singular_values":s[:20].tolist()}
def _construct(y,name):
 y=np.asarray(y,np.float64)
 if name.startswith("action_centered"):y=y-y.mean(0,keepdims=True)
 if name.endswith("row_normalized"):y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
 return y
def _pair(y):
 y=np.asarray(y,np.float64);y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12);g=y@y.T;n=len(y)
 if n<2:return {"mean_cosine":0.,"mean_absolute_cosine":0.,"angular_dispersion_degrees":0.}
 v=g[np.triu_indices(n,1)];return {"mean_cosine":float(v.mean()),"mean_absolute_cosine":float(np.abs(v).mean()),"angular_dispersion_degrees":float(np.degrees(np.arccos(np.clip(v,-1,1))).std())}
def _cka(x,y):
 x=x-x.mean(0);y=y-y.mean(0);num=np.linalg.norm(x.T@y)**2;den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y);return float(num/max(den,1e-30))
def _load(root,role,item):return np.load(root/SCRATCH/f"finite_{role}"/f"layer_{item['base_trial_id']}.npz")

def run(root:Path)->dict:
 cfg=verify(root)["config"];verify_stage(root,"design");d=json.loads((root/OUT/"design_v25.json").read_text());v24=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text());cand=json.loads((root/"results/v24/processed/bottleneck_candidate_selection_v24.json").read_text())
 scales=cand["normalization_scales"];constructions=cfg["rank_audit"]["constructions"];rank_rows=[];spectra=defaultdict(list)
 for item in d["rank_audit_states"]:
  z=_load(root,"validation",item)
  for state in ("P0","Pq"):
   residual=np.asarray(z[f"{state}_residual"][-32:],np.float64);t0=np.asarray(z[f"{state}_t0"][-32:],np.float64);vocab=np.asarray(z[f"{state}_vocab"][-32:],np.float64)
   for li,layer in enumerate(cfg["layers"]):
    broad=np.concatenate([residual[:,li]/scales["residual"][li],t0/scales["t0"],vocab/scales["vocab"]],1)
    for target,y in (("direct_residual_256",residual[:,li]),("T0_288",t0),("random_vocab_512",vocab),("broad_T4_1056",broad)):
     for construction in constructions:
      metric=_sv(_construct(y,construction));spectra[(target,construction,layer)].append(metric["singular_values"])
      rank_rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"state":state,"layer":layer,"target":target,"construction":construction,**{k:v for k,v in metric.items() if k!="singular_values"}})
  z.close()
 rank=pd.DataFrame(rank_rows);rank_path=root/OUT/"rank_construction_audit_v25.parquet";rank.to_parquet(rank_path,index=False,compression="zstd")
 med=rank.groupby(["target","construction","layer"]).median(numeric_only=True).reset_index();median_path=root/OUT/"rank_construction_medians_v25.parquet";med.to_parquet(median_path,index=False,compression="zstd")
 spectrum_rows=[]
 for (target,construction,layer),values in spectra.items():
  arr=np.asarray(values);spectrum_rows.append({"target":target,"construction":construction,"layer":layer,"median_first20":np.median(arr,axis=0).tolist()})
 spectrum_path=root/OUT/"rank_spectra_first20_v25.json";write_json_atomic(spectrum_path,{"spectra":spectrum_rows})
 def curve(target,construction,key="r95"):
  q=med[(med.target==target)&(med.construction==construction)].set_index("layer");return [float(q.loc[l,key]) for l in cfg["layers"]]
 audit={"matrix_orientation":"rows=independent held-out interventions within frozen state; columns=target coordinates","centering_axis":"subtract action-row mean separately within each state","per_row_normalization_used_in_primary":False,
  "v24_raw_broad_r95":curve("broad_T4_1056","raw"),"corrected_centered_broad_r95":curve("broad_T4_1056","action_centered"),"normalized_broad_r95":curve("broad_T4_1056","row_normalized"),"centered_normalized_broad_r95":curve("broad_T4_1056","action_centered_row_normalized"),
  "raw_top1_fraction":curve("broad_T4_1056","raw","top1_fraction"),"centered_top1_fraction":curve("broad_T4_1056","action_centered","top1_fraction")}
 audit["LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED"]=bool(max(audit["corrected_centered_broad_r95"])<=2)
 audit["V24_RANK1_STATUS"]="RECONFIRMED" if audit["LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED"] else "MEASUREMENT_DEFINITION_SPECIFIC_UNCENTERED_EFFECT"
 audit.update({"rank_rows_sha256":sha256_file(rank_path),"rank_medians_sha256":sha256_file(median_path),"spectra_sha256":sha256_file(spectrum_path)})
 write_json_atomic(root/OUT/"v24_rank_audit_v25.json",audit)

 # Held-out finite convergence profiles.
 per_layer={l:[] for l in cfg["layers"]};meta=[]
 for item in d["rank_audit_states"]:
  z=_load(root,"validation",item)
  for state in ("P0","Pq"):
   for li,l in enumerate(cfg["layers"]):per_layer[l].append(np.asarray(z[f"{state}_residual"][-32:,li],np.float64))
   meta.append((item["base_trial_id"],item["family"],state))
  z.close()
 early=np.concatenate([x-x.mean(0) for x in per_layer[0]])
 conv=[]
 for l in cfg["layers"]:
  blocks=per_layer[l];centered=[x-x.mean(0) for x in blocks];pooled=np.concatenate(centered);same=[_pair(x) for x in centered]
  across=[]
  for ai in range(32):across.append(_pair(np.stack([x[ai]-x.mean(0) for x in blocks])))
  sj=[]
  for i in range(0,len(blocks),2):
   a=centered[i];b=centered[i+1];cos=np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12);sj.extend(cos.tolist())
  metric=_sv(pooled);conv.append({"layer":l,"centered_r90":metric["r90"],"centered_r95":metric["r95"],"centered_r99":metric["r99"],"participation_ratio":metric["stable_rank"],"entropy_rank":metric["entropy_rank"],
   "same_state_mean_cosine":float(np.mean([x["mean_cosine"] for x in same])),"same_state_absolute_cosine":float(np.mean([x["mean_absolute_cosine"] for x in same])),"same_state_angular_dispersion_degrees":float(np.mean([x["angular_dispersion_degrees"] for x in same])),
   "different_state_same_action_absolute_cosine":float(np.mean([x["mean_absolute_cosine"] for x in across])),"same_J_P0_Pq_cosine":float(np.median(sj)),"centered_kernel_alignment_to_layer0":_cka(early,pooled)})
 convf=pd.DataFrame(conv);conv_path=root/OUT/"layerwise_causal_convergence_v25.parquet";convf.to_parquet(conv_path,index=False,compression="zstd")

 # Finite and exact-JVP adjacent-layer contraction.
 finite=[]
 for l in range(31):
  ratios=[];turn=[]
  for x,y in zip(per_layer[l],per_layer[l+1]):
   xc=x-x.mean(0);yc=y-y.mean(0);ratios.extend((np.linalg.norm(yc,axis=1)/np.maximum(np.linalg.norm(xc,axis=1),1e-12)).tolist());turn.extend((np.sum(xc*yc,1)/np.maximum(np.linalg.norm(xc,axis=1)*np.linalg.norm(yc,axis=1),1e-12)).tolist())
  finite.append({"from_layer":l,"to_layer":l+1,"norm_ratio_median":float(np.median(ratios)),"direction_cosine_median":float(np.median(turn))})
 jvp_blocks={l:[] for l in cfg["layers"]}
 for item in v24["jvp_states"]:
  z=np.load(root/SCRATCH/"jvp"/f"layer_jvp_{item['base_trial_id']}.npz")
  for state in ("P0","Pq"):
   for li,l in enumerate(cfg["layers"]):jvp_blocks[l].append(np.asarray(z[f"{state}_residual"][:,li],np.float64))
  z.close()
 jvp=[]
 for l in range(31):
  ratios=[];turn=[]
  for x,y in zip(jvp_blocks[l],jvp_blocks[l+1]):
   xc=x-x.mean(0);yc=y-y.mean(0);ratios.extend((np.linalg.norm(yc,axis=1)/np.maximum(np.linalg.norm(xc,axis=1),1e-12)).tolist());turn.extend((np.sum(xc*yc,1)/np.maximum(np.linalg.norm(xc,axis=1)*np.linalg.norm(yc,axis=1),1e-12)).tolist())
  jvp.append({"from_layer":l,"to_layer":l+1,"norm_ratio_median":float(np.median(ratios)),"direction_cosine_median":float(np.median(turn))})
 contraction={"finite":finite,"exact_jvp":jvp};write_json_atomic(root/OUT/"jvp_finite_contraction_v25.json",contraction)

 # Train-only linear mappings, evaluated on disjoint validation states.
 mapping=[]
 train_items=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text())["development_states"][:20]
 train_by={l:[] for l in cfg["mapping_layers"]}
 for item in train_items:
  z=_load(root,"development",item)
  for state in ("P0","Pq"):
   for l in cfg["mapping_layers"]:
    y=np.asarray(z[f"{state}_residual"][:32,l],np.float64);train_by[l].append(y-y.mean(0))
  z.close()
 test_by={l:np.concatenate([x-x.mean(0) for x in per_layer[l]]) for l in cfg["mapping_layers"]}
 for left,right in zip(cfg["mapping_layers"][:-1],cfg["mapping_layers"][1:]):
  x=np.concatenate(train_by[left]);y=np.concatenate(train_by[right]);beta=np.linalg.pinv(x,rcond=1e-6)@y;pred=test_by[left]@beta;truth=test_by[right];s=np.linalg.svd(beta,compute_uv=False);rel=float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1e-12))
  rng=np.random.default_rng(250027+left);pairs=rng.integers(0,len(truth),size=(5000,2));u=test_by[left]/np.maximum(np.linalg.norm(test_by[left],axis=1,keepdims=True),1e-12);v=truth/np.maximum(np.linalg.norm(truth,axis=1,keepdims=True),1e-12);uc=np.sum(u[pairs[:,0]]*u[pairs[:,1]],1);vc=np.sum(v[pairs[:,0]]*v[pairs[:,1]],1);mask=vc>.9
  mapping.append({"from_layer":left,"to_layer":right,"test_relative_l2":rel,"mapping_first20_singular_values":s[:20].tolist(),"mapping_r95":_sv(beta)["r95"],"upstream_centered_r95":_sv(test_by[left])["r95"],"downstream_centered_r95":_sv(truth)["r95"],"downstream_similar_pair_count":int(mask.sum()),"preimage_diverse_fraction_given_downstream_similar":float(np.mean(uc[mask]<.3)) if mask.any() else 0.})
 write_json_atomic(root/OUT/"many_to_one_mapping_v25.json",{"mappings":mapping})

 # Channel-conditioned convergence from train actions.
 channel=[]
 tags=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text())["action_channel_tags"];train_ids=json.loads((root/"results/v24/processed/bottleneck_design_v24.json").read_text())["train_action_ids"]
 indices={name:[i for i,a in enumerate(train_ids) if tags[a]==name][:32] for name in ("REC","Conv","KV","joint")}
 for l in cfg["mapping_layers"]:
  for name,idx in indices.items():
   blocks=[]
   for item in d["component_basis_states"]:
    z=_load(root,"development",item);blocks.extend([np.asarray(z[f"{s}_residual"][idx,l],np.float64) for s in ("P0","Pq")]);z.close()
   y=np.concatenate([x-x.mean(0) for x in blocks]);channel.append({"layer":l,"channel":name,"actions":len(idx),"centered_r95":_sv(y)["r95"],**_pair(y)})
 chf=pd.DataFrame(channel);chpath=root/OUT/"channel_convergence_v25.parquet";chf.to_parquet(chpath,index=False,compression="zstd")
 summary={"rank_audit":audit,"finite_convergence":{"early_r95":int(convf.iloc[0].centered_r95),"late_r95":int(convf.iloc[-1].centered_r95),"early_absolute_cosine":float(convf.iloc[0].same_state_absolute_cosine),"late_absolute_cosine":float(convf.iloc[-1].same_state_absolute_cosine)},
  "MANY_TO_ONE_MAPPING_DIAGNOSTIC":bool(any(x["preimage_diverse_fraction_given_downstream_similar"]>=.2 for x in mapping)),"rank_audit_sha256":sha256_file(rank_path),"convergence_sha256":sha256_file(conv_path),"channel_sha256":sha256_file(chpath),"historical_final_opened":False,"v25_independent_final_opened":False}
 target=root/OUT/"rank_convergence_summary_v25.json";write_json_atomic(target,summary)
 frozen=stage_freeze(root,"rank_convergence",[SOURCE,"artifacts/distributed_causal_interaction_v25_design.freeze.json",str(rank_path.relative_to(root)),str(median_path.relative_to(root)),str(spectrum_path.relative_to(root)),str(conv_path.relative_to(root)),"results/v25/processed/jvp_finite_contraction_v25.json","results/v25/processed/many_to_one_mapping_v25.json",str(chpath.relative_to(root)),str(target.relative_to(root))],summary)
 return {"freeze_digest":frozen["freeze_digest"],**summary}

if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
