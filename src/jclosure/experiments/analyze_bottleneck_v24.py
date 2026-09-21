"""Train-only bottleneck localization followed by frozen validation diagnostics."""
from __future__ import annotations
import argparse,json,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch
from jclosure.experiments import action_pool_v22 as actions
from jclosure.protocol_v24 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/analyze_bottleneck_v24.py"; OUT=Path("results/v24/processed"); SCRATCH=Path("/data/CSK/J-space-project/v24-output-bottleneck-work")
STATE_REPS=Path("/data/CSK/J-space-project/v21-action-geometry-work/state_representations_v21.npz")

def _ranks(y):
 a=np.asarray(y,np.float32)
 # Rank-only diagnostics need singular values, not singular vectors.  The
 # smaller Gram matrix is exactly equivalent and avoids repeatedly running a
 # tall 1056x256 SVD for every state/layer/probe count.  CUDA is used when the
 # analysis command is assigned a GPU; the persisted source arrays remain
 # unchanged and the eigenspectrum is sorted deterministically below.
 if torch.cuda.is_available():
  t=torch.as_tensor(a,device="cuda")
  gram=t.T@t if t.shape[0]>=t.shape[1] else t@t.T
  e=torch.linalg.eigvalsh(gram).clamp_min_(0).flip(0).double().cpu().numpy()
 else:
  gram=a.T@a if a.shape[0]>=a.shape[1] else a@a.T
  e=np.linalg.eigvalsh(gram.astype(np.float64))[::-1]; e=np.maximum(e,0)
 p=e/max(e.sum(),1e-30); c=np.cumsum(p)
 r=[int(np.searchsorted(c,x)+1) for x in (.90,.95,.99)]; stable=float(e.sum()/max(e[0],1e-30)); entropy=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))
 return r,stable,entropy
def _basis(rows,k):
 x=np.asarray(rows,np.float64); gram=x.T@x; values,vectors=np.linalg.eigh(gram); basis=vectors[:,np.argsort(values)[::-1][:min(k,x.shape[1])]]
 # Canonicalize the otherwise arbitrary eigenvector signs for stable hashes.
 for j in range(basis.shape[1]):
  pivot=int(np.argmax(np.abs(basis[:,j])))
  if basis[pivot,j]<0:basis[:,j]*=-1
 return basis
def _sub(a,b):
 n=min(a.shape[1],b.shape[1]); s=np.clip(np.linalg.svd(a[:,:n].T@b[:,:n],compute_uv=False),0,1); angle=np.degrees(np.arccos(s))
 return {"median_angle_degrees":float(np.median(angle)),"max_angle_degrees":float(np.max(angle)),"overlap":float(np.mean(s*s)),"projector_distance":float(np.linalg.norm(a[:,:n]@a[:,:n].T-b[:,:n]@b[:,:n].T))}
def _coverage(y,b):
 p=np.asarray(y)@b@b.T; residual=float(np.linalg.norm(y-p)/max(np.linalg.norm(y),1e-12)); yn=np.linalg.norm(y,axis=1); pn=np.linalg.norm(p,axis=1); cos=np.sum(y*p,axis=1)/np.maximum(yn*pn,1e-12)
 return {"explained_norm_fraction":float(1-residual*residual),"relative_residual":residual,"median_cosine":float(np.median(cos)),"sample_count":int(len(y))}
def _load(root,role,item): return np.load(root/SCRATCH/f"finite_{role}"/f"layer_{item['base_trial_id']}.npz")
def _rows_for(root,role,items,actions_count,state,layer,key="residual"):
 out=[]
 for item in items:
  z=_load(root,role,item); out.append(np.asarray(z[f"{state}_{key}"][:actions_count,layer] if key=="residual" else z[f"{state}_{key}"][:actions_count],np.float64)); z.close()
 return np.concatenate(out)
def _rank_rows(root,role,items,counts,layers,scales):
 rows=[]
 for item in items:
  z=_load(root,role,item)
  for state in ("P0","Pq"):
   for m in counts:
    residual=np.asarray(z[f"{state}_residual"][:m],np.float64); t0=np.asarray(z[f"{state}_t0"][:m],np.float64); vocab=np.asarray(z[f"{state}_vocab"][:m],np.float64)
    for li,layer in enumerate(layers):
     broad=np.concatenate([residual[:,li]/scales["residual"][li],t0/scales["t0"],vocab/scales["vocab"]],axis=1)
     for target,y in (("residual",residual[:,li]),("T0",t0),("vocab_random",vocab),("broad_T4",broad)):
      r,s,e=_ranks(y.T); rows.append({"role":role,"base_trial_id":item["base_trial_id"],"family":item["family"],"state":state,"m":m,"layer":layer,"target":target,"r90":r[0],"r95":r[1],"r99":r[2],"stable_rank":s,"entropy_rank":e})
  z.close()
 return rows

def select(root:Path)->dict:
 cfg=verify(root)["config"]; verify_stage(root,"layer_response_design"); d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); items=d["development_states"]
 # Development-only block balancing scales.
 norms={"residual":[[] for _ in cfg["layers"]],"t0":[],"vocab":[]}
 for item in items:
  z=_load(root,"development",item)
  for state in ("P0","Pq"):
   r=np.asarray(z[f"{state}_residual"][:256],np.float64); t=np.asarray(z[f"{state}_t0"][:256],np.float64); v=np.asarray(z[f"{state}_vocab"][:256],np.float64)
   for li in range(len(cfg["layers"])): norms["residual"][li].extend(np.linalg.norm(r[:,li],axis=1).tolist())
   norms["t0"].extend(np.linalg.norm(t,axis=1).tolist()); norms["vocab"].extend(np.linalg.norm(v,axis=1).tolist())
  z.close()
 scales={"residual":[max(float(np.median(x)),1e-8) for x in norms["residual"]],"t0":max(float(np.median(norms["t0"])),1e-8),"vocab":max(float(np.median(norms["vocab"])),1e-8)}
 parquet=root/OUT/"layerwise_rank_development_v24.parquet"
 if parquet.exists(): frame=pd.read_parquet(parquet)
 else:
  rows=_rank_rows(root,"development",items,cfg["probe_counts"],cfg["layers"],scales); frame=pd.DataFrame(rows); frame.to_parquet(parquet,index=False,compression="zstd")
 med=frame[(frame.m==256)].groupby(["target","layer"]).r95.median(); early=float(med[("broad_T4",0)]); candidates=[]
 for layer in cfg["layers"]:
  value=float(med[("broad_T4",layer)]); downstream=[float(med[("broad_T4",x)]) for x in cfg["layers"] if x>=layer]
  fam=frame[(frame.m==256)&(frame.layer==layer)&(frame.target=="broad_T4")].groupby("family").r95.median()
  passed=value<=early*(1-cfg["rank_collapse"]["minimum_reduction_fraction"]) and max(downstream)-min(downstream)<=cfg["rank_collapse"]["downstream_plateau_absolute_r95"] and float(fam.max()-fam.min())<=cfg["rank_collapse"]["family_max_r95_spread"]
  if passed:candidates.append(layer)
 formal_layer=min(candidates) if candidates else None
 diagnostic_layer=formal_layer if formal_layer is not None else int(min(cfg["layers"],key=lambda l:(float(med[("broad_T4",l)]),-l)))
 train_rows=[]; clean_rows=[]; family_rows=defaultdict(list)
 for item in items:
  z=_load(root,"development",item); li=cfg["layers"].index(diagnostic_layer)
  for state in ("P0","Pq"):
   y=np.asarray(z[f"{state}_residual"][:256,li],np.float64); train_rows.append(y); family_rows[item["family"]].append(y); clean_rows.append(np.asarray(z[f"{state}_baseline_residual"][li],np.float64))
  z.close()
 train_rows=np.concatenate(train_rows); clean_rows=np.stack(clean_rows)
 global_basis_full=_basis(train_rows,max(cfg["bottleneck_dimensions"]))
 coverage={str(k):_coverage(train_rows,global_basis_full[:,:k]) for k in cfg["bottleneck_dimensions"]}
 eligible=[k for k in cfg["bottleneck_dimensions"] if coverage[str(k)]["explained_norm_fraction"]>=cfg["coverage_gate"]["explained_norm_min"]]
 diagnostic_k=min(eligible) if eligible else max(cfg["bottleneck_dimensions"])
 layer_bases=[]
 for li,layer in enumerate(cfg["layers"]): layer_bases.append(_basis(_rows_for(root,"development",items,256,"P0",li),32).astype(np.float32))
 family_basis={f:_basis(np.concatenate(x),diagnostic_k).astype(np.float32) for f,x in family_rows.items()}
 rng=np.random.default_rng(240026); random_basis=np.linalg.qr(rng.standard_normal((256,diagnostic_k)))[0].astype(np.float32)
 variance_basis=_basis(clean_rows-clean_rows.mean(0),diagnostic_k).astype(np.float32)
 weights=rng.standard_normal((diagnostic_k,len(train_rows))); random_causal=np.linalg.qr(train_rows.T@weights.T)[0][:,:diagnostic_k].astype(np.float32)
 basis_path=root/OUT/"projected_bottleneck_bases_v24.npz"; np.savez_compressed(basis_path,dimensions=np.asarray(cfg["bottleneck_dimensions"]),global_basis=global_basis_full.astype(np.float32),
  layer_bases=np.stack(layer_bases),layers=np.asarray(cfg["layers"]),random_basis=random_basis,variance_basis=variance_basis,random_causal_basis=random_causal,
  **{f"family_{f}":b for f,b in family_basis.items()})
 result={"formal_candidate_layer":formal_layer,"diagnostic_layer":diagnostic_layer,"diagnostic_k":diagnostic_k,"development_early_broad_r95":early,
  "development_candidate_broad_r95":float(med[("broad_T4",diagnostic_layer)]),"rank_collapse_candidate_exists":formal_layer is not None,
  "development_global_coverage_by_k":coverage,"normalization_scales":scales,"selection_used_validation":False,
  "rank_rows_sha256":sha256_file(parquet),"projected_basis_sha256":sha256_file(basis_path),"historical_final_opened":False,"v24_independent_final_opened":False}
 target=root/OUT/"bottleneck_candidate_selection_v24.json"; write_json_atomic(target,result)
 frozen=stage_freeze(root,"candidate_selection",[SOURCE,"artifacts/causal_output_bottleneck_v24_layer_response_design.freeze.json",str(parquet),str(basis_path),str(target)],result)
 return {"freeze_digest":frozen["freeze_digest"],**result}

def validate(root:Path)->dict:
 cfg=verify(root)["config"]; candidate=verify_stage(root,"candidate_selection"); d=json.loads((root/OUT/"bottleneck_design_v24.json").read_text()); items=d["validation_states"]
 c=json.loads((root/OUT/"bottleneck_candidate_selection_v24.json").read_text()); scales=c["normalization_scales"]
 rows=_rank_rows(root,"validation",items,cfg["probe_counts"],cfg["layers"],scales); frame=pd.DataFrame(rows); parquet=root/OUT/"layerwise_rank_validation_v24.parquet"; frame.to_parquet(parquet,index=False,compression="zstd")
 layer=c["diagnostic_layer"]; li=cfg["layers"].index(layer); k=c["diagnostic_k"]
 with np.load(root/OUT/"projected_bottleneck_bases_v24.npz") as b:
  dims=list(map(int,b["dimensions"])); global_basis=np.asarray(b["global_basis"][:,:k],np.float64); random_basis=np.asarray(b["random_basis"],np.float64); variance_basis=np.asarray(b["variance_basis"],np.float64); random_causal=np.asarray(b["random_causal_basis"],np.float64)
  family_basis={f:np.asarray(b[f"family_{f}"],np.float64) for f in sorted({x["family"] for x in items})}
 # Representation maps for nearest-J and nearest-P baselines.
 reps=np.load(STATE_REPS); maps={}
 for split in ("train","validation"):
  for i,name in enumerate(reps[f"{split}_ids"]): maps[str(name)]={"S0":np.asarray(reps[f"S0_{split}"][i]),"S1":np.asarray(reps[f"S1_{split}"][i])}
 dev=d["development_states"]; dev_basis={}
 for item in dev:
  z=_load(root,"development",item)
  for state in ("P0","Pq"):
   dev_basis[(item["base_trial_id"],state)]=_basis(np.asarray(z[f"{state}_residual"][:256,li],np.float64),k)
  z.close()
 accum=defaultdict(list); local_bases={}; orientation=[]
 for item in items:
  z=_load(root,"validation",item)
  for state in ("P0","Pq"):
   ytrain=np.asarray(z[f"{state}_residual"][:256,li],np.float64); y=np.asarray(z[f"{state}_residual"][256:,li],np.float64); local=_basis(ytrain,k); local_bases[(item["base_trial_id"],state)]=local
   key=f"{item['base_trial_id']}::{'P0' if state=='P0' else item['q_name']}"; rep=maps[key]
   nearest={name:min(dev,key=lambda x:np.linalg.norm(rep[name]-maps[f"{x['base_trial_id']}::{'P0' if state=='P0' else x['q_name']}"][name])) for name in ("S0","S1")}
   wrong=family_basis[sorted(family_basis)[(sorted(family_basis).index(item["family"])+1)%len(family_basis)]]
   bases={"B_GLOBAL":global_basis,"B_FAMILY":family_basis[item["family"]],"B_LOCAL_J":dev_basis[(nearest["S0"]["base_trial_id"],state)],
          "B_LOCAL_P":dev_basis[(nearest["S1"]["base_trial_id"],state)],"B_STATE_ORACLE":local,"CONTROL_RANDOM":random_basis,
          "CONTROL_VARIANCE":variance_basis,"CONTROL_SHUFFLED_FAMILY":wrong,"CONTROL_RANDOM_CAUSAL":random_causal}
   for name,basis in bases.items(): accum[name].append((y,item["family"],_coverage(y,basis)))
  orientation.append({"base_trial_id":item["base_trial_id"],"family":item["family"],**_sub(local_bases[(item["base_trial_id"],"P0")],local_bases[(item["base_trial_id"],"Pq")])})
  z.close()
 coverage={}
 for name,blocks in accum.items():
  all_y=[]; all_p=[]
  # aggregate from already computed scalar blocks, weighted descriptive medians
  coverage[name]={"explained_norm_fraction_median":float(np.median([x[2]["explained_norm_fraction"] for x in blocks])),"relative_residual_median":float(np.median([x[2]["relative_residual"] for x in blocks])),
  "median_cosine":float(np.median([x[2]["median_cosine"] for x in blocks])),"family_relative_residual":{f:float(np.median([x[2]["relative_residual"] for x in blocks if x[1]==f])) for f in sorted({x[1] for x in blocks})}}
 # Orientation dynamics among state-oracle bases.
 pair_angles=[]
 keys=list(local_bases)
 for i,left in enumerate(keys):
  left_item=next(x for x in items if x["base_trial_id"]==left[0]); left_rep=maps[f"{left[0]}::{'P0' if left[1]=='P0' else left_item['q_name']}"]["S0"]
  for right in keys[i+1:]:
   right_item=next(x for x in items if x["base_trial_id"]==right[0]); right_rep=maps[f"{right[0]}::{'P0' if right[1]=='P0' else right_item['q_name']}"]["S0"]
   pair_angles.append({"left":left,"right":right,"same_family":left_item["family"]==right_item["family"],"same_state_role":left[1]==right[1],"j_distance":float(np.linalg.norm(left_rep-right_rep)),**_sub(local_bases[left],local_bases[right])})
 nearest=[]
 for key0 in keys:
  subset=[x for x in pair_angles if x["left"]==key0 or x["right"]==key0]
  nearest.append(min(subset,key=lambda x:x["j_distance"]))
 med=frame.groupby(["target","layer","m"]).r95.median(); curves={}
 for target in ("residual","T0","vocab_random","broad_T4"):
  curves[target]={str(m):[float(med[(target,l,m)]) for l in cfg["layers"]] for m in cfg["probe_counts"]}
 # Metric-corrected cross-target action-mode consistency at the candidate layer.
 directions=torch.load(root/"artifacts/causal/v13/probe_directions_v13.pt",map_location="cpu",weights_only=False); score=np.asarray(directions["score_directions"],np.float64); del directions
 x=np.stack([actions._score(d["action_specs"][aid],score)*float(d["action_alphas"][aid]) for aid in d["train_action_ids"]]); gram=x@x.T; ev,q=np.linalg.eigh(gram); order=np.argsort(ev)[::-1]; ev,q=ev[order],q[:,order]; keep=ev>max(ev[0]*1e-8,1e-12); w=q[:,keep]/np.sqrt(ev[keep])[None]
 cross=[]
 for item in items:
  z=_load(root,"validation",item)
  for state in ("P0","Pq"):
   ys={"residual":np.asarray(z[f"{state}_residual"][:256,li],np.float64),"T0":np.asarray(z[f"{state}_t0"][:256],np.float64),"vocab":np.asarray(z[f"{state}_vocab"][:256],np.float64)}
   modes={}
   for name,y in ys.items():
    _,s,vh=np.linalg.svd(y.T@w,full_matrices=False); r=_ranks(y.T@w)[0][1]; modes[name]=vh.T[:,:r]
   cross.append({"residual_T0":_sub(modes["residual"],modes["T0"]),"residual_vocab":_sub(modes["residual"],modes["vocab"]),"T0_vocab":_sub(modes["T0"],modes["vocab"])})
  z.close()
 # JVP diagnostic rank at 32 actions.
 jvp_rows=[]
 for item in d["jvp_states"]:
  z=np.load(root/SCRATCH/"jvp"/f"layer_jvp_{item['base_trial_id']}.npz")
  for state in ("P0","Pq"):
   for lj,l in enumerate(cfg["layers"]):
    r,s,e=_ranks(np.asarray(z[f"{state}_residual"][:,lj],np.float64).T); jvp_rows.append({"base_trial_id":item["base_trial_id"],"family":item["family"],"state":state,"layer":l,"r90":r[0],"r95":r[1],"r99":r[2],"stable_rank":s,"entropy_rank":e})
  z.close()
 jvp_frame=pd.DataFrame(jvp_rows); jvp_path=root/OUT/"layerwise_jvp_rank_v24.parquet"; jvp_frame.to_parquet(jvp_path,index=False,compression="zstd")
 # Natural clean-transition ranks for rows with two frozen historical teacher
 # tokens.  The append-only eligibility amendment records the excluded h1-only
 # rows; no token is inferred or generated after the design freeze.
 natural=[]
 for item in d["natural_transition_states"]:
  path=root/SCRATCH/"natural"/f"natural_{item['base_trial_id']}.npz"
  if not path.exists(): continue
  z=np.load(path); natural.append(np.asarray(z["natural_residual"],np.float64)); z.close()
 if not natural: raise RuntimeError("no eligible frozen two-token natural transitions")
 natural=np.stack(natural); natural_rank=[]
 for lj,l in enumerate(cfg["layers"]): natural_rank.append({"layer":l,"ranks":_ranks(natural[:,lj].T)[0]})
 natural_capture=_coverage(natural[:,li],global_basis)
 r95_128=float(np.median(frame[(frame.target=="T0")&(frame.m==128)].r95)); r95_256=float(np.median(frame[(frame.target=="T0")&(frame.m==256)].r95)); output_saturated=abs(r95_256-r95_128)<=2 and abs(r95_256-r95_128)/max(r95_128,1)<=.15
 broad256=curves["broad_T4"]["256"]; drops=np.diff(broad256); largest_drop_layer=int(cfg["layers"][int(np.argmin(drops))+1]) if len(drops) else None
 global_pass=coverage["B_GLOBAL"]["explained_norm_fraction_median"]>=cfg["coverage_gate"]["explained_norm_min"] and coverage["B_GLOBAL"]["relative_residual_median"]<=cfg["coverage_gate"]["relative_residual_max"] and coverage["B_GLOBAL"]["median_cosine"]>=cfg["coverage_gate"]["cosine_min"]
 oracle_pass=coverage["B_STATE_ORACLE"]["explained_norm_fraction_median"]>=cfg["coverage_gate"]["explained_norm_min"] and coverage["B_STATE_ORACLE"]["relative_residual_median"]<=cfg["coverage_gate"]["relative_residual_max"] and coverage["B_STATE_ORACLE"]["median_cosine"]>=cfg["coverage_gate"]["cosine_min"]
 representational=bool(c["rank_collapse_candidate_exists"] and oracle_pass)
 result={"candidate_layer":layer,"candidate_k":k,"formal_rank_collapse_layer":c["formal_candidate_layer"],"rank_curves":curves,"coverage":coverage,
  "heldout_global_pass":global_pass,"heldout_state_oracle_pass":oracle_pass,"LOW_RANK_OUTPUT_BOTTLENECK_REPRESENTATIONAL_PASS":representational,
  "global_vs_local_oracle_residual_gain":float(coverage["B_GLOBAL"]["relative_residual_median"]-coverage["B_STATE_ORACLE"]["relative_residual_median"]),
  "same_J_orientation":{"median_angle_degrees":float(np.median([x["median_angle_degrees"] for x in orientation])),"median_overlap":float(np.median([x["overlap"] for x in orientation])),"pair_count":len(orientation)},
  "orientation_dynamics":{"nearest_J_angle_median":float(np.median([x["median_angle_degrees"] for x in nearest])),"same_family_angle_median":float(np.median([x["median_angle_degrees"] for x in pair_angles if x["same_family"]])),
  "across_family_angle_median":float(np.median([x["median_angle_degrees"] for x in pair_angles if not x["same_family"]])),"transport_fidelity_median":float(np.median([max(0,1-x["projector_distance"]/math.sqrt(2*k)) for x in nearest]))},
  "cross_target_action_mode":{"residual_T0_overlap_median":float(np.median([x["residual_T0"]["overlap"] for x in cross])),"residual_vocab_overlap_median":float(np.median([x["residual_vocab"]["overlap"] for x in cross])),"T0_vocab_overlap_median":float(np.median([x["T0_vocab"]["overlap"] for x in cross]))},
  "jvp_candidate_r95_median":float(jvp_frame[jvp_frame.layer==layer].r95.median()),"natural_transition_layer_ranks":natural_rank,
  "natural_transition_candidate_coverage":natural_capture,"natural_transition_eligible_count":int(len(natural)),
  "natural_transition_excluded_count":int(len(d["natural_transition_states"])-len(natural)),"output_T0_rank_saturated":output_saturated,
  "rank_collapse_shape":{"largest_adjacent_drop_layer":largest_drop_layer,"largest_adjacent_r95_drop":float(np.min(drops)) if len(drops) else 0.0,"candidate_is_stable_minimum":bool(broad256[layer]==min(broad256))},
  "compression_profile":{"V23_input_JVP_r95_at_256":28.0,"V23_input_finite_r95_at_256":33.5,"early_hidden_broad_r95":float(broad256[0]),"candidate_hidden_broad_r95":float(broad256[layer]),"late_hidden_broad_r95":float(broad256[-1]),"late_T0_r95":float(curves["T0"]["256"][-1]),"late_vocab_r95":float(curves["vocab_random"]["256"][-1])},
  "rank_validation_sha256":sha256_file(parquet),"jvp_rows_sha256":sha256_file(jvp_path),"selection_digest":candidate["freeze_digest"],
  "historical_final_opened":False,"v24_independent_final_opened":False}
 target=root/OUT/"output_bottleneck_validation_v24.json"; write_json_atomic(target,result)
 return result

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("stage",choices=("select","validate")); a=p.parse_args(); ans=select(Path.cwd()) if a.stage=="select" else validate(Path.cwd()); print(json.dumps(ans,indent=2))
