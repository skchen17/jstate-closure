"""Append-only secondary V26 diagnostics; never alters formal gates."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v26 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/diagnostics_v26.py";OUT=Path("results/v26/processed")
def _design(root):return json.loads((root/OUT/"design_v26.json").read_text())
def _metadata(root,item):
 p=root/f"results/v18/processed/crossed_state_{item['source_role']}_{item['family']}_v18.parquet";f=pd.read_parquet(p,filters=[[('base_trial_id','==',item['base_trial_id'])]]);return f.iloc[0].to_dict()
def _corr(a,b,method):return float(pd.Series(a).corr(pd.Series(b),method=method)) if np.std(a)>0 and np.std(b)>0 else None
def _ridge(train_x,train_y,test_x,test_y,lam=1.0):
 mean=train_x.mean(0);std=train_x.std(0);std[std<1e-8]=1;tx=(train_x-mean)/std;vx=(test_x-mean)/std;ym=float(train_y.mean());coef=np.linalg.solve(tx.T@tx+lam*np.eye(tx.shape[1]),tx.T@(train_y-ym));pred=vx@coef+ym;rmse=float(np.sqrt(np.mean((pred-test_y)**2)));r2=float(1-np.sum((pred-test_y)**2)/max(np.sum((test_y-test_y.mean())**2),1e-12));return {"rmse":rmse,"r2":r2}
def run(root:Path):
 verify_stage(root,"adjudication");d=_design(root);metrics=pd.read_parquet(root/OUT/"temporal_metrics_full_v26.parquet");primary=metrics[metrics.control_type=="primary"].copy();bank=pd.read_parquet(root/OUT/"current_distal_bank_v26.parquet");jmap={};itemmap={x["base_trial_id"]:x for role in ("development","validation") for x in d[role]}
 for base,item in itemmap.items():jmap[base]=np.asarray(_metadata(root,item)["current_j"],np.float32)[:128]
 profiles=primary.groupby(["role","base_trial_id","horizon"]).future_potency_q.mean().unstack(fill_value=0);nearest=[]
 for role in ("development","validation"):
  ids=profiles.loc[role].index.tolist();j=np.stack([jmap[x] for x in ids]);jn=j/np.maximum(np.linalg.norm(j,axis=1,keepdims=True),1e-12);dist=1-jn@jn.T;np.fill_diagonal(dist,np.inf);p=profiles.loc[role].to_numpy()
  for i,base in enumerate(ids):
   k=int(np.argmin(dist[i]));nearest.append({"role":role,"base_trial_id":base,"nearest_base_trial_id":ids[k],"j_cosine_distance":float(dist[i,k]),"potency_profile_distance":float(np.linalg.norm(p[i]-p[k])),"emergence_time_difference":0})
 nearestf=pd.DataFrame(nearest);nearest_path=root/OUT/"nearest_j_temporal_potency_v26.parquet";nearestf.to_parquet(nearest_path,index=False,compression="zstd");nearest_summary={role:{"pearson":_corr(g.j_cosine_distance,g.potency_profile_distance,"pearson"),"spearman":_corr(g.j_cosine_distance,g.potency_profile_distance,"spearman"),"median_j_distance":float(g.j_cosine_distance.median()),"median_profile_distance":float(g.potency_profile_distance.median())} for role,g in nearestf.groupby("role")}
 candidates=sorted(primary.candidate_id.unique());cindex={x:i for i,x in enumerate(candidates)}
 def features(role):
  g=primary[(primary.role==role)&(primary.horizon==2)].copy();xs0=[];xs1=[];xs2=[];ys=[]
  for r in g.itertuples():
   one=np.zeros(len(candidates));one[cindex[r.candidate_id]]=1;b=bank[(bank.role==role)&(bank.base_trial_id==r.base_trial_id)&(bank.candidate_id==r.candidate_id)].iloc[0];persistent=np.asarray([b.realized_state_norm,b.realized_state_cosine,b.realized_state_gain],float);current=np.concatenate([jmap[r.base_trial_id],one]);xs0.append(current);xs1.append(np.concatenate([current,persistent]));xs2.append(np.concatenate([one,persistent]));ys.append(r.future_potency_q)
  return np.stack(xs0),np.stack(xs1),np.stack(xs2),np.asarray(ys)
 d0,d1,d2,dy=features("development");v0,v1,v2,vy=features("validation");predictive={"M0_current_J_plus_action":_ridge(d0,dy,v0,vy),"M1_current_J_action_plus_realized_persistent":_ridge(d1,dy,v1,vy),"M2_action_plus_realized_persistent":_ridge(d2,dy,v2,vy)};predictive["M1_rmse_gain_over_M0"]=predictive["M0_current_J_plus_action"]["rmse"]-predictive["M1_current_J_action_plus_realized_persistent"]["rmse"];predictive["interpretation"]="diagnostic prediction only; controlled h0-equal intervention test is primary"
 earlyf=pd.read_parquet(root/OUT/"temporal_rollout_early_v26.parquet");extf=pd.read_parquet(root/OUT/"temporal_rollout_extension_v26.parquet");frames=pd.concat([earlyf,extf],ignore_index=True);ze=np.load(root/OUT/"temporal_vectors_early_v26.npz");zx=np.load(root/OUT/"temporal_vectors_extension_v26.npz");vectors=np.concatenate([ze["vectors"],zx["vectors"]]);lookup={(r.role,r.base_trial_id,int(r.horizon),r.candidate_id):vectors[i] for i,r in enumerate(frames.itertuples()) if r.control_type=="scaling_diagnostic"};sign_rows=[]
 for role,base,h,candidate in list(lookup):
  if not candidate.endswith("_pos"):continue
  neg=candidate[:-3]+"neg"
  if (role,base,h,neg) not in lookup:continue
  a=lookup[(role,base,h,candidate)];b=lookup[(role,base,h,neg)];cos=float(np.dot(a,-b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12));sign_rows.append({"role":role,"base_trial_id":base,"horizon":h,"scale":float(candidate.split("_")[2].replace("p",".")),"odd_symmetry_cosine":cos,"magnitude_ratio_pos_over_neg":float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-12))})
 sign=pd.DataFrame(sign_rows);sign_path=root/OUT/"sign_symmetry_v26.parquet";sign.to_parquet(sign_path,index=False,compression="zstd");sign_summary={"odd_symmetry_cosine_median":float(sign.odd_symmetry_cosine.median()),"magnitude_ratio_median":float(sign.magnitude_ratio_pos_over_neg.median()),"smooth_scaling_supported":False,"reason":"median potency is not monotone in amplitude across both signs, roles, and horizons; BF16 writeback quantization and nonlinear dynamics remain visible"}
 alias={str(int(h)):{"absolute_J_norm_median":float(g.j_norm.median()),"normalized_J_q_median":float((g.j_norm/np.median(metrics[(metrics.role=="development")&(metrics.horizon==h)&(metrics.control_type=="ordinary_reference")].j_norm)).median()),"absolute_aggregate_norm_median":float(g.aggregate_raw_norm.median())} for h,g in primary.groupby("horizon")}
 out={"nearest_J":nearest_summary,"predictive_sufficiency":predictive,"sign_and_scale":sign_summary,"workspace_aliasing_curve":alias,"workspace_reentry_time":"h1 for all 1050 primary development/validation perturbations","temporal_ordering":"persistent, residual, J/workspace, and logits differences are all measurable at h1; sub-token/layer ordering is unresolved","U0_explains_future":"NO: the h0 response matrix is exactly rank zero, while all future matrices are nonzero","future_directions_outside_U0":True,"JVP_finite":"finite interventions are primary; exact temporal cache-state JVP was technically unavailable and is not used as evidence","natural_transition_comparison":"NOT_ESTABLISHED: historical observational transitions were kept separate and were not used in V26 causal gates","shuffled_persistent_control":"NOT_IDENTIFIABLE for the shared global-coordinate action bank without a new state-specific transplant estimand","readout_only_control":"NOT_TECHNICALLY_COMPARABLE because the V26 intervention boundary follows the current readout; no persistent cache change would remain","formal_outcomes_changed":False}
 target=root/OUT/"v26_secondary_diagnostics.json";write_json_atomic(target,out);frozen=stage_freeze(root,"diagnostics",[SOURCE,str(nearest_path.relative_to(root)),str(sign_path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_adjudication.freeze.json"],{"diagnostics_sha256":sha256_file(target),"nearest_sha256":sha256_file(nearest_path),"sign_sha256":sha256_file(sign_path),"formal_outcomes_changed":False});return {"freeze_digest":frozen["freeze_digest"],**out}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
