"""V26 gates, temporal geometry, controls, final opening, and adjudication."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v26 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/analyze_v26.py";OUT=Path("results/v26/processed")
TARGETS=("j","logits","semantic","workspace","broad_vocabulary","late_residual")
DIMS={"j":128,"logits":32,"semantic":32,"workspace":96,"broad_vocabulary":512,"late_residual":256}

def _refs(frame):
 dev=frame[(frame.role=="development")&(frame.control_type=="ordinary_reference")]
 return {int(h):{t:float(g[f"{t}_norm"].median()) for t in TARGETS} for h,g in dev.groupby("horizon")}
def _with_q(frame,refs,cfg):
 x=frame.copy();floors=x[x.control_type=="numerical_scale"].groupby(["role","horizon"]).aggregate_raw_norm.median().to_dict();qs=[];potent=[]
 for row in x.itertuples():
  q=[getattr(row,f"{t}_norm")/max(refs[int(row.horizon)][t],1e-12) for t in TARGETS];value=float(np.mean(q));qs.append(value);floor=float(floors.get((row.role,row.horizon),0.0));potent.append(bool(value>=cfg["future_gate"]["potency_threshold"] and row.aggregate_raw_norm>=cfg["future_gate"]["floor_safety_factor"]*floor))
 x["future_potency_q"]=qs;x["future_potent"]=potent;return x
def _lower(values,seed,n=1000):
 a=np.asarray(values,float);rng=np.random.default_rng(seed);return float(np.quantile([np.mean(rng.choice(a,len(a),replace=True)) for _ in range(n)],.025))
def _gates(frame,cfg):
 primary=frame[frame.control_type=="primary"];out={}
 for role in sorted(primary.role.unique()):
  out[role]={}
  for h,g in primary[primary.role==role].groupby("horizon"):
   fam={f:float(z.future_potent.mean()) for f,z in g.groupby("family")};frac=float(g.future_potent.mean());lower=_lower(g.future_potent.astype(float),260000+int(h)+(0 if role=="development" else 100))
   passed=bool(g.future_potency_q.median()>=cfg["future_gate"]["median_q_min"] and frac>=cfg["future_gate"]["potent_fraction_min"] and lower>=cfg["future_gate"]["bootstrap_lower_min"] and sum(v>=cfg["future_gate"]["potent_fraction_min"] for v in fam.values())>=cfg["future_gate"]["families_required"])
   out[role][str(int(h))]={"median_q":float(g.future_potency_q.median()),"potent_fraction":frac,"bootstrap_2_5_lower":lower,"family_fractions":fam,"replicated_families":sum(v>=cfg["future_gate"]["potent_fraction_min"] for v in fam.values()),"pass":passed}
 return out
def _load_phase(root,phase):
 frame=pd.read_parquet(root/OUT/f"temporal_rollout_{phase}_v26.parquet");z=np.load(root/OUT/f"temporal_vectors_{phase}_v26.npz");return frame,np.asarray(z["vectors"],np.float32)

def early(root:Path):
 verify_stage(root,"early_rollout");cfg=verify(root)["config"];frame,vectors=_load_phase(root,"early");refs=_refs(frame);enriched=_with_q(frame,refs,cfg);path=root/OUT/"temporal_metrics_early_v26.parquet";enriched.to_parquet(path,index=False,compression="zstd");gates=_gates(enriched,cfg);opened=bool(any(gates["development"][h]["pass"] and gates["validation"][h]["pass"] for h in ("1","2")))
 result={"reference_scales":refs,"gates":gates,"extension_opened":opened,"opening_rule":"same pre-registered h1 or h2 future-potency gate passes development and validation","future_based_bank_selection":False,"historical_final_opened":False,"v26_independent_final_opened":False};target=root/OUT/"early_temporal_analysis_v26.json";write_json_atomic(target,result);frozen=stage_freeze(root,"extension_opening",[SOURCE,str(path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_early_rollout.freeze.json"],{"extension_opened":opened,"early_metrics_sha256":sha256_file(path),**result});return {"freeze_digest":frozen["freeze_digest"],**result}

def _rank(s,threshold):
 if not len(s) or float(np.sum(s*s))==0:return 0
 return int(np.searchsorted(np.cumsum(s*s)/np.sum(s*s),threshold)+1)
def _normalize_vectors(vectors,horizons,refs):
 out=np.empty_like(vectors);start=0
 for t in TARGETS:
  stop=start+DIMS[t]
  for h in np.unique(horizons):out[horizons==h,start:stop]=vectors[horizons==h,start:stop]/max(refs[int(h)][t]*math.sqrt(DIMS[t]),1e-12)
  start=stop
 return out
def _geometry(frame,vectors,refs,cfg):
 mask=frame.control_type.eq("primary").to_numpy();f=frame[mask].reset_index(drop=True);v=_normalize_vectors(vectors[mask],f.horizon.to_numpy(),refs);bases={};summary={}
 for h in sorted(f.horizon.unique()):
  train=v[(f.role=="development")&(f.horizon==h)];center=train-train.mean(0,keepdims=True);_,s,vt=np.linalg.svd(center,full_matrices=False);r90=_rank(s,.9);r95=_rank(s,.95);r99=_rank(s,.99);basis=vt[:r95].T;bases[int(h)]=basis;val=v[(f.role=="validation")&(f.horizon==h)];coverage=float(np.median(np.linalg.norm(val@basis,axis=1)/np.maximum(np.linalg.norm(val,axis=1),1e-12))) if r95 else 0.0;summary[str(int(h))]={"r90":r90,"r95":r95,"r99":r99,"validation_projection_coverage":coverage,"top1_fraction":float(s[0]**2/max(np.sum(s*s),1e-12))}
 pairs={};hs=sorted(bases)
 for a,b in zip(hs[:-1],hs[1:]):
  k=min(32,bases[a].shape[1],bases[b].shape[1]);ua=bases[a][:,:k];ub=bases[b][:,:k];overlap=float(np.linalg.norm(ua.T@ub,"fro")**2/max(k,1));distance=float(math.sqrt(max(1-overlap,0)));pairs[f"{a}->{b}"]={"k":k,"grassmann_overlap":overlap,"projector_distance":distance,"principal_angle_cosines":np.linalg.svd(ua.T@ub,compute_uv=False).tolist()}
 distances=[x["projector_distance"] for x in pairs.values()];rotating=bool(distances and np.median(distances)>=cfg["geometry"]["rotation_projector_distance_min"]);fixed=bool(not rotating and all(x["validation_projection_coverage"]>=cfg["geometry"]["fixed_cross_horizon_coverage_min"] for x in summary.values()));return {"by_horizon":summary,"adjacent":pairs,"U0_rank":0,"future_requires_outside_U0":True,"ROTATING_FUTURE_POTENT_GEOMETRY":rotating,"FIXED_FUTURE_POTENT_GEOMETRY":fixed},v,f
def _channel_interactions(frame,vectors):
 lookup={(r.base_trial_id,int(r.horizon),r.candidate_id):vectors[i] for i,r in enumerate(frame.itertuples()) if r.control_type=="primary"};rows=[]
 for base,h,_ in sorted({(k[0],k[1],0) for k in lookup}):
  for joint,a,b in (("rec_conv","rec","conv"),("rec_kv","rec","kv"),("conv_kv","conv","kv")):
   if all((base,h,x) in lookup for x in (joint,a,b)):
    y=lookup[(base,h,joint)];inter=y-lookup[(base,h,a)]-lookup[(base,h,b)];rows.append({"base_trial_id":base,"horizon":h,"term":joint,"interaction_ratio":float(np.linalg.norm(inter)/max(np.linalg.norm(y),1e-12))})
 return pd.DataFrame(rows)

def full(root:Path):
 verify_stage(root,"extension_rollout");cfg=verify(root)["config"];earlyf,earlyv=_load_phase(root,"early");extf,extv=_load_phase(root,"extension");frame=pd.concat([earlyf,extf],ignore_index=True);vectors=np.concatenate([earlyv,extv]);refs=_refs(frame);enriched=_with_q(frame,refs,cfg);metrics_path=root/OUT/"temporal_metrics_full_v26.parquet";enriched.to_parquet(metrics_path,index=False,compression="zstd");gates=_gates(enriched,cfg)
 primary=enriched[enriched.control_type=="primary"];curves=primary.groupby(["role","horizon","family","channel"]).agg(median_q=("future_potency_q","median"),potent_fraction=("future_potent","mean"),median_raw=("aggregate_raw_norm","median")).reset_index();curves_path=root/OUT/"temporal_potency_curves_v26.parquet";curves.to_parquet(curves_path,index=False,compression="zstd")
 emergence=[]
 for keys,g in primary.groupby(["role","base_trial_id","family","candidate_id","channel"]):
  hs=sorted(int(h) for h in g[g.future_potent].horizon.unique());emergence.append(dict(zip(("role","base_trial_id","family","candidate_id","channel"),keys),emergence_time=str(hs[0]) if hs else "NEVER",max_q=float(g.future_potency_q.max())))
 emerge=pd.DataFrame(emergence);emerge_path=root/OUT/"emergence_times_v26.parquet";emerge.to_parquet(emerge_path,index=False,compression="zstd")
 geometry,norm_vectors,geom_frame=_geometry(enriched,vectors,refs,cfg);geom_path=root/OUT/"future_potent_subspaces_v26.json";write_json_atomic(geom_path,geometry)
 interactions=_channel_interactions(enriched,vectors);interaction_path=root/OUT/"channel_interaction_temporal_v26.parquet";interactions.to_parquet(interaction_path,index=False,compression="zstd")
 channel_profiles=primary.groupby(["role","family","channel","horizon"]).future_potency_q.median().reset_index();dev2=channel_profiles[(channel_profiles.role=="development")&(channel_profiles.horizon==2)];val2=channel_profiles[(channel_profiles.role=="validation")&(channel_profiles.horizon==2)];ranges_dev=dev2.groupby("family").future_potency_q.agg(lambda x:float(x.max()-x.min()));ranges_val=val2.groupby("family").future_potency_q.agg(lambda x:float(x.max()-x.min()));channel_pass=bool(sum(ranges_dev>=cfg["secondary_gates"]["channel_profile_range_min"])>=4 and sum(ranges_val>=cfg["secondary_gates"]["channel_profile_range_min"])>=4)
 state_cv={}
 for role in ("development","validation"):
  g=primary[(primary.role==role)&(primary.horizon==2)];vals=[];fams={}
  for (family,candidate),z in g.groupby(["family","candidate_id"]):
   cv=float(z.future_potency_q.std()/max(z.future_potency_q.mean(),1e-12));vals.append(cv);fams.setdefault(family,[]).append(cv)
  state_cv[role]={"median_within_action_across_state_cv":float(np.median(vals)),"family_medians":{f:float(np.median(v)) for f,v in fams.items()}}
 state_pass=bool(state_cv["development"]["median_within_action_across_state_cv"]>=cfg["secondary_gates"]["state_dependence_cv_min"] and state_cv["validation"]["median_within_action_across_state_cv"]>=cfg["secondary_gates"]["state_dependence_cv_min"] and sum(v>=cfg["secondary_gates"]["state_dependence_cv_min"] for v in state_cv["validation"]["family_medians"].values())>=4)
 samej={}
 for role in ("development","validation"):
  g=primary[(primary.role==role)&(primary.horizon>0)].copy();g["j_q"]=[r.j_norm/max(refs[int(r.horizon)]["j"],1e-12) for r in g.itertuples()];by={str(int(h)):{"median_j_q":float(z.j_q.median()),"divergent_fraction":float((z.j_q>=cfg["secondary_gates"]["same_j_reentry_median_q_min"]).mean())} for h,z in g.groupby("horizon")};samej[role]=by
 samej_pass=bool(any(samej["development"][h]["median_j_q"]>=.25 and samej["validation"][h]["median_j_q"]>=.25 for h in samej["development"]))
 devval=bool(any(gates["development"][str(h)]["pass"] and gates["validation"][str(h)]["pass"] for h in (1,2,4,8)));strict_pass=devval
 controls={name:{str(int(h)):float(g.aggregate_raw_norm.median()) for h,g in primary_frame.groupby("horizon")} for name,primary_frame in [(name,enriched[enriched.control_type==name]) for name in ("ordinary_reference","same_norm_random","numerical_scale")]}
 scale=enriched[enriched.control_type=="scaling_diagnostic"].groupby(["role","candidate_id","horizon"]).future_potency_q.median().reset_index();scale_path=root/OUT/"scale_sign_temporal_v26.parquet";scale.to_parquet(scale_path,index=False,compression="zstd")
 final_open=bool(devval);result={"reference_scales":refs,"gates":gates,"V26_A_DEVVAL":devval,"V26_B_DEVVAL":strict_pass,"V26_C_DEVVAL":samej_pass,"V26_D_ROTATING":geometry["ROTATING_FUTURE_POTENT_GEOMETRY"],"V26_E_FIXED":geometry["FIXED_FUTURE_POTENT_GEOMETRY"],"V26_F_CHANNEL_SPECIFIC":channel_pass,"V26_G_STATE_DEPENDENT":state_pass,"same_j":samej,"channel_profile_ranges":{"development":ranges_dev.to_dict(),"validation":ranges_val.to_dict()},"state_dependence":state_cv,"controls":controls,"channel_interaction_medians":interactions.groupby(["horizon","term"]).interaction_ratio.median().unstack().to_dict() if len(interactions) else {},"finite_jvp_status":"FINITE_PRIMARY; exact temporal cache-state JVP unavailable in current runtime and not used for gates","natural_transition_status":"OBSERVATIONAL_ONLY_NOT_USED_IN_CAUSAL_GATES","current_readout_conditional_test":"controlled h0-equal persistent interventions compared under identical teacher forcing","final_opened":final_open,"finalist":cfg["finalist"],"future_based_bank_selection":False,"historical_final_opened":False,"v26_independent_final_opened":False};target=root/OUT/"v26_devval_analysis.json";write_json_atomic(target,result)
 inputs=[SOURCE,str(metrics_path.relative_to(root)),str(curves_path.relative_to(root)),str(emerge_path.relative_to(root)),str(geom_path.relative_to(root)),str(interaction_path.relative_to(root)),str(scale_path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_extension_rollout.freeze.json"]
 frozen=stage_freeze(root,"devval_analysis",inputs,{"analysis_sha256":sha256_file(target),"V26_A_DEVVAL":devval,"V26_B_DEVVAL":strict_pass,"final_opened":final_open,"historical_final_opened":False,"v26_independent_final_opened":False})
 opening=stage_freeze(root,"final_opening",[str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_devval_analysis.freeze.json"],{"final_opened":final_open,"single_finalist":cfg["finalist"],"final_opening_hash":json.loads((root/OUT/"design_v26.json").read_text())["final_opening_hash"],"threshold_tuning_on_final":False,"historical_final_opened":False,"v26_independent_final_opened":False})
 return {"analysis_freeze":frozen["freeze_digest"],"final_opening_freeze":opening["freeze_digest"],**result}

def adjudicate(root:Path):
 verify_stage(root,"final_rollout");cfg=verify(root)["config"];dev=json.loads((root/OUT/"v26_devval_analysis.json").read_text());final=pd.read_parquet(root/OUT/"temporal_rollout_final_v26.parquet");refs={int(h):v for h,v in dev["reference_scales"].items()};enriched=_with_q(final,refs,cfg);path=root/OUT/"temporal_metrics_final_v26.parquet";enriched.to_parquet(path,index=False,compression="zstd");primary=enriched[enriched.control_type=="primary"];family={f:float(z.future_potent.mean()) for f,z in primary.groupby("family")};final_pass=bool(primary.future_potency_q.median()>=.25 and primary.future_potent.mean()>=.25 and sum(v>=.25 for v in family.values())>=4)
 A=bool(dev["V26_A_DEVVAL"] and final_pass);B=bool(dev["V26_B_DEVVAL"] and final_pass);C=bool(dev["V26_C_DEVVAL"] and final_pass);D=bool(dev["V26_D_ROTATING"]);E=bool(dev["V26_E_FIXED"]);F=bool(dev["V26_F_CHANNEL_SPECIFIC"]);G=bool(dev["V26_G_STATE_DEPENDENT"]);H=bool(not A and not B);I=bool(not A and not B and not C);flags={"V26_A":A,"V26_B":B,"V26_C":C,"V26_D":D,"V26_E":E,"V26_F":F,"V26_G":G,"V26_H":H,"V26_I":I};labels={"V26_A":"V26-A_TEMPORAL_READOUT_POTENCY_CONFIRMED","V26_B":"V26-B_STRICT_CURRENT_SILENT_FUTURE_POTENCY","V26_C":"V26-C_SAME_WORKSPACE_FUTURE_DIVERGENCE_CONFIRMED","V26_D":"V26-D_ROTATING_FUTURE_POTENT_GEOMETRY","V26_E":"V26-E_FIXED_FUTURE_POTENT_GEOMETRY","V26_F":"V26-F_CHANNEL_SPECIFIC_TEMPORAL_POTENCY","V26_G":"V26-G_STATE_DEPENDENT_TEMPORAL_POTENCY","V26_H":"V26-H_NO_TEMPORAL_POTENCY_DETECTED","V26_I":"V26-I_CURRENT_READOUT_SUFFICIENT_UNDER_TESTED_PANEL"};outcomes=[labels[k] for k,v in flags.items() if v];routing=bool(A or B)
 result={"formal_outcomes":outcomes,"primary_outcome":"+".join(outcomes),**flags,"final_confirmation":{"median_q":float(primary.future_potency_q.median()),"potent_fraction":float(primary.future_potent.mean()),"family_fractions":family,"pass":final_pass,"rows":len(primary)},"CURRENT_READOUT_CAUSAL_SUFFICIENCY_REJECTED_UNDER_TESTED_PANEL":bool(A or B or C),"wording":"future-relevant persistent distinction; no memory-variable or complete-state claim","H2_REMAINS":True,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"CAUSAL_ROUTING_V27_AUTHORIZED":routing,"historical_final_opened":False,"v26_independent_final_opened":True,"future_based_bank_selection":False};target=root/OUT/"v26_adjudication.json";write_json_atomic(target,result);frozen=stage_freeze(root,"adjudication",[SOURCE,str(path.relative_to(root)),str(target.relative_to(root)),"artifacts/temporal_readout_potency_v26_final_rollout.freeze.json"],{"adjudication_sha256":sha256_file(target),"formal_outcomes":outcomes,"CAUSAL_ROUTING_V27_AUTHORIZED":routing,"historical_final_opened":False,"v26_independent_final_opened":True});return {"freeze_digest":frozen["freeze_digest"],**result}

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=("early","full","adjudicate"));a=p.parse_args();print(json.dumps(early(Path.cwd()) if a.command=="early" else full(Path.cwd()) if a.command=="full" else adjudicate(Path.cwd()),indent=2))
