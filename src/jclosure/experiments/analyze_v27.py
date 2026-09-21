"""Frozen-gate V27A analysis, finalist selection and routing authorization."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v27 import verify,verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic
SOURCE="src/jclosure/experiments/analyze_v27.py";OUT=Path("results/v27/processed")
def boot(x,n,seed):
 if not len(x):return 0.0
 rng=np.random.default_rng(seed);return float(np.quantile([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(n)],.025))
def gate(f,cfg,seed):
 if f.empty:return {"rows":0,"median_q":0.0,"potent_fraction":0.0,"bootstrap_2_5_lower":0.0,"family_fractions":{},"replicated_families":0,"pass":False}
 potent=(f.h1_q>=cfg["potency_threshold"]).to_numpy(float);fam={k:float((v.h1_q>=cfg["potency_threshold"]).mean()) for k,v in f.groupby("family")};rep=sum(v>=cfg["potent_fraction_min"] for v in fam.values());med=float(f.h1_q.median());frac=float(potent.mean());lo=boot(potent,cfg["bootstrap_samples"],seed);return {"rows":len(f),"states":int(f.base_trial_id.nunique()),"median_q":med,"potent_fraction":frac,"bootstrap_2_5_lower":lo,"family_fractions":fam,"replicated_families":rep,"pass":bool(med>=cfg["median_q_min"] and frac>=cfg["potent_fraction_min"] and lo>=cfg["bootstrap_lower_min"] and rep>=cfg["families_required"])}
def run(root:Path):
 verify_stage(root,"h1_response");cfg=verify(root)["config"];bank=pd.read_parquet(root/OUT/"pre_readout_silent_bank_v27.parquet");f=pd.read_parquet(root/OUT/"pre_readout_h1_v27.parquet");gates={r:gate(f[f.role==r],cfg["future_gate"],cfg["seed"]+i) for i,r in enumerate(("development","validation"))};A=all(x["pass"] for x in gates.values())
 distal={}
 for role in ("development","validation"):
  accepted=bank[(bank.role==role)&bank.PRE_READOUT_DISTAL][["base_trial_id","candidate_id"]];distal[role]=gate(f[f.role==role].merge(accepted,on=["base_trial_id","candidate_id"]),cfg["future_gate"],cfg["seed"]+10+(role=="validation"))
 auth=A or all(x["pass"] for x in distal.values())
 dev=[]
 for cid,x in f[f.role=="development"].groupby("candidate_id"):
  z=gate(x,cfg["future_gate"],cfg["seed"]+20);dev.append({"candidate_id":cid,**z})
 eligible=[x for x in dev if x["pass"]];winner=sorted(eligible,key=lambda x:(-x["median_q"],x["candidate_id"]))[0]["candidate_id"] if eligible else None
 valwin=gate(f[(f.role=="validation")&(f.candidate_id==winner)],cfg["future_gate"],cfg["seed"]+30) if winner else gate(f.iloc[:0],cfg["future_gate"],0);final_open=A and winner is not None and valwin["pass"]
 timing={r:{"median_pre_h1_q":float(x.h1_q.median()),"median_post_h1_q":float(x.post_h1_q.median()),"median_direction_cosine":float(x.pre_post_cosine.dropna().median()),"median_magnitude_ratio":float(x.pre_post_magnitude_ratio.median())} for r,x in f.groupby("role")}
 analysis={"V27_A_DEVVAL":A,"DETAILED_ROUTING_AUTHORIZED":auth,"gates":gates,"distal_gates":distal,"candidate_development_gates":dev,"routing_candidate_id":winner,"routing_candidate_hash":sha256_file(root/OUT/"pre_readout_h1_v27.parquet") if winner else None,"winner_validation_gate":valwin,"final_opened":final_open,"timing_comparison":timing,"future_based_bank_selection":False,"historical_final_opened":False,"v27_independent_final_opened":False};p=root/OUT/"v27a_analysis.json";write_json_atomic(p,analysis);fr=stage_freeze(root,"v27a_analysis",[SOURCE,str(p.relative_to(root)),"results/v27/processed/pre_readout_h1_v27.parquet","artifacts/pre_readout_reentry_v27_h1_response.freeze.json"],analysis)
 opening={"final_opened":final_open,"candidate_id":winner,"opening_rule":"V27A passes development+validation and development-selected candidate passes validation","FINAL_OPENING_HASH":sha256_file(p),"historical_final_opened":False,"v27_independent_final_opened":False};op=root/OUT/"final_opening_v27.json";write_json_atomic(op,opening);of=stage_freeze(root,"final_opening",[SOURCE,str(op.relative_to(root)),str(p.relative_to(root)),"artifacts/pre_readout_reentry_v27_v27a_analysis.freeze.json"],opening)
 return {"analysis_freeze":fr["freeze_digest"],"opening_freeze":of["freeze_digest"],**analysis}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
