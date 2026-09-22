"""Frozen-gate reciprocal natural-write transplant adjudication."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from jclosure.protocol_v28 import verify,verify_stage,stage_freeze
from jclosure.provenance import write_json_atomic
SOURCE="src/jclosure/experiments/analyze_transplant_v28.py";OUT=Path("results/v28/processed");PRIMARY="REC+Conv+KV_last_slot_previous_copy"
def summarize(f,cfg):
 out={}
 for c,g in f.groupby("condition"):
  by={}
  for direction,x in g.groupby("direction"):
   success=(x.donor_target_cosine>=cfg["transplant_cosine_min"])&(x.magnitude_ratio>=cfg["transplant_magnitude_min"])&x.writeback_pass;fam={name:float(((v.donor_target_cosine>=cfg["transplant_cosine_min"])&(v.magnitude_ratio>=cfg["transplant_magnitude_min"])&v.writeback_pass).mean()) for name,v in x.groupby("family")};by[direction]={"rows":len(x),"median_cosine":float(x.donor_target_cosine.median()),"median_magnitude_ratio":float(x.magnitude_ratio.median()),"median_relative_l2":float(x.relative_l2_to_donor.median()),"success_fraction":float(success.mean()),"family_fractions":fam,"replicated_families":sum(v>=cfg["future_fraction_min"] for v in fam.values()),"all_writeback":bool(x.writeback_pass.all()),"pass":bool(float(x.donor_target_cosine.median())>=cfg["transplant_cosine_min"] and float(x.magnitude_ratio.median())>=cfg["transplant_magnitude_min"] and float(success.mean())>=cfg["future_fraction_min"] and sum(v>=cfg["future_fraction_min"] for v in fam.values())>=cfg["families_required"] and bool(x.writeback_pass.all()))}
  out[c]={"directions":by,"reciprocal_pass":all(v["pass"] for v in by.values())}
 return out
def development(root):
 verify_stage(root,"transplant_development");cfg=verify(root)["config"];f=pd.read_parquet(root/OUT/"natural_write_transplant_development_v28.parquet");profiles=summarize(f,cfg);x={"profiles":profiles,"primary_transplant_condition":PRIMARY,"primary_selected_by_architecture_not_response":True,"primary_development_pass":profiles[PRIMARY]["reciprocal_pass"],"validation_responses_observed_before_freeze":0,"pilot_state_excluded":True};p=root/OUT/"transplant_development_analysis_v28.json";write_json_atomic(p,x);fr=stage_freeze(root,"transplant_development_analysis",[SOURCE,str(p.relative_to(root)),"results/v28/processed/natural_write_transplant_development_v28.parquet","artifacts/token_state_transaction_v28_transplant_development.freeze.json"],x);return {"freeze_digest":fr["freeze_digest"],"primary_development_pass":x["primary_development_pass"],"profiles":profiles}
def validation(root):
 verify_stage(root,"transplant_validation");dev=verify_stage(root,"transplant_development_analysis");cfg=verify(root)["config"];f=pd.read_parquet(root/OUT/"natural_write_transplant_validation_v28.parquet");profiles=summarize(f,cfg);x={"profiles":profiles,"primary_transplant_condition":PRIMARY,"primary_validation_pass":profiles[PRIMARY]["reciprocal_pass"],"V28_C_DEVVAL":bool(dev["primary_development_pass"] and profiles[PRIMARY]["reciprocal_pass"]),"historical_final_opened":False};p=root/OUT/"transplant_validation_analysis_v28.json";write_json_atomic(p,x);fr=stage_freeze(root,"transplant_validation_analysis",[SOURCE,str(p.relative_to(root)),"results/v28/processed/natural_write_transplant_validation_v28.parquet","artifacts/token_state_transaction_v28_transplant_validation.freeze.json"],x);return {"freeze_digest":fr["freeze_digest"],"primary_validation_pass":x["primary_validation_pass"],"V28_C_DEVVAL":x["V28_C_DEVVAL"],"profiles":profiles}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("role",choices=("development","validation"));a=p.parse_args();print(json.dumps((development if a.role=="development" else validation)(Path.cwd()),indent=2))
