"""Append-only clamp amendment for finite-dimensional V25 rank counters."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from jclosure.protocol_v25 import verify_stage,stage_freeze
from jclosure.provenance import sha256_file,write_json_atomic

SOURCE="src/jclosure/experiments/rank_convergence_amendment_v25.py";OUT=Path("results/v25/processed")
def run(root:Path)->dict:
 verify_stage(root,"rank_convergence");source=root/OUT/"layerwise_causal_convergence_v25.parquet";frame=pd.read_parquet(source)
 corrections={}
 for name in ("centered_r90","centered_r95","centered_r99"):
  before=frame[name].copy();frame[name]=frame[name].clip(upper=256).astype(int);corrections[name]=int((before!=frame[name]).sum())
 target=root/OUT/"layerwise_causal_convergence_corrected_v25.parquet";frame.to_parquet(target,index=False,compression="zstd")
 mapping=json.loads((root/OUT/"many_to_one_mapping_v25.json").read_text())
 for row in mapping["mappings"]:
  for name in ("mapping_r95","upstream_centered_r95","downstream_centered_r95"):row[name]=min(int(row[name]),256)
 mapping_target=root/OUT/"many_to_one_mapping_corrected_v25.json";write_json_atomic(mapping_target,mapping)
 result={"reason":"floating cumulative energy ended infinitesimally below one; searchsorted returned dimension+1","ambient_dimension":256,"corrections":corrections,"early_r95":int(frame.iloc[0].centered_r95),"late_r95":int(frame.iloc[-1].centered_r95),"corrected_convergence_sha256":sha256_file(target),"corrected_mapping_sha256":sha256_file(mapping_target),"original_records_overwritten":False,"historical_final_opened":False,"v25_independent_final_opened":False}
 result_path=root/OUT/"rank_convergence_amendment_v25.json";write_json_atomic(result_path,result);frozen=stage_freeze(root,"rank_convergence_amendment",[SOURCE,"artifacts/distributed_causal_interaction_v25_rank_convergence.freeze.json",str(target.relative_to(root)),str(mapping_target.relative_to(root)),str(result_path.relative_to(root))],result);return {"freeze_digest":frozen["freeze_digest"],**result}
if __name__=="__main__":print(json.dumps(run(Path.cwd()),indent=2))
