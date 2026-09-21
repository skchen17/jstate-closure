"""Append-only V25 freezes; V1--V24 remain immutable inputs."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import yaml
from jclosure.provenance import sha256_file,write_json_atomic

PARENT="2197d58dc6699d3a6bba467f703f267fa3dcabc7"
CONFIG=Path("configs/distributed_causal_interaction_v25.yaml")
FREEZE=Path("artifacts/distributed_causal_interaction_v25.freeze.json")
PREFIX="distributed_causal_interaction_v25"
INPUTS=(
 "reports/V24_COMPLETE_REPORT.md","reports/FINAL_REPORT.md",
 "artifacts/causal_output_bottleneck_v24_final.freeze.json",
 "artifacts/causal_output_bottleneck_v24_candidate_selection.freeze.json",
 "artifacts/causal_output_bottleneck_v24_mediation_protocol.freeze.json",
 "artifacts/causal_output_bottleneck_v24_mediation_results_amendment.freeze.json",
 "results/v24/processed/bottleneck_design_v24.json",
 "results/v24/processed/layerwise_rank_development_v24.parquet",
 "results/v24/processed/layerwise_rank_validation_v24.parquet",
 "results/v24/processed/layerwise_jvp_rank_v24.parquet",
 "results/v24/processed/output_bottleneck_validation_v24.json",
 "results/v24/processed/mediation_branches_v24.parquet",
 "results/v24/processed/causal_bottleneck_mediation_v24.json",
 "results/v24/processed/raw_hidden_bottleneck_basis_v24.npz",
 "results/v24/processed/strict_writeback_audit_v24.parquet",
 "results/v24/processed/v24_adjudication.json",
 "results/v23/processed/input_rank_scaling_v23.json",
 "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
 "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
 "artifacts/counterfactual_workspace_v19_splits.freeze.json",
)

def _digest(value:dict[str,Any])->str:
 body={k:v for k,v in value.items() if k!="freeze_digest"}
 return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def freeze(root:Path)->dict[str,Any]:
 target=root/FREEZE
 if target.exists(): raise RuntimeError("V25 base freeze already exists")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if head!=PARENT: raise RuntimeError(f"V25 parent mismatch: {head}")
 config=yaml.safe_load((root/CONFIG).read_text())
 if config["parent_commit"]!=PARENT or config["authorization"]["H3_AUTHORIZED"] or config["authorization"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"]:
  raise RuntimeError("V25 parent/authorization mismatch")
 value={"schema_version":34,"protocol_version":config["protocol_version"],"parent_commit":PARENT,
  "created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),
  "protocol_code_sha256":sha256_file(root/"src/jclosure/protocol_v25.py"),
  "frozen_input_sha256":{p:sha256_file(root/p) for p in INPUTS},"config":config,
  "V1_V24_records_mutated":False,"historical_final_opened":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False}
 value["freeze_digest"]=_digest(value); write_json_atomic(target,value); return value

def verify(root:Path)->dict[str,Any]:
 value=json.loads((root/FREEZE).read_text())
 if value["freeze_digest"]!=_digest(value): raise RuntimeError("V25 digest mismatch")
 if sha256_file(root/CONFIG)!=value["config_sha256"] or sha256_file(root/"src/jclosure/protocol_v25.py")!=value["protocol_code_sha256"]: raise RuntimeError("V25 source drift")
 for p,h in value["frozen_input_sha256"].items():
  if p!="reports/FINAL_REPORT.md" and sha256_file(root/p)!=h: raise RuntimeError(f"V25 frozen input drift: {p}")
 return value

def stage_freeze(root:Path,name:str,inputs:list[str],payload:dict[str,Any])->dict[str,Any]:
 base=verify(root); target=root/f"artifacts/{PREFIX}_{name}.freeze.json"
 if target.exists(): raise RuntimeError(f"V25 stage exists: {name}")
 value={"schema_version":34,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":base["freeze_digest"],
  "created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{p:sha256_file(root/p) for p in inputs},**payload}
 value["freeze_digest"]=_digest(value); write_json_atomic(target,value); return value

def verify_stage(root:Path,name:str)->dict[str,Any]:
 base=verify(root); value=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
 if value["freeze_digest"]!=_digest(value) or value["base_freeze_digest"]!=base["freeze_digest"]: raise RuntimeError(f"V25 stage mismatch: {name}")
 for p,h in value["input_sha256"].items():
  if sha256_file(root/p)!=h: raise RuntimeError(f"V25 stage source drift: {name}:{p}")
 return value

if __name__=="__main__":
 import sys
 print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
