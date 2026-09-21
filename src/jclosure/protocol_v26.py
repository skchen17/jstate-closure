"""Append-only V26 protocol and stage freezes."""
from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from jclosure.provenance import sha256_file, write_json_atomic

PARENT="04f70cebcb66a9bc2431052b44c488b1ce608d07"
CONFIG=Path("configs/temporal_readout_potency_v26.yaml")
FREEZE=Path("artifacts/temporal_readout_potency_v26.freeze.json")
PREFIX="temporal_readout_potency_v26"
INPUTS=(
 "reports/V25_COMPLETE_REPORT.md","reports/FINAL_REPORT.md",
 "artifacts/distributed_causal_interaction_v25_final.freeze.json",
 "results/v25/processed/v24_rank_audit_v25.json",
 "results/v25/processed/same_j_reconstruction_v25.parquet",
 "results/v25/processed/readout_compression_v25.json",
 "results/v25/processed/interaction_estimand_amendment_v25.json",
 "results/v24/processed/causal_bottleneck_mediation_v24.json",
 "results/v23/processed/input_rank_scaling_v23.json",
 "artifacts/causal_action_manifold_v22_final.freeze.json",
 "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
 "artifacts/counterfactual_workspace_v19_splits.freeze.json",
 "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
 "results/v24/processed/target_projections_v24.npz",
 "results/v24/processed/bottleneck_design_v24.json",
 "results/v18/processed/crossed_state_train_boolean_logic_v18.parquet",
 "results/v18/processed/crossed_state_validation_boolean_logic_v18.parquet",
)

def _digest(value:dict[str,Any])->str:
 body={k:v for k,v in value.items() if k!="freeze_digest"}
 return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def freeze(root:Path)->dict[str,Any]:
 target=root/FREEZE
 if target.exists():raise RuntimeError("V26 base freeze already exists")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if head!=PARENT:raise RuntimeError(f"V26 parent mismatch: {head}")
 config=yaml.safe_load((root/CONFIG).read_text())
 auth=config["authorization"]
 if config["parent_commit"]!=PARENT or auth["H3_AUTHORIZED"] or auth["DYNAMIC_STATE_SEARCH_AUTHORIZED"] or auth["CAUSAL_ROUTING_V27_AUTHORIZED_default"]:raise RuntimeError("V26 authorization mismatch")
 value={"schema_version":35,"protocol_version":config["protocol_version"],"parent_commit":PARENT,"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),"protocol_code_sha256":sha256_file(root/"src/jclosure/protocol_v26.py"),"frozen_input_sha256":{p:sha256_file(root/p) for p in INPUTS},"config":config,"V1_V25_records_mutated":False,"historical_final_opened":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"CAUSAL_ROUTING_V27_AUTHORIZED":False}
 value["freeze_digest"]=_digest(value);write_json_atomic(target,value);return value

def verify(root:Path)->dict[str,Any]:
 value=json.loads((root/FREEZE).read_text())
 if value["freeze_digest"]!=_digest(value):raise RuntimeError("V26 digest mismatch")
 if sha256_file(root/CONFIG)!=value["config_sha256"] or sha256_file(root/"src/jclosure/protocol_v26.py")!=value["protocol_code_sha256"]:raise RuntimeError("V26 source drift")
 for p,h in value["frozen_input_sha256"].items():
  if p!="reports/FINAL_REPORT.md" and sha256_file(root/p)!=h:raise RuntimeError(f"V26 frozen input drift: {p}")
 return value

def stage_freeze(root:Path,name:str,inputs:list[str],payload:dict[str,Any])->dict[str,Any]:
 base=verify(root);target=root/f"artifacts/{PREFIX}_{name}.freeze.json"
 if target.exists():raise RuntimeError(f"V26 stage exists: {name}")
 value={"schema_version":35,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":base["freeze_digest"],"created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{p:sha256_file(root/p) for p in inputs},**payload}
 value["freeze_digest"]=_digest(value);write_json_atomic(target,value);return value

def verify_stage(root:Path,name:str)->dict[str,Any]:
 base=verify(root);value=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
 if value["freeze_digest"]!=_digest(value) or value["base_freeze_digest"]!=base["freeze_digest"]:raise RuntimeError(f"V26 stage mismatch: {name}")
 for p,h in value["input_sha256"].items():
  if sha256_file(root/p)!=h:raise RuntimeError(f"V26 stage source drift: {name}:{p}")
 return value

if __name__=="__main__":
 import sys
 print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
