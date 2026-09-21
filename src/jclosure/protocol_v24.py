"""Append-only V24 freezes; V1--V23 remain immutable inputs."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import yaml
from jclosure.provenance import sha256_file,write_json_atomic

PARENT="4dcc412d85fb2995237c48ec755beb34fd662941"
CONFIG=Path("configs/causal_output_bottleneck_v24.yaml")
FREEZE=Path("artifacts/causal_output_bottleneck_v24.freeze.json")
PREFIX="causal_output_bottleneck_v24"
INPUTS=("reports/V23_COMPLETE_REPORT.md","reports/FINAL_REPORT.md",
"artifacts/oracle_local_action_charts_v23_final.freeze.json",
"results/v23/processed/input_rank_scaling_v23.json","results/v23/processed/oracle_local_action_charts_v23.json",
"results/v23/processed/probe_selection_v23.json","results/v23/processed/probe_candidate_design_v23.json",
"results/v22/processed/action_pool_calibration_v22.json","results/v22/processed/action_selection_v22.json",
"artifacts/action_coordinate_geometry_v21_roles.freeze.json","artifacts/compact_causal_response_operator_v20_splits.freeze.json",
"artifacts/compact_causal_response_operator_v20_operator_design.freeze.json","artifacts/causal/v13/probe_directions_v13.pt")

def _digest(value:dict[str,Any])->str:
 body={k:v for k,v in value.items() if k!="freeze_digest"}
 return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def freeze(root:Path)->dict[str,Any]:
 target=root/FREEZE
 if target.exists(): raise RuntimeError("V24 base freeze already exists")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if head!=PARENT: raise RuntimeError(f"V24 parent mismatch: {head}")
 config=yaml.safe_load((root/CONFIG).read_text())
 if config["parent_commit"]!=PARENT or config["authorization"]["H3_AUTHORIZED"] or config["authorization"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"]:
  raise RuntimeError("V24 parent/authorization mismatch")
 value={"schema_version":33,"protocol_version":config["protocol_version"],"parent_commit":PARENT,
 "created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),
 "protocol_code_sha256":sha256_file(root/"src/jclosure/protocol_v24.py"),
 "frozen_input_sha256":{p:sha256_file(root/p) for p in INPUTS},"config":config,
 "V1_V23_records_mutated":False,"historical_final_opened":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False}
 value["freeze_digest"]=_digest(value); write_json_atomic(target,value); return value

def verify(root:Path)->dict[str,Any]:
 value=json.loads((root/FREEZE).read_text())
 if value["freeze_digest"]!=_digest(value): raise RuntimeError("V24 digest mismatch")
 if sha256_file(root/CONFIG)!=value["config_sha256"] or sha256_file(root/"src/jclosure/protocol_v24.py")!=value["protocol_code_sha256"]: raise RuntimeError("V24 source drift")
 for p,h in value["frozen_input_sha256"].items():
  if p!="reports/FINAL_REPORT.md" and sha256_file(root/p)!=h: raise RuntimeError(f"V24 frozen input drift: {p}")
 return value

def stage_freeze(root:Path,name:str,inputs:list[str],payload:dict[str,Any])->dict[str,Any]:
 base=verify(root); target=root/f"artifacts/{PREFIX}_{name}.freeze.json"
 if target.exists(): raise RuntimeError(f"V24 stage exists: {name}")
 value={"schema_version":33,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":base["freeze_digest"],
 "created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{p:sha256_file(root/p) for p in inputs},**payload}
 value["freeze_digest"]=_digest(value); write_json_atomic(target,value); return value

def verify_stage(root:Path,name:str)->dict[str,Any]:
 base=verify(root); value=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
 if value["freeze_digest"]!=_digest(value) or value["base_freeze_digest"]!=base["freeze_digest"]: raise RuntimeError(f"V24 stage mismatch: {name}")
 for p,h in value["input_sha256"].items():
  if sha256_file(root/p)!=h: raise RuntimeError(f"V24 stage source drift: {name}:{p}")
 return value

if __name__=="__main__":
 import sys
 print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
