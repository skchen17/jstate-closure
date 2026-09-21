"""Append-only V27 protocol and sequential stage freezes."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import yaml
from jclosure.provenance import sha256_file,write_json_atomic

PARENT="0657f731e8a165425baa231d4880bab36bf01169"
CONFIG=Path("configs/pre_readout_reentry_v27.yaml")
FREEZE=Path("artifacts/pre_readout_reentry_v27.freeze.json")
PREFIX="pre_readout_reentry_v27"
INPUTS=("reports/V26_COMPLETE_REPORT.md","reports/FINAL_REPORT.md","artifacts/temporal_readout_potency_v26_final.freeze.json","results/v26/processed/current_distal_bank_v26.parquet","results/v26/processed/v26_devval_analysis.json","results/v26/processed/v26_adjudication.json","results/v25/processed/readout_compression_v25.json","results/v24/processed/target_projections_v24.npz","artifacts/counterfactual_workspace_v19_splits.freeze.json")

def digest(value:dict[str,Any])->str:
 body={k:v for k,v in value.items() if k!="freeze_digest"}
 return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def freeze(root:Path)->dict[str,Any]:
 target=root/FREEZE
 if target.exists():raise RuntimeError("V27 base freeze already exists")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if head!=PARENT:raise RuntimeError(f"V27 parent mismatch: {head}")
 cfg=yaml.safe_load((root/CONFIG).read_text())
 if cfg["parent_commit"]!=PARENT or cfg["authorization"]["DETAILED_ROUTING_AUTHORIZED_default"]:raise RuntimeError("V27 authorization mismatch")
 value={"schema_version":36,"protocol_version":cfg["protocol_version"],"parent_commit":PARENT,"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),"protocol_code_sha256":sha256_file(root/"src/jclosure/protocol_v27.py"),"frozen_input_sha256":{p:sha256_file(root/p) for p in INPUTS},"config":cfg,"V1_V26_records_mutated":False,"historical_final_opened":False,"DETAILED_ROUTING_AUTHORIZED":False,"H3_AUTHORIZED":False,"DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"AUTONOMOUS_CONTROLLER_AUTHORIZED":False}
 value["freeze_digest"]=digest(value);write_json_atomic(target,value);return value
def verify(root:Path)->dict[str,Any]:
 value=json.loads((root/FREEZE).read_text())
 if value["freeze_digest"]!=digest(value):raise RuntimeError("V27 digest mismatch")
 if sha256_file(root/CONFIG)!=value["config_sha256"] or sha256_file(root/"src/jclosure/protocol_v27.py")!=value["protocol_code_sha256"]:raise RuntimeError("V27 source drift")
 for p,h in value["frozen_input_sha256"].items():
  if p!="reports/FINAL_REPORT.md" and sha256_file(root/p)!=h:raise RuntimeError(f"V27 frozen input drift: {p}")
 return value
def stage_freeze(root:Path,name:str,inputs:list[str],payload:dict[str,Any])->dict[str,Any]:
 base=verify(root);target=root/f"artifacts/{PREFIX}_{name}.freeze.json"
 if target.exists():raise RuntimeError(f"V27 stage exists: {name}")
 value={"schema_version":36,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":base["freeze_digest"],"created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{p:sha256_file(root/p) for p in inputs},**payload};value["freeze_digest"]=digest(value);write_json_atomic(target,value);return value
def verify_stage(root:Path,name:str)->dict[str,Any]:
 base=verify(root);value=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
 if value["freeze_digest"]!=digest(value) or value["base_freeze_digest"]!=base["freeze_digest"]:raise RuntimeError(f"V27 stage mismatch: {name}")
 for p,h in value["input_sha256"].items():
  if sha256_file(root/p)!=h:raise RuntimeError(f"V27 stage source drift: {name}:{p}")
 return value
if __name__=="__main__":
 import sys
 print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
