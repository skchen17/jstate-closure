"""Append-only V28 protocol with source and stage integrity checks."""
from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import yaml
from jclosure.provenance import sha256_file,write_json_atomic
PARENT="83235995faa5aefb0f9fd9dab6f865a3506df2e1";CONFIG=Path("configs/token_state_transaction_v28.yaml");PREFIX="token_state_transaction_v28";BASE=Path(f"artifacts/{PREFIX}.freeze.json")
INPUTS=("reports/V27_COMPLETE_REPORT.md","reports/FINAL_REPORT.md","artifacts/pre_readout_reentry_v27_final.freeze.json","results/v27/processed/v27_adjudication.json","results/v27/processed/pre_vs_post_timing_v27.parquet","results/v26/processed/v26_devval_analysis.json","results/v25/processed/readout_compression_v25.json","results/v24/processed/target_projections_v24.npz")
def digest(x):return hashlib.sha256(json.dumps({k:v for k,v in x.items() if k!="freeze_digest"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def freeze(root):
 if (root/BASE).exists():raise RuntimeError("V28 base freeze exists")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
 if head!=PARENT:raise RuntimeError(f"V28 parent mismatch: {head}")
 cfg=yaml.safe_load((root/CONFIG).read_text())
 if cfg["parent_commit"]!=PARENT or cfg["authorization"]["H3_AUTHORIZED"]:raise RuntimeError("V28 config mismatch")
 x={"schema_version":37,"protocol_version":cfg["protocol_version"],"parent_commit":PARENT,"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),"protocol_sha256":sha256_file(root/"src/jclosure/protocol_v28.py"),"frozen_input_sha256":{p:sha256_file(root/p) for p in INPUTS},"config":cfg,"historical_final_opened":False,"V1_V27_records_mutated":False};x["freeze_digest"]=digest(x);write_json_atomic(root/BASE,x);return x
def verify(root):
 x=json.loads((root/BASE).read_text())
 if x["freeze_digest"]!=digest(x) or x["config_sha256"]!=sha256_file(root/CONFIG) or x["protocol_sha256"]!=sha256_file(root/"src/jclosure/protocol_v28.py"):raise RuntimeError("V28 base drift")
 for p,h in x["frozen_input_sha256"].items():
  if p!="reports/FINAL_REPORT.md" and sha256_file(root/p)!=h:raise RuntimeError(f"V28 historical drift: {p}")
 return x
def stage_freeze(root,name,inputs,payload):
 b=verify(root);p=root/f"artifacts/{PREFIX}_{name}.freeze.json"
 if p.exists():raise RuntimeError(f"V28 stage exists: {name}")
 x={"schema_version":37,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":b["freeze_digest"],"created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{q:sha256_file(root/q) for q in inputs},**payload};x["freeze_digest"]=digest(x);write_json_atomic(p,x);return x
def verify_stage(root,name):
 b=verify(root);x=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
 if x["freeze_digest"]!=digest(x) or x["base_freeze_digest"]!=b["freeze_digest"]:raise RuntimeError(f"V28 stage mismatch {name}")
 for p,h in x["input_sha256"].items():
  if sha256_file(root/p)!=h:raise RuntimeError(f"V28 stage input drift {name}:{p}")
 return x
if __name__=="__main__":
 import sys
 print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
