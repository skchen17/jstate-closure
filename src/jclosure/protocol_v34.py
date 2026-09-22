"""V34 append-only functional-mediation protocol and stage seals."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PREFIX="functional_mediation_v34"
PARENT="b8ee1284e72ec4a79d64b624a189de2ca57d2278"
CONFIG=Path("configs/functional_mediation_v34.yaml")
BASE=Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY=(
    "reports/V33_COMPLETE_REPORT.md",
    "reports/V33_ALL_REPORTS.md",
    "results/v33/processed/v33_adjudication.json",
    "results/v33/processed/v33_integrity_index.json",
    "results/v33/processed/phase_b_instrumentation_v33.json",
    "artifacts/cross_model_rec_conv_v33_adjudication.freeze.json",
    "reports/V32_COMPLETE_REPORT.md",
    "results/v32/processed/v32_adjudication.json",
)


def digest(record):
    return hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!="freeze_digest"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def freeze(root:Path):
    if (root/BASE).exists():raise RuntimeError("V34 base already frozen")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if head!=PARENT:raise RuntimeError(f"V34 parent mismatch: {head}")
    cfg=yaml.safe_load((root/CONFIG).read_text())
    for model in cfg["models"].values():
        path=Path(model["local_path"])
        for filename,expected in model["weight_sha256"].items():
            if sha256_file(path/filename)!=expected:raise RuntimeError(f"V34 model weight drift {filename}")
        if sha256_file(path/"tokenizer.json")!=model["tokenizer_sha256"]:raise RuntimeError("V34 tokenizer drift")
        if sha256_file(path/"config.json")!=model["config_sha256"]:raise RuntimeError("V34 model config drift")
    v33=json.loads((root/"results/v33/processed/v33_adjudication.json").read_text())
    for flag in ("V33-A_CROSS_MODEL_REC_CORRECTION_REPLICATED","V33-B_CONV_DOMINANT_HANDOFF_REPLICATED"):
        if not v33["formal_outcomes"][flag]:raise RuntimeError("V34 not authorized by V33")
    record={"schema_version":43,"protocol_version":PREFIX,"parent_commit":PARENT,"created_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256_file(root/CONFIG),"protocol_sha256":sha256_file(root/"src/jclosure/protocol_v34.py"),"frozen_history_sha256":{p:sha256_file(root/p) for p in HISTORY},"cumulative_report_at_start_sha256":sha256_file(root/"reports/FINAL_REPORT.md"),"config":cfg,"V1_V33_records_mutated":False,"historical_final_opened":False}
    record["freeze_digest"]=digest(record)
    write_json_atomic(root/BASE,record)
    return record


def verify(root:Path):
    record=json.loads((root/BASE).read_text())
    if record["freeze_digest"]!=digest(record) or record["config_sha256"]!=sha256_file(root/CONFIG) or record["protocol_sha256"]!=sha256_file(root/"src/jclosure/protocol_v34.py"):
        raise RuntimeError("V34 base drift")
    for path,expected in record["frozen_history_sha256"].items():
        if sha256_file(root/path)!=expected:raise RuntimeError(f"V34 historical drift: {path}")
    return record


def stage_freeze(root:Path,name:str,inputs,payload):
    base=verify(root)
    path=root/f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():raise RuntimeError(f"V34 stage already exists: {name}")
    record={"schema_version":43,"protocol_version":f"{PREFIX}_{name}","base_freeze_digest":base["freeze_digest"],"created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":{p:sha256_file(root/p) for p in inputs},**payload}
    record["freeze_digest"]=digest(record)
    write_json_atomic(path,record)
    return record


def verify_stage(root:Path,name:str):
    base=verify(root)
    record=json.loads((root/f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if record["freeze_digest"]!=digest(record) or record["base_freeze_digest"]!=base["freeze_digest"]:raise RuntimeError(f"V34 stage mismatch: {name}")
    for path,expected in record["input_sha256"].items():
        if sha256_file(root/path)!=expected:raise RuntimeError(f"V34 stage input drift: {name}:{path}")
    return record


if __name__=="__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:]==["freeze"] else verify(Path.cwd()))["freeze_digest"])
