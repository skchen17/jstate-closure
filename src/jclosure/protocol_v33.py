"""Append-only V33 cross-model replication protocol and integrity freezes."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "cross_model_rec_conv_v33"
PARENT = "09a00a67e5a286a05e47730c43586bce94b8095a"
CONFIG = Path("configs/cross_model_rec_conv_v33.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY = (
    "reports/V32_COMPLETE_REPORT.md",
    "reports/V32_ALL_REPORTS.md",
    "results/v32/processed/v32_adjudication.json",
    "results/v32/processed/v32_integrity_index.json",
    "artifacts/rec_conv_mechanism_v32_final_opening.freeze.json",
    "reports/V31_COMPLETE_REPORT.md",
)


def digest(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "freeze_digest"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path):
    path = root / BASE
    if path.exists():
        raise RuntimeError("V33 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V33 parent mismatch: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text())
    model_path = Path(cfg["model2"]["local_path"])
    for filename, key in (("model.safetensors", "checkpoint_sha256"), ("tokenizer.json", "tokenizer_sha256"), ("config.json", "config_sha256")):
        if sha256_file(model_path / filename) != cfg["model2"][key]:
            raise RuntimeError(f"V33 model-2 frozen file drift: {filename}")
    v32 = json.loads((root / "results/v32/processed/v32_adjudication.json").read_text())
    if not v32["formal_outcomes"]["V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED"] or not v32["cross_model_replication_authorized"]:
        raise RuntimeError("V33 cross-model work not authorized by V32")
    record = {"schema_version": 42, "protocol_version": PREFIX, "parent_commit": PARENT, "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256_file(root / CONFIG), "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v33.py"), "frozen_history_sha256": {p: sha256_file(root / p) for p in HISTORY}, "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"), "model_file_sha256": {name: sha256_file(model_path / name) for name in ("model.safetensors", "tokenizer.json", "config.json")}, "config": cfg, "V1_V32_records_mutated": False, "historical_final_opened": False}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify(root: Path):
    record = json.loads((root / BASE).read_text())
    if record["freeze_digest"] != digest(record) or record["config_sha256"] != sha256_file(root / CONFIG) or record["protocol_sha256"] != sha256_file(root / "src/jclosure/protocol_v33.py"):
        raise RuntimeError("V33 base drift")
    for path, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V33 historical drift: {path}")
    return record


def stage_freeze(root: Path, name: str, inputs, payload):
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V33 stage already exists: {name}")
    record = {"schema_version": 42, "protocol_version": f"{PREFIX}_{name}", "base_freeze_digest": base["freeze_digest"], "created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": {p: sha256_file(root / p) for p in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, name: str):
    base = verify(root)
    record = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V33 stage mismatch: {name}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V33 stage input drift: {name}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
