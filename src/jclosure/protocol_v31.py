"""Append-only V31 protocol and integrity freezes."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PARENT = "5113e800cd4f287648cd52edc3577eb77cd60776"
PREFIX = "compositional_natural_writes_v31"
CONFIG = Path("configs/compositional_natural_writes_v31.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY = (
    "reports/V30_COMPLETE_REPORT.md",
    "reports/V30_ALL_REPORTS.md",
    "artifacts/transferable_natural_writes_v30_final_opening.freeze.json",
    "results/v30/processed/v30_integrity_index.json",
    "results/v30/processed/v30_adjudication.json",
    "reports/V29_COMPLETE_REPORT.md",
    "reports/V28_COMPLETE_REPORT.md",
)


def digest(record):
    data = {k: v for k, v in record.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path):
    path = root / BASE
    if path.exists():
        raise RuntimeError("V31 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V31 parent mismatch: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text())
    if cfg["parent_commit"] != PARENT or cfg["authorization"]["H3_AUTHORIZED"]:
        raise RuntimeError("V31 configuration/authorization mismatch")
    record = {"schema_version": 40, "protocol_version": PREFIX, "parent_commit": PARENT, "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256_file(root / CONFIG), "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v31.py"), "frozen_history_sha256": {p: sha256_file(root / p) for p in HISTORY}, "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"), "config": cfg, "historical_final_opened": False, "V1_V30_records_mutated": False}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify(root: Path):
    record = json.loads((root / BASE).read_text())
    if record["freeze_digest"] != digest(record) or record["config_sha256"] != sha256_file(root / CONFIG) or record["protocol_sha256"] != sha256_file(root / "src/jclosure/protocol_v31.py"):
        raise RuntimeError("V31 base protocol drift")
    for path, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"historical drift: {path}")
    return record


def stage_freeze(root: Path, name: str, inputs, payload):
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V31 stage already exists: {name}")
    record = {"schema_version": 40, "protocol_version": f"{PREFIX}_{name}", "base_freeze_digest": base["freeze_digest"], "created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": {p: sha256_file(root / p) for p in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, name: str):
    base = verify(root)
    record = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V31 stage mismatch: {name}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V31 stage input drift: {name}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
