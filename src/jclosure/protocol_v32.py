"""Immutable V32 base and stage records; historic versions are read-only."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "rec_conv_mechanism_v32"
PARENT = "774a867eb4ffb91ef749e971fb78922ed312a285"
CONFIG = Path("configs/rec_conv_mechanism_v32.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY = (
    "reports/V31_COMPLETE_REPORT.md",
    "reports/V31_ALL_REPORTS.md",
    "results/v31/processed/v31_integrity_index.json",
    "artifacts/compositional_natural_writes_v31_final_opening.freeze.json",
    "reports/V30_COMPLETE_REPORT.md",
    "reports/V29_COMPLETE_REPORT.md",
    "reports/V28_COMPLETE_REPORT.md",
)


def digest(record):
    body = {k: v for k, v in record.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path):
    path = root / BASE
    if path.exists():
        raise RuntimeError("V32 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V32 parent mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text())
    if config["parent_commit"] != PARENT or any(config["authorization"][k] for k in ("H3_AUTHORIZED", "DYNAMIC_STATE_SEARCH_AUTHORIZED", "AUTONOMOUS_CONTROLLER_AUTHORIZED", "CROSS_MODEL_REPLICATION_AUTHORIZED")):
        raise RuntimeError("V32 config authorization mismatch")
    record = {
        "schema_version": 41,
        "protocol_version": PREFIX,
        "parent_commit": PARENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v32.py"),
        "frozen_history_sha256": {p: sha256_file(root / p) for p in HISTORY},
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": config,
        "historical_final_opened": False,
        "V1_V31_records_mutated": False,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify(root: Path):
    record = json.loads((root / BASE).read_text())
    if record["freeze_digest"] != digest(record) or record["config_sha256"] != sha256_file(root / CONFIG) or record["protocol_sha256"] != sha256_file(root / "src/jclosure/protocol_v32.py"):
        raise RuntimeError("V32 base drift")
    for path, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V32 historical drift: {path}")
    return record


def stage_freeze(root: Path, name: str, inputs, payload):
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V32 stage exists: {name}")
    record = {
        "schema_version": 41,
        "protocol_version": f"{PREFIX}_{name}",
        "base_freeze_digest": base["freeze_digest"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {p: sha256_file(root / p) for p in inputs},
        **payload,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, name: str):
    base = verify(root)
    record = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V32 stage mismatch: {name}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V32 stage input drift: {name}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
