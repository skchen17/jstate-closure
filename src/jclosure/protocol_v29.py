"""Append-only V29 provenance and stage freezes."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PARENT = "8d3954ef05957ab22f27261ea7eaa217497b2b62"
PREFIX = "natural_write_content_v29"
CONFIG = Path("configs/natural_write_content_v29.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY = (
    "reports/V28_COMPLETE_REPORT.md",
    "artifacts/token_state_transaction_v28_final.freeze.json",
    "results/v28/processed/v28_adjudication.json",
    "results/v28/processed/v28_integrity_index.json",
    "reports/V27_COMPLETE_REPORT.md",
    "reports/V26_COMPLETE_REPORT.md",
    "reports/V25_COMPLETE_REPORT.md",
)


def digest(record):
    content = {k: v for k, v in record.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path):
    path = root / BASE
    if path.exists():
        raise RuntimeError("V29 protocol already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V29 parent mismatch: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text())
    if cfg["parent_commit"] != PARENT or cfg["authorization"]["H3_AUTHORIZED"]:
        raise RuntimeError("V29 authorization/config mismatch")
    x = {
        "schema_version": 38,
        "protocol_version": PREFIX,
        "parent_commit": PARENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v29.py"),
        "frozen_history_sha256": {p: sha256_file(root / p) for p in HISTORY},
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": cfg,
        "historical_final_opened": False,
        "V1_V28_records_mutated": False,
    }
    x["freeze_digest"] = digest(x)
    write_json_atomic(path, x)
    return x


def verify(root: Path):
    x = json.loads((root / BASE).read_text())
    if x["freeze_digest"] != digest(x):
        raise RuntimeError("V29 protocol freeze digest mismatch")
    if x["config_sha256"] != sha256_file(root / CONFIG) or x["protocol_sha256"] != sha256_file(root / "src/jclosure/protocol_v29.py"):
        raise RuntimeError("V29 protocol drift")
    for p, h in x["frozen_history_sha256"].items():
        if sha256_file(root / p) != h:
            raise RuntimeError(f"historical drift: {p}")
    return x


def stage_freeze(root: Path, name: str, inputs, payload):
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V29 stage already frozen: {name}")
    x = {
        "schema_version": 38,
        "protocol_version": f"{PREFIX}_{name}",
        "base_freeze_digest": base["freeze_digest"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {p: sha256_file(root / p) for p in inputs},
        **payload,
    }
    x["freeze_digest"] = digest(x)
    write_json_atomic(path, x)
    return x


def verify_stage(root: Path, name: str):
    base = verify(root)
    x = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if x["freeze_digest"] != digest(x) or x["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V29 stage mismatch: {name}")
    for p, h in x["input_sha256"].items():
        if sha256_file(root / p) != h:
            raise RuntimeError(f"V29 stage input drift: {name}:{p}")
    return x


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
