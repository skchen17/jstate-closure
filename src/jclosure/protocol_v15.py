"""Immutable V15 protocol and provenance checks; V1--V14 are read-only parents."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

CONFIG = Path("configs/quantization_aware_actuation_v15.yaml")
FREEZE = Path("artifacts/quantization_aware_actuation_v15.freeze.json")
PARENT = "832d1a57f316a59bd961739f7caea4c4e0c702c8"
PARENT_INPUTS = [
    "reports/V14_COMPLETE_REPORT.md",
    "artifacts/finite_causal_control_v14.freeze.json",
    "artifacts/finite_causal_control_v14_finalist_decision.freeze.json",
    "results/v14/processed/numerical_snr_summary_v14.json",
    "results/v14/processed/jvp_finite_writeback_audit_v14.parquet",
    "artifacts/causal_geometry_v13_jvp.freeze.json",
    "results/v13/processed/causal_probe_scaling_v13.parquet",
    "reports/FINAL_REPORT.md",
]
SOURCES = [
    str(CONFIG),
    "src/jclosure/protocol_v15.py",
    "src/jclosure/experiments/actuation_v15.py",
]


def digest(value: dict[str, Any]) -> str:
    clean = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V15 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    if head != PARENT:
        raise RuntimeError(f"V15 parent HEAD mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT:
        raise RuntimeError("V15 config parent mismatch")
    value = {
        "schema_version": 24,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "source_hashes": {name: sha256_file(root / name) for name in SOURCES},
        "parent_hashes": {name: sha256_file(root / name) for name in PARENT_INPUTS},
        "parent_final_report_hash_is_pre_v15": True,
        "config": config,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / FREEZE, value)
    return value


def verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != digest(value):
        raise RuntimeError("V15 base freeze digest mismatch")
    for name, expected in value["source_hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V15 frozen source changed: {name}")
    for name, expected in value["parent_hashes"].items():
        if name == "reports/FINAL_REPORT.md":
            continue  # cumulative report is the one declared mutable parent
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V15 parent changed: {name}")
    return value


def stage_freeze(root: Path, name: str, inputs: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    parent = verify(root)
    path = Path(f"artifacts/quantization_aware_actuation_v15_{name}.freeze.json")
    if (root / path).exists():
        raise RuntimeError(f"V15 stage already frozen: {path}")
    value = {
        "schema_version": 24,
        "protocol_version": f"quantization_aware_causal_actuation_v15_{name}",
        "parent_freeze_digest": parent["freeze_digest"],
        "input_hashes": {item: sha256_file(root / item) for item in inputs},
        **detail,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / path, value)
    return value
