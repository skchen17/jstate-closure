"""Immutable V17 base and stage freezes; V1--V16 are read-only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

CONFIG = Path("configs/interventional_state_sufficiency_v17.yaml")
FREEZE = Path("artifacts/interventional_state_sufficiency_v17.freeze.json")
PARENT = "60060cb481b6fe2fe6a9bbf123611d37d4445fea"
PARENT_INPUTS = [
    "reports/V16_COMPLETE_REPORT.md",
    "artifacts/nonlinear_finite_causal_action_v16.freeze.json",
    "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json",
    "results/v16/processed/finite_action_bank_v16.parquet",
    "results/v16/processed/v16_analysis.json",
    "results/v16/processed/v16_adjudication.json",
    "artifacts/causal_geometry_v13_jvp.freeze.json",
    "results/v13/processed/causal_capture_train_v13.json",
    "results/v13/processed/causal_capture_validation_v13.json",
    "reports/FINAL_REPORT.md",
]


def digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V17 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V17 parent HEAD mismatch: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["parent_commit"] != PARENT:
        raise RuntimeError("V17 config parent mismatch")
    value = {
        "schema_version": 26,
        "protocol_version": cfg["protocol_version"],
        "parent_commit": PARENT,
        "config_hash": sha256_file(root / CONFIG),
        "protocol_code_hash": sha256_file(root / "src/jclosure/protocol_v17.py"),
        "parent_hashes": {name: sha256_file(root / name) for name in PARENT_INPUTS},
        "parent_final_report_hash_is_pre_v17": True,
        "config": cfg,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / FREEZE, value)
    return value


def verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != digest(value):
        raise RuntimeError("V17 base freeze digest mismatch")
    if sha256_file(root / CONFIG) != value["config_hash"]:
        raise RuntimeError("V17 config changed")
    if sha256_file(root / "src/jclosure/protocol_v17.py") != value["protocol_code_hash"]:
        raise RuntimeError("V17 protocol code changed")
    for name, expected in value["parent_hashes"].items():
        if name != "reports/FINAL_REPORT.md" and sha256_file(root / name) != expected:
            raise RuntimeError(f"V17 parent changed: {name}")
    return value


def stage_freeze(root: Path, stage: str, paths: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = Path(f"artifacts/interventional_state_sufficiency_v17_{stage}.freeze.json")
    if (root / target).exists():
        raise RuntimeError(f"V17 stage already frozen: {target}")
    value = {
        "schema_version": 26,
        "protocol_version": f"interventional_state_sufficiency_v17_{stage}",
        "base_freeze_digest": base["freeze_digest"],
        "input_hashes": {path: sha256_file(root / path) for path in paths},
        **detail,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / target, value)
    return value


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "verify"
    print((freeze if command == "freeze" else verify)(Path.cwd())["freeze_digest"])
