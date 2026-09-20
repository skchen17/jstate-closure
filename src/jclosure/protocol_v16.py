"""V16 immutable protocol and staged provenance. V1--V15 are read-only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

CONFIG = Path("configs/nonlinear_finite_causal_action_v16.yaml")
FREEZE = Path("artifacts/nonlinear_finite_causal_action_v16.freeze.json")
PARENT = "f1986b1b806112b1c337eb1084b3b82ae11ffb1f"
PARENT_INPUTS = [
    "reports/V15_COMPLETE_REPORT.md",
    "artifacts/quantization_aware_actuation_v15.freeze.json",
    "artifacts/quantization_aware_actuation_v15_finalist_decision.freeze.json",
    "results/v15/processed/finite_response_linearity_v15.json",
    "results/v15/processed/finite_operator_columns_v15.parquet",
    "results/v15/processed/actuator_transfer_corrected_v15.json",
    "artifacts/causal_geometry_v13_jvp.freeze.json",
    "results/v13/processed/probe_directions_v13.json",
    "reports/FINAL_REPORT.md",
]


def digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V16 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V16 parent HEAD mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT:
        raise RuntimeError("V16 config parent mismatch")
    value = {
        "schema_version": 25,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "config_hash": sha256_file(root / CONFIG),
        "protocol_code_hash": sha256_file(root / "src/jclosure/protocol_v16.py"),
        "parent_hashes": {name: sha256_file(root / name) for name in PARENT_INPUTS},
        "parent_final_report_hash_is_pre_v16": True,
        "config": config,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / FREEZE, value)
    return value


def verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != digest(value):
        raise RuntimeError("V16 base freeze digest mismatch")
    if sha256_file(root / CONFIG) != value["config_hash"]:
        raise RuntimeError("V16 config changed")
    if sha256_file(root / "src/jclosure/protocol_v16.py") != value["protocol_code_hash"]:
        raise RuntimeError("V16 protocol code changed")
    for name, expected in value["parent_hashes"].items():
        if name != "reports/FINAL_REPORT.md" and sha256_file(root / name) != expected:
            raise RuntimeError(f"V16 parent changed: {name}")
    return value


def stage_freeze(root: Path, stage: str, paths: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = Path(f"artifacts/nonlinear_finite_causal_action_v16_{stage}.freeze.json")
    if (root / target).exists():
        raise RuntimeError(f"V16 stage already frozen: {target}")
    value = {
        "schema_version": 25,
        "protocol_version": f"nonlinear_finite_causal_action_v16_{stage}",
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
