"""Append-only V20 freeze protocol; V1–V19 records are strictly read-only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

CONFIG = Path("configs/compact_causal_response_operator_v20.yaml")
FREEZE = Path("artifacts/compact_causal_response_operator_v20.freeze.json")
PARENT = "3cefc7c61429770eeef7fd6544adf4b9b17a1de5"
PARENT_INPUTS = (
    "reports/V19_COMPLETE_REPORT.md",
    "reports/FINAL_REPORT.md",
    "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
    "artifacts/counterfactual_workspace_v19_splits.freeze.json",
    "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json",
    "artifacts/causal_bank_v13_splits.freeze.json",
    "data/v13/causal_bank_selected.json",
    "results/v19/processed/v19_adjudication.json",
    "results/v19/processed/factorial_summary_validation_v19.json",
    "results/v16/processed/finite_action_bank_v16.parquet",
    "artifacts/causal/v13/probe_directions_v13.pt",
)


def digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V20 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V20 parent mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT:
        raise RuntimeError("V20 config parent mismatch")
    value = {
        "schema_version": 29,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": sha256_file(root / CONFIG),
        "protocol_code_hash": sha256_file(root / "src/jclosure/protocol_v20.py"),
        "parent_hashes": {path: sha256_file(root / path) for path in PARENT_INPUTS},
        "config": config,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / FREEZE, value)
    return value


def verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if value["freeze_digest"] != digest(value):
        raise RuntimeError("V20 base digest mismatch")
    if sha256_file(root / CONFIG) != value["config_hash"]:
        raise RuntimeError("V20 config changed")
    if sha256_file(root / "src/jclosure/protocol_v20.py") != value["protocol_code_hash"]:
        raise RuntimeError("V20 protocol code changed")
    for path, expected in value["parent_hashes"].items():
        if path != "reports/FINAL_REPORT.md" and sha256_file(root / path) != expected:
            raise RuntimeError(f"V20 parent changed: {path}")
    return value


def stage_freeze(root: Path, name: str, paths: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = root / f"artifacts/compact_causal_response_operator_v20_{name}.freeze.json"
    if target.exists():
        raise RuntimeError(f"V20 stage already frozen: {target}")
    value = {
        "schema_version": 29,
        "protocol_version": f"compact_causal_response_operator_v20_{name}",
        "base_freeze_digest": base["freeze_digest"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": {path: sha256_file(root / path) for path in paths},
        **detail,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(target, value)
    return value


def verify_stage(root: Path, name: str) -> dict[str, Any]:
    base = verify(root)
    value = json.loads((root / f"artifacts/compact_causal_response_operator_v20_{name}.freeze.json").read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V20 {name} digest mismatch")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V20 {name} input changed: {path}")
    return value


if __name__ == "__main__":
    import sys

    print((freeze if sys.argv[1:] == ["freeze"] else verify)(Path.cwd())["freeze_digest"])
