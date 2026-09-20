"""Immutable V18 base and staged protocol freezes; V1--V17 remain read-only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

CONFIG = Path("configs/strong_state_context_ceiling_v18.yaml")
FREEZE = Path("artifacts/strong_state_context_ceiling_v18.freeze.json")
PARENT = "a41592fef6f6499f002efe1356fb3a4b8edb53d3"
PARENT_INPUTS = [
    "reports/V17_COMPLETE_REPORT.md",
    "artifacts/interventional_state_sufficiency_v17.freeze.json",
    "artifacts/interventional_state_sufficiency_v17_splits.freeze.json",
    "results/v17/processed/state_context_ceiling_v17.json",
    "results/v17/processed/conditional_raw_residual_v17.json",
    "results/v17/processed/v17_adjudication.json",
    "results/v16/processed/finite_action_bank_v16.parquet",
    "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json",
    "results/v13/processed/causal_capture_train_v13.json",
    "results/v13/processed/causal_capture_validation_v13.json",
    "reports/FINAL_REPORT.md",
]


def digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V18 base freeze exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V18 parent HEAD mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT:
        raise RuntimeError("V18 config parent mismatch")
    value = {
        "schema_version": 27,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "config_hash": sha256_file(root / CONFIG),
        "protocol_code_hash": sha256_file(root / "src/jclosure/protocol_v18.py"),
        "parent_hashes": {name: sha256_file(root / name) for name in PARENT_INPUTS},
        "parent_final_report_hash_is_pre_v18": True,
        "config": config,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / FREEZE, value)
    return value


def verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if value["freeze_digest"] != digest(value):
        raise RuntimeError("V18 base freeze digest mismatch")
    if sha256_file(root / CONFIG) != value["config_hash"]:
        raise RuntimeError("V18 config changed")
    if sha256_file(root / "src/jclosure/protocol_v18.py") != value["protocol_code_hash"]:
        raise RuntimeError("V18 protocol code changed")
    for path, expected in value["parent_hashes"].items():
        if path != "reports/FINAL_REPORT.md" and sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 parent changed: {path}")
    return value


def stage_freeze(root: Path, stage: str, paths: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = Path(f"artifacts/strong_state_context_ceiling_v18_{stage}.freeze.json")
    if (root / target).exists():
        raise RuntimeError(f"V18 stage already frozen: {target}")
    value = {"schema_version": 27, "protocol_version": f"strong_state_context_ceiling_v18_{stage}",
             "base_freeze_digest": base["freeze_digest"],
             "input_hashes": {path: sha256_file(root / path) for path in paths}, **detail}
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / target, value)
    return value


if __name__ == "__main__":
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "verify"
    print((freeze if command == "freeze" else verify)(Path.cwd())["freeze_digest"])
