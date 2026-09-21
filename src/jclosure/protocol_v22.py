"""Append-only V22 freezes; V1--V21 remain immutable inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PARENT = "351da6f061c3ed13e90324f50afc30f53a8796de"
CONFIG = Path("configs/causal_action_manifold_v22.yaml")
FREEZE = Path("artifacts/causal_action_manifold_v22.freeze.json")
PREFIX = "causal_action_manifold_v22"
INPUTS = (
    "reports/V21_COMPLETE_REPORT.md",
    "reports/FINAL_REPORT.md",
    "artifacts/action_coordinate_geometry_v21.freeze.json",
    "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
    "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json",
    "artifacts/action_coordinate_geometry_v21_state_representations.freeze.json",
    "results/v21/processed/paired_geometry_analysis_v21.json",
    "results/v21/processed/paired_jvp_finite_operator_v21.parquet",
    "results/v21/processed/v21_adjudication.json",
    "artifacts/compact_causal_response_operator_v20_actions.freeze.json",
    "artifacts/causal/v13/probe_directions_v13.pt",
)


def _digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    target = root / FREEZE
    if target.exists():
        raise RuntimeError("V22 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V22 parent commit mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT or config["authorization"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"]:
        raise RuntimeError("V22 parent or authorization mismatch")
    detail = {
        "schema_version": 31,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_code_sha256": sha256_file(root / "src/jclosure/protocol_v22.py"),
        "frozen_input_sha256": {name: sha256_file(root / name) for name in INPUTS},
        "config": config,
        "V1_V21_records_mutated": False,
        "historical_final_six_opened": False,
        "dynamic_state_search_authorized": False,
    }
    detail["freeze_digest"] = _digest(detail)
    write_json_atomic(target, detail)
    return detail


def verify(root: Path) -> dict[str, Any]:
    detail = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if detail["freeze_digest"] != _digest(detail):
        raise RuntimeError("V22 base digest mismatch")
    if sha256_file(root / CONFIG) != detail["config_sha256"]:
        raise RuntimeError("V22 config drift")
    if sha256_file(root / "src/jclosure/protocol_v22.py") != detail["protocol_code_sha256"]:
        raise RuntimeError("V22 protocol drift")
    for name, expected in detail["frozen_input_sha256"].items():
        if name != "reports/FINAL_REPORT.md" and sha256_file(root / name) != expected:
            raise RuntimeError(f"V22 frozen parent input changed: {name}")
    return detail


def stage_freeze(root: Path, name: str, inputs: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if target.exists():
        raise RuntimeError(f"V22 stage already frozen: {name}")
    detail = {
        "schema_version": 31,
        "protocol_version": f"{PREFIX}_{name}",
        "base_freeze_digest": base["freeze_digest"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {path: sha256_file(root / path) for path in inputs},
        **payload,
    }
    detail["freeze_digest"] = _digest(detail)
    write_json_atomic(target, detail)
    return detail


def verify_stage(root: Path, name: str) -> dict[str, Any]:
    base = verify(root)
    detail = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if detail["freeze_digest"] != _digest(detail) or detail["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V22 stage digest mismatch: {name}")
    for path, expected in detail["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V22 stage source drift: {name}:{path}")
    return detail


if __name__ == "__main__":
    import sys
    answer = freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd())
    print(answer["freeze_digest"])
