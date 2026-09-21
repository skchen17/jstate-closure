"""Append-only V23 freezes; V1--V22 remain immutable inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PARENT = "c755b4d7b0baf3aa0291456fcf49169087b8c863"
CONFIG = Path("configs/oracle_local_action_charts_v23.yaml")
FREEZE = Path("artifacts/oracle_local_action_charts_v23.freeze.json")
PREFIX = "oracle_local_action_charts_v23"
INPUTS = (
    "reports/V22_COMPLETE_REPORT.md",
    "reports/FINAL_REPORT.md",
    "artifacts/causal_action_manifold_v22_final.freeze.json",
    "artifacts/causal_action_manifold_v22_action_selection.freeze.json",
    "artifacts/causal_action_manifold_v22_expanded_bank_design.freeze.json",
    "results/v22/processed/action_pool_v22.json",
    "results/v22/processed/action_pool_calibration_v22.json",
    "results/v22/processed/action_selection_v22.json",
    "results/v22/processed/expanded_bank_design_v22.json",
    "results/v22/processed/expanded_response_bank_index_v22.parquet",
    "results/v22/processed/scale_composition_design_v22.json",
    "results/v22/processed/action_scale_law_v22.json",
    "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
    "artifacts/action_coordinate_geometry_v21_state_representations.freeze.json",
    "artifacts/causal/v13/probe_directions_v13.pt",
)


def _digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    target = root / FREEZE
    if target.exists():
        raise RuntimeError("V23 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V23 parent commit mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT:
        raise RuntimeError("V23 parent mismatch")
    if config["authorization"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"] or config["authorization"]["H3_AUTHORIZED"]:
        raise RuntimeError("V23 dynamic/H3 authorization must remain false")
    detail = {
        "schema_version": 32,
        "protocol_version": config["protocol_version"],
        "parent_commit": PARENT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_code_sha256": sha256_file(root / "src/jclosure/protocol_v23.py"),
        "frozen_input_sha256": {name: sha256_file(root / name) for name in INPUTS},
        "config": config,
        "V1_V22_records_mutated": False,
        "historical_final_actions_opened": False,
        "dynamic_state_search_authorized": False,
        "H3_authorized": False,
    }
    detail["freeze_digest"] = _digest(detail)
    write_json_atomic(target, detail)
    return detail


def verify(root: Path) -> dict[str, Any]:
    detail = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if detail["freeze_digest"] != _digest(detail):
        raise RuntimeError("V23 base digest mismatch")
    if sha256_file(root / CONFIG) != detail["config_sha256"]:
        raise RuntimeError("V23 config drift")
    if sha256_file(root / "src/jclosure/protocol_v23.py") != detail["protocol_code_sha256"]:
        raise RuntimeError("V23 protocol drift")
    for name, expected in detail["frozen_input_sha256"].items():
        if name != "reports/FINAL_REPORT.md" and sha256_file(root / name) != expected:
            raise RuntimeError(f"V23 frozen parent input changed: {name}")
    return detail


def stage_freeze(root: Path, name: str, inputs: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if target.exists():
        raise RuntimeError(f"V23 stage already frozen: {name}")
    detail = {
        "schema_version": 32,
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
        raise RuntimeError(f"V23 stage digest mismatch: {name}")
    for path, expected in detail["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V23 stage source drift: {name}:{path}")
    return detail


if __name__ == "__main__":
    import sys
    answer = freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd())
    print(answer["freeze_digest"])
