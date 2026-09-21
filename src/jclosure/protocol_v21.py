"""Append-only V21 diagnostic freezes; V1–V20 are immutable inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PARENT = "5d8ec3cf291856ab708f9f04cb73fe4a7aa56a58"
CONFIG = Path("configs/action_coordinate_geometry_v21.yaml")
FREEZE = Path("artifacts/action_coordinate_geometry_v21.freeze.json")
INPUTS = (
    "reports/V20_COMPLETE_REPORT.md",
    "reports/FINAL_REPORT.md",
    "artifacts/compact_causal_response_operator_v20.freeze.json",
    "artifacts/compact_causal_response_operator_v20_actions.freeze.json",
    "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
    "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json",
    "artifacts/compact_causal_response_operator_v20_operator_analysis.freeze.json",
    "artifacts/compact_causal_response_operator_v20_q_locality_scales.freeze.json",
    "results/v20/processed/oracle_operator_search_v20.json",
    "results/v20/processed/operator_geometry_v20.json",
    "results/v20/processed/v20_adjudication_amendment_1.json",
    "results/v20/processed/response_operator_operator_train_v20.json",
    "results/v20/processed/response_operator_operator_validation_v20.json",
    "results/v20/processed/action_descriptors_v20.json",
    "artifacts/causal/v13/probe_directions_v13.pt",
)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "freeze_digest"},
                                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    if (root / FREEZE).exists():
        raise RuntimeError("V21 base freeze already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V21 parent commit mismatch: {head}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT or config["authorization"]["DYNAMIC_STATE_SEARCH_AUTHORIZED"]:
        raise RuntimeError("V21 config parent or dynamic authorization invalid")
    detail = {"schema_version": 30, "protocol_version": config["protocol_version"],
              "parent_commit": PARENT, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256_file(root / CONFIG),
              "protocol_code_sha256": sha256_file(root / "src/jclosure/protocol_v21.py"),
              "frozen_input_sha256": {name: sha256_file(root / name) for name in INPUTS},
              "config": config,
              "V20_final_six_action_responses_opened": False,
              "V21_dynamic_state_search_authorized": False}
    detail["freeze_digest"] = _digest(detail)
    write_json_atomic(root / FREEZE, detail)
    return detail


def verify(root: Path) -> dict[str, Any]:
    detail = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    if detail["freeze_digest"] != _digest(detail):
        raise RuntimeError("V21 base freeze digest mismatch")
    if sha256_file(root / CONFIG) != detail["config_sha256"]:
        raise RuntimeError("V21 config drift")
    if sha256_file(root / "src/jclosure/protocol_v21.py") != detail["protocol_code_sha256"]:
        raise RuntimeError("V21 protocol drift")
    for name, expected in detail["frozen_input_sha256"].items():
        if name != "reports/FINAL_REPORT.md" and sha256_file(root / name) != expected:
            raise RuntimeError(f"V21 parent input changed: {name}")
    return detail


def stage_freeze(root: Path, name: str, inputs: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    base = verify(root)
    target = root / f"artifacts/action_coordinate_geometry_v21_{name}.freeze.json"
    if target.exists():
        raise RuntimeError(f"V21 stage already frozen: {name}")
    detail = {"schema_version": 30, "protocol_version": f"action_coordinate_geometry_v21_{name}",
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {path: sha256_file(root / path) for path in inputs}, **payload}
    detail["freeze_digest"] = _digest(detail)
    write_json_atomic(target, detail)
    return detail


def verify_stage(root: Path, name: str) -> dict[str, Any]:
    base = verify(root)
    detail = json.loads((root / f"artifacts/action_coordinate_geometry_v21_{name}.freeze.json").read_text())
    if detail["freeze_digest"] != _digest(detail) or detail["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V21 stage digest mismatch: {name}")
    for path, expected in detail["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V21 stage source drift: {name}:{path}")
    return detail


if __name__ == "__main__":
    import sys
    result = freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd())
    print(result["freeze_digest"])
