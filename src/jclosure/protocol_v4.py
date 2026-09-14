"""Freeze and verification helpers for the additive v4 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.datasets_v4 import PROTOCOL_V4
from jclosure.provenance import sha256_file, write_json_atomic

FREEZE_PATH = Path("artifacts/program_tasks_v4.freeze.json")


def _payload_digest(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "freeze_digest"}
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_program_freeze(
    root: Path,
    config: dict[str, Any],
    *,
    calibration_summary: Path,
    data_paths: dict[str, Path],
    selection: dict[str, Any],
) -> dict[str, Any]:
    tracked = [
        Path(config["_config_path"]),
        root / "src/jclosure/datasets_v4.py",
        root / "src/jclosure/experiments/teacher_tasks_v4.py",
        calibration_summary,
        *data_paths.values(),
    ]
    hashes = {
        str(path.resolve().relative_to(root)): sha256_file(path) for path in tracked
    }
    payload: dict[str, Any] = {
        "schema_version": 6,
        "protocol_version": PROTOCOL_V4,
        "config_digest": config_digest(config),
        "model_revision": config["model"]["revision"],
        "lens_revision": config["lens"]["revision"],
        "calibration_summary": str(calibration_summary.relative_to(root)),
        "formal_data": {
            domain: str(path.relative_to(root)) for domain, path in data_paths.items()
        },
        "selection": selection,
        "hashes": hashes,
    }
    payload["freeze_digest"] = _payload_digest(payload)
    write_json_atomic(root / FREEZE_PATH, payload)
    return payload


def verify_program_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / FREEZE_PATH
    if not path.is_file():
        raise RuntimeError("v4 program-task freeze is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_V4:
        raise RuntimeError("v4 program-task freeze has the wrong protocol")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("v4 configuration changed after data freeze")
    if payload.get("freeze_digest") != _payload_digest(payload):
        raise RuntimeError("v4 freeze payload digest mismatch")
    mismatches = []
    for relative, expected in payload.get("hashes", {}).items():
        candidate = root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"v4 frozen artifact mismatch: {mismatches}")
    return payload
