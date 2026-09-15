"""Immutable corrective protocol for the v7 endpoint serialization defect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.protocol_v7 import digest
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_V7_CORRECTIVE = "persistent_channel_compression_corrective_v7_1"
SCHEMA_VERSION_V7_CORRECTIVE = 10
FREEZE_PATH = Path("artifacts/arch_compression_v7_corrective.freeze.json")


def _capture_artifact(root: Path) -> Path:
    summary = json.loads(
        (root / "results/v7/processed/raw_arch_channel_ceiling_v7.json").read_text()
    )
    return root / str(summary["artifact"])


def frozen_paths(root: Path) -> list[Path]:
    return [
        root / "configs/persistent_channels_v7_corrective.yaml",
        root / "artifacts/arch_compression_v7.freeze.json",
        root / "results/v7/processed/raw_arch_channel_ceiling_v7.json",
        root / "results/v7/processed/persistent_channel_attribution_v7.json",
        root
        / "artifacts/causal/v6/causal-endpoint-v6-20260914T161620Z-a20b5a8c-s20260828-full-66/causal_endpoint_f32.npz",
        root / "src/jclosure/protocol_v7_corrective.py",
        root / "src/jclosure/experiments/arch_compression_v7_corrective.py",
        root / "scripts/run_arch_compression_v7_corrective.sh",
        _capture_artifact(root),
    ]


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = frozen_paths(root)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"corrective freeze inputs missing: {missing}")
    source = json.loads(
        (root / "results/v7/processed/raw_arch_channel_ceiling_v7.json").read_text()
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V7_CORRECTIVE,
        "protocol_version": PROTOCOL_V7_CORRECTIVE,
        "config_digest": config_digest(config),
        "parent_freeze_digest": json.loads(
            (root / "artifacts/arch_compression_v7.freeze.json").read_text()
        )["freeze_digest"],
        "source_capture_run_id": source["run_id"],
        "source_capture_artifact_sha256": source["artifact_sha256"],
        "correction": {
            "defect": "all serialized v7 channel endpoint arrays aliased the final full trajectory",
            "unchanged_valid_results": "scalar and curve attribution records",
            "correct_target": "v6 j_intervened[:,1] - v6 j_clean[:,1]",
            "ridge_alpha": 1.0,
            "split": "unchanged 33 localization_fit / 33 attribution_test",
        },
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    payload["freeze_digest"] = digest(payload)
    write_json_atomic(root / FREEZE_PATH, payload)
    return payload


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads((root / FREEZE_PATH).read_text())
    if payload.get("protocol_version") != PROTOCOL_V7_CORRECTIVE:
        raise RuntimeError("v7 corrective protocol mismatch")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("v7 corrective config changed after freeze")
    if payload.get("freeze_digest") != digest(payload):
        raise RuntimeError("v7 corrective freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"v7 corrective frozen input mismatch: {failures}")
    return payload
