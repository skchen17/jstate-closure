"""Result-conditioned freezes for v7 localization and compression stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.protocol_v7 import digest
from jclosure.protocol_v7 import verify_freeze as verify_attribution_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v7 import PROTOCOL_V7, SCHEMA_VERSION_V7

LOCALIZATION_FREEZE = Path("artifacts/channel_localization_v7.freeze.json")
COMPRESSION_FREEZE = Path("artifacts/arch_compression_v7.freeze.json")


def _paths(root: Path, kind: str) -> list[Path]:
    common = [
        root / "configs/persistent_channels_v7.yaml",
        root / "src/jclosure/cache_v7.py",
        root / "src/jclosure/records_v7.py",
        root / "src/jclosure/protocol_v7_stage2.py",
        root / "results/v7/processed/persistent_channel_attribution_v7.json",
        root / "results/v7/processed/cache_restore_v7.json",
        root / "artifacts/persistent_channels_v7.freeze.json",
    ]
    if kind == "localization":
        return [
            *common,
            root / "src/jclosure/experiments/localize_channels_v7.py",
            root / "scripts/run_arch_state_v7.sh",
        ]
    if kind == "compression":
        return [
            *common,
            root / "results/v7/processed/channel_localization_v7.json",
            root / "src/jclosure/experiments/arch_compression_v7.py",
            root / "src/jclosure/arch_compression_v7.py",
            root / "scripts/run_arch_state_v7.sh",
        ]
    raise ValueError(f"unknown v7 stage-2 freeze kind {kind}")


def freeze_path(kind: str) -> Path:
    return LOCALIZATION_FREEZE if kind == "localization" else COMPRESSION_FREEZE


def build_stage_freeze(root: Path, config: dict[str, Any], kind: str) -> dict[str, Any]:
    parent = verify_attribution_freeze(root, config)
    paths = _paths(root, kind)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{kind} freeze inputs missing: {missing}")
    attribution = json.loads(
        (
            root / "results/v7/processed/persistent_channel_attribution_v7.json"
        ).read_text(encoding="utf-8")
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V7,
        "protocol_version": PROTOCOL_V7,
        "stage": kind,
        "config_digest": config_digest(config),
        "parent_freeze_digest": parent["freeze_digest"],
        "channel_classification": attribution["classification"],
        "kv_stable": attribution["kv_stable"],
        "recurrent_stable": attribution["recurrent_stable"],
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    payload["freeze_digest"] = digest(payload)
    write_json_atomic(root / freeze_path(kind), payload)
    return payload


def verify_stage_freeze(
    root: Path, config: dict[str, Any], kind: str
) -> dict[str, Any]:
    verify_attribution_freeze(root, config)
    payload = json.loads((root / freeze_path(kind)).read_text(encoding="utf-8"))
    if payload.get("stage") != kind or payload.get("protocol_version") != PROTOCOL_V7:
        raise RuntimeError(f"v7 {kind} protocol mismatch")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError(f"v7 {kind} config changed after freeze")
    if payload.get("freeze_digest") != digest(payload):
        raise RuntimeError(f"v7 {kind} freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"v7 {kind} frozen input mismatch: {failures}")
    return payload
