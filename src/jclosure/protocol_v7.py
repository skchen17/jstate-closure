"""Freeze, split, and immutable-history guards for protocol v7."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v7 import PROTOCOL_V7, SCHEMA_VERSION_V7

V6_BASELINE_COMMIT = "ac7f7c5511af954cd76b9d331042851236f56e59"
V6_GUARD_PATH = Path("artifacts/v6_immutable.sha256.json")
FREEZE_PATH = Path("artifacts/persistent_channels_v7.freeze.json")
SPLIT_PATH = Path("data/v7/persistent_channel_split.json")


def digest(payload: dict[str, Any]) -> str:
    clean = {
        key: value
        for key, value in payload.items()
        if key not in {"freeze_digest", "manifest_digest"}
    }
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _v6_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative in (
        "configs/peripheral_v6.yaml",
        "artifacts/peripheral_v6.freeze.json",
        "artifacts/v5_immutable.sha256.json",
        "data/v6",
        "results/v6",
    ):
        path = root / relative
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(value for value in path.rglob("*") if value.is_file())
    for pattern in (
        "reports/*V6.md",
        "src/jclosure/*v6.py",
        "src/jclosure/experiments/*v6.py",
        "scripts/*v6.sh",
        "schemas/*v6*",
    ):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(paths))


def build_v6_guard(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != V6_BASELINE_COMMIT:
        raise RuntimeError(
            f"v6 guard must be created at {V6_BASELINE_COMMIT}, observed {head}"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V7,
        "purpose": "byte-level guard for frozen v6 protocol/results",
        "baseline_commit": V6_BASELINE_COMMIT,
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in _v6_paths(root)
        },
    }
    payload["manifest_digest"] = digest(payload)
    write_json_atomic(root / V6_GUARD_PATH, payload)
    return payload


def verify_v6_guard(root: Path) -> dict[str, Any]:
    payload = json.loads((root / V6_GUARD_PATH).read_text(encoding="utf-8"))
    if payload.get("manifest_digest") != digest(payload):
        raise RuntimeError("v6 immutable manifest digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"frozen v6 files changed: {failures}")
    return payload


def write_split_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    section = config["persistent_channels_v7"]
    artifact = root / section["source_v6_artifact"]
    if sha256_file(artifact) != section["source_v6_artifact_sha256"]:
        raise RuntimeError("v6 float32 causal artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as values:
        items = []
        for index in range(len(values["base_trial_id"])):
            old_role = str(values["split"][index])
            items.append(
                {
                    "artifact_index": index,
                    "base_trial_id": str(values["base_trial_id"][index]),
                    "prompt_id": str(values["prompt_id"][index]),
                    "family": str(values["family"][index]),
                    "role": "localization_fit"
                    if old_role == "causal_fit"
                    else "attribution_test",
                    "source_role": old_role,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V7,
        "protocol_version": PROTOCOL_V7,
        "seed": int(section["attribution_split_seed"]),
        "source_artifact": str(section["source_v6_artifact"]),
        "source_artifact_sha256": sha256_file(artifact),
        "items": sorted(items, key=lambda item: item["base_trial_id"]),
    }
    payload["manifest_digest"] = digest(payload)
    path = root / SPLIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return payload


def _frozen_code_paths(root: Path) -> list[Path]:
    relatives = (
        "src/jclosure/cache_v7.py",
        "src/jclosure/records_v7.py",
        "src/jclosure/protocol_v7.py",
        "src/jclosure/experiments/persistent_channels_v7.py",
        "schemas/protocol-v7-record.schema.json",
        "scripts/run_persistent_channels_v7.sh",
    )
    return [root / relative for relative in relatives]


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_v6_guard(root)
    split = write_split_manifest(root, config)
    section = config["persistent_channels_v7"]
    paths = [
        root / config["_config_path"],
        root / SPLIT_PATH,
        *_frozen_code_paths(root),
    ]
    sources = [
        root / section[key]
        for key in (
            "source_v6_summary",
            "source_v6_records",
            "source_v6_trials",
            "source_v6_artifact",
            "source_tasks",
        )
    ]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v7 freeze inputs missing: {missing}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V7,
        "protocol_version": PROTOCOL_V7,
        "config_digest": config_digest(config),
        "model_revision": config["model"]["revision"],
        "lens_revision": config["lens"]["revision"],
        "v6_guard_digest": guard["manifest_digest"],
        "split_manifest": str(SPLIT_PATH),
        "split_manifest_digest": split["manifest_digest"],
        "thresholds": section["channel_gate"],
        "hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [*paths, *sources]
        },
    }
    payload["freeze_digest"] = digest(payload)
    write_json_atomic(root / FREEZE_PATH, payload)
    return payload


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_v6_guard(root)
    payload = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_V7:
        raise RuntimeError("v7 protocol mismatch")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("v7 config changed after freeze")
    if payload.get("freeze_digest") != digest(payload):
        raise RuntimeError("v7 freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"v7 frozen input mismatch: {failures}")
    return payload
