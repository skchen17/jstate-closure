"""Additive freeze and immutable-history contract for protocol v9."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic

SCHEMA_VERSION_V9 = 12
PROTOCOL_V9 = "conditional_sufficiency_dimension_protocol_v9"
BASELINE_COMMIT = "7db2599eff165c4b048a450572625aa898c4ce75"
GUARD_PATH = Path("artifacts/v8_immutable.sha256.json")
FREEZE_PATH = Path("artifacts/sufficiency_protocol_v9.freeze.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"digest", "freeze_digest"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_guard(root: Path) -> dict[str, Any]:
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT],
        cwd=root,
        text=True,
    ).splitlines()
    paths = [name for name in names if name != "reports/FINAL_REPORT.md"]
    missing = [name for name in paths if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"v9 baseline files missing: {missing[:10]}")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V9,
        "purpose": "byte-level guard for every tracked v1-v8 file at the v9 baseline",
        "baseline_commit": BASELINE_COMMIT,
        "declared_exception": "reports/FINAL_REPORT.md",
        "hashes": {name: sha256_file(root / name) for name in paths},
    }
    value["digest"] = _digest(value)
    write_json_atomic(root / GUARD_PATH, value)
    return value


def verify_guard(root: Path) -> dict[str, Any]:
    value = json.loads((root / GUARD_PATH).read_text(encoding="utf-8"))
    if value.get("digest") != _digest(value):
        raise RuntimeError("v9 history guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen v1-v8 files changed: {changed[:20]}")
    return value


def _code_paths(root: Path) -> list[Path]:
    names = (
        "configs/sufficiency_v9.yaml",
        "schemas/protocol-v9-record.schema.json",
        "scripts/run_sufficiency_v9.sh",
        "src/jclosure/protocol_v9.py",
        "src/jclosure/experiments/sufficiency_v9.py",
        "src/jclosure/experiments/report_v9.py",
        "src/jclosure/reporting_v9.py",
        "tests/test_v9.py",
    )
    return [root / name for name in names]


def _source_paths(root: Path) -> list[Path]:
    names = [
        "artifacts/persistent_state_v8.freeze.json",
        "artifacts/compression_protocol_v8_1.freeze.json",
        "artifacts/persistent/v8/structured_features_v8.npz",
        "results/v8/processed/structured_features_v8.json",
        "results/v8/processed/structured_component_screen_v8.json",
        "results/v8/processed/structured_component_screen_v8.parquet",
        "results/v8/processed/persistent_state_compression_v8.json",
        "results/v8/processed/persistent_state_compression_v8.parquet",
    ]
    paths = [root / name for name in names]
    for split in ("train", "validation", "final_test"):
        summary = root / f"results/v8/processed/persistent_capture_{split}_v8.json"
        paths.append(summary)
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            paths.append(root / payload["endpoint_artifact"])
    return paths


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = build_guard(root)
    paths = [*_code_paths(root), *_source_paths(root)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v9 freeze inputs missing: {missing}")
    feature_manifest = json.loads(
        (root / "results/v8/processed/structured_features_v8.json").read_text(
            encoding="utf-8"
        )
    )
    requested = [
        int(value) for value in config["sufficiency_v9"]["requested_dimensions"]
    ]
    rank = int(feature_manifest["dimension"])
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V9,
        "protocol_version": PROTOCOL_V9,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_digest": config_digest(config),
        "history_guard_digest": guard["digest"],
        "v8_source_freeze_digest": feature_manifest["source_freeze_digest"],
        "effective_feature_rank": rank,
        "requested_dimensions": requested,
        "identified_dimensions": [value for value in requested if value <= rank],
        "rank_limited_dimensions": [value for value in requested if value > rank],
        "thresholds": config["sufficiency_v9"],
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    return value


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    value = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v9 freeze digest mismatch")
    if value.get("history_guard_digest") != guard["digest"]:
        raise RuntimeError("v9 history guard mismatch")
    if value.get("config_digest") != config_digest(config):
        raise RuntimeError("v9 config changed after freeze")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v9 frozen input mismatch: {changed[:20]}")
    return value
