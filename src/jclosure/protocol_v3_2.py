"""Freeze and verification contracts for additive protocol v3.2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jclosure.baseline_guard import verify_manifest as verify_v2_manifest
from jclosure.config import config_digest, load_config
from jclosure.provenance import git_commit, sha256_file, write_json_atomic
from jclosure.runtime_v3_2 import MEMORY_PROTOCOL_V32, PROTOCOL_V32

CALIBRATION_FREEZE = Path("artifacts/closure_protocol_v3_2.calibration.freeze.json")
CLOSURE_FREEZE = Path("artifacts/closure_protocol_v3_2.freeze.json")
MEMORY_FREEZE = Path("artifacts/compact_memory_v3_2.freeze.json")

CLOSURE_SOURCES = (
    "configs/closure_v3_2.yaml",
    "schemas/trial-record-v3-2.schema.json",
    "src/jclosure/records_v3_2.py",
    "src/jclosure/clamp_v3_2.py",
    "src/jclosure/statistics_v3_2.py",
    "src/jclosure/runtime_v3_2.py",
    "src/jclosure/protocol_v3_2.py",
    "src/jclosure/experiments/prepare_v3_2.py",
    "src/jclosure/experiments/calibrate_v3_2.py",
    "src/jclosure/experiments/closure_v3_2.py",
)
MEMORY_SOURCES = (
    "configs/compact_memory_v3_2.yaml",
    "schemas/compact-memory-record-v3-2.schema.json",
    "src/jclosure/compact_memory_v3_2.py",
    "src/jclosure/experiments/compact_memory_v3_2.py",
    "src/jclosure/reporting_v3_2.py",
)


def _normalized_digest(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    if isinstance(value.get("model"), dict):
        value["model"].pop("device", None)
    return config_digest(value)


def _hashes(root: Path, paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v3.2 freeze inputs missing: {missing}")
    return {path: sha256_file(root / path) for path in paths}


def _verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    failures = []
    for relative, expected in hashes.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    if failures:
        raise RuntimeError(f"v3.2 freeze hash mismatch: {failures}")


def _verify_old_results(root: Path) -> None:
    verify_v2_manifest(root, root / "artifacts/phase0_v2_immutable.sha256.json")
    v3 = json.loads((root / "artifacts/v3_immutable.sha256.json").read_text())
    _verify_hashes(root, v3["files"])


def create_calibration_freeze(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    _verify_old_results(repository)
    config = load_config(repository / "configs/closure_v3_2.yaml")
    data = json.loads((repository / "artifacts/v3_2_data_manifest.json").read_text())
    data_paths = [
        "artifacts/v3_2_data_manifest.json",
        *(value["path"] for value in data["causal_domains"].values()),
        "results/processed/concept_vocabulary_v2_4096.json",
    ]
    payload = {
        "schema_version": 5,
        "protocol_version": PROTOCOL_V32,
        "status": "FROZEN_FOR_CALIBRATION",
        "baseline_commit": "a0d164e1167fae21a7d3e7ec41dae9fd7e819019",
        "freeze_created_from_commit": git_commit(repository),
        "config_path": "configs/closure_v3_2.yaml",
        "config_digest": _normalized_digest(config),
        "source_hashes": _hashes(repository, CLOSURE_SOURCES),
        "data_hashes": _hashes(repository, data_paths),
        "model": config["model"],
        "lens": config["lens"],
        "seeds": config["v3_2_data"],
        "thresholds": config["v3_2"],
    }
    write_json_atomic(repository / CALIBRATION_FREEZE, payload)
    return payload


def verify_calibration_freeze(
    root: str | Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    repository = Path(root).resolve()
    payload = json.loads((repository / CALIBRATION_FREEZE).read_text())
    if payload.get("protocol_version") != PROTOCOL_V32:
        raise RuntimeError("v3.2 calibration freeze protocol mismatch")
    _verify_old_results(repository)
    _verify_hashes(repository, payload["source_hashes"])
    _verify_hashes(repository, payload["data_hashes"])
    if config is not None and _normalized_digest(config) != payload["config_digest"]:
        raise RuntimeError("runtime config differs from v3.2 calibration freeze")
    return payload


def create_closure_freeze(
    root: str | Path,
    calibration_path: str | Path = "results/v3_2/processed/closure_v3_2_calibration.json",
) -> dict[str, Any]:
    repository = Path(root).resolve()
    calibration_freeze = verify_calibration_freeze(repository)
    calibration_file = repository / calibration_path
    calibration = json.loads(calibration_file.read_text())
    protocols = calibration.get("authorized_protocols", [])
    if not calibration.get("behavioral_authorized") or len(protocols) != 1:
        raise RuntimeError("v3.2 calibration did not authorize exactly one protocol")
    bank = str(calibration["activation_bank_manifest"])
    records = str(calibration["records"])
    payload = {
        **calibration_freeze,
        "status": "BEHAVIORAL_AUTHORIZED",
        "selected_protocol": protocols[0],
        "calibration_hashes": _hashes(
            repository,
            [str(calibration_file.relative_to(repository)), records, bank],
        ),
    }
    write_json_atomic(repository / CLOSURE_FREEZE, payload)
    return payload


def verify_closure_freeze(
    root: str | Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    repository = Path(root).resolve()
    payload = json.loads((repository / CLOSURE_FREEZE).read_text())
    verify_calibration_freeze(repository, config)
    _verify_hashes(repository, payload["calibration_hashes"])
    if payload.get("status") != "BEHAVIORAL_AUTHORIZED":
        raise RuntimeError("v3.2 behavioral protocol is not authorized")
    return payload


def create_memory_freeze(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    _verify_old_results(repository)
    config = load_config(repository / "configs/compact_memory_v3_2.yaml")
    group = config["compact_memory_v3_2"]["source_shard_group_id"]
    paths: list[str] = []
    for manifest_path in sorted((repository / "results/v3_1/raw").glob("compact-memory-v3-1-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETED_TRACE_SHARD" and manifest.get("shard_group_id") == group:
            paths.extend(
                [
                    str(manifest_path.relative_to(repository)),
                    str(manifest["records"]),
                ]
            )
    if len(paths) != 12:
        raise RuntimeError(f"expected six completed source shards, found {len(paths) // 2}")
    payload = {
        "schema_version": 5,
        "protocol_version": MEMORY_PROTOCOL_V32,
        "status": "FROZEN_FOR_EXECUTION",
        "baseline_commit": "a0d164e1167fae21a7d3e7ec41dae9fd7e819019",
        "freeze_created_from_commit": git_commit(repository),
        "config_path": "configs/compact_memory_v3_2.yaml",
        "config_digest": _normalized_digest(config),
        "source_hashes": _hashes(repository, MEMORY_SOURCES),
        "trace_record_hashes": _hashes(repository, paths),
        "source_shard_group_id": group,
    }
    write_json_atomic(repository / MEMORY_FREEZE, payload)
    return payload


def verify_memory_freeze(
    root: str | Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    repository = Path(root).resolve()
    payload = json.loads((repository / MEMORY_FREEZE).read_text())
    if payload.get("protocol_version") != MEMORY_PROTOCOL_V32:
        raise RuntimeError("compact-memory v3.2 freeze protocol mismatch")
    _verify_old_results(repository)
    _verify_hashes(repository, payload["source_hashes"])
    _verify_hashes(repository, payload["trace_record_hashes"])
    if config is not None and _normalized_digest(config) != payload["config_digest"]:
        raise RuntimeError("runtime config differs from compact-memory v3.2 freeze")
    return payload
