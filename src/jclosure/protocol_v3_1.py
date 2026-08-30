"""Freeze contracts for the two independent protocol-v3.1 experiment arms."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jclosure.baseline_guard import verify_manifest as verify_v2_manifest
from jclosure.config import config_digest, load_config
from jclosure.provenance import git_commit, sha256_file, write_json_atomic

CLOSURE_PROTOCOL = "corrective_exploratory_protocol_v3_1"
MEMORY_PROTOCOL = "compact_memory_exploratory_v3_1"
CLOSURE_FREEZE = Path("artifacts/closure_protocol_v3_1.freeze.json")
MEMORY_FREEZE = Path("artifacts/compact_memory_v3_1.freeze.json")

CLOSURE_SOURCES = (
    "configs/closure_v3_1.yaml",
    "schemas/trial-record-v3-1.schema.json",
    "src/jclosure/records_v3_1.py",
    "src/jclosure/clamp_v3_1.py",
    "src/jclosure/datasets_v3_1.py",
    "src/jclosure/runtime_v3_1.py",
    "src/jclosure/statistics_v3_1.py",
    "src/jclosure/protocol_v3_1.py",
    "src/jclosure/experiments/prepare_v3_1.py",
    "src/jclosure/experiments/calibrate_v3_1.py",
    "src/jclosure/experiments/closure_v3_1.py",
)
MEMORY_SOURCES = (
    "configs/compact_memory_v3_1.yaml",
    "schemas/compact-memory-record-v3-1.schema.json",
    "src/jclosure/records_v3_1.py",
    "src/jclosure/datasets_v3_1.py",
    "src/jclosure/compact_memory_v3_1.py",
    "src/jclosure/protocol_v3_1.py",
    "src/jclosure/experiments/compact_memory_v3_1.py",
)


def _hashes(root: Path, paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v3.1 freeze inputs missing: {missing}")
    return {path: sha256_file(root / path) for path in paths}


def _normalized_config_digest(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    if isinstance(value.get("model"), dict):
        value["model"].pop("device", None)
    return config_digest(value)


def _verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    failures = []
    for relative, expected in hashes.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(
                {"path": relative, "expected": expected, "observed": observed}
            )
    if failures:
        raise RuntimeError(f"v3.1 freeze hash mismatch: {failures}")


def _verify_immutable_guards(root: Path) -> None:
    verify_v2_manifest(root, root / "artifacts/phase0_v2_immutable.sha256.json")
    v3_manifest = json.loads(
        (root / "artifacts/v3_immutable.sha256.json").read_text(encoding="utf-8")
    )
    _verify_hashes(root, v3_manifest["files"])


def create_closure_freeze(
    root: str | Path,
    *,
    calibration_path: str
    | Path = "results/v3_1/processed/closure_v3_1_calibration.json",
) -> dict[str, Any]:
    repository = Path(root).resolve()
    _verify_immutable_guards(repository)
    calibration_file = repository / calibration_path
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    if calibration.get("protocol_version") != CLOSURE_PROTOCOL:
        raise ValueError("calibration protocol mismatch")
    authorized = calibration.get("authorized_protocols", [])
    if not calibration.get("behavioral_authorized") or len(authorized) != 1:
        raise RuntimeError("v3.1 calibration did not authorize exactly one protocol")
    data_manifest = json.loads(
        (repository / "artifacts/v3_1_data_manifest.json").read_text(encoding="utf-8")
    )
    data_paths = [
        "artifacts/v3_1_data_manifest.json",
        *(value["path"] for value in data_manifest["causal_domains"].values()),
        str(calibration_file.relative_to(repository)),
        str(calibration["records"]),
        str(calibration["activation_bank_manifest"]),
        "results/processed/concept_vocabulary_v2_4096.json",
    ]
    config = load_config(repository / "configs/closure_v3_1.yaml")
    payload = {
        "schema_version": 4,
        "protocol_version": CLOSURE_PROTOCOL,
        "status": "BEHAVIORAL_AUTHORIZED",
        "baseline_commit": "6919d445ba82f8604df109d394517f51bdf2dc0a",
        "freeze_created_from_commit": git_commit(repository),
        "config_path": "configs/closure_v3_1.yaml",
        "config_digest": _normalized_config_digest(config),
        "selected_protocol": authorized[0],
        "source_hashes": _hashes(repository, CLOSURE_SOURCES),
        "data_hashes": _hashes(repository, data_paths),
        "model": config["model"],
        "lens": config["lens"],
        "seeds": config["v3_1_data"],
        "thresholds": {
            key: config["v3_1"][key]
            for key in (
                "dense_cosine_threshold",
                "top10_overlap_threshold",
                "rms_drift_threshold",
                "formal_displacement_fraction",
                "naturality_quantile",
            )
        },
    }
    write_json_atomic(repository / CLOSURE_FREEZE, payload)
    return payload


def create_memory_freeze(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    _verify_immutable_guards(repository)
    data_manifest = json.loads(
        (repository / "artifacts/v3_1_data_manifest.json").read_text(encoding="utf-8")
    )
    data_paths = [
        "artifacts/v3_1_data_manifest.json",
        *(value["path"] for value in data_manifest["memory_splits"].values()),
        "results/processed/concept_vocabulary_v2_4096.json",
    ]
    config = load_config(repository / "configs/compact_memory_v3_1.yaml")
    payload = {
        "schema_version": 4,
        "protocol_version": MEMORY_PROTOCOL,
        "status": "FROZEN_FOR_EXECUTION",
        "baseline_commit": "6919d445ba82f8604df109d394517f51bdf2dc0a",
        "freeze_created_from_commit": git_commit(repository),
        "config_path": "configs/compact_memory_v3_1.yaml",
        "config_digest": _normalized_config_digest(config),
        "source_hashes": _hashes(repository, MEMORY_SOURCES),
        "data_hashes": _hashes(repository, data_paths),
        "model": config["model"],
        "lens": config["lens"],
        "controller_seeds": config["compact_memory"]["controller_seeds"],
    }
    write_json_atomic(repository / MEMORY_FREEZE, payload)
    return payload


def verify_freeze(
    root: str | Path,
    *,
    kind: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = repository / (CLOSURE_FREEZE if kind == "closure" else MEMORY_FREEZE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = CLOSURE_PROTOCOL if kind == "closure" else MEMORY_PROTOCOL
    if payload.get("protocol_version") != expected:
        raise RuntimeError("v3.1 freeze protocol mismatch")
    _verify_immutable_guards(repository)
    _verify_hashes(repository, payload.get("source_hashes", {}))
    _verify_hashes(repository, payload.get("data_hashes", {}))
    frozen_config = load_config(repository / payload["config_path"])
    if _normalized_config_digest(frozen_config) != payload.get("config_digest"):
        raise RuntimeError("v3.1 frozen config digest mismatch")
    if config is not None and _normalized_config_digest(config) != payload.get(
        "config_digest"
    ):
        raise RuntimeError("runtime config differs from v3.1 freeze")
    return payload
