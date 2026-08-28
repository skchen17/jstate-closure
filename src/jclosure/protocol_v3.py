"""Freeze and verification contract for exploratory Phase 3 protocol v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jclosure.baseline_guard import verify_manifest as verify_v2_manifest
from jclosure.config import config_digest, load_config
from jclosure.provenance import git_commit, sha256_file, write_json_atomic

PROTOCOL_VERSION = "exploratory_protocol_v3"
FREEZE_RELATIVE_PATH = Path("artifacts/phase3_protocol_v3.freeze.json")
PROTOCOL_CODE_PATHS = (
    "configs/geometry_v3.yaml",
    "schemas/geometry-record-v3.schema.json",
    "schemas/trial-record-v3.schema.json",
    "src/jclosure/baseline_guard.py",
    "src/jclosure/clamp_v3.py",
    "src/jclosure/geometry.py",
    "src/jclosure/protocol_v3.py",
    "src/jclosure/records.py",
    "src/jclosure/experiments/geometry_v3.py",
    "src/jclosure/experiments/clamp_v3_calibration.py",
    "src/jclosure/experiments/closure_v3.py",
)


def _existing_hashes(root: Path, paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v3 freeze inputs missing: {missing}")
    return {path: sha256_file(root / path) for path in paths}


def verify_hashes(root: str | Path, hashes: dict[str, str]) -> None:
    repository = Path(root).resolve()
    failures: list[dict[str, str]] = []
    for relative, expected in hashes.items():
        candidate = repository / relative
        observed = "MISSING" if not candidate.is_file() else sha256_file(candidate)
        if observed != expected:
            failures.append(
                {"path": relative, "expected": str(expected), "observed": observed}
            )
    if failures:
        raise RuntimeError(f"Phase 3 v3 freeze hash mismatch: {failures}")


def create_v3_freeze(
    root: str | Path,
    *,
    config_path: str | Path,
    calibration_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    config_file = Path(config_path).resolve()
    calibration_file = Path(calibration_path).resolve()
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    if calibration.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("calibration is not exploratory protocol v3")
    verify_v2_manifest(
        repository, repository / "artifacts/phase0_v2_immutable.sha256.json"
    )
    source_hashes = _existing_hashes(repository, PROTOCOL_CODE_PATHS)
    candidate_path = repository / calibration["candidate_records"]
    bank_path = repository / calibration["activation_bank_manifest"]
    data_hashes = _existing_hashes(
        repository,
        [
            str(calibration_file.relative_to(repository)),
            str(candidate_path.relative_to(repository)),
            str(bank_path.relative_to(repository)),
            "artifacts/phase0_v2_immutable.sha256.json",
        ],
    )
    config = load_config(config_file)
    payload = {
        "schema_version": 3,
        "protocol_version": PROTOCOL_VERSION,
        "status": (
            "BEHAVIORAL_AUTHORIZED"
            if calibration.get("behavioral_authorized_protocols")
            else "GEOMETRY_CALIBRATION_ONLY"
        ),
        "baseline_commit": "d504eaa14af45f9df32101cf4599c55d3fac8707",
        "freeze_created_from_commit": git_commit(repository),
        "config_path": str(config_file.relative_to(repository)),
        "config_digest": config_digest(config),
        "calibration_run_id": calibration.get("run_id"),
        "thresholds": {
            "dense": config["v3_state"]["dense"],
            "sparse": config["v3_state"]["sparse"],
            "rms_drift": config["v3_state"]["rms_drift_threshold"],
            "formal_displacement": config["v3_state"]["formal_displacement_fraction"],
            "small_perturbation_min": config["v3_state"][
                "small_perturbation_min_fraction"
            ],
            "naturality_quantile": config["geometry"]["naturality_quantile"],
        },
        "eligible_protocols": {
            key: value
            for key, value in calibration.get("protocols", {}).items()
            if value.get("behavioral_authorized")
        },
        "position_scopes": config["clamp_v3"]["position_scopes"],
        "clamp_modes": config["clamp_v3"]["modes"],
        "source_hashes": source_hashes,
        "data_hashes": data_hashes,
    }
    target = (
        repository / FREEZE_RELATIVE_PATH
        if output_path is None
        else Path(output_path).resolve()
    )
    write_json_atomic(target, payload)
    return payload


def verify_v3_freeze(
    root: str | Path,
    *,
    require_behavioral_authorization: bool = True,
    freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = (
        repository / FREEZE_RELATIVE_PATH
        if freeze_path is None
        else Path(freeze_path).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError("Phase 3 v3 freeze manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Phase 3 freeze protocol mismatch")
    for group in ("source_hashes", "data_hashes"):
        verify_hashes(repository, payload.get(group, {}))
    config = load_config(repository / payload["config_path"])
    if config_digest(config) != payload.get("config_digest"):
        raise RuntimeError("Phase 3 v3 frozen configuration digest mismatch")
    verify_v2_manifest(
        repository, repository / "artifacts/phase0_v2_immutable.sha256.json"
    )
    if require_behavioral_authorization and not payload.get("eligible_protocols"):
        raise RuntimeError(
            "no v3 operational state passed calibration; behavioral closure is gated"
        )
    return payload
