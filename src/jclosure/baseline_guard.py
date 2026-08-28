"""Byte-level regression guard for immutable Phase 0 v1/v2 artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.provenance import sha256_file, write_json_atomic

BASELINE_COMMIT = "d504eaa14af45f9df32101cf4599c55d3fac8707"
IMMUTABLE_PREFIXES = (
    "artifacts/MANIFEST.json",
    "artifacts/phase0_protocol_v2.freeze.json",
    "configs/confirm_v2.yaml",
    "configs/phase0_v1_replay.yaml",
    "configs/phase0_v2_calibration.yaml",
    "configs/phase0_v2_confirmatory.yaml",
    "configs/pilot_v2.yaml",
    "configs/qwen3_5_4b.yaml",
    "data/phase0_v2/",
    "reports/PHASE0_PROTOCOL_AUDIT.md",
    "reports/PHASE0_VALIDATION.md",
    "reports/PHASE0_V2_CALIBRATION.md",
    "reports/PHASE0_V2_CONFIRMATORY.md",
    "reports/LAYER_CALIBRATION.md",
    "results/processed/concept_vocabulary_v2_",
    "results/processed/layer_calibration",
    "results/processed/phase0_",
    "results/processed/phase0_v2_",
    "results/raw/layer-calibration-",
    "results/raw/phase0-",
    "results/raw/phase0-v2-",
    "results/v1_replay/",
)


def immutable_paths(root: str | Path, commit: str = BASELINE_COMMIT) -> tuple[str, ...]:
    repository = Path(root).resolve()
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        path
        for path in result.stdout.splitlines()
        if any(path == prefix or path.startswith(prefix) for prefix in IMMUTABLE_PREFIXES)
    )


def create_manifest(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    files = immutable_paths(repository)
    missing = [path for path in files if not (repository / path).is_file()]
    if missing:
        raise FileNotFoundError(f"immutable v2 files missing: {missing}")
    return {
        "schema_version": 3,
        "guard": "phase0_v1_v2_byte_regression",
        "baseline_commit": BASELINE_COMMIT,
        "file_count": len(files),
        "files": {path: sha256_file(repository / path) for path in files},
    }


def verify_manifest(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("v2 guard baseline commit mismatch")
    failures: list[dict[str, str]] = []
    for relative, expected in manifest.get("files", {}).items():
        path = repository / relative
        observed = "MISSING" if not path.is_file() else sha256_file(path)
        if observed != expected:
            failures.append(
                {"path": relative, "expected": str(expected), "observed": observed}
            )
    if failures:
        raise RuntimeError(f"immutable v2 byte regression: {failures}")
    return {
        "status": "PASSED",
        "baseline_commit": BASELINE_COMMIT,
        "verified_files": len(manifest.get("files", {})),
    }


def write_manifest(root: str | Path, target: str | Path) -> None:
    write_json_atomic(target, create_manifest(root))
