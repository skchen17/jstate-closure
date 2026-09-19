"""V14 immutable provenance and stage-freeze helpers.

V13 records are read-only parents; every V14 estimand change requires a new
versioned freeze rather than modifying a previous freeze or result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

BASELINE_COMMIT = "37df399e4bba8afaca6d99721fe8ceee653e78ee"
CONFIG = Path("configs/finite_causal_control_v14.yaml")
BASE_FREEZE = Path("artifacts/finite_causal_control_v14.freeze.json")


def digest(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    names = [
        str(CONFIG),
        "reports/V13_COMPLETE_REPORT.md",
        "artifacts/causal_geometry_v13_jvp.freeze.json",
        "artifacts/causal_geometry_v13_jvp_scalar_sharded.freeze.json",
        "artifacts/causal_geometry_v13_finalists.freeze.json",
        "artifacts/causal_bank_v13_capture.freeze.json",
        "results/v13/processed/causal_probe_scaling_v13.parquet",
        "results/v13/processed/moving_tangent_oracle_confirmatory_v13.parquet",
        "src/jclosure/experiments/numerics_v14.py",
    ]
    return {name: sha256_file(root / name) for name in names}


def freeze_base(root: Path) -> dict[str, Any]:
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        != BASELINE_COMMIT
    ):
        raise RuntimeError("V14 baseline commit changed before base freeze")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != BASELINE_COMMIT:
        raise RuntimeError("V14 config baseline mismatch")
    value = {
        "schema_version": 23,
        "protocol_version": "finite_causal_control_v14",
        "purpose": "freeze V14 numerical audit and predeclared control candidates",
        "baseline_commit": BASELINE_COMMIT,
        "no_historical_recomputation": True,
        "source_hashes": _source_hashes(root),
        "config": config,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / BASE_FREEZE, value)
    return value


def verify_base(root: Path) -> dict[str, Any]:
    value = json.loads((root / BASE_FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != digest(value):
        raise RuntimeError("V14 base freeze digest mismatch")
    for name, expected in value["source_hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V14 frozen source changed: {name}")
    return value


def freeze_stage(
    root: Path, name: str, sources: list[str], payload: dict[str, Any]
) -> dict[str, Any]:
    base = verify_base(root)
    path = Path(f"artifacts/finite_causal_control_v14_{name}.freeze.json")
    if (root / path).exists():
        raise RuntimeError(f"stage freeze already exists: {path}")
    value = {
        "schema_version": 23,
        "protocol_version": f"finite_causal_control_v14_{name}",
        "parent_freeze_digest": base["freeze_digest"],
        "source_hashes": {item: sha256_file(root / item) for item in sources},
        **payload,
    }
    value["freeze_digest"] = digest(value)
    write_json_atomic(root / path, value)
    return value
