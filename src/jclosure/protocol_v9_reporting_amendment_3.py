"""Additive freeze for corrected protocol-v9 figures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.protocol_v9_reporting_amendment_2 import verify_amendment_2
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path(
    "artifacts/sufficiency_protocol_v9_reporting_amendment_3.freeze.json"
)
AMENDMENT_VERSION = "conditional_sufficiency_v9_reporting_amendment_3"


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "amendment_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _paths(root: Path) -> list[Path]:
    names = (
        "configs/sufficiency_v9.yaml",
        "artifacts/sufficiency_protocol_v9_reporting_amendment_2.freeze.json",
        "results/v9/processed/sufficiency_dimension_sweep_v9.json",
        "results/v9/processed/sufficiency_dimension_sweep_v9.parquet",
        "results/v9/processed/residual_information_localization_v9.json",
        "results/v9/processed/residual_information_localization_v9.parquet",
        "scripts/run_sufficiency_v9_reporting_amendment_3.sh",
        "src/jclosure/protocol_v9_reporting_amendment_3.py",
        "src/jclosure/reporting_v9_amendment_3.py",
        "src/jclosure/experiments/report_v9_amendment_3.py",
        "tests/test_v9_reporting_amendment_3.py",
    )
    return [root / name for name in names]


def build_amendment(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    previous = verify_amendment_2(root, config)
    paths = _paths(root)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v9 reporting amendment-3 inputs missing: {missing}")
    value: dict[str, Any] = {
        "schema_version": 12,
        "amendment_version": AMENDMENT_VERSION,
        "purpose": "prevent incompatible v8/v9 conditional curves from being joined",
        "base_freeze_digest": previous["base_freeze_digest"],
        "previous_amendment_digest": previous["amendment_digest"],
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["amendment_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    return value


def verify_amendment_3(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    previous = verify_amendment_2(root, config)
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("amendment_digest") != _digest(value):
        raise RuntimeError("v9 reporting amendment-3 digest mismatch")
    if value.get("previous_amendment_digest") != previous["amendment_digest"]:
        raise RuntimeError("v9 reporting amendment-3 chain mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v9 reporting amendment-3 input mismatch: {changed[:20]}")
    return value
