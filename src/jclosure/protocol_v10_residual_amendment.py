"""Frozen additive amendment for the v10 residual-localization estimand."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jclosure.protocol_v10 import FREEZE_PATH, verify_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_V10_RESIDUAL_AMENDMENT = "residual_estimand_amendment_v10_1"
AMENDMENT_FREEZE_PATH = Path(
    "artifacts/causal_sufficiency_v10_residual_amendment.freeze.json"
)
ARCHIVED_RESIDUAL_JSON = Path(
    "results/v10/processed/residual_localization_audit_v10_pre_estimand_fix.json"
)
ARCHIVED_RESIDUAL_PARQUET = Path(
    "results/v10/processed/residual_localization_audit_v10_pre_estimand_fix.parquet"
)


def _digest(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"digest", "freeze_digest"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _paths(root: Path) -> list[Path]:
    names = (
        "src/jclosure/protocol_v10_residual_amendment.py",
        "src/jclosure/experiments/residual_audit_v10_amendment.py",
        "scripts/run_causal_sufficiency_v10_residual_amendment.sh",
        "tests/test_v10_residual_amendment.py",
        str(FREEZE_PATH),
        "results/v10/processed/corrected_sub512_sufficiency_v10.json",
        "results/v10/processed/corrected_sub512_sufficiency_v10.parquet",
        str(ARCHIVED_RESIDUAL_JSON),
        str(ARCHIVED_RESIDUAL_PARQUET),
    )
    return [root / name for name in names]


def build_amendment_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    source_json = root / "results/v10/processed/residual_localization_audit_v10.json"
    source_parquet = root / "results/v10/processed/residual_localization_audit_v10.parquet"
    for source, archive in (
        (source_json, root / ARCHIVED_RESIDUAL_JSON),
        (source_parquet, root / ARCHIVED_RESIDUAL_PARQUET),
    ):
        if not archive.exists():
            shutil.copy2(source, archive)
    paths = _paths(root)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v10 residual-amendment inputs missing: {missing}")
    value: dict[str, Any] = {
        "protocol_version": PROTOCOL_V10_RESIDUAL_AMENDMENT,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "base_freeze_digest": base["freeze_digest"],
        "reason": (
            "The first residual audit refit the base coefficients in an augmented "
            "coordinate system, allowing ridge regularization changes to appear as "
            "incremental information. The amendment freezes the base prediction, "
            "uses OOF target residuals, and makes combined_reference exactly equal "
            "to the unified full-minus-compact comparator."
        ),
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_FREEZE_PATH, value)
    return value


def verify_amendment_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    value = json.loads((root / AMENDMENT_FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v10 residual-amendment freeze digest mismatch")
    if value.get("base_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError("v10 residual-amendment base mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v10 residual-amendment input mismatch: {changed[:20]}")
    return value
