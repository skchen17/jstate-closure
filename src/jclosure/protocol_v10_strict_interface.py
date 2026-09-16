"""Freeze the derived strict-interface three-way state audit for v10."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.protocol_v10 import verify_candidate_freeze
from jclosure.protocol_v10_causal_amendment import verify_causal_amendment_freeze
from jclosure.provenance import sha256_file, write_json_atomic

STRICT_INTERFACE_FREEZE = Path("artifacts/strict_interface_audit_v10.freeze.json")
PROTOCOL = "strict_interface_three_way_audit_v10_1"


def _digest(value: dict[str, Any]) -> str:
    payload = {k: v for k, v in value.items() if k not in {"digest", "freeze_digest"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _paths(root: Path) -> list[Path]:
    return [
        root / "src/jclosure/protocol_v10_strict_interface.py",
        root / "src/jclosure/experiments/strict_interface_v10.py",
        root / "scripts/run_strict_interface_v10.sh",
        root / "tests/test_strict_interface_v10.py",
        root / "artifacts/causal_sufficiency_v10_candidates.freeze.json",
        root / "artifacts/causal_sufficiency_v10_causal_metadata_amendment.freeze.json",
        root / "results/v10/processed/compact_decoder_v10.json",
        root / "results/v10/processed/decoded_causal_state_validation_v10.json",
        root / "results/v10/processed/decoded_causal_state_validation_v10.parquet",
    ]


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    candidate = verify_candidate_freeze(root, config)
    causal_amendment = verify_causal_amendment_freeze(root, config)
    paths = _paths(root)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"strict-interface freeze inputs missing: {missing}")
    value: dict[str, Any] = {
        "protocol_version": PROTOCOL,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "candidate_freeze_digest": candidate["freeze_digest"],
        "causal_amendment_freeze_digest": causal_amendment["freeze_digest"],
        "three_way_definition": {
            "compact_only": "clean cache plus decoded compact delta",
            "full_raw": "clean cache plus frozen teacher raw delta",
            "compact_plus_raw_residual": "decoded delta plus (raw minus decoded)",
        },
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / STRICT_INTERFACE_FREEZE, value)
    return value


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    candidate = verify_candidate_freeze(root, config)
    value = json.loads((root / STRICT_INTERFACE_FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("strict-interface freeze digest mismatch")
    if value.get("candidate_freeze_digest") != candidate["freeze_digest"]:
        raise RuntimeError("strict-interface candidate mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"strict-interface frozen input mismatch: {changed[:20]}")
    return value
