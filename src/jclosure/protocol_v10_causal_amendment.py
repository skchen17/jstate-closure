"""Freeze the additive v10 fix for resolving v8 prompt metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.protocol_v10 import verify_candidate_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_V10_CAUSAL_AMENDMENT = "causal_metadata_mapping_amendment_v10_1"
CAUSAL_AMENDMENT_FREEZE = Path(
    "artifacts/causal_sufficiency_v10_causal_metadata_amendment.freeze.json"
)
DECODER_SUMMARY = Path("results/v10/processed/compact_decoder_v10.json")
FORMAL_DATA = Path("data/v8/persistent_causal_formal.json")


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
    paths = [
        root / "src/jclosure/protocol_v10_causal_amendment.py",
        root / "src/jclosure/experiments/decoded_causal_v10_amendment.py",
        root / "scripts/run_causal_sufficiency_v10_causal_amendment.sh",
        root / "tests/test_v10_causal_amendment.py",
        root / "artifacts/causal_sufficiency_v10_candidates.freeze.json",
        root / DECODER_SUMMARY,
        root / FORMAL_DATA,
    ]
    for split in ("train", "validation", "final_test"):
        summary = root / f"results/v8/processed/persistent_capture_{split}_v8.json"
        paths.append(summary)
        value = json.loads(summary.read_text(encoding="utf-8"))
        paths.append(root / value["pair_records"])
    return paths


def build_causal_amendment_freeze(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    candidate = verify_candidate_freeze(root, config)
    decoder = json.loads((root / DECODER_SUMMARY).read_text(encoding="utf-8"))
    if decoder["candidate_freeze_digest"] != candidate["freeze_digest"]:
        raise RuntimeError("decoder and candidate freeze mismatch")
    paths = _paths(root)
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"causal metadata amendment inputs missing: {missing}")
    value: dict[str, Any] = {
        "protocol_version": PROTOCOL_V10_CAUSAL_AMENDMENT,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "candidate_freeze_digest": candidate["freeze_digest"],
        "decoder_run_id": decoder["run_id"],
        "mapping_definition": (
            "pair_record.prompt_id -> ProgramTraceTask.example_id loaded from "
            "the frozen v8 formal dataset"
        ),
        "reason": (
            "The frozen base runner incorrectly loaded the legacy v7 replication "
            "task table although all selected pairs are protocol-v8 tasks."
        ),
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / CAUSAL_AMENDMENT_FREEZE, value)
    return value


def verify_causal_amendment_freeze(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    candidate = verify_candidate_freeze(root, config)
    value = json.loads((root / CAUSAL_AMENDMENT_FREEZE).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("causal metadata amendment freeze digest mismatch")
    if value.get("candidate_freeze_digest") != candidate["freeze_digest"]:
        raise RuntimeError("causal metadata amendment candidate mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"causal metadata amendment input mismatch: {changed[:20]}")
    return value
