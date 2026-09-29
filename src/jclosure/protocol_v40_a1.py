"""Append-only V40 startup amendment: prompt-level collision rejection."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jclosure.protocol_v40 import digest, verify as verify_initial
from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40_a1"
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
SOURCES = ("src/jclosure/protocol_v40_a1.py",
           "src/jclosure/experiments/sample_pool_v40_a1.py",
           "src/jclosure/experiments/design_v40_a1.py")


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 amendment already frozen")
    initial = verify_initial(root)
    if any((root / "data/v40").glob("*.json")):
        raise RuntimeError("Cannot amend after pool records exist")
    record = {
        "schema_version": 1, "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "initial_base_freeze_digest": initial["freeze_digest"],
        "reason": "Initial pool run stopped before writing records: distinct transition programs could share a recipient prompt.",
        "correction": "Reject recipient/donor prompt collisions during response-blind generation; keep initial freeze immutable.",
        "initial_sample_pool_files_written": False,
        "intervention_outcomes_observed": False,
        "source_sha256": {p: sha256_file(root / p) for p in SOURCES},
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    initial = verify_initial(root)
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record) or record["initial_base_freeze_digest"] != initial["freeze_digest"]:
        raise RuntimeError("V40 amendment digest drift")
    for p, expected in record["source_sha256"].items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 amendment source drift: {p}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 amended stage already frozen: {stage}")
    record = {"schema_version": 1, "stage": stage,
              "amendment_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {p: sha256_file(root / p) for p in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, stage: str) -> dict:
    base = verify(root)
    record = json.loads((root / f"artifacts/{PREFIX}_{stage}.freeze.json").read_text())
    if record["freeze_digest"] != digest(record) or record["amendment_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V40 amended stage drift: {stage}")
    for p, expected in record["input_sha256"].items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 amended stage input drift: {stage}:{p}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
