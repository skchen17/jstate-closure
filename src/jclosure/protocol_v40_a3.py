"""Append-only V40 sequential-continuation calibration correction."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.protocol_v40 import digest
from jclosure.protocol_v40_a2 import verify as verify_a2
from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40_a3"
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
CONFIG = Path("configs/channel_function_v40_calibration_a3.yaml")
SOURCES = ("src/jclosure/protocol_v40_a3.py",
           "src/jclosure/experiments/calibration_v40_a3.py",
           "scripts/audit_v40_continuation.py")
AUDITS = ("results/v40/processed/continuation_audit_Q_a2.json",
          "results/v40/processed/continuation_audit_F_a2.json")


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 a3 already frozen")
    parent = verify_a2(root)
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["operative_continuation"] != "sequential single-token native cached steps for both models":
        raise RuntimeError("V40 a3 continuation drift")
    if cfg["pilot_delays"] != [0, 1, 32] or cfg["full_calibration_delays"] != [0]:
        raise RuntimeError("V40 a3 delay drift")
    record = {"schema_version": 1, "protocol_version": PREFIX,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "parent_a2_digest": parent["freeze_digest"],
              "config_sha256": sha256_file(root / CONFIG),
              "source_sha256": {p: sha256_file(root / p) for p in SOURCES},
              "a2_continuation_audit_sha256": {p: sha256_file(root / p) for p in AUDITS},
              "a2_raw_pilot_is_not_formal_v40_evidence": True,
              "formal_development_outcomes_observed": False,
              "validation_and_final_unopened": True,
              "config": cfg}
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    parent = verify_a2(root)
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record) or record["parent_a2_digest"] != parent["freeze_digest"]:
        raise RuntimeError("V40 a3 digest drift")
    for p, expected in {str(CONFIG): record["config_sha256"],
                        **record["source_sha256"],
                        **record["a2_continuation_audit_sha256"]}.items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 a3 frozen input drift: {p}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 a3 stage already frozen: {stage}")
    record = {"schema_version": 1, "stage": stage,
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {p: sha256_file(root / p) for p in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
