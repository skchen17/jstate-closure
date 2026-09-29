"""Append-only V40 model calibration protocol seal."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.protocol_v40 import digest
from jclosure.protocol_v40_a1 import verify as verify_a1
from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40_a2"
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
CONFIG = Path("configs/channel_function_v40_calibration_a2.yaml")
SOURCES = ("src/jclosure/protocol_v40_a2.py",
           "src/jclosure/experiments/calibration_v40_a2.py",
           "scripts/preflight_v40_a2.py",
           "src/jclosure/experiments/runtime_v34.py",
           "src/jclosure/cache_v7.py")


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 calibration protocol already frozen")
    parent = verify_a1(root)
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    if head != cfg["parent_startup_commit"] or cfg["role"] != "calibration":
        raise RuntimeError("V40 a2 parent/role drift")
    if cfg["conditions"] != ["BASE", "REC_only", "Conv_only", "KV_only",
                             "REC_Conv", "REC_Conv_KV"] or cfg["calibration_delays"] != [0, 1, 32]:
        raise RuntimeError("V40 a2 calibration design drift")
    if cfg["neutral_text"] != " neutral":
        raise RuntimeError("V40 a2 neutral text drift")
    record = {"schema_version": 1, "protocol_version": PREFIX,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "parent_amendment_digest": parent["freeze_digest"],
              "config_sha256": sha256_file(root / CONFIG),
              "source_sha256": {p: sha256_file(root / p) for p in SOURCES},
              "design_Q_seal_sha256": sha256_file(root / "artifacts/channel_function_v40_a1_design_Q.freeze.json"),
              "design_F_seal_sha256": sha256_file(root / "artifacts/channel_function_v40_a1_design_F.freeze.json"),
              "calibration_outcomes_observed_before_freeze": False,
              "validation_and_final_unopened": True,
              "config": cfg}
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    parent = verify_a1(root)
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record) or record["parent_amendment_digest"] != parent["freeze_digest"]:
        raise RuntimeError("V40 a2 digest drift")
    for p, expected in {str(CONFIG): record["config_sha256"],
                        **record["source_sha256"],
                        "artifacts/channel_function_v40_a1_design_Q.freeze.json": record["design_Q_seal_sha256"],
                        "artifacts/channel_function_v40_a1_design_F.freeze.json": record["design_F_seal_sha256"]}.items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 a2 frozen source drift: {p}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 a2 stage already frozen: {stage}")
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
