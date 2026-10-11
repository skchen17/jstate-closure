"""V40 Qwen trajectory-factorial design and immutable run seals."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from jclosure.protocol_v40 import digest
from jclosure.protocol_v40_a4 import verify as verify_a4
from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40_a5"
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
CONFIG = Path("configs/channel_function_v40_trajectory_a5.yaml")
SOURCES = ("src/jclosure/experiments/trajectory_v40_a5.py",
           "scripts/analyze_v40_trajectory_a5.py")


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 a5 already frozen")
    parent = verify_a4(root)
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (cfg["model"] != "Q" or cfg["delays"] != [0, 8]
            or len(cfg["conditions"]) != 8 or cfg["readout"] != "after_one_space"
            or cfg["task_grounded_gate"] != {
                "minimum_native_qualified_states": 16,
                "positive_bootstrap_lower_bound": 0.0,
                "positive_family_medians_required": 3}):
        raise RuntimeError("V40 a5 design drift")
    if list((root / "results/v40/raw").glob("trajectory_Q_*_a5-*.jsonl")):
        raise RuntimeError("V40 a5 trajectory outcome exists before freeze")
    record = {"schema_version": 1, "protocol_version": PREFIX,
              "created_utc": datetime.now(UTC).isoformat(),
              "parent_a4_digest": parent["freeze_digest"],
              "config_sha256": sha256_file(root / CONFIG),
              "source_sha256": {name: sha256_file(root / name) for name in SOURCES},
              "formal_trajectory_outcomes_observed_before_freeze": False,
              "config": cfg}
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    parent = verify_a4(root)
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if (record["freeze_digest"] != digest(record)
            or record["parent_a4_digest"] != parent["freeze_digest"]):
        raise RuntimeError("V40 a5 digest drift")
    for path, expected in ((str(CONFIG), record["config_sha256"]),
                           *record["source_sha256"].items()):
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V40 a5 frozen input drift: {path}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 a5 stage already frozen: {stage}")
    record = {"schema_version": 1, "stage": stage,
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(UTC).isoformat(),
              "input_sha256": {p: sha256_file(root / p) for p in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, stage: str) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["freeze_digest"] != digest(record)
            or record["base_freeze_digest"] != base["freeze_digest"]):
        raise RuntimeError(f"V40 a5 stage drift: {stage}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V40 a5 stage input drift: {stage}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
