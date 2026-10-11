"""Append-only V40 formal design seal and result-partition seals."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.protocol_v40 import digest
from jclosure.protocol_v40_a3 import verify as verify_a3
from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40_a4"
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
CONFIG = Path("configs/channel_function_v40_formal_a4.yaml")
DESIGN = Path("results/v40/processed/formal_design_a4_v40.json")
SOURCES = (
    "src/jclosure/protocol_v40_a4.py",
    "src/jclosure/experiments/design_v40_a4.py",
    "src/jclosure/experiments/formal_v40_a4.py",
)


def verify_answer_stage(root: Path, parent: dict) -> dict:
    path = root / "artifacts/channel_function_v40_a3_answer_form_adjudication.freeze.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["freeze_digest"] != digest(record)
            or record["base_freeze_digest"] != parent["freeze_digest"]):
        raise RuntimeError("V40 answer-form stage digest drift")
    for name, expected in record["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V40 answer-form stage input drift: {name}")
    return record


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 a4 already frozen")
    parent = verify_a3(root)
    answer = verify_answer_stage(root, parent)
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (cfg["readout"] != "after_one_space" or cfg["primary_model"] != "Q"
            or cfg["temporal_delays"] != [0, 1, 2, 4, 8, 16, 32]):
        raise RuntimeError("V40 a4 formal config drift")
    design = json.loads((root / DESIGN).read_text(encoding="utf-8"))
    if not design["no_model_forward_or_intervention_outcome"]:
        raise RuntimeError("V40 a4 design not response-blind")
    for key in ("Q", "F"):
        if {role: len(design["models"][key]["roles"][role]) for role in
            ("development", "validation", "independent_final")} != {
                "development": 64, "validation": 32, "independent_final": 32}:
            raise RuntimeError("V40 a4 role count drift")
    if list((root / "results/v40/raw").glob("formal_*_a4-*.jsonl")):
        raise RuntimeError("V40 a4 formal outcome exists before design freeze")
    record = {
        "schema_version": 1, "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_a3_digest": parent["freeze_digest"],
        "answer_form_digest": answer["freeze_digest"],
        "config_sha256": sha256_file(root / CONFIG),
        "design_sha256": sha256_file(root / DESIGN),
        "source_sha256": {p: sha256_file(root / p) for p in SOURCES},
        "pool_manifest_sha256": sha256_file(root / "data/v40/sample_pool_manifest_v40.json"),
        "formal_outcomes_observed_before_freeze": False,
        "validation_and_final_unopened": True,
        "config": cfg,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    parent = verify_a3(root)
    answer = verify_answer_stage(root, parent)
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if (record["freeze_digest"] != digest(record)
            or record["parent_a3_digest"] != parent["freeze_digest"]
            or record["answer_form_digest"] != answer["freeze_digest"]):
        raise RuntimeError("V40 a4 digest drift")
    paths = {str(CONFIG): record["config_sha256"],
             str(DESIGN): record["design_sha256"],
             "data/v40/sample_pool_manifest_v40.json": record["pool_manifest_sha256"],
             **record["source_sha256"]}
    for path, expected in paths.items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V40 a4 frozen input drift: {path}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 a4 stage already frozen: {stage}")
    record = {"schema_version": 1, "stage": stage,
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
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
        raise RuntimeError(f"V40 a4 stage drift: {stage}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V40 a4 stage input drift: {stage}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
