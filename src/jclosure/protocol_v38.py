"""Independent V38 protocol; V1–V37 reports and results remain immutable."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic


PREFIX = "trajectory_composition_v38"
CONFIG = Path("configs/trajectory_v38.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
PARENT = "ef075e85025eb634b0a02fecadc16c8a4a6811e9"
FROZEN_HISTORY = (
    "reports/V36_COMPLETE_REPORT.md",
    "reports/V37_COMPLETE_REPORT.md",
    "reports/V37_ALL_REPORTS.md",
    "results/v37/processed/v37_adjudication.json",
    "results/v37/processed/v37_integrity_index.json",
    "artifacts/computational_origin_v37.freeze.json",
    "artifacts/computational_origin_v37_report.freeze.json",
)


def digest(value: dict) -> str:
    body = {k: v for k, v in value.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V38 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"Unexpected V38 parent {head}; expected {PARENT}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT or config["primary_groups"] != ["Q2", "Q3", "Q4"]:
        raise RuntimeError("V38 config drift")
    expected = {"R000": [], "R100": ["Q2"], "R010": ["Q3"], "R001": ["Q4"],
                "R110": ["Q2", "Q3"], "R101": ["Q2", "Q4"],
                "R011": ["Q3", "Q4"], "R111": ["Q2", "Q3", "Q4"]}
    if config["conditions"] != expected:
        raise RuntimeError("V38 frozen eight-condition mapping drift")
    model_spec = yaml.safe_load((root / config["model_spec_source"]).read_text(encoding="utf-8"))
    specs = model_spec["models"]
    for key in config["models"]:
        spec = specs[key]
        location = Path(spec["local_path"])
        for filename, expected_hash in spec["weight_sha256"].items():
            if sha256_file(location / filename) != expected_hash:
                raise RuntimeError(f"V38 model weight drift: {key}:{filename}")
        for filename, expected_key in (("tokenizer.json", "tokenizer_sha256"),
                                       ("config.json", "config_sha256")):
            if sha256_file(location / filename) != spec[expected_key]:
                raise RuntimeError(f"V38 model {filename} drift: {key}")
        if len(spec["recurrent_layers"]) != 24:
            raise RuntimeError(f"V38 recurrent architecture drift: {key}")
    record = {
        "schema_version": 1,
        "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_commit": PARENT,
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v38.py"),
        "generator_sha256": sha256_file(root / "src/jclosure/datasets_v8.py"),
        "model_spec_sha256": sha256_file(root / config["model_spec_source"]),
        "model_specs": specs,
        "frozen_history_sha256": {p: sha256_file(root / p) for p in FROZEN_HISTORY},
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": config,
        "prior_v37_formal_results_excluded_from_v38": True,
        "historical_v1_v37_mutated": False,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record):
        raise RuntimeError("V38 base digest mismatch")
    for name, expected in ((CONFIG, record["config_sha256"]),
                           (Path("src/jclosure/protocol_v38.py"), record["protocol_sha256"]),
                           (Path("src/jclosure/datasets_v8.py"), record["generator_sha256"])):
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V38 frozen source drift: {name}")
    for name, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"Historical V1–V37 drift: {name}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V38 stage already frozen: {stage}")
    record = {
        "schema_version": 1,
        "stage": stage,
        "base_freeze_digest": base["freeze_digest"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {name: sha256_file(root / name) for name in inputs},
        **payload,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, stage: str) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V38 stage drift: {stage}")
    for name, expected in record["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V38 stage input drift: {stage}:{name}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
