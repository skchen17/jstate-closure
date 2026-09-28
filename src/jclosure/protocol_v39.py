"""Frozen V39 protocol and append-only stage seals."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic


PREFIX = "interaction_genesis_v39"
CONFIG = Path("configs/genesis_v39.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
PARENT = "3c33f945b22b7f66a90835861bdb829072a1ea38"
FROZEN_HISTORY = (
    "reports/V38_COMPLETE_REPORT.md",
    "reports/V38_ALL_REPORTS.md",
    "results/v38/processed/final_gate_v38.json",
    "results/v38/processed/v38_integrity_index.json",
    "artifacts/trajectory_composition_v38.freeze.json",
    "artifacts/trajectory_composition_v38_report.freeze.json",
)
AUDIT_FILES = (
    "reports/V38_METRIC_AMENDMENT_V39.md",
    "reports/V39_V38_INTERACTION_METRIC_AUDIT.md",
    "results/v39/processed/v38_interaction_metric_audit_v39.json",
    "results/v39/processed/v38_interaction_metric_audit_v39.parquet",
)


def digest(value: dict) -> str:
    body = {k: v for k, v in value.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V39 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"Unexpected V39 parent {head}; expected {PARENT}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT or config["primary_groups"] != ["Q2", "Q3", "Q4"]:
        raise RuntimeError("V39 config drift")
    expected = {"R000": [], "R100": ["Q2"], "R010": ["Q3"], "R001": ["Q4"],
                "R110": ["Q2", "Q3"], "R101": ["Q2", "Q4"],
                "R011": ["Q3", "Q4"], "R111": ["Q2", "Q3", "Q4"]}
    if config["conditions"] != expected:
        raise RuntimeError("V39 condition mapping drift")
    roles = config["roles_per_family"]
    if roles != {"calibration": 4, "development": 16, "validation": 8,
                 "independent_final": 8}:
        raise RuntimeError("V39 sample counts drift")
    formal = config["formal_generator_horizon"]
    if set(formal) != set(config["families"]):
        raise RuntimeError("V39 family horizon drift")
    spec_path = root / config["model_spec_source"]
    specs = yaml.safe_load(spec_path.read_text(encoding="utf-8"))["models"]
    for key in config["models"]:
        spec = specs[key]
        location = Path(spec["local_path"])
        for name, expected_hash in spec["weight_sha256"].items():
            if sha256_file(location / name) != expected_hash:
                raise RuntimeError(f"V39 model weight drift: {key}:{name}")
        for name, field in (("tokenizer.json", "tokenizer_sha256"),
                            ("config.json", "config_sha256")):
            if sha256_file(location / name) != spec[field]:
                raise RuntimeError(f"V39 model {name} drift: {key}")
        if len(spec["recurrent_layers"]) != 24:
            raise RuntimeError(f"V39 recurrent topology drift: {key}")
    frozen_files = {name: sha256_file(root / name) for name in (*FROZEN_HISTORY, *AUDIT_FILES)}
    record = {
        "schema_version": 1,
        "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_commit": PARENT,
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v39.py"),
        "generator_sha256": sha256_file(root / "src/jclosure/datasets_v8.py"),
        "model_spec_sha256": sha256_file(spec_path),
        "model_specs": {key: specs[key] for key in config["models"]},
        "frozen_history_and_audit_sha256": frozen_files,
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": config,
        "historical_v38_formal_results_excluded_from_v39": True,
        "historical_v1_v38_mutated": False,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record):
        raise RuntimeError("V39 base digest mismatch")
    for name, expected in ((CONFIG, record["config_sha256"]),
                           (Path("src/jclosure/protocol_v39.py"), record["protocol_sha256"]),
                           (Path("src/jclosure/datasets_v8.py"), record["generator_sha256"])):
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V39 frozen source drift: {name}")
    for name, expected in record["frozen_history_and_audit_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V39 historical/audit drift: {name}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V39 stage already frozen: {stage}")
    record = {"schema_version": 1, "stage": stage,
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {name: sha256_file(root / name) for name in inputs},
              **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, stage: str) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V39 stage drift: {stage}")
    for name, expected in record["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V39 stage input drift: {stage}:{name}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
