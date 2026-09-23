"""Independent V37 protocol; V1–V36 files and findings remain untouched."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic


PREFIX = "computational_origin_v37"
CONFIG = Path("configs/operator_v37.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
PARENT = "d7b93f041706a60895f0294c9e75c9cb6363a4f6"
FROZEN_HISTORY = (
    "reports/V35_COMPLETE_REPORT.md",
    "reports/V36_COMPLETE_REPORT.md",
    "reports/V36_ALL_REPORTS.md",
    "results/v35/processed/v35_adjudication.json",
    "results/v36/processed/v36_adjudication.json",
    "results/v36/processed/v36_full_integrity_index.json",
    "artifacts/computational_origin_v36.freeze.json",
    "reports/V36_V35_ADJUDICATION_AUDIT.md",
)


def digest(value: dict) -> str:
    body = {k: v for k, v in value.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V37 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"Unexpected parent {head}; expected {PARENT}")
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if config["parent_commit"] != PARENT or config["exclude_versions"][-1] != 36:
        raise RuntimeError("V37 config drift")
    model_spec = yaml.safe_load((root / config["model_spec_source"]).read_text(encoding="utf-8"))
    model_specs = model_spec["models"]
    for key in config["models"]:
        spec = model_specs[key]
        location = Path(spec["local_path"])
        for filename, expected in spec["weight_sha256"].items():
            if sha256_file(location / filename) != expected:
                raise RuntimeError(f"V37 model weight drift: {key}:{filename}")
        for filename, expected_key in (("tokenizer.json", "tokenizer_sha256"),
                                       ("config.json", "config_sha256")):
            if sha256_file(location / filename) != spec[expected_key]:
                raise RuntimeError(f"V37 model {filename} drift: {key}")
        if len(spec["recurrent_layers"]) != 24:
            raise RuntimeError(f"V37 recurrent architecture drift: {key}")
    record = {
        "schema_version": 1,
        "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_commit": PARENT,
        "config_sha256": sha256_file(root / CONFIG),
        "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v37.py"),
        "model_spec_sha256": sha256_file(root / config["model_spec_source"]),
        "model_specs": model_specs,
        "frozen_history_sha256": {p: sha256_file(root / p) for p in FROZEN_HISTORY},
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": config,
        "prior_v36_validation_excluded_from_v37": True,
        "prior_v36_independent_final_reopened": False,
        "historical_v1_v36_mutated": False,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record):
        raise RuntimeError("V37 base digest mismatch")
    for name, expected in ((CONFIG, record["config_sha256"]),
                           (Path("src/jclosure/protocol_v37.py"), record["protocol_sha256"])):
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V37 frozen source drift: {name}")
    for name, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"Historical V35/V36 drift: {name}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V37 stage already frozen: {stage}")
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
        raise RuntimeError(f"V37 stage drift: {stage}")
    for name, expected in record["input_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V37 stage input drift: {stage}:{name}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
