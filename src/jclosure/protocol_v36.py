"""Append-only V36 computational read-operator protocol and stage seals."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "computational_origin_v36"
PARENT = "9c272d501f6c0beb6ee0bf6c643dbdc77aef9475"
CONFIG = Path("configs/operator_v36.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
HISTORY = (
    "reports/V35_COMPLETE_REPORT.md", "reports/V35_ALL_REPORTS.md",
    "results/v35/processed/v35_adjudication.json",
    "results/v35/processed/v35_integrity_index.json",
    "artifacts/hierarchical_read_v35_adjudication.freeze.json",
    "reports/V36_V35_ADJUDICATION_AUDIT.md",
    "reports/V34_COMPLETE_REPORT.md", "results/v34/processed/v34_adjudication.json",
    "reports/V33_COMPLETE_REPORT.md", "results/v33/processed/v33_adjudication.json",
    "reports/V32_COMPLETE_REPORT.md", "results/v32/processed/v32_adjudication.json",
)


def digest(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "freeze_digest"},
                                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def freeze(root: Path):
    if (root / BASE).exists():
        raise RuntimeError("V36 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V36 parent mismatch: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text())
    if cfg["parent_commit"] != PARENT:
        raise RuntimeError("V36 config parent drift")
    if len(cfg["local_representative_positions"]) != 4 or len(cfg["relative_depth_groups"]) != 4:
        raise RuntimeError("V36 prospective local layer design incomplete")
    source_paths = {
        "Q": Path("/home/user/anaconda3/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"),
        "F": Path("/home/user/anaconda3/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py"),
    }
    for key, path in source_paths.items():
        if sha256_file(path) != cfg["native_equations"]["source_sha256"][key]:
            raise RuntimeError(f"V36 installed native source drift {key}")
    v34cfg = yaml.safe_load((root / cfg["model_spec_source"]).read_text())
    for key in cfg["models"]:
        model = v34cfg["models"][key]
        location = Path(model["local_path"])
        for filename, expected in model["weight_sha256"].items():
            if sha256_file(location / filename) != expected:
                raise RuntimeError(f"V36 model weight drift: {key}:{filename}")
        if sha256_file(location / "tokenizer.json") != model["tokenizer_sha256"]:
            raise RuntimeError(f"V36 tokenizer drift: {key}")
        if sha256_file(location / "config.json") != model["config_sha256"]:
            raise RuntimeError(f"V36 model config drift: {key}")
        if len(model["recurrent_layers"]) != 24:
            raise RuntimeError(f"V36 recurrent depth drift: {key}")
    v34 = json.loads((root / "results/v34/processed/v34_adjudication.json").read_text())
    for key in ("V34-A_SHARED_FUNCTIONAL_MEDIATOR_IDENTIFIED", "V34-E_SHARED_RECURRENT_READ_MEDIATION"):
        if not v34["formal_outcomes"][key]:
            raise RuntimeError("V36 not authorized by V34")
    record = {"schema_version": 45, "protocol_version": PREFIX, "parent_commit": PARENT,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "config_sha256": sha256_file(root / CONFIG),
              "protocol_sha256": sha256_file(root / "src/jclosure/protocol_v36.py"),
              "v34_model_spec_sha256": sha256_file(root / cfg["model_spec_source"]),
              "model_specs": v34cfg["models"],
              "frozen_history_sha256": {path: sha256_file(root / path) for path in HISTORY},
              "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
              "config": cfg, "V1_V35_records_mutated": False,
              "historical_independent_finals_reopened": False}
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path):
    record = json.loads((root / BASE).read_text())
    if (record["freeze_digest"] != digest(record) or
            record["config_sha256"] != sha256_file(root / CONFIG) or
            record["protocol_sha256"] != sha256_file(root / "src/jclosure/protocol_v36.py") or
            record["v34_model_spec_sha256"] != sha256_file(root / record["config"]["model_spec_source"])):
        raise RuntimeError("V36 base drift")
    for path, expected in record["frozen_history_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V36 historical drift: {path}")
    return record


def stage_freeze(root: Path, name: str, inputs, payload):
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{name}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V36 stage already exists: {name}")
    record = {"schema_version": 45, "protocol_version": f"{PREFIX}_{name}",
              "base_freeze_digest": base["freeze_digest"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": {name: sha256_file(root / name) for name in inputs}, **payload}
    record["freeze_digest"] = digest(record)
    write_json_atomic(path, record)
    return record


def verify_stage(root: Path, name: str):
    base = verify(root)
    record = json.loads((root / f"artifacts/{PREFIX}_{name}.freeze.json").read_text())
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V36 stage mismatch: {name}")
    for path, expected in record["input_sha256"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V36 stage input drift: {name}:{path}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
