"""V40 starting-point and response-blind sample/design seals."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from jclosure.provenance import sha256_file, write_json_atomic

PREFIX = "channel_function_v40"
CONFIG = Path("configs/channel_function_v40.yaml")
BASE = Path(f"artifacts/{PREFIX}.freeze.json")
PARENT = "0dea89c9296737ba713411c11de97184ba8c9eba"
SOURCES = ("src/jclosure/protocol_v40.py", "src/jclosure/datasets_v40.py",
           "src/jclosure/experiments/sample_pool_v40.py",
           "src/jclosure/experiments/design_v40.py")
HISTORY = ("reports/V39_ALL_REPORTS.md", "reports/V39_COMPLETE_REPORT.md",
           "results/v39/processed/v39_adjudication.json",
           "artifacts/interaction_genesis_v39_report.freeze.json")


def digest(value: dict) -> str:
    body = {k: v for k, v in value.items() if k != "freeze_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def freeze(root: Path) -> dict:
    if (root / BASE).exists():
        raise RuntimeError("V40 base already frozen")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    if head != PARENT:
        raise RuntimeError(f"V40 parent drift: {head}")
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["parent_commit"] != PARENT or cfg["primary_model"] != "Q":
        raise RuntimeError("V40 identity drift")
    if cfg["delays"] != [0, 1, 2, 4, 8, 16, 32]:
        raise RuntimeError("V40 delay drift")
    if cfg["conditions"] != {"BASE": [], "REC_only": ["REC"],
                             "Conv_only": ["Conv"], "KV_only": ["KV"],
                             "REC_Conv": ["REC", "Conv"],
                             "REC_Conv_KV": ["REC", "Conv", "KV"]}:
        raise RuntimeError("V40 channel mapping drift")
    specs_path = root / cfg["model_spec_source"]
    specs = yaml.safe_load(specs_path.read_text(encoding="utf-8"))["models"]
    for key in cfg["models"]:
        spec = specs[key]
        model_dir = Path(spec["local_path"])
        for filename, expected in spec["weight_sha256"].items():
            if sha256_file(model_dir / filename) != expected:
                raise RuntimeError(f"V40 model weight drift: {key}:{filename}")
        for filename, field in (("tokenizer.json", "tokenizer_sha256"),
                                ("config.json", "config_sha256")):
            if sha256_file(model_dir / filename) != spec[field]:
                raise RuntimeError(f"V40 model {filename} drift: {key}")
    record = {
        "schema_version": 1, "protocol_version": PREFIX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_commit": PARENT,
        "config_sha256": sha256_file(root / CONFIG),
        "source_sha256": {p: sha256_file(root / p) for p in SOURCES},
        "model_spec_source_sha256": sha256_file(specs_path),
        "model_specs": {key: specs[key] for key in cfg["models"]},
        "v39_starting_point_sha256": {p: sha256_file(root / p) for p in HISTORY},
        "cumulative_report_at_start_sha256": sha256_file(root / "reports/FINAL_REPORT.md"),
        "config": cfg, "formal_intervention_outcomes_observed_before_freeze": False,
        "v39_formal_results_count_as_v40": False,
    }
    record["freeze_digest"] = digest(record)
    write_json_atomic(root / BASE, record)
    return record


def verify(root: Path) -> dict:
    record = json.loads((root / BASE).read_text(encoding="utf-8"))
    if record["freeze_digest"] != digest(record):
        raise RuntimeError("V40 base digest drift")
    for p, expected in {str(CONFIG): record["config_sha256"],
                        **record["source_sha256"],
                        record["config"]["model_spec_source"]:
                        record["model_spec_source_sha256"],
                        **record["v39_starting_point_sha256"]}.items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 frozen input drift: {p}")
    return record


def stage_freeze(root: Path, stage: str, inputs: list[str], payload: dict) -> dict:
    base = verify(root)
    path = root / f"artifacts/{PREFIX}_{stage}.freeze.json"
    if path.exists():
        raise RuntimeError(f"V40 stage already frozen: {stage}")
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
    if record["freeze_digest"] != digest(record) or record["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V40 stage drift: {stage}")
    for p, expected in record["input_sha256"].items():
        if sha256_file(root / p) != expected:
            raise RuntimeError(f"V40 stage input drift: {stage}:{p}")
    return record


if __name__ == "__main__":
    import sys
    print((freeze(Path.cwd()) if sys.argv[1:] == ["freeze"] else verify(Path.cwd()))["freeze_digest"])
