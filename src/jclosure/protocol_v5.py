"""Freeze and fail-closed verification for peripheral-state protocol v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.datasets_v5 import PROTOCOL_V5, write_replication_tasks
from jclosure.provenance import sha256_file, write_json_atomic

FREEZE_PATH_V5 = Path("artifacts/peripheral_v5.freeze.json")
REPLICATION_DATA_V5 = Path("data/v5/h2_replication.json")


def _digest(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "freeze_digest"}
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_peripheral_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    replication_path = root / REPLICATION_DATA_V5
    replication_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = write_replication_tasks(root, config, replication_path)
    trace_summary_path = root / config["peripheral_v5"]["source_trace_summary"]
    derived_summary_path = root / config["peripheral_v5"]["derived_trace_summary"]
    if not derived_summary_path.is_file():
        raise RuntimeError("derive the layer-23 v5 traces before freezing protocol")
    teacher_summary_path = root / config["peripheral_v5"]["source_teacher_summary"]
    trace_summary = json.loads(trace_summary_path.read_text(encoding="utf-8"))
    derived_summary = json.loads(derived_summary_path.read_text(encoding="utf-8"))
    tracked = [
        Path(config["_config_path"]),
        root / "src/jclosure/datasets_v5.py",
        root / "src/jclosure/protocol_v5.py",
        root / "src/jclosure/peripheral_v5.py",
        root / "src/jclosure/records_v5.py",
        root / "src/jclosure/reporting_v5.py",
        root / "src/jclosure/experiments/peripheral_v5.py",
        root / "src/jclosure/experiments/h2_replication_v5.py",
        root / "schemas/protocol-v5-record.schema.json",
        root / "scripts/run_peripheral_v5.sh",
        root / "scripts/run_h2_replication_v5.sh",
        root / "scripts/build_report_v5.sh",
        trace_summary_path,
        derived_summary_path,
        teacher_summary_path,
        replication_path,
        root / "artifacts/program_tasks_v4.freeze.json",
        root / "artifacts/v4_immutable.sha256.json",
    ]
    hashes = {
        str(path.resolve().relative_to(root)): sha256_file(path) for path in tracked
    }
    source_domains: dict[str, Any] = {}
    for record in derived_summary["domains"]:
        domain = str(record["domain"])
        source_domains[domain] = {
            "tensor_path": record["tensor_path"],
            "tensor_sha256": record["tensor_sha256"],
            "step_metadata": record["step_metadata"],
            "step_metadata_sha256": record["step_metadata_sha256"],
            "trajectories": record["trajectories"],
            "trajectories_sha256": record["trajectories_sha256"],
            "teacher_correct_trajectories": record["teacher_correct_trajectories"],
            "steps": record["steps"],
        }
    payload: dict[str, Any] = {
        "schema_version": 7,
        "protocol_version": PROTOCOL_V5,
        "config_digest": config_digest(config),
        "model_revision": config["model"]["revision"],
        "lens_revision": config["lens"]["revision"],
        "dictionary_size": int(config["peripheral_v5"]["dictionary_size"]),
        "dictionary_hash": derived_summary["dictionary_hash"],
        "source_protocol": derived_summary["protocol_version"],
        "source_program_freeze_digest": trace_summary["program_freeze_digest"],
        "source_v4_trace_sha256": sha256_file(trace_summary_path),
        "derived_trace_sha256": sha256_file(derived_summary_path),
        "source_domains": source_domains,
        "analysis_split_roles": {
            "train": "train",
            "validation": "validation",
            "causal_fidelity": "causal_test",
            "rollout_test": "rollout_test",
        },
        "replication_data": str(REPLICATION_DATA_V5),
        "replication_count": len(tasks),
        "replication_program_hashes": sorted(value.program_hash for value in tasks),
        "thresholds": {
            key: config["peripheral_v5"][key]
            for key in (
                "full_remainder_minimum_cosine_gain",
                "maximum_action_accuracy_loss",
                "compact_gap_closed",
                "compact_maximum_conditional_gain",
                "compact_minimum_causal_direction_cosine",
                "compact_minimum_output_sign_agreement",
                "recurrent_maximum_state_dimension",
            )
        },
        "hashes": hashes,
    }
    payload["freeze_digest"] = _digest(payload)
    write_json_atomic(root / FREEZE_PATH_V5, payload)
    return payload


def verify_peripheral_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / FREEZE_PATH_V5
    if not path.is_file():
        raise RuntimeError("peripheral v5 freeze is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_V5:
        raise RuntimeError("peripheral v5 freeze protocol mismatch")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("peripheral v5 config changed after freeze")
    if payload.get("freeze_digest") != _digest(payload):
        raise RuntimeError("peripheral v5 freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        candidate = root / relative
        observed = sha256_file(candidate) if candidate.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    for record in payload["source_domains"].values():
        for path_key, hash_key in (
            ("tensor_path", "tensor_sha256"),
            ("step_metadata", "step_metadata_sha256"),
            ("trajectories", "trajectories_sha256"),
        ):
            candidate = root / record[path_key]
            observed = sha256_file(candidate) if candidate.is_file() else "MISSING"
            if observed != record[hash_key]:
                failures.append(str(record[path_key]))
    if failures:
        raise RuntimeError(f"peripheral v5 frozen artifact mismatch: {failures}")
    return payload
