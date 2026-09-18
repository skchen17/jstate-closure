"""Immutable protocol and split freezes for causal path geometry V13."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_V13 = "causal_path_geometry_v13"
SCHEMA_VERSION_V13 = 22
GUARD_PATH = Path("artifacts/v12_immutable.sha256.json")
BASE_FREEZE_PATH = Path("artifacts/causal_path_geometry_v13.freeze.json")
SPLIT_FREEZE_PATH = Path("artifacts/causal_bank_v13_splits.freeze.json")
TASKS_PATH = Path("data/v13/causal_bank_candidates.json")
TEACHER_SUMMARY = Path("results/v13/processed/teacher_screen_v13.json")
SELECTION_PATH = Path("data/v13/causal_bank_selected.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_ids(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _historical_paths(root: Path) -> list[Path]:
    output: list[Path] = []
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    for name in tracked:
        if not name:
            continue
        path = root / name
        if not path.is_file():
            continue
        if name == "reports/FINAL_REPORT.md":
            continue
        if name.startswith(("results/v13/", "artifacts/causal/v13/", "data/v13/")):
            continue
        if "v13" in name.lower():
            continue
        if name.startswith(
            (
                "src/",
                "tests/",
                "scripts/",
                "configs/",
                "schemas/",
                "reports/",
                "artifacts/",
                "data/",
            )
        ):
            output.append(path)
    return sorted(output)


def build_guard(root: Path, baseline_commit: str) -> dict[str, Any]:
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _historical_paths(root)
    }
    value = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "purpose": "protect every V1-V12 source, freeze, result, and standalone report",
        "baseline_commit": baseline_commit,
        "excluded_mutable_paths": ["reports/FINAL_REPORT.md"],
        "file_count": len(hashes),
        "hashes": hashes,
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / GUARD_PATH, value)
    return value


def verify_guard(root: Path) -> dict[str, Any]:
    value = json.loads((root / GUARD_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V12 immutable guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"V1-V12 immutable files changed: {changed[:8]}")
    return value


def _code_paths(root: Path) -> list[Path]:
    names = [
        "configs/causal_geometry_v13.yaml",
        "src/jclosure/protocol_v13.py",
        "src/jclosure/experiments/bank_v13.py",
        "src/jclosure/experiments/geometry_v13.py",
        "src/jclosure/experiments/runtime_v13.py",
        "src/jclosure/reporting_v13.py",
        "scripts/run_causal_geometry_v13.sh",
        "schemas/protocol-v13-record.schema.json",
        "tests/test_v13.py",
    ]
    return [root / name for name in names]


def build_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    selected = json.loads((root / SELECTION_PATH).read_text(encoding="utf-8"))
    section = config["causal_geometry_v13"]
    by_split: dict[str, list[str]] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "final_test"):
        rows = [row for row in selected["items"] if row["split"] == split]
        ids = [str(row["example_id"]) for row in rows]
        by_split[split] = ids
        family_counts[split] = {
            family: sum(row["family"] == family for row in rows)
            for family in sorted({str(row["family"]) for row in rows})
        }
    nested = {
        str(size): by_split["train"][: int(size)]
        for size in section["train_sizes"]
        if int(size) <= len(by_split["train"])
    }
    paths = _code_paths(root) + [
        root / TASKS_PATH,
        root / TEACHER_SUMMARY,
        root / SELECTION_PATH,
    ]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "purpose": "freeze V13 independent causal bank, estimands, probes, gates, and analyses",
        "baseline_commit": section["baseline_commit"],
        "parent_guard_digest": guard["freeze_digest"],
        "config_digest": hashlib.sha256(
            json.dumps(section, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "hashes": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in paths
        },
        "splits": {
            split: {
                "count": len(ids),
                "id_sha256": _hash_ids(ids),
                "base_trial_ids": ids,
                "family_counts": family_counts[split],
            }
            for split, ids in by_split.items()
        },
        "nested_train_ids": nested,
        "train_sizes": section["train_sizes"],
        "dimensions": section["dimensions"],
        "probe_sizes": section["jvp"]["probe_sizes"],
        "probe_families": section["jvp"]["probe_families"],
        "target_bundles": section["target_bundles"],
        "saturation": section["saturation"],
        "causal_gates": section["causal_gates"],
        "confirmatory_used_for_selection": False,
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / BASE_FREEZE_PATH, value)
    write_json_atomic(
        root / SPLIT_FREEZE_PATH,
        value["splits"] | {"parent_freeze_digest": value["freeze_digest"]},
    )
    return value


def verify_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_guard(root)
    value = json.loads((root / BASE_FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 base freeze digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 frozen input changed: {name}")
    section = config["causal_geometry_v13"]
    digest = hashlib.sha256(
        json.dumps(section, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != value["config_digest"]:
        raise RuntimeError("V13 config changed after freeze")
    return value


def build_stage_freeze(
    root: Path,
    config: dict[str, Any],
    *,
    path: Path,
    purpose: str,
    inputs: list[Path],
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = verify_base_freeze(root, config)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "purpose": purpose,
        "parent_freeze_digest": base["freeze_digest"],
        "hashes": {str(item): sha256_file(root / item) for item in inputs},
        **payload,
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / path, value)
    return value


def verify_stage_freeze(
    root: Path, config: dict[str, Any], path: Path
) -> dict[str, Any]:
    base = verify_base_freeze(root, config)
    value = json.loads((root / path).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError(f"V13 stage freeze digest mismatch: {path}")
    if value["parent_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError(f"V13 stage parent mismatch: {path}")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 stage input changed: {name}")
    return value
