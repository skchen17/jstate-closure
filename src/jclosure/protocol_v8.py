"""Immutable-history guard and freeze contract for protocol v8."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v8 import PROTOCOL_V8, SCHEMA_VERSION_V8

BASELINE_COMMIT = "2326885c1a502e68f3f707ae1960d9c9d902ddf8"
GUARD_PATH = Path("artifacts/v7_immutable.sha256.json")
FREEZE_PATH = Path("artifacts/persistent_state_v8.freeze.json")
SELECTION_PATH = Path("data/v8/persistent_causal_selection.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"digest", "freeze_digest"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_guard(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != BASELINE_COMMIT:
        raise RuntimeError(
            f"v8 history guard must be created at {BASELINE_COMMIT}; got {head}"
        )
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT], cwd=root, text=True
    ).splitlines()
    paths = [
        name
        for name in names
        if name not in {"README.md", "EXPERIMENT_SPEC.md", "reports/FINAL_REPORT.md"}
        and not name.startswith(("results/v8/", "data/v8/"))
    ]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V8,
        "purpose": "byte-level guard for every tracked v1-v7 artifact at the v8 baseline",
        "baseline_commit": BASELINE_COMMIT,
        "hashes": {name: sha256_file(root / name) for name in paths},
    }
    value["digest"] = _digest(value)
    write_json_atomic(root / GUARD_PATH, value)
    return value


def verify_guard(root: Path) -> dict[str, Any]:
    value = json.loads((root / GUARD_PATH).read_text(encoding="utf-8"))
    if value.get("digest") != _digest(value):
        raise RuntimeError("v7 immutable guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen v1-v7 files changed: {changed}")
    return value


def build_selection(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    formal_path = root / "data/v8/persistent_causal_formal.json"
    teacher_path = root / "results/v8/processed/teacher_formal_v8.json"
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (root / teacher["records"]).read_text(encoding="utf-8").splitlines()
    ]
    correct = {row["prompt_id"] for row in records if row["full_trajectory_correct"]}
    targets = config["persistent_state_v8"]["formal"]["target_valid_per_family"]
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "final_test"):
        counts[split] = {}
        for family in sorted({str(item["family"]) for item in formal["items"]}):
            candidates = sorted(
                (
                    item
                    for item in formal["items"]
                    if item["split"] == split
                    and item["family"] == family
                    and item["example_id"] in correct
                ),
                key=lambda item: hashlib.sha256(
                    str(item["example_id"]).encode()
                ).hexdigest(),
            )
            target = int(targets[split])
            if len(candidates) < target:
                raise RuntimeError(
                    f"teacher-correct selection shortfall {split}/{family}: {len(candidates)}/{target}"
                )
            # Retain deterministic reserves because intervention eligibility is not guaranteed.
            selected.extend(candidates)
            counts[split][family] = len(candidates)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V8,
        "protocol_version": PROTOCOL_V8,
        "formal_data": str(formal_path.relative_to(root)),
        "formal_data_sha256": sha256_file(formal_path),
        "teacher_summary": str(teacher_path.relative_to(root)),
        "teacher_summary_sha256": sha256_file(teacher_path),
        "target_valid_per_family": targets,
        "teacher_correct_candidates": counts,
        "items": selected,
    }
    payload["digest"] = _digest(payload)
    write_json_atomic(root / SELECTION_PATH, payload)
    return payload


def _code_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    names = (
        "README.md",
        "EXPERIMENT_SPEC.md",
        "configs/persistent_state_v8.yaml",
        "data/v8/difficulty_calibration_r6.json",
        "data/v8/persistent_causal_formal_r6_attempt.json",
        "results/v8/processed/difficulty_calibration_v8_r6.json",
        "results/v8/processed/teacher_formal_v8_r6_attempt.json",
        "results/v8/processed/teacher_formal_v8_competence_r2_strict_parser_attempt.json",
        "src/jclosure/datasets_v8.py",
        "src/jclosure/records_v8.py",
        "src/jclosure/protocol_v8.py",
        "src/jclosure/persistent_state_v8.py",
        "src/jclosure/experiments/calibrate_v8.py",
        "src/jclosure/experiments/persistent_state_v8.py",
        "src/jclosure/experiments/compress_persistent_v8.py",
        "src/jclosure/experiments/report_v8.py",
        "src/jclosure/reporting_v8.py",
        "schemas/protocol-v8-record.schema.json",
        "scripts/run_persistent_state_v8.sh",
        "tests/test_v8.py",
    )
    del config
    return [root / name for name in names]


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    selection = build_selection(root, config)
    paths = [root / SELECTION_PATH, *_code_paths(root, config)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v8 freeze inputs missing: {missing}")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V8,
        "protocol_version": PROTOCOL_V8,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_digest": config_digest(config),
        "model_revision": config["model"]["revision"],
        "lens_revision": config["lens"]["revision"],
        "history_guard_digest": guard["digest"],
        "selection_manifest": str(SELECTION_PATH),
        "selection_digest": selection["digest"],
        "thresholds": config["persistent_state_v8"],
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    return value


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_guard(root)
    value = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v8 freeze digest mismatch")
    if value.get("config_digest") != config_digest(config):
        raise RuntimeError("v8 config changed after freeze")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v8 frozen input mismatch: {changed}")
    return value
