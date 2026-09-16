"""Immutable-history, base-freeze, and candidate-freeze contracts for v10."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic

SCHEMA_VERSION_V10 = 13
PROTOCOL_V10 = "corrected_causal_sufficiency_protocol_v10"
BASELINE_COMMIT = "69dc044ac080483a8df5c6d2a95fc0a3cf1c4682"
GUARD_PATH = Path("artifacts/v9_immutable.sha256.json")
FREEZE_PATH = Path("artifacts/causal_sufficiency_v10.freeze.json")
CANDIDATE_FREEZE_PATH = Path("artifacts/causal_sufficiency_v10_candidates.freeze.json")
OBSERVATIONAL_SUMMARY = Path(
    "results/v10/processed/corrected_sub512_sufficiency_v10.json"
)
SCREEN_PATH = Path("results/v8/processed/structured_component_screen_v8.parquet")


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
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT],
        cwd=root,
        text=True,
    ).splitlines()
    paths = [name for name in names if name != "reports/FINAL_REPORT.md"]
    missing = [name for name in paths if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"v10 baseline files missing: {missing[:20]}")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V10,
        "purpose": "byte-level guard for every tracked v1-v9 file at v10 baseline",
        "baseline_commit": BASELINE_COMMIT,
        "declared_exception": "reports/FINAL_REPORT.md",
        "hashes": {name: sha256_file(root / name) for name in paths},
    }
    value["digest"] = _digest(value)
    write_json_atomic(root / GUARD_PATH, value)
    return value


def verify_guard(root: Path) -> dict[str, Any]:
    value = json.loads((root / GUARD_PATH).read_text(encoding="utf-8"))
    if value.get("digest") != _digest(value):
        raise RuntimeError("v10 history guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen v1-v9 files changed: {changed[:20]}")
    return value


def _code_paths(root: Path) -> list[Path]:
    names = (
        "configs/causal_sufficiency_v10.yaml",
        "schemas/protocol-v10-record.schema.json",
        "scripts/run_causal_sufficiency_v10.sh",
        "src/jclosure/protocol_v10.py",
        "src/jclosure/experiments/sufficiency_v10.py",
        "src/jclosure/experiments/decoded_causal_v10.py",
        "src/jclosure/experiments/report_v10.py",
        "src/jclosure/reporting_v10.py",
        "tests/test_v10.py",
    )
    return [root / name for name in names]


def _source_paths(root: Path) -> list[Path]:
    names = [
        "artifacts/persistent_state_v8.freeze.json",
        "artifacts/sufficiency_protocol_v9.freeze.json",
        "artifacts/sufficiency_protocol_v9_reporting_amendment_3.freeze.json",
        "artifacts/persistent/v8/structured_features_v8.npz",
        "results/v8/processed/structured_features_v8.json",
        "results/v8/processed/structured_component_screen_v8.parquet",
        "results/v8/processed/structured_component_screen_v8.json",
        "results/v9/processed/sufficiency_dimension_sweep_v9.json",
        "results/v9/processed/sufficiency_dimension_sweep_v9.parquet",
        "results/v9/processed/residual_information_localization_v9.json",
        "results/v9/processed/residual_information_localization_v9.parquet",
    ]
    paths = [root / name for name in names]
    for split in ("train", "validation", "final_test"):
        summary = root / f"results/v8/processed/persistent_capture_{split}_v8.json"
        paths.append(summary)
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            paths.append(root / payload["endpoint_artifact"])
    return paths


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = build_guard(root)
    paths = [*_code_paths(root), *_source_paths(root)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v10 freeze inputs missing: {missing}")
    feature_manifest = json.loads(
        (root / "results/v8/processed/structured_features_v8.json").read_text()
    )
    section = config["causal_sufficiency_v10"]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_digest": config_digest(config),
        "history_guard_digest": guard["digest"],
        "v8_source_freeze_digest": feature_manifest["source_freeze_digest"],
        "v9_source_freeze_digest": json.loads(
            (root / "artifacts/sufficiency_protocol_v9.freeze.json").read_text()
        )["freeze_digest"],
        "effective_feature_rank": int(feature_manifest["dimension"]),
        "thresholds": section,
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    return value


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    value = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v10 base freeze digest mismatch")
    if value.get("history_guard_digest") != guard["digest"]:
        raise RuntimeError("v10 history guard mismatch")
    if value.get("config_digest") != config_digest(config):
        raise RuntimeError("v10 config changed after freeze")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v10 frozen input mismatch: {changed[:20]}")
    return value


def _confirmatory_selection(
    root: Path, section: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, Any]]:
    frame = pd.read_parquet(root / SCREEN_PATH)
    frame = frame[(frame["split"] == "final_test") & (frame["condition"] == "R7")]
    columns = [str(value) for value in section["score_columns"]]
    per_family = int(section["per_family"])
    enriched_per_family = int(section["effect_enriched_per_family"])
    confirmatory: list[str] = []
    enriched: list[str] = []
    details: dict[str, Any] = {}
    for family, values in frame.groupby("family", sort=True):
        normalized = []
        for column in columns:
            raw = values[column].to_numpy(dtype=float)
            scale = raw.std()
            normalized.append((raw - raw.mean()) / (scale if scale > 1e-12 else 1.0))
        score = np.mean(np.stack(normalized, axis=1), axis=1)
        ranked = values.assign(_score=score).sort_values(
            ["_score", "base_trial_id"], ascending=[False, True]
        )
        high = ranked.head(enriched_per_family)
        high_ids = high["base_trial_id"].astype(str).tolist()
        remaining = ranked[~ranked["base_trial_id"].astype(str).isin(high_ids)].copy()
        remaining["_hash"] = remaining["base_trial_id"].map(
            lambda value: hashlib.sha256(
                f"{section['random_seed']}:{value}".encode()
            ).hexdigest()
        )
        random_part = remaining.sort_values("_hash").head(
            per_family - enriched_per_family
        )
        random_ids = random_part["base_trial_id"].astype(str).tolist()
        selected = [*high_ids, *random_ids]
        if len(selected) != per_family:
            raise RuntimeError(f"v10 confirmatory shortfall for {family}")
        enriched.extend(high_ids)
        confirmatory.extend(selected)
        details[str(family)] = {
            "available": int(len(ranked)),
            "effect_enriched": high_ids,
            "hash_sampled": random_ids,
            "minimum_enriched_score": float(high["_score"].min()),
        }
    return confirmatory, enriched, details


def build_candidate_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    summary = json.loads((root / OBSERVATIONAL_SUMMARY).read_text(encoding="utf-8"))
    if summary.get("source_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError("v10 observational result belongs to another freeze")
    candidates = [int(value) for value in summary["selected_confirmatory_candidates"]]
    if not candidates or len(candidates) > int(
        config["causal_sufficiency_v10"]["candidate_selection"]["maximum_candidates"]
    ):
        raise RuntimeError("v10 candidate selection is empty or exceeds frozen maximum")
    confirmatory, enriched, details = _confirmatory_selection(
        root, config["causal_sufficiency_v10"]["confirmatory_subset"]
    )
    paths = [
        root / OBSERVATIONAL_SUMMARY,
        root / summary["records"],
        root / "results/v10/processed/residual_localization_audit_v10.json",
        root / "results/v10/processed/residual_localization_audit_v10.parquet",
        root / "results/v10/processed/semantic_sufficiency_audit_v10.json",
        root / "results/v10/processed/semantic_sufficiency_audit_v10.parquet",
        root / SCREEN_PATH,
    ]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "base_freeze_digest": base["freeze_digest"],
        "candidate_selection_status": summary["candidate_selection_status"],
        "candidate_dimensions": candidates,
        "selection_split": "validation",
        "causal_confirmatory_base_trial_ids": confirmatory,
        "effect_enriched_base_trial_ids": enriched,
        "confirmatory_selection": details,
        "confirmatory_base_trial_id_sha256": hashlib.sha256(
            "\n".join(sorted(confirmatory)).encode()
        ).hexdigest(),
        "effect_enriched_base_trial_id_sha256": hashlib.sha256(
            "\n".join(sorted(enriched)).encode()
        ).hexdigest(),
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / CANDIDATE_FREEZE_PATH, value)
    return value


def verify_candidate_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base = verify_freeze(root, config)
    value = json.loads((root / CANDIDATE_FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v10 candidate freeze digest mismatch")
    if value.get("base_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError("v10 candidate freeze base mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v10 candidate-freeze input mismatch: {changed[:20]}")
    return value
