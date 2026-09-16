"""Immutable history, independent split, and staged freezes for protocol v11."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic

SCHEMA_VERSION_V11 = 14
PROTOCOL_V11 = "architecture_resolved_causal_geometry_v11"
BASELINE_COMMIT = "10c24a475852ecf6370c8ad0d5e1cae5ad3cc92e"
GUARD_PATH = Path("artifacts/v10_immutable.sha256.json")
BASE_FREEZE_PATH = Path("artifacts/causal_geometry_v11.freeze.json")
PREPARED_FREEZE_PATH = Path("artifacts/causal_geometry_v11_prepared.freeze.json")
STAGE2_FREEZE_PATH = Path("artifacts/causal_geometry_v11_stage2.freeze.json")
CONFIRM_FREEZE_PATH = Path("artifacts/causal_geometry_v11_confirmatory.freeze.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"digest", "freeze_digest"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tracked_at_baseline(root: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT],
        cwd=root,
        text=True,
    ).splitlines()


def build_guard(root: Path) -> dict[str, Any]:
    names = [
        name for name in _tracked_at_baseline(root) if name != "reports/FINAL_REPORT.md"
    ]
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"v11 baseline files missing: {missing[:20]}")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V11,
        "purpose": "byte-level guard for every tracked v1-v10 file",
        "baseline_commit": BASELINE_COMMIT,
        "declared_exception": "reports/FINAL_REPORT.md",
        "hashes": {name: sha256_file(root / name) for name in names},
    }
    value["digest"] = _digest(value)
    write_json_atomic(root / GUARD_PATH, value)
    return value


def verify_guard(root: Path) -> dict[str, Any]:
    value = json.loads((root / GUARD_PATH).read_text(encoding="utf-8"))
    if value.get("digest") != _digest(value):
        raise RuntimeError("v11 history guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen v1-v10 files changed: {changed[:20]}")
    return value


def _code_paths(root: Path) -> list[Path]:
    names = (
        "configs/causal_geometry_v11.yaml",
        "schemas/protocol-v11-record.schema.json",
        "scripts/run_causal_geometry_v11.sh",
        "src/jclosure/protocol_v11.py",
        "src/jclosure/state_models_v11.py",
        "src/jclosure/experiments/prepare_v11.py",
        "src/jclosure/experiments/causal_v11.py",
        "src/jclosure/experiments/analyze_v11.py",
        "src/jclosure/reporting_v11.py",
        "tests/test_v11.py",
    )
    return [root / name for name in names]


def _source_paths(root: Path) -> list[Path]:
    names = [
        "artifacts/causal_sufficiency_v10.freeze.json",
        "artifacts/causal_sufficiency_v10_candidates.freeze.json",
        "artifacts/causal_sufficiency_v10_causal_metadata_amendment.freeze.json",
        "artifacts/strict_interface_audit_v10.freeze.json",
        "artifacts/persistent/v8/structured_features_v8.npz",
        "results/v8/processed/structured_features_v8.json",
        "results/v8/processed/structured_component_screen_v8.parquet",
        "data/v8/persistent_causal_formal.json",
        "results/v10/processed/decoded_causal_state_validation_v10.json",
        "results/v10/processed/decoded_causal_state_validation_v10.parquet",
        "results/v10/processed/compact_decoder_v10.json",
    ]
    paths = [root / name for name in names]
    for split in ("train", "validation", "final_test"):
        summary = root / f"results/v8/processed/persistent_capture_{split}_v8.json"
        paths.append(summary)
        value = json.loads(summary.read_text(encoding="utf-8"))
        paths.extend((root / value["pair_records"], root / value["effect_records"]))
    return paths


def _program_hashes(root: Path) -> dict[str, str]:
    payload = json.loads(
        (root / "data/v8/persistent_causal_formal.json").read_text(encoding="utf-8")
    )
    return {
        str(item["example_id"]): str(item["program_hash"]) for item in payload["items"]
    }


def _pair_rows(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "final_test"):
        summary = json.loads(
            (
                root / f"results/v8/processed/persistent_capture_{split}_v8.json"
            ).read_text()
        )
        for line in (root / summary["pair_records"]).read_text().splitlines():
            row = json.loads(line)
            if row["valid"]:
                output[str(row["base_trial_id"])] = row
    return output


def _select_confirmatory(
    root: Path, section: dict[str, Any], development_ids: list[str]
) -> dict[str, Any]:
    frame = pd.read_parquet(
        root / "results/v8/processed/structured_component_screen_v8.parquet"
    )
    frame = frame[(frame["split"] == "final_test") & (frame["condition"] == "R7")]
    frame = frame[~frame["base_trial_id"].astype(str).isin(development_ids)].copy()
    pair_rows = _pair_rows(root)
    program_hashes = _program_hashes(root)
    selected: list[str] = []
    enriched: list[str] = []
    details: dict[str, Any] = {}
    per_family = int(section["per_family"])
    enriched_count = int(section["effect_enriched_per_family"])
    seed = int(section["selection_seed"])
    score_columns = (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_abs_change",
    )
    for family, values in frame.groupby("family", sort=True):
        normalized = []
        for column in score_columns:
            raw = values[column].astype(float)
            scale = float(raw.std())
            normalized.append(
                (raw - float(raw.mean())) / (scale if scale > 1e-12 else 1.0)
            )
        score = sum(normalized) / len(normalized)
        ranked = values.assign(_score=score).sort_values(
            ["_score", "base_trial_id"], ascending=[False, True]
        )
        high = ranked.head(enriched_count)
        high_ids = high["base_trial_id"].astype(str).tolist()
        remaining = ranked[~ranked["base_trial_id"].astype(str).isin(high_ids)].copy()
        remaining["_hash"] = remaining["base_trial_id"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        hash_ids = (
            remaining.sort_values("_hash")
            .head(per_family - enriched_count)["base_trial_id"]
            .astype(str)
            .tolist()
        )
        family_ids = [*high_ids, *hash_ids]
        if len(family_ids) != per_family:
            raise RuntimeError(f"v11 confirmatory shortfall for {family}")
        selected.extend(family_ids)
        enriched.extend(high_ids)
        details[str(family)] = {
            "effect_enriched": high_ids,
            "hash_sampled": hash_ids,
            "available_after_v10_exclusion": int(len(ranked)),
        }
    prompts = [str(pair_rows[value]["prompt_id"]) for value in selected]
    programs = [program_hashes[value] for value in prompts]
    return {
        "base_trial_ids": selected,
        "effect_enriched_base_trial_ids": enriched,
        "prompt_ids": prompts,
        "program_hashes": programs,
        "selection": details,
        "base_trial_id_sha256": hashlib.sha256(
            "\n".join(sorted(selected)).encode()
        ).hexdigest(),
        "program_hash_sha256": hashlib.sha256(
            "\n".join(sorted(programs)).encode()
        ).hexdigest(),
    }


def build_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = build_guard(root)
    paths = [*_code_paths(root), *_source_paths(root)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v11 freeze inputs missing: {missing}")
    v10 = json.loads(
        (root / "artifacts/causal_sufficiency_v10_candidates.freeze.json").read_text()
    )
    development_ids = [
        str(value) for value in v10["causal_confirmatory_base_trial_ids"]
    ]
    section = config["causal_geometry_v11"]
    confirmatory = _select_confirmatory(root, section["confirmatory"], development_ids)
    if set(development_ids) & set(confirmatory["base_trial_ids"]):
        raise RuntimeError("v11 confirmatory bank overlaps v10 development bank")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_digest": config_digest(config),
        "history_guard_digest": guard["digest"],
        "thresholds": section,
        "development": {
            "role": "method development and validation only",
            "source": section["development_source"],
            "base_trial_ids": development_ids,
            "base_trial_id_sha256": hashlib.sha256(
                "\n".join(sorted(development_ids)).encode()
            ).hexdigest(),
        },
        "confirmatory": {
            "role": "independent v11 final causal bank; inaccessible to method selection",
            **confirmatory,
        },
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / BASE_FREEZE_PATH, value)
    return value


def verify_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    value = json.loads((root / BASE_FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v11 base freeze digest mismatch")
    if value.get("history_guard_digest") != guard["digest"]:
        raise RuntimeError("v11 history guard mismatch")
    if value.get("config_digest") != config_digest(config):
        raise RuntimeError("v11 config changed after freeze")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v11 frozen input mismatch: {changed[:20]}")
    return value


def build_derived_freeze(
    root: Path,
    config: dict[str, Any],
    *,
    path: Path,
    purpose: str,
    inputs: list[Path],
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = verify_base_freeze(root, config)
    paths = [root / value for value in inputs]
    missing = [str(value.relative_to(root)) for value in paths if not value.is_file()]
    if missing:
        raise RuntimeError(f"{purpose} inputs missing: {missing}")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "purpose": purpose,
        "base_freeze_digest": base["freeze_digest"],
        **payload,
        "hashes": {str(value.relative_to(root)): sha256_file(value) for value in paths},
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / path, value)
    return value


def verify_derived_freeze(
    root: Path, config: dict[str, Any], path: Path
) -> dict[str, Any]:
    base = verify_base_freeze(root, config)
    value = json.loads((root / path).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError(f"derived freeze digest mismatch: {path}")
    if value.get("base_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError(f"derived freeze base mismatch: {path}")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"derived freeze input mismatch: {changed[:20]}")
    return value
