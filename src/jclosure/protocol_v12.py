"""Frozen splits, estimands, and immutable-history guard for protocol v12."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic

SCHEMA_VERSION_V12 = 15
PROTOCOL_V12 = "local_causal_geometry_v12"
BASELINE_COMMIT = "f5118642f12e3b60e22df52930610867773b108a"
GUARD_PATH = Path("artifacts/v11_immutable.sha256.json")
BASE_FREEZE_PATH = Path("artifacts/causal_geometry_v12.freeze.json")
PREPARED_FREEZE_PATH = Path("artifacts/causal_geometry_v12_prepared.freeze.json")
JVP_FREEZE_PATH = Path("artifacts/causal_geometry_v12_jvp.freeze.json")
CONFIRM_FREEZE_PATH = Path("artifacts/causal_geometry_v12_confirmatory.freeze.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"digest", "freeze_digest"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def build_guard(root: Path) -> dict[str, Any]:
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT],
        cwd=root,
        text=True,
    ).splitlines()
    names = [name for name in names if name != "reports/FINAL_REPORT.md"]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V12,
        "purpose": "byte guard for every tracked v1-v11 file",
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
        raise RuntimeError("v12 history guard digest mismatch")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen v1-v11 files changed: {changed[:20]}")
    return value


def _code_paths(root: Path) -> list[Path]:
    names = (
        "configs/causal_geometry_v12.yaml",
        "schemas/protocol-v12-record.schema.json",
        "scripts/run_causal_geometry_v12.sh",
        "src/jclosure/protocol_v12.py",
        "src/jclosure/state_models_v12.py",
        "src/jclosure/experiments/prepare_v12.py",
        "src/jclosure/experiments/jvp_v12.py",
        "src/jclosure/experiments/causal_v12.py",
        "src/jclosure/experiments/analyze_v12.py",
        "src/jclosure/reporting_v12.py",
        "tests/test_v12.py",
    )
    return [root / name for name in names]


def _source_paths(root: Path) -> list[Path]:
    names = [
        "artifacts/causal_geometry_v11.freeze.json",
        "artifacts/causal_geometry_v11_confirmatory.freeze.json",
        "artifacts/causal_geometry_v11_manifold_amendment.freeze.json",
        "artifacts/persistent/v8/structured_features_v8.npz",
        "data/v8/persistent_causal_formal.json",
        "data/v8/persistent_causal_selection.json",
        "results/v8/processed/structured_component_screen_v8.parquet",
        "results/v11/processed/causal_confirmatory_v11.parquet",
        "results/v11/processed/causal_confirmatory_v11.json",
        "reports/V11_COMPLETE_REPORT.md",
    ]
    paths = [root / name for name in names]
    for split in ("train", "validation", "final_test"):
        summary = root / f"results/v8/processed/persistent_capture_{split}_v8.json"
        paths.append(summary)
        value = json.loads(summary.read_text(encoding="utf-8"))
        paths.extend((root / value["pair_records"], root / value["endpoint_artifact"]))
    return paths


def _pair_rows(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
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


def _screen(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        root / "results/v8/processed/structured_component_screen_v8.parquet"
    )
    return frame[frame["condition"] == "R7"].copy()


def _hash_select(frame: pd.DataFrame, *, per_family: int, seed: int) -> list[str]:
    output = []
    for _, values in frame.groupby("family", sort=True):
        values = values.copy()
        values["_hash"] = values["base_trial_id"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        output.extend(
            values.sort_values("_hash").head(per_family)["base_trial_id"].astype(str)
        )
    return list(output)


def _confirmatory(
    root: Path, section: dict[str, Any], excluded: set[str]
) -> dict[str, Any]:
    frame = _screen(root)
    frame = frame[(frame["split"] == "final_test")]
    frame = frame[~frame["base_trial_id"].astype(str).isin(excluded)].copy()
    score_columns = (
        "output_js_divergence",
        "future_j_trajectory_divergence",
        "target_log_odds_abs_change",
    )
    selected, enriched = [], []
    details = {}
    for family, values in frame.groupby("family", sort=True):
        normalized = []
        for column in score_columns:
            raw = values[column].astype(float)
            scale = max(float(raw.std()), 1e-12)
            normalized.append((raw - float(raw.mean())) / scale)
        ranked = values.assign(_score=sum(normalized) / len(normalized)).sort_values(
            ["_score", "base_trial_id"], ascending=[False, True]
        )
        high = ranked.head(int(section["effect_enriched_per_family"]))
        high_ids = high["base_trial_id"].astype(str).tolist()
        remaining = ranked[~ranked["base_trial_id"].astype(str).isin(high_ids)].copy()
        seed = int(section["selection_seed"])
        remaining["_hash"] = remaining["base_trial_id"].map(
            lambda value, seed=seed: hashlib.sha256(
                f"{seed}:{value}".encode()
            ).hexdigest()
        )
        random_ids = (
            remaining.sort_values("_hash")
            .head(int(section["per_family"]) - len(high_ids))["base_trial_id"]
            .astype(str)
            .tolist()
        )
        family_ids = [*high_ids, *random_ids]
        selected.extend(family_ids)
        enriched.extend(high_ids)
        details[str(family)] = {
            "effect_enriched": high_ids,
            "hash_sampled": random_ids,
            "available_after_exclusion": int(len(values)),
        }
    return {
        "base_trial_ids": selected,
        "effect_enriched_base_trial_ids": enriched,
        "base_trial_id_sha256": _hash_ids(selected),
        "selection": details,
    }


def _nested_train_ids(
    frame: pd.DataFrame, sizes: list[int], seed: int
) -> dict[str, list[str]]:
    train = frame[frame["split"] == "train"].copy()
    family_orders = {}
    for family, values in train.groupby("family", sort=True):
        values = values.copy()
        values["_hash"] = values["base_trial_id"].map(
            lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        family_orders[str(family)] = (
            values.sort_values("_hash")["base_trial_id"].astype(str).tolist()
        )
    output = {}
    available = sum(len(value) for value in family_orders.values())
    for size in sizes:
        if size > available or size % len(family_orders):
            continue
        per_family = size // len(family_orders)
        if any(len(value) < per_family for value in family_orders.values()):
            continue
        output[str(size)] = [
            item
            for family in sorted(family_orders)
            for item in family_orders[family][:per_family]
        ]
    return output


def build_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = build_guard(root)
    paths = [*_code_paths(root), *_source_paths(root)]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v12 freeze inputs missing: {missing}")
    v11 = json.loads((root / "artifacts/causal_geometry_v11.freeze.json").read_text())
    v10 = json.loads(
        (root / "artifacts/causal_sufficiency_v10_candidates.freeze.json").read_text()
    )
    excluded = set(v10["causal_confirmatory_base_trial_ids"])
    excluded.update(v11["development"]["base_trial_ids"])
    excluded.update(v11["confirmatory"]["base_trial_ids"])
    section = config["causal_geometry_v12"]
    screen = _screen(root)
    validation = screen[screen["split"] == "validation"].copy()
    validation_panel = _hash_select(
        validation,
        per_family=int(section["validation_panel"]["per_family"]),
        seed=int(section["validation_panel"]["selection_seed"]),
    )
    remaining_validation = validation[
        ~validation["base_trial_id"].astype(str).isin(validation_panel)
    ]
    anchors = _hash_select(
        remaining_validation,
        per_family=int(section["jvp"]["anchors_per_family"]),
        seed=int(section["jvp"]["selection_seed"]),
    )
    train_sets = _nested_train_ids(
        screen,
        [int(value) for value in section["train_sizes"]],
        int(section["train_subset_seed"]),
    )
    requested_sizes = [int(value) for value in section["train_sizes"]]
    unavailable = [value for value in requested_sizes if str(value) not in train_sets]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_digest": config_digest(config),
        "history_guard_digest": guard["digest"],
        "estimands": {
            "predictive_fidelity": "P_t -> held-out future target",
            "writeback_fidelity": "teacher raw delta effect vs reconstructed delta effect",
            "replacement_fidelity": "continue(raw P_t) vs continue(reconstructed P_hat_t)",
            "non_interchangeability": True,
        },
        "thresholds": section,
        "train_scaling": {
            "nested_base_trial_ids": train_sets,
            "hashes": {key: _hash_ids(ids) for key, ids in train_sets.items()},
            "unavailable_requested_sizes": unavailable,
            "unavailable_status": "NOT_IDENTIFIED_BANK_SIZE",
        },
        "validation_panel": {
            "base_trial_ids": validation_panel,
            "base_trial_id_sha256": _hash_ids(validation_panel),
        },
        "jvp_anchors": {
            "base_trial_ids": anchors,
            "base_trial_id_sha256": _hash_ids(anchors),
        },
        "development": {
            "base_trial_ids": [
                str(value) for value in v11["development"]["base_trial_ids"]
            ],
            "base_trial_id_sha256": v11["development"]["base_trial_id_sha256"],
            "role": "observed development bank only",
        },
        "confirmatory": {
            "role": "new independent v12 final bank",
            **_confirmatory(root, section["confirmatory"], excluded),
        },
        "hashes": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    if set(value["confirmatory"]["base_trial_ids"]) & excluded:
        raise RuntimeError("v12 confirmatory bank overlaps an observed v10/v11 bank")
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / BASE_FREEZE_PATH, value)
    return value


def verify_base_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_guard(root)
    value = json.loads((root / BASE_FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v12 base freeze digest mismatch")
    if value.get("history_guard_digest") != guard["digest"]:
        raise RuntimeError("v12 history guard mismatch")
    if value.get("config_digest") != config_digest(config):
        raise RuntimeError("v12 config changed after freeze")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"v12 frozen input mismatch: {changed[:20]}")
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
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
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
        raise RuntimeError(f"derived v12 freeze digest mismatch: {path}")
    if value.get("base_freeze_digest") != base["freeze_digest"]:
        raise RuntimeError(f"derived v12 freeze base mismatch: {path}")
    changed = [
        name
        for name, expected in value["hashes"].items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if changed:
        raise RuntimeError(f"derived v12 input mismatch: {changed[:20]}")
    return value
