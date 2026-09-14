"""Freeze and immutable-history checks for the additive v6 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from jclosure.config import config_digest
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.records_v6 import PROTOCOL_V6, SCHEMA_VERSION_V6

FREEZE_PATH = Path("artifacts/peripheral_v6.freeze.json")
PAIR_MANIFEST = Path("data/v6/causal_pair_split.json")
V5_GUARD_PATH = Path("artifacts/v5_immutable.sha256.json")


def _digest(payload: dict[str, Any]) -> str:
    clean = {
        key: value
        for key, value in payload.items()
        if key not in {"freeze_digest", "manifest_digest"}
    }
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _v5_paths(root: Path) -> list[Path]:
    tracked = []
    for relative in (
        "configs/peripheral_v5.yaml",
        "artifacts/peripheral_v5.freeze.json",
        "reports/PERIPHERAL_STATE_V5.md",
    ):
        path = root / relative
        if path.is_file():
            tracked.append(path)
    for base in (root / "results/v5", root / "data/v5"):
        if base.is_dir():
            tracked.extend(path for path in base.rglob("*") if path.is_file())
    for pattern in (
        "src/jclosure/*v5.py",
        "src/jclosure/experiments/*v5.py",
        "scripts/*v5.sh",
        "schemas/*v5*",
    ):
        tracked.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(tracked))


def build_v5_guard(root: Path, *, baseline_commit: str) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION_V6,
        "purpose": "byte-level guard for frozen v5 inputs and results",
        "baseline_commit": baseline_commit,
        "hashes": {
            str(path.relative_to(root)): sha256_file(path) for path in _v5_paths(root)
        },
    }
    payload["manifest_digest"] = _digest(payload)
    write_json_atomic(root / V5_GUARD_PATH, payload)
    return payload


def verify_v5_guard(root: Path) -> dict[str, Any]:
    payload = json.loads((root / V5_GUARD_PATH).read_text(encoding="utf-8"))
    if payload.get("manifest_digest") != _digest(payload):
        raise RuntimeError("v5 immutable manifest digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"frozen v5 files changed: {failures}")
    return payload


def write_pair_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = root / config["peripheral_v6"]["source_h2_records"]
    frame = pd.read_parquet(source)
    frame = frame[(frame["condition"] == "j_preserving") & frame["valid"]]
    seed = int(config["peripheral_v6"]["causal_split_seed"])
    fraction = float(config["peripheral_v6"]["causal_fit_fraction"])
    items = []
    for family, group in frame.groupby("family", sort=True):
        ordered = sorted(
            group.to_dict("records"),
            key=lambda value: hashlib.sha256(
                f"{seed}:{value['base_trial_id']}".encode()
            ).hexdigest(),
        )
        cut = max(1, min(len(ordered) - 1, round(fraction * len(ordered))))
        for index, value in enumerate(ordered):
            items.append(
                {
                    "base_trial_id": str(value["base_trial_id"]),
                    "prompt_id": str(value["prompt_id"]),
                    "donor_id": str(value["donor_id"]),
                    "family": str(family),
                    "horizon": int(value["horizon"]),
                    "role": "causal_fit" if index < cut else "causal_test",
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION_V6,
        "protocol_version": PROTOCOL_V6,
        "seed": seed,
        "source": str(source.relative_to(root)),
        "source_sha256": sha256_file(source),
        "items": sorted(items, key=lambda value: str(value["base_trial_id"])),
    }
    payload["manifest_digest"] = _digest(payload)
    path = root / PAIR_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return payload


def build_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = verify_v5_guard(root)
    pairs = write_pair_manifest(root, config)
    files = [
        Path(config["_config_path"]),
        root / "src/jclosure/records_v6.py",
        root / "src/jclosure/protocol_v6.py",
        root / "src/jclosure/peripheral_v6.py",
        root / "src/jclosure/experiments/causal_endpoint_v6.py",
        root / "src/jclosure/experiments/peripheral_ceiling_v6.py",
        root / "src/jclosure/reporting_v6.py",
        root / "schemas/protocol-v6-record.schema.json",
        root / "scripts/run_peripheral_v6.sh",
        root / "scripts/build_report_v6.sh",
        root / PAIR_MANIFEST,
    ]
    section = config["peripheral_v6"]
    source_paths = [
        root / section[key]
        for key in (
            "source_h2_summary",
            "source_h2_records",
            "source_mediation_summary",
            "source_reference_summary",
            "source_trace_summary",
        )
    ]
    hashes = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in [*files, *source_paths]
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V6,
        "protocol_version": PROTOCOL_V6,
        "config_digest": config_digest(config),
        "model_revision": config["model"]["revision"],
        "lens_revision": config["lens"]["revision"],
        "dictionary_size": int(section["dictionary_size"]),
        "pair_manifest": str(PAIR_MANIFEST),
        "pair_manifest_digest": pairs["manifest_digest"],
        "v5_guard_digest": guard["manifest_digest"],
        "thresholds": section["ceiling_rules"],
        "hashes": hashes,
    }
    payload["freeze_digest"] = _digest(payload)
    write_json_atomic(root / FREEZE_PATH, payload)
    return payload


def verify_freeze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verify_v5_guard(root)
    payload = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_V6:
        raise RuntimeError("v6 protocol mismatch")
    if payload.get("config_digest") != config_digest(config):
        raise RuntimeError("v6 config changed after freeze")
    if payload.get("freeze_digest") != _digest(payload):
        raise RuntimeError("v6 freeze digest mismatch")
    failures = []
    for relative, expected in payload["hashes"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"v6 frozen input mismatch: {failures}")
    return payload
