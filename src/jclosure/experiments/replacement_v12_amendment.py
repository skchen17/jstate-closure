"""Formal V12 strict-replacement non-identification record.

V12 encodes intervention deltas, not absolute complete cache states.  Reusing the
writeback experiment as if it were compact-only full-state replacement would be a raw
clean-state scaffold bypass.  This amendment records that boundary explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.config import load_config
from jclosure.protocol_v12 import CONFIRM_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL = "strict_state_replacement_v12_amendment_1"
SCHEMA_VERSION = 22
FREEZE_PATH = Path("artifacts/strict_state_replacement_v12_amendment_1.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/replacement_v12_amendment.py")
CONFIRM_RECORDS = Path("results/v12/processed/causal_confirmatory_v12.parquet")
CONFIRM_SUMMARY = Path("results/v12/processed/causal_confirmatory_v12.json")
OUTPUT_RECORDS = Path("results/v12/processed/strict_state_replacement_v12.parquet")
OUTPUT_SUMMARY = Path("results/v12/processed/strict_state_replacement_v12.json")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze(root: Path) -> dict[str, Any]:
    base = verify_base_freeze(
        root, load_config(root / "configs/causal_geometry_v12.yaml")
    )
    inputs = [CODE_PATH, CONFIRM_FREEZE_PATH, CONFIRM_RECORDS, CONFIRM_SUMMARY]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL,
        "parent_base_freeze_digest": base["freeze_digest"],
        "decision": "FULL_STATE_REPLACEMENT_NOT_IDENTIFIED",
        "reason": (
            "the frozen V12 representations encode persistent intervention deltas; "
            "they do not encode an absolute complete REC/conv/KV/model-cache state"
        ),
        "raw_clean_scaffold_counts_as_bypass": True,
        "hashes": {str(path): sha256_file(root / path) for path in inputs},
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    return value


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("strict-replacement amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"strict-replacement amendment input changed: {name}")
    return value


def _run(root: Path) -> dict[str, Any]:
    freeze = _verify(root)
    confirm = json.loads((root / CONFIRM_SUMMARY).read_text(encoding="utf-8"))
    confirm_freeze = json.loads(
        (root / CONFIRM_FREEZE_PATH).read_text(encoding="utf-8")
    )
    best = max(
        confirm_freeze["method_specs"],
        key=lambda value: float(value.get("development_stage2_score", 0.0)),
    )
    method = str(best["method"])
    frame = pd.read_parquet(root / CONFIRM_RECORDS)
    diagnostic = frame[frame["method"] == method].copy()
    diagnostic["condition"] = "reconstructed_persistent_channels"
    diagnostic["replacement_status"] = "INTERVENTION_DIAGNOSTIC_NOT_FULL_REPLACEMENT"
    diagnostic["raw_state_bypass"] = False
    status_rows = []
    for horizon in (1, 2, 4, 8):
        status_rows.extend(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_version": PROTOCOL,
                    "condition": "raw_absolute_state",
                    "horizon": horizon,
                    "replacement_status": "REFERENCE_ONLY",
                    "identified": True,
                    "raw_state_bypass": False,
                    "authorization_path": False,
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_version": PROTOCOL,
                    "condition": "compact_only_reconstructed_absolute_state",
                    "horizon": horizon,
                    "replacement_status": "NOT_IDENTIFIED_ABSOLUTE_STATE_NOT_ENCODED",
                    "identified": False,
                    "raw_state_bypass": False,
                    "authorization_path": True,
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_version": PROTOCOL,
                    "condition": "compact_plus_raw_residual",
                    "horizon": horizon,
                    "replacement_status": "NOT_RUN_DIAGNOSTIC_NOT_AUTHORIZATION_PATH",
                    "identified": False,
                    "raw_state_bypass": True,
                    "authorization_path": False,
                },
            ]
        )
    status = pd.DataFrame(status_rows)
    diagnostic["identified"] = True
    diagnostic["authorization_path"] = False
    output = pd.concat((diagnostic, status), ignore_index=True, sort=False)
    path = root / OUTPUT_RECORDS
    output.to_parquet(path, index=False, compression="zstd")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL,
        "source_freeze_digest": freeze["freeze_digest"],
        "replacement_scope": (
            "formal audit of absolute full persistent/model-cache replacement; "
            "writeback diagnostic retained separately"
        ),
        "conditions": {
            "raw": "REFERENCE_ONLY",
            "reconstructed": "INTERVENTION_DIAGNOSTIC_NOT_FULL_REPLACEMENT",
            "compact_only_reconstructed": ("NOT_IDENTIFIED_ABSOLUTE_STATE_NOT_ENCODED"),
            "compact_plus_raw_residual": ("NOT_RUN_DIAGNOSTIC_NOT_AUTHORIZATION_PATH"),
        },
        "diagnostic_method": method,
        "aggregates": {method: confirm["aggregates"][method]},
        "authorization": {method: confirm["authorization"][method]},
        "authorized_methods": [],
        "compact_only_full_state_verified": False,
        "raw_state_bypass_detected": False,
        "strict_full_state_replacement_pass": False,
        "reason": freeze["reason"],
        "records": str(OUTPUT_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(root / OUTPUT_SUMMARY, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    root = Path.cwd()
    value = _freeze(root) if args.stage == "freeze" else _run(root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
