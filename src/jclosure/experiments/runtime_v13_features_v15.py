"""Single-decompression endpoint amendment for the V13 feature runtime."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features_v6 as amendment_6
from jclosure.experiments import runtime_v13_features_v7 as amendment_7
from jclosure.experiments import runtime_v13_features_v14 as amendment_14
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_15"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_15.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v15.py")
PARENT_AMENDMENT_PATH = amendment_14.AMENDMENT_PATH
METADATA_FIELDS = {"base_trial_id", "prompt_id", "family"}


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _endpoints_single_decompression(
    root: Path, ids: list[str]
) -> dict[str, np.ndarray]:
    """Decode each endpoint NPZ member once and preserve the frozen ID order."""

    from jclosure.experiments import geometry_v13 as geometry

    locations: dict[str, tuple[dict[str, np.ndarray], int]] = {}
    for capture in geometry._capture_manifests(root):
        with np.load(root / capture["endpoint_artifact"], allow_pickle=False) as data:
            base_ids = data["base_trial_id"].astype(str)
            arrays = {
                name: np.asarray(data[name])
                for name in data.files
                if name not in METADATA_FIELDS
            }
        for index, base_id in enumerate(base_ids):
            locations[str(base_id)] = (arrays, index)
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    for base_id in ids:
        arrays, index = locations[base_id]
        for name, values in arrays.items():
            output[name].append(values[index])
    return {
        name: np.stack(values)
        for name, values in output.items()
        if len(values) == len(ids)
    }


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_14._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "decode each V13 endpoint NPZ member exactly once",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "endpoint collation before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "floating_reduction_schedule_changed": True,
        "scope": (
            "I/O schedule only: cache each identical NPZ endpoint member after one "
            "decompression, then select the same rows in the same frozen base-trial "
            "order; endpoint values, dtypes, names, stacking, and PCA outputs are "
            "unchanged"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-14-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "the row loop repeatedly decompressed each complete NPZ member; the "
                "run was stopped after PCA/eigh but before endpoint or feature output"
            ),
        },
        "runtime": {
            "endpoint_member_decompressions": "once_per_capture_member",
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_14.CODE_PATH): sha256_file(root / amendment_14.CODE_PATH),
            str(PARENT_AMENDMENT_PATH): sha256_file(root / PARENT_AMENDMENT_PATH),
            str(BASE_FREEZE_PATH): sha256_file(root / BASE_FREEZE_PATH),
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> None:
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 feature-memory amendment-15 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-15 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v15 --target freeze|geometry ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    torch.set_num_threads(amendment_7.INTRAOP_THREADS)
    torch.set_num_interop_threads(amendment_7.INTEROP_THREADS)
    from jclosure.experiments import geometry_v13 as module

    module._load_raw_deltas = amendment_6._load_frozen_workspace
    module._dual_pca_scores = amendment_14._dual_pca_scores_cpu_eigh
    module._endpoints = _endpoints_single_decompression
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
