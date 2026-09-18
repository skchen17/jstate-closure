"""Two-way execution sharding for the frozen scalar exact-JVP V13 stage."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.runtime_v13 import (
    AMENDMENT_PATH as RUNTIME_AMENDMENT_PATH,
)
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.experiments.runtime_v13 import _verify as _verify_runtime
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path("artifacts/causal_geometry_v13_jvp_scalar_sharded.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_scalar_sharded.py")
MERGE_PATH = Path("scripts/merge_v13_jvp_scalar_shards.py")
PROTOCOL = "causal_path_geometry_v13_scalar_jvp_sharded_amendment_1"
SHARD_COUNT = 2
ANCHORS_PER_SHARD = 5


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze(root: Path) -> dict[str, Any]:
    runtime = _verify_runtime(root)
    jvp = json.loads((root / geometry.GEOMETRY_FREEZE).read_text())
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "purpose": "split frozen scalar exact-JVP anchors into two disjoint execution shards",
        "parent_runtime_freeze_digest": runtime["freeze_digest"],
        "parent_jvp_freeze_digest": jvp["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "probe_operator_changed": False,
        "target_bundle_changed": False,
        "derivative_backend": "torch.autograd.functional.jvp",
        "direction_batch_size": 1,
        "shard_count": SHARD_COUNT,
        "anchors_per_shard": ANCHORS_PER_SHARD,
        "anchor_partition": "contiguous slices of the frozen sorted anchor list",
        "matrix_roots": [
            f"results/v13/processed/jvp_matrices_scalar_shard_{index}"
            for index in range(SHARD_COUNT)
        ],
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(MERGE_PATH): sha256_file(root / MERGE_PATH),
            str(RUNTIME_AMENDMENT_PATH): sha256_file(root / RUNTIME_AMENDMENT_PATH),
            str(geometry.GEOMETRY_FREEZE): sha256_file(root / geometry.GEOMETRY_FREEZE),
            "configs/causal_geometry_v13.yaml": sha256_file(
                root / "configs/causal_geometry_v13.yaml"
            ),
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    return value


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / AMENDMENT_PATH).read_text())
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 scalar-JVP sharding amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 scalar-JVP sharding input changed: {name}")
    _verify_runtime(root)
    return value


def main() -> None:
    root = Path.cwd()
    if "--freeze-amendment" in sys.argv:
        print(_freeze(root)["freeze_digest"])
        return
    if "--shard-index" not in sys.argv:
        raise SystemExit("--shard-index is required")
    position = sys.argv.index("--shard-index")
    shard_index = int(sys.argv[position + 1])
    del sys.argv[position : position + 2]
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise SystemExit(f"shard index must be in [0, {SHARD_COUNT})")
    _verify(root)
    original_anchor_ids = geometry._anchor_ids

    def sharded_anchor_ids(data: dict[str, Any], per_family: int) -> list[str]:
        anchors = original_anchor_ids(data, per_family)
        start = shard_index * ANCHORS_PER_SHARD
        stop = start + ANCHORS_PER_SHARD
        selected = anchors[start:stop]
        if len(selected) != ANCHORS_PER_SHARD:
            raise RuntimeError("frozen anchor partition has an unexpected size")
        return selected

    geometry._anchor_ids = sharded_anchor_ids
    geometry._load_encoder = _load_encoder_memory_efficient
    geometry.load_model_bundle = _load_model
    geometry.PROTOCOL_V13 = PROTOCOL
    geometry.JVP_MATRIX_ROOT = Path(
        f"results/v13/processed/jvp_matrices_scalar_shard_{shard_index}"
    )
    geometry.JVP_RECORDS = Path(
        f"results/v13/processed/causal_probe_scaling_scalar_shard_{shard_index}_v13.parquet"
    )
    geometry.JVP_SUMMARY = Path(
        f"results/v13/processed/causal_probe_scaling_scalar_shard_{shard_index}_v13.json"
    )
    geometry.main()


if __name__ == "__main__":
    main()
