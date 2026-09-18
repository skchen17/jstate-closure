"""Row-bounded copy amendment for the V13 feature runtime."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features_v6 as amendment_6
from jclosure.experiments import runtime_v13_features_v7 as amendment_7
from jclosure.experiments import runtime_v13_features_v12 as amendment_12
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_13"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_13.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v13.py")
PARENT_AMENDMENT_PATH = amendment_12.AMENDMENT_PATH
COPY_ROW_CHUNK = 32


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_row_bounded(
    blocks: list[torch.Tensor], train: Any, *, device: str
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Run amendment-12 with bounded row panels for large CPU copies."""

    original_copy = torch.Tensor.copy_

    def row_bounded_copy(
        output: torch.Tensor,
        source: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            output.device.type == "cpu"
            and source.device.type == "cpu"
            and output.ndim == 2
            and source.ndim == 2
            and output.shape == source.shape
            and output.shape[0] > COPY_ROW_CHUNK
            and output.shape[1] >= 65_536
        ):
            for start in range(0, output.shape[0], COPY_ROW_CHUNK):
                stop = min(start + COPY_ROW_CHUNK, output.shape[0])
                original_copy(output[start:stop], source[start:stop], *args, **kwargs)
            return output
        return original_copy(output, source, *args, **kwargs)

    torch.Tensor.copy_ = row_bounded_copy  # type: ignore[method-assign]
    try:
        return amendment_12._dual_pca_scores_single_pool(blocks, train, device=device)
    finally:
        torch.Tensor.copy_ = original_copy  # type: ignore[method-assign]


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_12._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bound allocator panels in the V13 dual-PCA CPU copies",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "dual-PCA before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "floating_reduction_schedule_changed": True,
        "scope": (
            "allocation schedule only: split each identical large two-dimensional "
            "CPU copy into 32-row panels so allocator temporaries are reused; copy "
            "values and row order are unchanged, and amendment-12 still controls all "
            "casts, reductions, normalization, GPU Gram updates, and eigensolver"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-12-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "large CPU copies retained one 38.4 MiB conversion panel per internal "
                "tile; the run was stopped before any estimand output"
            ),
        },
        "runtime": {
            "copy_row_chunk": COPY_ROW_CHUNK,
            "dual_pca_cpu_threads": 1,
            "torch_interop_threads": amendment_7.INTEROP_THREADS,
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_12.CODE_PATH): sha256_file(root / amendment_12.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-13 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-13 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v13 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_row_bounded
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
