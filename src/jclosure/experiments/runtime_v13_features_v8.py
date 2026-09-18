"""Pinned-transfer amendment for the bounded V13 feature runtime."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features_v6 as amendment_6
from jclosure.experiments import runtime_v13_features_v7 as amendment_7
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_8"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_8.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v8.py")
PARENT_AMENDMENT_PATH = amendment_7.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_pinned(
    blocks: list[torch.Tensor], train: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run the frozen buffered implementation with one reusable pinned held buffer."""

    count = int(blocks[0].shape[0])
    original_empty = torch.empty

    def pinned_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        size = args[0] if args else kwargs.get("size")
        dtype = kwargs.get("dtype")
        if (
            isinstance(size, tuple)
            and len(size) == 2
            and int(size[0]) == count
            and dtype == torch.float32
            and "device" not in kwargs
        ):
            kwargs["pin_memory"] = True
        return original_empty(*args, **kwargs)

    torch.empty = pinned_empty  # type: ignore[assignment]
    try:
        return amendment_6._dual_pca_scores_contiguous(blocks, train, device=device)
    finally:
        torch.empty = original_empty  # type: ignore[assignment]


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_7._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "reuse one pinned CPU transfer buffer for frozen V13 dual PCA",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "dual-PCA before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "floating_reduction_schedule_changed": False,
        "scope": (
            "allocation placement only: the existing [5550,width] float32 values "
            "buffer is CUDA-pinned and reused, avoiding per-transfer pageable staging; "
            "all tensor values and operations are unchanged"
        ),
        "runtime": {
            "torch_intraop_threads": amendment_7.INTRAOP_THREADS,
            "torch_interop_threads": amendment_7.INTEROP_THREADS,
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_7.CODE_PATH): sha256_file(root / amendment_7.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-8 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-8 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v8 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_pinned
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
