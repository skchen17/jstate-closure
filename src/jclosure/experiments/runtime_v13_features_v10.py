"""Identity-gather amendment for the bounded V13 feature runtime."""

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
from jclosure.experiments import runtime_v13_features_v9 as amendment_9
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_10"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_10.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v10.py")
PARENT_AMENDMENT_PATH = amendment_9.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_identity_copy(
    blocks: list[torch.Tensor], train: Any, *, device: str
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Replace only the two frozen identity gathers with direct BF16 copies."""

    original_index_select = torch.index_select

    def identity_copy(
        input_tensor: torch.Tensor,
        dim: int,
        index: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            out is not None
            and dim == 0
            and input_tensor.dtype == torch.bfloat16
            and out.dtype == torch.bfloat16
            and len(index) == out.shape[0]
            and len(index) <= input_tensor.shape[0]
            and torch.equal(index, torch.arange(len(index), dtype=index.dtype))
        ):
            out.copy_(input_tensor[: len(index)])
            return out
        return original_index_select(input_tensor, dim, index, out=out)

    torch.index_select = identity_copy  # type: ignore[assignment]
    try:
        return amendment_9._dual_pca_scores_staged(blocks, train, device=device)
    finally:
        torch.index_select = original_index_select  # type: ignore[assignment]


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_9._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "avoid allocator retention in the two V13 BF16 identity gathers",
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
            "allocation placement only: the frozen train and all-row indices are "
            "identity prefixes, so copy the same BF16 column slices directly into "
            "the amendment-9 BF16 staging buffers; all subsequent casts, reductions, "
            "normalization, chunk order, Gram updates, and eigensolver are unchanged"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-9-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "the identity index_select kernel retained internal gather blocks; "
                "the run was stopped before OOM and before any estimand output"
            ),
        },
        "runtime": {
            "torch_intraop_threads": amendment_7.INTRAOP_THREADS,
            "torch_interop_threads": amendment_7.INTEROP_THREADS,
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_9.CODE_PATH): sha256_file(root / amendment_9.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-10 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-10 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v10 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_identity_copy
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
