"""CPU-eigensolver amendment for the bounded V13 feature runtime."""

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
from jclosure.experiments import runtime_v13_features_v13 as amendment_13
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_14"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_14.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v14.py")
PARENT_AMENDMENT_PATH = amendment_13.AMENDMENT_PATH
EIGEN_THREADS = 8


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_cpu_eigh(
    blocks: list[torch.Tensor], train: Any, *, device: str
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Run amendment-12 while solving large symmetric eigenproblems on CPU."""

    original_eigh = torch.linalg.eigh

    def cpu_eigh(
        input_tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            input_tensor.device.type == "cuda"
            and input_tensor.ndim == 2
            and input_tensor.shape[0] >= 4_000
            and input_tensor.shape[0] == input_tensor.shape[1]
        ):
            previous = torch.get_num_threads()
            torch.set_num_threads(EIGEN_THREADS)
            try:
                values, vectors = original_eigh(input_tensor.cpu(), *args, **kwargs)
                return values.to(input_tensor.device), vectors.to(input_tensor.device)
            finally:
                torch.set_num_threads(previous)
        return original_eigh(input_tensor, *args, **kwargs)

    torch.linalg.eigh = cpu_eigh  # type: ignore[assignment]
    try:
        return amendment_12._dual_pca_scores_single_pool(blocks, train, device=device)
    finally:
        torch.linalg.eigh = original_eigh  # type: ignore[assignment]


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_13._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bound V13 large symmetric-eigensolver host workspace",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "dual-PCA eigensolver before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": True,
        "floating_reduction_schedule_changed": True,
        "scope": (
            "numerical backend only: copy each already-computed symmetric float32 "
            "Gram matrix to CPU, solve the same torch.linalg.eigh problem with the "
            "eight-thread CPU LAPACK backend, and return eigenpairs to the original "
            "CUDA device; Gram values, ordering rule, rank threshold, score formula, "
            "samples, and target estimand are unchanged"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-13-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "the CUDA/MAGMA 4,800-square eigensolver retained one 38.4 MiB host "
                "workspace panel per internal tile; stopped before any output"
            ),
        },
        "runtime": {
            "dual_pca_cpu_threads": 1,
            "eigensolver_device": "cpu",
            "eigensolver_threads": EIGEN_THREADS,
            "torch_interop_threads": amendment_7.INTEROP_THREADS,
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_12.CODE_PATH): sha256_file(root / amendment_12.CODE_PATH),
            str(amendment_13.CODE_PATH): sha256_file(root / amendment_13.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-14 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-14 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v14 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_cpu_eigh
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
