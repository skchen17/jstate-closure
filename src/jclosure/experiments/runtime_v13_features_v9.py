"""Explicit-staging amendment for the bounded V13 feature runtime."""

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
from jclosure.experiments import runtime_v13_features_v5 as amendment_5
from jclosure.experiments import runtime_v13_features_v6 as amendment_6
from jclosure.experiments import runtime_v13_features_v7 as amendment_7
from jclosure.experiments import runtime_v13_features_v8 as amendment_8
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_9"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_9.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v9.py")
PARENT_AMENDMENT_PATH = amendment_8.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_staged(
    blocks: list[torch.Tensor], train: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run the frozen dual PCA with explicit, reusable conversion staging."""

    count = int(blocks[0].shape[0])
    fit_count = int(len(train))
    if not np.array_equal(train, np.arange(fit_count, dtype=train.dtype)):
        raise RuntimeError(
            "V13 frozen workspace train rows are not a contiguous prefix"
        )
    train_index = torch.arange(fit_count, dtype=torch.long)
    all_index = torch.arange(count, dtype=torch.long)
    total_gram = torch.zeros((fit_count, fit_count), device=device, dtype=torch.float32)
    total_cross = torch.zeros((count, fit_count), device=device, dtype=torch.float32)
    block_products: list[tuple[torch.Tensor, torch.Tensor]] = []
    metadata = []
    buffers: dict[
        int,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ] = {}

    def work(
        width: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if width not in buffers:
            buffers[width] = (
                torch.empty((fit_count, width), dtype=torch.bfloat16),
                torch.empty((fit_count, width), dtype=torch.float64),
                torch.empty((fit_count, width), dtype=torch.float32),
                torch.empty((count, width), dtype=torch.bfloat16),
                torch.empty((count, width), dtype=torch.float32, pin_memory=True),
                torch.empty((fit_count, width), dtype=torch.float32, pin_memory=True),
            )
        return buffers[width]

    for block_index, block in enumerate(blocks):
        flat = block.reshape(count, -1)
        dimension = int(flat.shape[1])
        sum_values = torch.zeros(dimension, dtype=torch.float64)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            width = stop - start
            train_bf16, train_double, _, _, _, _ = work(width)
            torch.index_select(flat[:, start:stop], 0, train_index, out=train_bf16)
            train_double.copy_(train_bf16)
            sum_values[start:stop] = train_double.sum(dim=0)
        mean = (sum_values / fit_count).float()
        squared = 0.0
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            width = stop - start
            train_bf16, _, train_float, _, _, _ = work(width)
            torch.index_select(flat[:, start:stop], 0, train_index, out=train_bf16)
            train_float.copy_(train_bf16)
            train_float.sub_(mean[start:stop])
            train_float.square_()
            squared += float(train_float.sum().item())
        scale = max((squared / (fit_count * dimension)) ** 0.5, 1e-12)
        gram = torch.zeros_like(total_gram)
        cross = torch.zeros_like(total_cross)
        normalization = scale * (dimension**0.5)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            width = stop - start
            _, _, _, held_bf16, values, fit_cpu = work(width)
            torch.index_select(flat[:, start:stop], 0, all_index, out=held_bf16)
            values.copy_(held_bf16)
            values.sub_(mean[start:stop])
            values.div_(normalization)
            fit_cpu.copy_(values[:fit_count])
            fit = fit_cpu.to(device)
            held = values.to(device)
            gram.add_(fit @ fit.T)
            cross.add_(held @ fit.T)
            del fit, held
            torch.cuda.empty_cache()
        total_gram.add_(gram)
        total_cross.add_(cross)
        block_products.append((gram, cross))
        metadata.append(
            {
                "block_index": block_index,
                "elements": dimension,
                "train_rms": scale,
            }
        )

    def scores(
        gram: torch.Tensor, cross: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(values, descending=True)
        values = values[order].clamp_min(0)
        vectors = vectors[:, order]
        keep = values > values[0].clamp_min(1e-20) * 1e-8
        values, vectors = values[keep], vectors[:, keep]
        return cross @ vectors / torch.sqrt(values).clamp_min(1e-12), values

    combined, eigenvalues = scores(total_gram, total_cross)
    structured = [scores(gram, cross)[0] for gram, cross in block_products]
    return (
        combined.cpu().numpy().astype(np.float32),
        torch.cat(structured, dim=1).cpu().numpy().astype(np.float32),
        [
            *metadata,
            {
                "combined_rank": int(combined.shape[1]),
                "largest_eigenvalue": float(eigenvalues[0]),
                "smallest_retained_eigenvalue": float(eigenvalues[-1]),
            },
        ],
    )


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_8._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bound BF16-to-float CPU conversion temporaries in V13 dual PCA",
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
            "allocation placement only: gather the identical BF16 slices into fixed "
            "reusable BF16 buffers before the frozen float64/float32 conversions; "
            "reuse pinned held/train transfer buffers; sample order, 65,536-wide "
            "chunks, casts, reductions, normalization, Gram updates, and eigensolver "
            "are unchanged"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-8-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "implicit CPU conversion temporaries accumulated until swap was full; "
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
            str(amendment_5.CODE_PATH): sha256_file(root / amendment_5.CODE_PATH),
            str(amendment_6.CODE_PATH): sha256_file(root / amendment_6.CODE_PATH),
            str(amendment_8.CODE_PATH): sha256_file(root / amendment_8.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-9 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-9 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v9 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_staged
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
