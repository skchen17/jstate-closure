"""Fixed-buffer amendment for exact V13 dual PCA."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features_v3 as amendment_3
from jclosure.experiments import runtime_v13_features_v4 as amendment_4
from jclosure.experiments.bank_v13 import CAPTURE_FREEZE_PATH
from jclosure.protocol_v13 import (
    BASE_FREEZE_PATH,
    verify_base_freeze,
    verify_stage_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_5"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_5.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v5.py")
PARENT_AMENDMENT_PATH = amendment_4.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_buffered(
    blocks: list[torch.Tensor], train: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Frozen exact dual PCA using reusable, width-keyed CPU work buffers."""

    count = int(blocks[0].shape[0])
    fit_count = int(len(train))
    train_index = torch.as_tensor(train, dtype=torch.long)
    total_gram = torch.zeros((fit_count, fit_count), device=device, dtype=torch.float32)
    total_cross = torch.zeros((count, fit_count), device=device, dtype=torch.float32)
    block_products: list[tuple[torch.Tensor, torch.Tensor]] = []
    metadata = []
    buffers: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    def work(
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if width not in buffers:
            buffers[width] = (
                torch.empty((fit_count, width), dtype=torch.bfloat16),
                torch.empty((fit_count, width), dtype=torch.float64),
                torch.empty((fit_count, width), dtype=torch.float32),
                torch.empty((count, width), dtype=torch.float32),
                torch.empty((fit_count, width), dtype=torch.float32),
            )
        return buffers[width]

    for block_index, block in enumerate(blocks):
        flat = block.reshape(count, -1)
        dimension = int(flat.shape[1])
        sum_values = torch.zeros(dimension, dtype=torch.float64)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            width = stop - start
            train_bf16, train_double, _, _, _ = work(width)
            torch.index_select(flat[:, start:stop], 0, train_index, out=train_bf16)
            train_double.copy_(train_bf16)
            sum_values[start:stop] = train_double.sum(dim=0)
        mean = (sum_values / fit_count).float()
        squared = 0.0
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            width = stop - start
            train_bf16, _, train_float, _, _ = work(width)
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
            _, _, _, values, fit_cpu = work(width)
            values.copy_(flat[:, start:stop])
            values.sub_(mean[start:stop])
            values.div_(normalization)
            torch.index_select(values, 0, train_index, out=fit_cpu)
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
    capture = verify_stage_freeze(root, config, CAPTURE_FREEZE_PATH)
    amendment_4._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bound exact dual-PCA temporary memory with reusable frozen-width buffers",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_capture_freeze_digest": capture["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "dual-PCA before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "numerical_equivalence": (
            "retain exact sample order, 65,536-dimensional boundaries, cast order, "
            "reductions, normalization, Gram updates, and eigendecomposition while "
            "reusing fixed gather/double/float/held work tensors"
        ),
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_4.CODE_PATH): sha256_file(root / amendment_4.CODE_PATH),
            str(PARENT_AMENDMENT_PATH): sha256_file(root / PARENT_AMENDMENT_PATH),
            str(BASE_FREEZE_PATH): sha256_file(root / BASE_FREEZE_PATH),
            str(CAPTURE_FREEZE_PATH): sha256_file(root / CAPTURE_FREEZE_PATH),
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> None:
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 feature-memory amendment-5 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-5 input changed: {name}")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v5 --target freeze|geometry ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import geometry_v13 as module

    module._load_raw_deltas = amendment_3._load_raw_deltas_file_backed
    module._dual_pca_scores = _dual_pca_scores_buffered
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
