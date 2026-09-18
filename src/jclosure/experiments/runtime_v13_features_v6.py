"""Frozen-workspace and contiguous-train amendment for exact V13 dual PCA."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features_v3 as amendment_3
from jclosure.experiments import runtime_v13_features_v5 as amendment_5
from jclosure.experiments.bank_v13 import CAPTURE_FREEZE_PATH
from jclosure.protocol_v13 import (
    BASE_FREEZE_PATH,
    verify_base_freeze,
    verify_stage_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_6"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_6.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v6.py")
PARENT_AMENDMENT_PATH = amendment_5.AMENDMENT_PATH
WORKSPACE = amendment_3.WORKSPACE
WORKSPACE_SHAPES = {
    "recurrent": (5_550, 6, 32, 128, 128),
    "conv": (5_550, 6, 8_192, 4),
    "kv": (5_550, 2, 2, 4, 256, 256),
}


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _workspace_paths(root: Path) -> dict[str, Path]:
    return {name: root / WORKSPACE / f"{name}.bf16" for name in WORKSPACE_SHAPES}


def _pair_metadata(root: Path) -> dict[str, dict[str, Any]]:
    from jclosure.experiments import geometry_v13 as geometry

    output = {}
    for capture in geometry._capture_manifests(root):
        for line in (
            (root / capture["pair_records"]).read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            output[str(row["base_trial_id"])] = row
    return output


def _load_frozen_workspace(
    root: Path,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    from jclosure.experiments import geometry_v13 as geometry

    metadata = _pair_metadata(root)
    ids: list[str] = []
    families: list[str] = []
    splits: list[str] = []
    for capture in geometry._capture_manifests(root):
        for declaration in capture["state_shards"]:
            for base_id in declaration["base_trial_ids"]:
                row = metadata[str(base_id)]
                ids.append(str(base_id))
                families.append(str(row["family"]))
                splits.append(str(row["split"]))
    if len(ids) != WORKSPACE_SHAPES["recurrent"][0] or len(set(ids)) != len(ids):
        raise RuntimeError("V13 frozen workspace metadata count/uniqueness mismatch")
    tensors = {}
    for name, shape in WORKSPACE_SHAPES.items():
        path = _workspace_paths(root)[name]
        expected_bytes = math.prod(shape) * 2
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"V13 workspace size mismatch: {path}")
        tensors[name] = torch.from_file(
            str(path), shared=True, size=math.prod(shape), dtype=torch.bfloat16
        ).reshape(shape)
    return tensors, ids, families, splits


def _dual_pca_scores_contiguous(
    blocks: list[torch.Tensor], train: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Frozen exact dual PCA using the frozen contiguous train prefix and fixed buffers."""

    count = int(blocks[0].shape[0])
    fit_count = int(len(train))
    if not np.array_equal(train, np.arange(fit_count, dtype=train.dtype)):
        raise RuntimeError(
            "V13 frozen workspace train rows are not the contiguous prefix"
        )
    total_gram = torch.zeros((fit_count, fit_count), device=device, dtype=torch.float32)
    total_cross = torch.zeros((count, fit_count), device=device, dtype=torch.float32)
    block_products: list[tuple[torch.Tensor, torch.Tensor]] = []
    metadata = []
    buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def work(width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if width not in buffers:
            buffers[width] = (
                torch.empty((fit_count, width), dtype=torch.float64),
                torch.empty((fit_count, width), dtype=torch.float32),
                torch.empty((count, width), dtype=torch.float32),
            )
        return buffers[width]

    for block_index, block in enumerate(blocks):
        flat = block.reshape(count, -1)
        dimension = int(flat.shape[1])
        sum_values = torch.zeros(dimension, dtype=torch.float64)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            train_double, _, _ = work(stop - start)
            train_double.copy_(flat[:fit_count, start:stop])
            sum_values[start:stop] = train_double.sum(dim=0)
        mean = (sum_values / fit_count).float()
        squared = 0.0
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            _, train_float, _ = work(stop - start)
            train_float.copy_(flat[:fit_count, start:stop])
            train_float.sub_(mean[start:stop])
            train_float.square_()
            squared += float(train_float.sum().item())
        scale = max((squared / (fit_count * dimension)) ** 0.5, 1e-12)
        gram = torch.zeros_like(total_gram)
        cross = torch.zeros_like(total_cross)
        normalization = scale * (dimension**0.5)
        for start in range(0, dimension, 65_536):
            stop = min(start + 65_536, dimension)
            _, _, values = work(stop - start)
            values.copy_(flat[:, start:stop])
            values.sub_(mean[start:stop])
            values.div_(normalization)
            fit = values[:fit_count].to(device)
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
    amendment_5._verify(root)
    workspace_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in _workspace_paths(root).values()
    }
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "freeze completed file-backed tensors and use their contiguous train prefix",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_capture_freeze_digest": capture["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "dual-PCA before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "train_rows": "frozen contiguous prefix [0,4800)",
        "workspace_shapes": {
            name: list(shape) for name, shape in WORKSPACE_SHAPES.items()
        },
        "workspace_hashes": workspace_hashes,
        "numerical_equivalence": (
            "the frozen capture order places all 4,800 train rows first; direct prefix "
            "views contain the same BF16 values in the same order as advanced indexing, "
            "with unchanged casts, reductions, normalizations, Gram updates, and eigensolver"
        ),
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_5.CODE_PATH): sha256_file(root / amendment_5.CODE_PATH),
            str(PARENT_AMENDMENT_PATH): sha256_file(root / PARENT_AMENDMENT_PATH),
            str(BASE_FREEZE_PATH): sha256_file(root / BASE_FREEZE_PATH),
            str(CAPTURE_FREEZE_PATH): sha256_file(root / CAPTURE_FREEZE_PATH),
            **workspace_hashes,
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> None:
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 feature-memory amendment-6 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-6 input changed: {name}")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v6 --target freeze|geometry ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import geometry_v13 as module

    module._load_raw_deltas = _load_frozen_workspace
    module._dual_pca_scores = _dual_pca_scores_contiguous
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
