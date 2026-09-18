"""Serial-conversion amendment for the bounded V13 feature runtime."""

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
from jclosure.experiments import runtime_v13_features_v10 as amendment_10
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_11"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_11.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v11.py")
PARENT_AMENDMENT_PATH = amendment_10.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dual_pca_scores_serial_conversion(
    blocks: list[torch.Tensor], train: Any, *, device: str
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Run amendment-10 with serial elementwise BF16 conversion copies."""

    original_copy = torch.Tensor.copy_

    def serial_conversion_copy(
        output: torch.Tensor,
        source: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            output.ndim == 2
            and source.ndim == 2
            and source.dtype == torch.bfloat16
            and output.dtype in {torch.float32, torch.float64}
            and output.device.type == "cpu"
            and source.device.type == "cpu"
        ):
            previous = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                return original_copy(output, source, *args, **kwargs)
            finally:
                torch.set_num_threads(previous)
        return original_copy(output, source, *args, **kwargs)

    torch.Tensor.copy_ = serial_conversion_copy  # type: ignore[method-assign]
    try:
        return amendment_10._dual_pca_scores_identity_copy(blocks, train, device=device)
    finally:
        torch.Tensor.copy_ = original_copy  # type: ignore[method-assign]


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    amendment_10._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bound allocator retention during V13 BF16 conversion copies",
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
            "allocation schedule only: execute the elementwise BF16-to-float64/"
            "float32 copies with one CPU worker, then restore the frozen eight-worker "
            "setting before every sum, square, Gram update, and eigensolver; values, "
            "casts, reduction schedule, chunks, and operation order are unchanged"
        ),
        "observed_parent_failure": {
            "parent_run_suffix": "features-memory-amendment-10-gpu1",
            "feature_artifact_written": False,
            "cause": (
                "parallel conversion copies retained bounded but excessive per-worker "
                "temporary blocks; the run was stopped before any estimand output"
            ),
        },
        "runtime": {
            "conversion_copy_threads": 1,
            "reduction_and_linear_algebra_threads": amendment_7.INTRAOP_THREADS,
            "torch_interop_threads": amendment_7.INTEROP_THREADS,
            "MALLOC_ARENA_MAX": amendment_7.MALLOC_ARENA_MAX,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_10.CODE_PATH): sha256_file(root / amendment_10.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-11 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-11 input changed: {name}")
    if os.environ.get("MALLOC_ARENA_MAX") != amendment_7.MALLOC_ARENA_MAX:
        raise RuntimeError("run V13 feature runtime with MALLOC_ARENA_MAX=2")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v11 --target freeze|geometry ..."
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
    module._dual_pca_scores = _dual_pca_scores_serial_conversion
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
