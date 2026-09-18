"""Frozen memory-order amendment for V13 feature construction.

The base implementation retains both a per-example padded KV list and its final
stack.  At the frozen 5,550-example bank size that transient representation can
exceed host RAM.  This wrapper preallocates the identical zero-padded tensor and
fills it once.  It changes neither values nor the downstream estimand.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from jclosure.config import load_config
from jclosure.experiments.bank_v13 import CAPTURE_FREEZE_PATH
from jclosure.protocol_v13 import (
    BASE_FREEZE_PATH,
    verify_base_freeze,
    verify_stage_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_1"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_1.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features.py")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_raw_deltas_memory_efficient(
    root: Path,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    from jclosure.experiments import geometry_v13 as geometry

    captures = geometry._capture_manifests(root)
    count = sum(int(capture["valid_count"]) for capture in captures)
    maximum_tokens = 256
    recurrent: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    ids: list[str] = []
    families: list[str] = []
    splits: list[str] = []
    offset = 0
    for capture in captures:
        for declaration in capture["state_shards"]:
            path = root / declaration["path"]
            if sha256_file(path) != declaration["sha256"]:
                raise RuntimeError(f"V13 state shard hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for row in payload["rows"]:
                clean, perturbed = row["clean"], row["perturbed"]
                recurrent_delta = perturbed["recurrent"] - clean["recurrent"]
                conv_delta = perturbed["conv"] - clean["conv"]
                current = torch.stack(
                    [
                        torch.stack(
                            [
                                perturbed["kv"][layer][name] - clean["kv"][layer][name]
                                for name in ("keys", "values")
                            ]
                        )
                        for layer in ("27", "31")
                    ]
                )
                if int(current.shape[-2]) > maximum_tokens:
                    raise RuntimeError(
                        f"V13 prompt cache length {current.shape[-2]} exceeds frozen "
                        f"KV analysis width {maximum_tokens}"
                    )
                if recurrent is None:
                    recurrent = torch.empty(
                        (count, *recurrent_delta.shape), dtype=torch.bfloat16
                    )
                    conv = torch.empty((count, *conv_delta.shape), dtype=torch.bfloat16)
                    kv = torch.zeros(
                        (
                            count,
                            *current.shape[:-2],
                            maximum_tokens,
                            current.shape[-1],
                        ),
                        dtype=torch.bfloat16,
                    )
                assert conv is not None and kv is not None
                recurrent[offset].copy_(recurrent_delta)
                conv[offset].copy_(conv_delta)
                kv[offset, ..., : current.shape[-2], :].copy_(current)
                ids.append(str(row["base_trial_id"]))
                families.append(str(row["family"]))
                splits.append(str(row["split"]))
                offset += 1
            del payload
    if recurrent is None or conv is None or kv is None or offset != count:
        raise RuntimeError(f"V13 preallocated load count mismatch: {offset} != {count}")
    return {"recurrent": recurrent, "conv": conv, "kv": kv}, ids, families, splits


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    capture = verify_stage_freeze(root, config, CAPTURE_FREEZE_PATH)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "remove duplicate host-memory materialization before any V13 feature output",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_capture_freeze_digest": capture["freeze_digest"],
        "failure_stage": "feature loading before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "numerical_equivalence": (
            "zero-initialize one [N,...,256,D] BF16 tensor and copy each exact raw "
            "delta once instead of constructing the same per-row padded tensors and "
            "then stacking them"
        ),
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment input changed: {name}")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features --target freeze|geometry ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import geometry_v13 as module

    module._load_raw_deltas = _load_raw_deltas_memory_efficient
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
