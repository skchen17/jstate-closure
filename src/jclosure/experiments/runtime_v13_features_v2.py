"""Arena-release amendment layered on the V13 preallocated feature loader."""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from jclosure.config import load_config
from jclosure.experiments import runtime_v13_features as amendment_1
from jclosure.experiments.bank_v13 import CAPTURE_FREEZE_PATH
from jclosure.protocol_v13 import (
    BASE_FREEZE_PATH,
    verify_base_freeze,
    verify_stage_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "causal_path_geometry_v13_feature_memory_amendment_2"
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v13_feature_memory_amendment_2.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_features_v2.py")
PARENT_AMENDMENT_PATH = amendment_1.AMENDMENT_PATH


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _malloc_trim() -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)


class _TrimmingDict(dict[Any, Any]):
    """Release tensor references before returning their arenas to the OS."""

    def __del__(self) -> None:
        self.clear()
        _malloc_trim()


def _load_raw_deltas_with_arena_release(
    root: Path,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str]]:
    original = torch.load

    def trimming_load(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        if isinstance(value, dict) and "rows" in value:
            return _TrimmingDict(value)
        return value

    torch.load = trimming_load  # type: ignore[assignment]
    try:
        return amendment_1._load_raw_deltas_memory_efficient(root)
    finally:
        torch.load = original  # type: ignore[assignment]
        _malloc_trim()


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    capture = verify_stage_freeze(root, config, CAPTURE_FREEZE_PATH)
    amendment_1._verify(root)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "release freed torch.load arenas during preallocated V13 feature loading",
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_capture_freeze_digest": capture["freeze_digest"],
        "parent_feature_memory_amendment_sha256": sha256_file(
            root / PARENT_AMENDMENT_PATH
        ),
        "failure_stage": "feature loading before the first V13 feature artifact",
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "numerical_equivalence": (
            "clear each already-consumed shard mapping and call glibc malloc_trim; "
            "all copied BF16 tensors and their order are unchanged"
        ),
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(amendment_1.CODE_PATH): sha256_file(root / amendment_1.CODE_PATH),
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
        raise RuntimeError("V13 feature-memory amendment-2 digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 feature-memory amendment-2 input changed: {name}")


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v13_features_v2 --target freeze|geometry ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import geometry_v13 as module

    module._load_raw_deltas = _load_raw_deltas_with_arena_release
    module.PROTOCOL_V13 = PROTOCOL_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
