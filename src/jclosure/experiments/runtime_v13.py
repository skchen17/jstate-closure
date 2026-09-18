"""Frozen dual-GPU runtime placement for V13 JVP and finite writeback stages."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jclosure.config import load_config
from jclosure.experiments.runtime_v12_amendment import _load_encoder_memory_efficient
from jclosure.model import load_model_bundle as _original_load_model_bundle
from jclosure.protocol_v13 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path("artifacts/causal_geometry_v13_runtime_dual_gpu.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v13.py")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _placement() -> dict[str, int]:
    mapping = {
        "model.embed_tokens": 0,
        "lm_head": 0,
        "model.norm": 0,
        "model.rotary_emb": 0,
    }
    mapping.update(
        {f"model.layers.{index}": 1 if index < 16 else 0 for index in range(32)}
    )
    return mapping


def _load_model(config: dict[str, Any]) -> Any:
    amended = copy.deepcopy(config)
    amended["model"]["device_map"] = _placement()
    amended["model"].pop("max_memory", None)
    amended["model"].pop("offload_folder", None)
    return _original_load_model_bundle(amended)


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    base = verify_base_freeze(root, config)
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": "causal_path_geometry_v13_runtime_dual_gpu",
        "purpose": "place target/readout layers and frozen cache edits on cuda:0",
        "parent_base_freeze_digest": base["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "model_weights_changed": False,
        "model_dtype_changed": False,
        "device_map": _placement(),
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(BASE_FREEZE_PATH): sha256_file(root / BASE_FREEZE_PATH),
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / AMENDMENT_PATH).read_text())
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 runtime amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 runtime amendment input changed: {name}")
    return value


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit("usage: runtime_v13 --target freeze|geometry ...") from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "geometry":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import geometry_v13 as module

    module._load_encoder = _load_encoder_memory_efficient
    module.load_model_bundle = _load_model
    module.PROTOCOL_V13 = "causal_path_geometry_v13_runtime_dual_gpu"
    module.main()


if __name__ == "__main__":
    main()
