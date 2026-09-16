"""Frozen two-GPU placement for exact V12 JVPs with targets on cuda:0."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jclosure.config import load_config
from jclosure.experiments.runtime_v12_amendment import _load_encoder_memory_efficient
from jclosure.experiments.runtime_v12_amendment_5 import (
    AMENDMENT_PATH as AMENDMENT_5_PATH,
)
from jclosure.model import load_model_bundle as _original_load_model_bundle
from jclosure.protocol_v12 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "local_causal_geometry_v12_runtime_amendment_6_jvp_dual_gpu"
SCHEMA_AMENDMENT = 21
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v12_runtime_amendment_6_jvp_dual_gpu.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v12_amendment_6.py")


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


def _load_model_jvp(config: dict[str, Any]) -> Any:
    amended = copy.deepcopy(config)
    amended["model"]["device_map"] = _placement()
    amended["model"].pop("max_memory", None)
    amended["model"].pop("offload_folder", None)
    return _original_load_model_bundle(amended)


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v12.yaml")
    base = verify_base_freeze(root, config)
    parents = [BASE_FREEZE_PATH, AMENDMENT_5_PATH]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_AMENDMENT,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "place JVP target layer 30 and frozen input channels on cuda:0",
        "parent_base_freeze_digest": base["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "model_weights_changed": False,
        "model_dtype_changed": False,
        "algorithm_changed": False,
        "device_map": _placement(),
        "cpu_offload": False,
        "probe": {
            "prefill_finite": True,
            "target_layer_30_device": "cuda:0",
            "first_parameter_device": "cuda:0",
            "records_emitted_by_probe": False,
        },
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            **{str(path): sha256_file(root / path) for path in parents},
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v12 JVP dual-GPU amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"v12 JVP dual-GPU amendment input changed: {name}")
    return value


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v12_amendment_6 --target freeze|jvp ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    if target != "jvp":
        raise SystemExit(f"unknown target: {target}")
    _verify(Path.cwd())
    from jclosure.experiments import jvp_v12 as module

    module._load_encoder = _load_encoder_memory_efficient
    module.load_model_bundle = _load_model_jvp
    module.PROTOCOL_V12 = PROTOCOL_AMENDMENT
    module.SCHEMA_VERSION_V12 = SCHEMA_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
