"""Frozen bounded hybrid CPU/GPU fallback for V12 model execution."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jclosure.config import load_config
from jclosure.experiments.runtime_v12_amendment import _load_encoder_memory_efficient
from jclosure.experiments.runtime_v12_amendment_2 import (
    AMENDMENT_PATH as AMENDMENT_2_PATH,
)
from jclosure.model import load_model_bundle as _original_load_model_bundle
from jclosure.protocol_v12 import BASE_FREEZE_PATH, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "local_causal_geometry_v12_runtime_amendment_3_hybrid"
SCHEMA_AMENDMENT = 18
AMENDMENT_PATH = Path(
    "artifacts/causal_geometry_v12_runtime_amendment_3_hybrid.freeze.json"
)
CODE_PATH = Path("src/jclosure/experiments/runtime_v12_amendment_3.py")
OFFLOAD_PATH = Path("results/v12/raw/hybrid_offload")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_model_hybrid(config: dict[str, Any]) -> Any:
    amended = copy.deepcopy(config)
    amended["model"]["device_map"] = "auto"
    amended["model"]["max_memory"] = {0: "6GiB", "cpu": "64GiB"}
    amended["model"]["offload_folder"] = str(Path.cwd() / OFFLOAD_PATH)
    return _original_load_model_bundle(amended)


def _freeze() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v12.yaml")
    base = verify_base_freeze(root, config)
    parents = [BASE_FREEZE_PATH, AMENDMENT_2_PATH]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_AMENDMENT,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "bounded hybrid execution after CPU throughput and shared-GPU probes",
        "parent_base_freeze_digest": base["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "model_weights_changed": False,
        "model_dtype_changed": False,
        "algorithm_changed": False,
        "execution_device": "accelerate auto-map with physical GPU1 plus CPU",
        "visible_gpu_memory_limit": "6GiB",
        "cpu_memory_limit": "64GiB",
        "probe": {
            "single_prefill_succeeded": True,
            "cpu_prefill_seconds": 6.61,
            "hybrid_prefill_seconds": 2.14,
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
        raise RuntimeError("v12 hybrid amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"v12 hybrid amendment input changed: {name}")
    return value


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v12_amendment_3 --target freeze|causal|jvp ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    _verify(Path.cwd())
    if target == "causal":
        from jclosure.experiments import causal_v12 as module
    elif target == "jvp":
        from jclosure.experiments import jvp_v12 as module
    else:
        raise SystemExit(f"unknown target: {target}")
    module._load_encoder = _load_encoder_memory_efficient
    module.load_model_bundle = _load_model_hybrid
    module.PROTOCOL_V12 = PROTOCOL_AMENDMENT
    module.SCHEMA_VERSION_V12 = SCHEMA_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
