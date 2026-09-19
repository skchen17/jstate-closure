"""Test architecture-compatible FP32 REC/conv state with BF16 KV cache."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import numerics_v14 as original
from jclosure.protocol_v14 import freeze_stage, verify_base

EPSILONS = [0.001, 0.01, 0.1, 1.0]
OUT = Path("results/v14/processed/numerical_partial_fp32_extension_2")


def _partial_fp32(cache: Any) -> Any:
    output = clone_hybrid_cache(cache)
    for layer in output.layers:
        for name in ("recurrent_states", "conv_states"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, name, value.float())
    return output


def main() -> None:
    root = Path.cwd()
    stage = (
        root
        / "artifacts/finite_causal_control_v14_numerical_partial_fp32_extension_2.freeze.json"
    )
    if not stage.exists():
        amendment = freeze_stage(
            root,
            "numerical_partial_fp32_extension_2",
            [
                "src/jclosure/experiments/numerics_v14_extension_2.py",
                "src/jclosure/experiments/numerics_v14.py",
                "results/v14/processed/jvp_finite_writeback_audit_v14.parquet",
            ],
            {
                "reason": "all-FP32 cache rejected by BF16 attention query; isolate REC/conv precision",
                "precision_mode": "fp32_recurrent_and_conv_bf16_kv",
                "epsilons": EPSILONS,
                "model_weights_changed": False,
            },
        )
    else:
        amendment = json.loads(stage.read_text(encoding="utf-8"))
    base = copy.deepcopy(verify_base(root))
    base["config"]["numerical_audit"]["epsilons"] = EPSILONS
    base["config"]["numerical_audit"]["precision_modes"] = ["fp32_cache_if_supported"]
    base["freeze_digest"] = amendment["freeze_digest"]
    original.verify_base = lambda _: base
    original._fp32_cache = _partial_fp32
    original.OUT = OUT
    print(json.dumps(original.audit(root), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
