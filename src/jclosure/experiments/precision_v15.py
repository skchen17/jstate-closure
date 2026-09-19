"""Separate addition/storage/consumption precision without altering model weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_v15 as operator
from jclosure.experiments.actuation_v15 import _mode_cache
from jclosure.experiments.numerics_v14 import _panels
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")
MODES = (
    "native_fp32_add_bf16_writeback",
    "cast_after_add",
    "native_bf16_add",
    "fp32_shadow_rec_conv",
    "fp32_rec_only",
    "fp32_conv_only",
    "fp32_kv_only",
)
EPSILONS = (0.1, 1.0)
DIRECTIONS = (0, 206)


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_precision.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "precision", ["src/jclosure/experiments/precision_v15.py", "results/v15/processed/actuator_transfer_v15.parquet"], {"role": "diagnostic_train", "modes": MODES, "epsilons": EPSILONS, "directions": DIRECTIONS, "fp32_shadow_is_altered_forward_semantics": True, "pretrained_weights_changed": False})
    else:
        stage = json.loads(stage_path.read_text())
    _, bundle, dense_map, metadata, values, measured, recurrent, attention = operator._setup(root)
    panel = _panels(root, int(base["config"]["diagnostic"]["anchors_per_family"]))
    rows = []
    for anchor_index, declaration in enumerate(panel):
        cache, kwargs, baseline, matrix, slices = operator._anchor(root, declaration, bundle, dense_map, metadata, values, measured)
        for direction_index in DIRECTIONS:
            row = {name: values["directions"][name][direction_index] for name in ("recurrent", "conv", "kv")}
            for epsilon in EPSILONS:
                for mode in MODES:
                    source = _mode_cache(cache, mode)
                    status = "SUPPORTED_DIAGNOSTIC" if mode.startswith("fp32_") else "SUPPORTED_CANONICAL_WRITEBACK"
                    error = None
                    try:
                        plus, minus, response, plus_cache, minus_cache = operator._effects(source, kwargs, baseline, row, recurrent, attention, epsilon, mode)
                        realized = operator._realization(source, plus_cache, row, recurrent, attention, epsilon)
                        del plus_cache, minus_cache
                    except (RuntimeError, ValueError, TypeError) as exc:
                        error = f"{type(exc).__name__}: {str(exc)[:300]}"
                        status = "UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT" if mode == "fp32_kv_only" else "ERROR"
                    for target in operator.TARGETS:
                        ideal = matrix[slices[target], direction_index]
                        common = {
                            "role": "diagnostic_train", "freeze_digest": stage["freeze_digest"],
                            "base_trial_id": str(declaration["base_trial_id"]),
                            "direction_index": direction_index, "mode": mode,
                            "epsilon": epsilon, "target": target, "status": status,
                            "error": error,
                            "model_weights_changed": False,
                            "construction_precision": "fp32" if mode != "native_bf16_add" else "bf16",
                            "storage_precision": "fp32_rec_conv" if mode == "fp32_shadow_rec_conv" else "fp32_rec" if mode == "fp32_rec_only" else "fp32_conv" if mode == "fp32_conv_only" else "fp32_kv" if mode == "fp32_kv_only" else "bf16",
                            "addition_precision": "bf16" if mode == "native_bf16_add" else "fp32",
                            "consumption_precision": "mixed_altered" if mode.startswith("fp32_") else "native_bf16",
                            "readout_precision": "fp32",
                        }
                        if error is None:
                            finite = response[target]
                            common.update(
                                finite_norm=float(np.linalg.norm(finite)),
                                cosine=operator.cosine(ideal, finite),
                                relative_l2=float(np.linalg.norm(finite - ideal) / max(float(np.linalg.norm(ideal)), 1e-20)),
                                effect_plus_norm=float(np.linalg.norm(plus[target])),
                                effect_minus_norm=float(np.linalg.norm(minus[target])),
                                realized_state_cosine=realized["cosine"],
                                realized_state_gain=realized["gain"],
                            )
                        rows.append(common)
                    del source
        write_json_atomic(root / OUT / "precision_progress_v15.json", {"completed_anchors": anchor_index + 1, "total_anchors": len(panel)})
    frame = pd.DataFrame(rows)
    path = root / OUT / "precision_modes_v15.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    pooled = frame[frame.status != "ERROR"].groupby(["mode", "epsilon", "target", "status"]).agg(
        count=("status", "size"), median_cosine=("cosine", "median"),
        median_relative_l2=("relative_l2", "median"),
        median_realized_state_cosine=("realized_state_cosine", "median"),
        median_realized_state_gain=("realized_state_gain", "median"),
    ).reset_index()
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "records": str(path), "records_sha256": sha256_file(path), "pooled": pooled.astype(object).where(pd.notnull(pooled), None).to_dict("records"), "fp32_kv_consumption": "UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT", "diagnostic_high_precision_changes_forward_semantics": True}
    write_json_atomic(root / OUT / "precision_modes_v15.json", summary)
    print(json.dumps(summary, sort_keys=True)[:3000])


if __name__ == "__main__":
    main()
