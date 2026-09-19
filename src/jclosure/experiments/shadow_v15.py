"""FP32 shadow accumulation with one final BF16 cast versus repeated writes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.actuation_v15 import (
    apply,
    components,
    load_context,
    transfer_metrics,
)
from jclosure.experiments.numerics_v14 import _panels
from jclosure.experiments.persistent_channels_v7 import _prefill
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")
INCREMENTS = (0.001, 0.01, 0.03)
STEPS = (1, 2, 4, 8, 16)


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_shadow_accumulation.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "shadow_accumulation", ["src/jclosure/experiments/shadow_v15.py", "results/v15/processed/actuator_transfer_v15.parquet"], {"role": "diagnostic_train", "increments": INCREMENTS, "step_counts": STEPS, "shadow_definition": "FP32 accumulate all increments, one BF16 cast immediately before model consumption; for one final write equals cast-after-add of total delta", "model_forward_executed": False})
    else:
        stage = json.loads(stage_path.read_text())
    bundle, dense_map, config, metadata, values = load_context(root)
    v8 = config["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    panel = _panels(root, int(base["config"]["diagnostic"]["anchors_per_family"]))
    rows = []
    for declaration in panel:
        base_id = str(declaration["base_trial_id"])
        task = values["tasks"][str(metadata[base_id]["prompt_id"])]
        clean = _prefill(bundle, task.prompt, measured_layers=measured, dense_map=dense_map, intervention_layer=int(v8["intervention"]["layer"]), candidate=None)
        cache = clean["cache"]
        for direction_index in base["config"]["diagnostic"]["direction_indices"]:
            direction = {name: values["directions"][name][int(direction_index)] for name in ("recurrent", "conv", "kv")}
            parts = components(cache, direction, recurrent, attention)
            for increment in INCREMENTS:
                for count in STEPS:
                    shadow = apply(cache, count * increment, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
                    repeated = cache
                    for _ in range(count):
                        repeated = apply(repeated, increment, direction, recurrent, attention, "native_fp32_add_bf16_writeback")
                    for mode, edited in (("fp32_shadow_cast_at_consumption", shadow), ("repeated_bf16_storage_write", repeated)):
                        for channel, layer, name, old, probe in parts:
                            actual = getattr(edited.layers[layer], name)
                            if channel in ("keys", "values"):
                                actual = actual[..., : old.shape[-2], :]
                            rows.append({"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "base_trial_id": base_id, "direction_index": int(direction_index), "mode": mode, "increment": increment, "steps": count, "total_requested_scale": increment * count, "channel": channel, "layer": layer, **transfer_metrics(old, probe, actual, increment * count)})
    path = root / OUT / "shadow_accumulation_v15.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    frame = pd.DataFrame(rows)
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "records": str(path), "records_sha256": sha256_file(path), "pooled": frame.groupby(["mode", "channel", "increment", "steps"]).agg(median_gain=("gain", "median"), median_cosine=("cosine", "median"), median_surviving_fraction=("surviving_fraction", "median")).reset_index().to_dict("records"), "interpretation": "shadow advantages concern accumulated requests before one BF16 consumption cast; not a high-precision forward semantics change"}
    write_json_atomic(root / OUT / "shadow_accumulation_v15.json", summary)
    print(json.dumps({"records": summary["records"], "sha256": summary["records_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
