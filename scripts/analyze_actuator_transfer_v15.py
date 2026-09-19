"""Preserve raw V15 records and issue a frozen correction to its empty summary.

The initial raw runner used ``frame.mode`` (the DataFrame method) where the
``mode`` column was intended. No measured rows are changed by this amendment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    input_path = root / "results/v15/processed/actuator_transfer_v15.parquet"
    original = root / "results/v15/processed/actuator_transfer_v15.json"
    frozen_path = root / "artifacts/quantization_aware_actuation_v15_transfer_summary_correction.freeze.json"
    if not frozen_path.exists():
        stage = stage_freeze(
            root,
            "transfer_summary_correction",
            ["scripts/analyze_actuator_transfer_v15.py", str(input_path.relative_to(root)), str(original.relative_to(root))],
            {"reason": "original summary selected DataFrame.mode method instead of mode column", "original_records_preserved": True, "measured_estimand_unchanged": True},
        )
    else:
        stage = json.loads(frozen_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(input_path)
    absolute = frame[(frame["scale_kind"] == "absolute") & (frame["mode"] == "native_fp32_add_bf16_writeback")]
    pooled = absolute.groupby(["channel", "scale"]).agg(
        median_gain=("gain", "median"), median_cosine=("cosine", "median"),
        median_zero_fraction=("zero_fraction", "median"),
        median_surviving_fraction=("surviving_fraction", "median"),
        median_below_one_ulp=("below_one_ulp_fraction", "median"),
        median_below_half_ulp=("below_half_ulp_fraction", "median"), count=("gain", "size"),
    ).reset_index()
    section = base["config"]["diagnostic"]
    deadzones = {}
    for channel, group in pooled.groupby("channel"):
        group = group.sort_values("scale")
        dead = group[group.median_zero_fraction >= section["deadzone_zero_fraction_threshold"]]
        change = group[group.median_surviving_fraction > 0]
        reliable = group[
            (group.median_gain >= section["reliable_state_gain_min"])
            & (group.median_gain <= section["reliable_state_gain_max"])
            & (group.median_cosine >= section["reliable_state_cosine_min"])
        ]
        deadzones[channel] = {
            "ACTUATOR_DEADZONE_MAX_TESTED_SCALE": float(dead.scale.max()) if len(dead) else None,
            "MIN_STATE_CHANGE_SCALE": float(change.scale.min()) if len(change) else None,
            "MIN_RELIABLE_STATE_SCALE": float(reliable.scale.min()) if len(reliable) else None,
            "REALIZED_REQUESTED_GAIN_AT_SCALE_1": float(group.loc[group.scale == 1.0, "median_gain"].iloc[0]),
            "SURVIVING_FRACTION_AT_SCALE_1": float(group.loc[group.scale == 1.0, "median_surviving_fraction"].iloc[0]),
        }
    modes = frame[frame["scale_kind"] == "absolute"].groupby(["mode", "channel", "scale"]).agg(
        median_gain=("gain", "median"), median_cosine=("cosine", "median"),
        median_surviving_fraction=("surviving_fraction", "median"),
    ).reset_index()
    layer = absolute.groupby(["channel", "layer", "scale"]).agg(
        median_gain=("gain", "median"), median_cosine=("cosine", "median"),
        median_zero_fraction=("zero_fraction", "median"), count=("gain", "size"),
    ).reset_index()
    output = {
        "protocol_version": base["protocol_version"], "freeze_digest": stage["freeze_digest"],
        "role": "diagnostic_train", "original_summary_sha256": sha256_file(original),
        "original_summary_error": "empty deadzones/curves caused by DataFrame.mode method collision",
        "raw_records": str(input_path.relative_to(root)), "raw_records_sha256": sha256_file(input_path),
        "deadzones": deadzones, "curves": pooled.to_dict("records"),
        "precision_modes": modes.to_dict("records"), "layer_curves": layer.to_dict("records"),
        "fp32_kv_consumption": "UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT",
        "fp32_shadow_is_diagnostic_not_canonical": True,
    }
    path = root / "results/v15/processed/actuator_transfer_corrected_v15.json"
    write_json_atomic(path, output)
    print(json.dumps(deadzones, sort_keys=True))


if __name__ == "__main__":
    main()
