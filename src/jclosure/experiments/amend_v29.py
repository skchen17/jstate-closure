"""Prospective V29 all-layer interface amendment after diagnostic calibration."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/amend_v29.py"
OUT = Path("results/v29/processed")


def run(root):
    verify_stage(root, "forks_calibration")
    verify_stage(root, "execution_plan")
    cfg = verify(root)["config"]
    g = pd.read_parquet(root / OUT / "geometry_calibration_v29.parquet")
    t = pd.read_parquet(root / OUT / "transfer_calibration_v29.parquet")
    old_layers = {c: sorted(int(v) for v in x.layer.unique()) for c, x in g.groupby("channel")}
    full = t[t.condition == "REC+Conv+KV"]
    payload = {
        "amendment_kind": "architecture_interface_coverage_not_hypothesis_or_threshold_change",
        "diagnostic_calibration_states": 25,
        "diagnostic_REC_Conv_KV_layer_sets": old_layers,
        "diagnostic_named_full_median_relative_l2": float(full.relative_l2_to_donor.median()),
        "diagnostic_named_full_was_not_full_native_cache": True,
        "reason": "V28 selected posterior layers sufficed for perturbations confined to those layers; token forks change earlier native REC/Conv/KV fields too.",
        "corrected_channel_semantics": "all initialized native cache layers; REC/Conv full tensor, KV newly appended slot with identical prior slots",
        "corrected_implementation": "src/jclosure/experiments/forks_formal_v29.py",
        "corrected_full_ceiling_required": "bitwise donor equality of every native cache field and future relative L2 <= 1e-6",
        "formal_development_saved_before_amendment": False,
        "transient_development_states_processed_before_interrupt": 53,
        "transient_development_metrics_saved": False,
        "transient_run_status": "interrupted_invalid_interface_diagnostic; no formal inference",
        "validation_responses_observed_before_amendment": False,
        "thresholds_unchanged": True,
        "role_assignments_unchanged": True,
        "fork_tokens_unchanged": True,
        "fixed_finalist_unchanged": "REC+Conv",
        "historical_final_opened": False,
        "H3_AUTHORIZED": cfg["authorization"]["H3_AUTHORIZED"],
    }
    path = root / OUT / "interface_amendment_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "interface_amendment", [SOURCE, "src/jclosure/experiments/forks_formal_v29.py", str(path.relative_to(root)), "artifacts/natural_write_content_v29_forks_calibration.freeze.json", "artifacts/natural_write_content_v29_execution_plan.freeze.json"], payload)
    return {"freeze_digest": fr["freeze_digest"], **payload}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
