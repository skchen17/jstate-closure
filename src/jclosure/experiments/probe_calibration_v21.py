"""Train-only BF16 calibration for 64 frozen paired JVP/finite probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.response_operator_v20 import _action_ok
from jclosure.protocol_v21 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/probe_calibration_v21.py"
ROWS = bank.OUT / "probe_calibration_v21.parquet"
SUMMARY = bank.OUT / "probe_calibration_v21.json"


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    config = verify(root)["config"]
    ids = []
    for family in sorted({x["family"] for x in roles["jvp_development"]}):
        ids.extend([x["base_trial_id"] for x in roles["jvp_development"] if x["family"] == family][:2])
    if len(ids) != 10:
        raise RuntimeError("V21 action calibration must use 2 development states per family")
    return stage_freeze(root, "probe_calibration_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json"],
                        {"base_ids": ids, "probe_sha256": roles["jvp_probe_sha256"],
                         "alpha_candidates": config["state_panels"]["jvp_alpha_candidates"],
                         "V20_actuator_thresholds": json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())["action_reliability"],
                         "selection_rule": "smallest_alpha_with_reliable_fraction_at_least_0.95; if_none_mark_unreliable_and_keep_in_audit",
                         "V20_selected_action_alpha_not_changed": True,
                         "V21_JVP_or_finite_responses_observed": 0,
                         "final_six_action_responses_opened": False})


def calibrate(root: Path) -> dict:
    design = verify_stage(root, "probe_calibration_design")
    roles = verify_stage(root, "roles")
    prompts = bank_prompt_index(root)
    if not callable(prompts):
        raise RuntimeError("V21 prompt source unavailable")
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    directions = values["directions"]
    by_id = {x["base_trial_id"]: x for x in roles["jvp_development"]}
    rows = []
    for number, base_id in enumerate(design["base_ids"], 1):
        item = by_id[base_id]
        prompt = prompts(base_id)
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        p0 = clean["cache"]
        for index in roles["jvp_probe_direction_indices"]:
            for alpha in design["alpha_candidates"]:
                action = v19._action_row(directions, int(index), float(alpha), 1)
                edited = v19.apply(p0, 1.0, action, rec, att, "native_fp32_add_bf16_writeback")
                actual = _readback(p0, edited, action, rec, att)
                rows.append({"base_trial_id": base_id, "family": item["family"],
                             "direction_index": int(index), "alpha": float(alpha),
                             "reliable": bool(_action_ok(actual, design["V20_actuator_thresholds"])),
                             "requested_norm": actual["requested_state_norm"],
                             "realized_norm": actual["realized_state_norm"],
                             "realized_cosine": actual["realized_state_cosine"],
                             "realized_gain": actual["realized_state_gain"],
                             "channel_survival": json.dumps(actual["channel_survival"])})
        print(f"V21 probe calibration {number}/{len(design['base_ids'])}", flush=True)
    frame = pd.DataFrame(rows)
    target = root / ROWS
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    fixed = {int(x["direction_index"]): float(x["base_alpha"]) for x in roles["train_actions"] + roles["validation_actions"]}
    selected = {}
    stats = []
    for index in roles["jvp_probe_direction_indices"]:
        group = frame[frame.direction_index == index]
        fractions = {str(alpha): float(group[group.alpha == alpha].reliable.mean()) for alpha in design["alpha_candidates"]}
        allowed = [float(alpha) for alpha in design["alpha_candidates"] if fractions[str(alpha)] >= 0.95]
        alpha = fixed.get(int(index), min(allowed) if allowed else None)
        reliable_at_alpha = fractions.get(str(alpha)) if alpha is not None else None
        selected[str(index)] = alpha
        stats.append({"direction_index": int(index), "selected_alpha": alpha,
                      "selected_reliable_fraction": reliable_at_alpha,
                      "candidate_reliable_fractions": fractions,
                      "V20_fixed_alpha": int(index) in fixed})
    result = {"design_digest": design["freeze_digest"], "probe_sha256": roles["jvp_probe_sha256"],
              "base_states": len(design["base_ids"]), "calibration_rows": len(frame),
              "selected_alpha_by_direction": selected, "by_probe": stats,
              "reliable_selected_probe_count": sum(x["selected_reliable_fraction"] is not None and x["selected_reliable_fraction"] >= 0.95 for x in stats),
              "calibration_path": str(ROWS), "calibration_sha256": sha256_file(target),
              "final_six_action_responses_opened": False}
    write_json_atomic(root / SUMMARY, result)
    freeze = stage_freeze(root, "probe_scales",
                          [SOURCE, str(ROWS), str(SUMMARY),
                           "artifacts/action_coordinate_geometry_v21_probe_calibration_design.freeze.json"],
                          {"selected_alpha_by_direction": selected,
                           "reliable_selected_probe_count": result["reliable_selected_probe_count"],
                           "calibration_sha256": result["calibration_sha256"],
                           "calibration_summary_sha256": sha256_file(root / SUMMARY),
                           "V21_JVP_or_finite_validation_responses_observed": 0,
                           "final_six_action_responses_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "reliable_selected_probe_count": result["reliable_selected_probe_count"]}


def bank_prompt_index(root: Path):
    from jclosure.experiments.operator_bank_v20 import prompt_index
    lookup = prompt_index(root)
    return lambda base_id: str(lookup[base_id]["prompt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate"))
    args = parser.parse_args()
    result = {"prepare": prepare, "calibrate": calibrate}[args.stage](Path.cwd())
    print(json.dumps(result, indent=2))
