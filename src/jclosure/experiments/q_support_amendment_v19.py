"""Append-only correction of channel support after inspecting frozen q tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.protocol_v19 import stage_freeze, verify, verify_stage

SOURCE = "src/jclosure/experiments/q_support_amendment_v19.py"
DESIGN = {"name": "rec_conv_causal1_amended", "direction_index": 1, "channel": "rec_conv"}


def run(root: Path) -> dict:
    split = verify_stage(root, "splits")
    previous = verify_stage(root, "q_amendment_1")
    cfg = verify(root)["config"]
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = bank._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    rows = []
    for item in split["calibration"]:
        old = bank._state_metadata(root, item)
        clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        tokens = [int(x) for x in old["teacher_tokens_h8_or_h1"][:1]]
        _, y00 = bank._trajectory(bundle, dense, clean["cache"], None, tokens, 1, clean["prompt_length"],
                                  jids, lids, ws_layers, ws_count, max(measured))
        action = split["actions"][0]
        action_row = bank._action_row(values["directions"], int(action["direction_index"]),
                                      float(cfg["actions"]["alpha"]), 1)
        p0a = apply(clean["cache"], 1.0, action_row, rec, att, "native_fp32_add_bf16_writeback")
        _, y01 = bank._trajectory(bundle, dense, p0a, None, tokens, 1, clean["prompt_length"],
                                  jids, lids, ws_layers, ws_count, max(measured))
        r = stack({key: y01[1][key] - y00[1][key] for key in TARGETS}, scales)
        for alpha in cfg["q"]["alpha_candidates"]:
            qrow = bank._q_row(values["directions"], DESIGN, float(alpha))
            pq = apply(clean["cache"], 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
            metrics = _readback(clean["cache"], pq, qrow, rec, att)
            _, y10 = bank._trajectory(bundle, dense, pq, None, tokens, 1, clean["prompt_length"],
                                      jids, lids, ws_layers, ws_count, max(measured))
            n = stack({key: y10[1][key] - y00[1][key] for key in TARGETS}, scales)
            rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                         "q_name": DESIGN["name"], "q_channel": DESIGN["channel"], "alpha": float(alpha),
                         "natural_to_action_ratio": float(np.linalg.norm(n) / max(np.linalg.norm(r), 1e-8)),
                         "natural_stack_norm": float(np.linalg.norm(n)),
                         "clean_action_stack_norm": float(np.linalg.norm(r)),
                         **metrics, "reliable": bank._reliable(metrics, cfg["q"]),
                         "finite_endpoint": bool(np.isfinite(n).all())})
    frame = pd.DataFrame(rows)
    path = root / bank.OUT / "q_support_calibration_v19.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    choices = []
    for alpha in cfg["q"]["alpha_candidates"]:
        sub = frame[frame.alpha == alpha]
        choices.append({"alpha": float(alpha), "reliable_fraction": float((sub.reliable & sub.finite_endpoint).mean()),
                        "median_natural_to_action_ratio": float(sub.natural_to_action_ratio.median()),
                        "median_raw_state_delta_norm": float(sub.realized_state_norm.median())})
    valid = [x for x in choices if x["reliable_fraction"] >= 0.8
             and x["median_raw_state_delta_norm"] >= cfg["q"]["min_raw_state_delta_norm"]]
    selected = max(valid, key=lambda x: x["median_natural_to_action_ratio"]) if valid else max(choices, key=lambda x: x["reliable_fraction"])
    corrected = []
    for q in previous["q"]:
        if q["name"] == "joint_arch4":
            continue  # exactly duplicates the isolated REC-4 correction
        if q["name"] == "rec_conv_arch24":
            q = {**q, "name": "conv_arch24_corrected", "channel": "conv"}
        corrected.append(q)
    corrected.append({**DESIGN, **selected,
                      "calibration_status": "RELIABLE" if valid else "NOT_RELIABLY_ACTUATABLE",
                      "natural_active_train_pilot": selected["median_natural_to_action_ratio"] >= cfg["q"]["natural_active_ratio_min"]})
    return stage_freeze(root, "q_amendment_2",
                        [SOURCE, str(path.relative_to(root)), "artifacts/counterfactual_workspace_v19_q_amendment_1.freeze.json",
                         "scripts/audit_selected_q_v19.py"],
                        {"why": "Frozen tensor support audit found direction 24 is Conv-only, not REC+Conv, and architecture direction 4 is REC-only, duplicating the isolated REC proposal. Relabel 24 as Conv, remove duplicate 4 joint proposal, and add train-calibrated REC+Conv direction 1.",
                         "when": "before any V19 factorial/validation response",
                         "responses_already_observed": {"initial_q_calibration": 480, "q_amendment_1_calibration": 120,
                                                        "factorial_train": 0, "factorial_validation": 0},
                         "prior_amendment_digest": previous["freeze_digest"],
                         "q": corrected, "q_calibration_rows": len(frame),
                         "q_calibration_state_count": int(frame.base_trial_id.nunique()),
                         "probe_actions": split["actions"], "action_matching_threshold": cfg["actions"],
                         "thresholds": cfg["targets"], "validation_responses_observed": 0,
                         "independent_final_opened": False})


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
