"""Historical train-calibrated shared actions and V20 train-only actuator checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as bank
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file

SOURCE = "src/jclosure/experiments/actions_v20.py"
CALIBRATION = bank.OUT / "action_calibration_v20.parquet"


def _digest_rows(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def amend_selection(root: Path) -> dict:
    cfg = verify(root)["config"]["action_bank"]
    if cfg["min_train_reliable_fraction"] != 1.0 or 28 not in cfg["selected_coordinates"]:
        raise RuntimeError("V20 action-selection amendment precondition changed")
    old = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    rows = old[(old.role == "train") & (old.design == "single") & (old.coordinate_index == 28)]
    if len(rows) != 200 or int((rows.reliability_status == "RELIABLE").sum()) != 198:
        raise RuntimeError("V20 V16 random coordinate 28 calibration differs from amendment evidence")
    independent_observed = len(list((bank.SCRATCH / "independent_v19_confirmation").glob("factorial_*.parquet")))
    return stage_freeze(root, "action_selection_amendment_1",
                        [SOURCE, "results/v16/processed/finite_action_bank_v16.parquet",
                         "artifacts/compact_causal_response_operator_v20_splits.freeze.json"],
                        {"why": "The intended third random control (coordinate 28) was 198/200 reliable, not 200/200, in frozen V16 train calibration; use a pre-operator 0.99 minimum and retain row-level actuator failures in denominators.",
                         "old_train_reliable_fraction_min": 1.0, "new_train_reliable_fraction_min": 0.99,
                         "coordinate_28_V16_reliable": 198, "coordinate_28_V16_total": 200,
                         "responses_already_observed": {"independent_V19_confirmation_states": independent_observed,
                                                        "V20_operator_states": 0, "V20_operator_action_rows": 0},
                         "state_splits_changed": False, "action_directions_changed": False})


def prepare(root: Path) -> dict:
    cfg = verify(root)["config"]["action_bank"]
    amendment = verify_stage(root, "action_selection_amendment_1")
    split = verify_stage(root, "splits")
    v16 = json.loads((root / cfg["source"]).read_text())
    frame = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
    frame = frame[(frame.role == "train") & (frame.design == "single")]
    selected = cfg["selected_coordinates"]
    partitions = {name: cfg[f"{name}_coordinates"] for name in ("train", "validation", "final_heldout")}
    if len(selected) != len(set(selected)) or set().union(*map(set, partitions.values())) != set(selected):
        raise RuntimeError("V20 action partition incomplete or duplicate")
    if sum(len(part) for part in partitions.values()) != len(selected):
        raise RuntimeError("V20 action partition overlaps")
    actions = []
    for coord in selected:
        group = frame[frame.coordinate_index == coord]
        alpha = float(cfg["random_alpha"] if coord in cfg["random_alpha_one_coordinates"] else cfg["default_alpha"])
        reliable = float((group.reliability_status == "RELIABLE").mean())
        if (len(group) != 200 or reliable < amendment["new_train_reliable_fraction_min"]
                or float(group.calibration_alpha.median()) != alpha):
            raise RuntimeError(f"V20 action {coord} fails frozen V16 train calibration")
        actions.append({"coordinate_index": coord, "direction_index": int(v16["direction_indices"][coord]),
                        "proposal_family": v16["direction_families"][coord], "base_alpha": alpha,
                        "V16_train_reliable_fraction": reliable,
                        "V16_train_median_j_effect_norm": float(group.j_effect_norm.median()),
                        "V16_train_median_realized_cosine": float(group.realized_state_cosine.median()),
                        "V16_train_median_gain": float(group.realized_state_gain.median())})
    by_coord = {row["coordinate_index"]: row for row in actions}
    partition_actions = {name: [by_coord[i] for i in coords] for name, coords in partitions.items()}
    excluded = []
    for coord in range(len(v16["direction_indices"])):
        if coord in selected:
            continue
        group = frame[frame.coordinate_index == coord]
        excluded.append({"coordinate_index": coord, "V16_reliable_fraction": float((group.reliability_status == "RELIABLE").mean()),
                         "V16_calibration_alpha": float(group.calibration_alpha.median()),
                         "reason": "outside_reliable_base_alpha_or_balanced_24_direction_panel"})
    return stage_freeze(root, "actions",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_action_selection_amendment_1.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
                         cfg["source"], "results/v16/processed/finite_action_bank_v16.parquet",
                         "artifacts/causal/v13/probe_directions_v13.pt"],
                        {"actions": actions, "partitions": partition_actions,
                         "action_hashes": {name: _digest_rows(rows) for name, rows in partition_actions.items()},
                         "all_action_hash": _digest_rows(actions),
                         "base_direction_count": len(actions),
                         "measured_development_direction_count": len(partition_actions["train"]) + len(partition_actions["validation"]),
                         "signs_measured": cfg["signs_measured"],
                         "decoder_train_sign": cfg["decoder_train_sign"],
                         "decoder_train_scale": cfg["decoder_train_scale"],
                         "diagnostic_scale_multiplier": cfg["diagnostic_scale_multiplier"],
                         "candidate_composition_pairs": verify(root)["config"]["operator_analysis"]["composition_pairs"],
                         "reliability_thresholds": {"cosine_min": cfg["reliability_min_cosine"],
                                                    "gain_min": cfg["reliability_gain_min"],
                                                    "gain_max": cfg["reliability_gain_max"]},
                         "excluded_candidates": excluded,
                         "selection_source": "V16_train_only_finite_action_BF16_calibration",
                         "heldout_action_labels_used": False,
                         "final_heldout_action_responses_observed": 0,
                         "V20_operator_responses_already_observed": 0,
                         "split_freeze_digest": split["freeze_digest"]})


def calibrate(root: Path) -> dict:
    action_design = verify_stage(root, "actions")
    split = verify_stage(root, "splits")
    prompts = bank.prompt_index(root)
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    cfg = verify(root)["config"]["action_bank"]
    rows = []
    scale_coords = set(action_design["partitions"]["validation"][i]["coordinate_index"] for i in range(6))
    scale_coords.update((0, 1, 2, 8))
    for number, item in enumerate(split["q_calibration"], 1):
        prompt = str(prompts[item["base_trial_id"]]["prompt"])
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        p0 = clean["cache"]
        for action in action_design["actions"]:
            coord = int(action["coordinate_index"])
            multipliers = (1.0, 2.0) if coord in scale_coords else (1.0,)
            for multiplier in multipliers:
                for sign in cfg["signs_measured"]:
                    alpha = float(action["base_alpha"] * multiplier)
                    arow = v19._action_row(values["directions"], int(action["direction_index"]), alpha, int(sign))
                    edited = v19.apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                    readback = _readback(p0, edited, arow, rec, att)
                    reliable = (readback["realized_state_cosine"] is not None
                                and readback["realized_state_cosine"] >= cfg["reliability_min_cosine"]
                                and readback["realized_state_gain"] is not None
                                and cfg["reliability_gain_min"] <= readback["realized_state_gain"] <= cfg["reliability_gain_max"])
                    rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                                 "coordinate_index": coord, "proposal_family": action["proposal_family"],
                                 "sign": int(sign), "alpha": alpha, "scale_multiplier": multiplier,
                                 "reliable": bool(reliable), **readback})
        if number % 5 == 0 or number == len(split["q_calibration"]):
            print(f"V20 action calibration {number}/{len(split['q_calibration'])}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / CALIBRATION
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    by_action = []
    for (coord, multiplier), group in frame.groupby(["coordinate_index", "scale_multiplier"], sort=True):
        by_action.append({"coordinate_index": int(coord), "scale_multiplier": float(multiplier),
                          "reliable_fraction": float(group.reliable.mean()),
                          "median_realized_cosine": float(group.realized_state_cosine.median()),
                          "median_gain": float(group.realized_state_gain.median()),
                          "median_raw_norm": float(group.realized_state_norm.median())})
    return stage_freeze(root, "action_calibration",
                        [SOURCE, str(CALIBRATION),
                         "artifacts/compact_causal_response_operator_v20_actions.freeze.json"],
                        {"calibration_path": str(CALIBRATION), "calibration_sha256": sha256_file(path),
                         "state_count": int(frame.base_trial_id.nunique()), "rows": len(frame),
                         "by_action_scale": by_action,
                         "base_scale_min_reliable_fraction": min(x["reliable_fraction"] for x in by_action if x["scale_multiplier"] == 1.0),
                         "train_only": True, "operator_validation_responses_observed": 0,
                         "final_heldout_action_responses_observed": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("amend_selection", "prepare", "calibrate"))
    args = parser.parse_args()
    result = {"amend_selection": amend_selection, "prepare": prepare, "calibrate": calibrate}[args.stage](Path.cwd())
    print(json.dumps({"freeze_digest": result["freeze_digest"], "stage": args.stage}, indent=2))
