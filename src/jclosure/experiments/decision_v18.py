"""Predeclared V18 stage-gate and formal-decision logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.crossed_bank_v18 import OUT, _split
from jclosure.protocol_v18 import digest, stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/decision_v18.py"
FREEZE = Path("artifacts/strong_state_context_ceiling_v18_decision.freeze.json")


def prepare(root: Path) -> dict:
    split = _split(root)
    gates = verify(root)["config"]["gates"]
    return stage_freeze(root, "decision", [SOURCE, "artifacts/strong_state_context_ceiling_v18_splits.freeze.json",
                                            "artifacts/strong_state_context_ceiling_v18_models.freeze.json",
                                            "artifacts/strong_state_context_ceiling_v18_strict_match.freeze.json",
                                            "artifacts/strong_state_context_ceiling_v18_history.freeze.json"],
                        {"split_freeze_digest": split["freeze_digest"],
                         "material_raw_gain_threshold": float(gates["material_raw_context_absolute_rel_l2_gain_min"]),
                         "raw_gate_targets": ["j", "stacked_normalized"],
                         "raw_gate_requires_positive_bootstrap_lower_and_nonnegative_all_family": True,
                         "strict_match_min_independent_state_pairs": 20,
                         "strict_match_median_divergence_min": float(gates["strict_response_divergence_min"]),
                         "strict_match_near_raw_control_gap_min": 0.10,
                         "history_incremental_gain_threshold": 0.05,
                         "history_formal_requires_positive_bootstrap_lower_and_nonnegative_all_family": True,
                         "no_V18_A_or_G_if_strict_match_not_identified": True,
                         "compact_authorization": "material_h1_or_any_h2_h4_h8_or_strict_match_positive",
                         "independent_final_requires_frozen_compact_dynamic_finalist": True,
                         "bootstrap_unit": "state", "bootstrap_replicates": 1000, "seed": 1822})


def _freeze(root: Path) -> dict:
    base = verify(root)
    value = json.loads((root / FREEZE).read_text())
    if value["freeze_digest"] != digest(value) or value["base_freeze_digest"] != base["freeze_digest"]:
        raise RuntimeError("V18 decision freeze invalid")
    for path, expected in value["input_hashes"].items():
        if sha256_file(root / path) != expected:
            raise RuntimeError(f"V18 decision input changed: {path}")
    return value


def _history_bootstrap(root: Path, horizon: int, context: str) -> dict:
    data = pd.read_parquet(root / OUT / "history_context_state_errors_v18.parquet")
    subset = data[(data.horizon == horizon) & (data.target == "stacked_normalized")]
    left = subset[subset.context == "current_j_raw"].sort_values("base_trial_id").reset_index(drop=True)
    right = subset[subset.context == context].sort_values("base_trial_id").reset_index(drop=True)
    if not left.base_trial_id.equals(right.base_trial_id):
        raise RuntimeError("V18 history bootstrap state alignment mismatch")
    values = np.stack((left.actual_squared_norm, left.error_squared_norm, right.error_squared_norm), axis=1)
    rng = np.random.default_rng(1822 + horizon)
    sample = rng.integers(0, len(values), size=(1000, len(values)))
    totals = values[sample].sum(axis=1)
    gain = (np.sqrt(totals[:, 1]) - np.sqrt(totals[:, 2])) / np.sqrt(np.maximum(totals[:, 0], 1e-24))
    family = {}
    for name in sorted(left.family.unique()):
        mask = left.family == name
        denominator = np.sqrt(float(left.loc[mask, "actual_squared_norm"].sum()))
        family[name] = (np.sqrt(float(left.loc[mask, "error_squared_norm"].sum())) -
                        np.sqrt(float(right.loc[mask, "error_squared_norm"].sum()))) / max(denominator, 1e-12)
    point = (np.sqrt(values[:, 1].sum()) - np.sqrt(values[:, 2].sum())) / np.sqrt(max(values[:, 0].sum(), 1e-24))
    return {"gain": float(point), "bootstrap_ci95": np.quantile(gain, [0.025, 0.975]).tolist(),
            "family_gain": family}


def run(root: Path) -> dict:
    freeze = _freeze(root)
    h1 = json.loads((root / OUT / "strong_h1_ceiling_v18.json").read_text())
    horizons = json.loads((root / OUT / "horizon_context_localization_v18.json").read_text())
    matched = json.loads((root / OUT / "strict_matched_intervention_v18.json").read_text())
    history = json.loads((root / OUT / "history_vs_snapshot_v18.json").read_text())
    match_a = matched["pair_types"]["near_J_different_raw"]
    match_b = matched["pair_types"]["near_J_near_raw"]
    strict_positive = (match_a["selected_one_to_one_state_pairs"] >= freeze["strict_match_min_independent_state_pairs"]
                       and match_b["selected_one_to_one_state_pairs"] >= freeze["strict_match_min_independent_state_pairs"]
                       and match_a["median_response_relative_divergence"] is not None
                       and match_b["median_response_relative_divergence"] is not None
                       and match_a["median_response_relative_divergence"] >= freeze["strict_match_median_divergence_min"]
                       and match_a["median_response_relative_divergence"] - match_b["median_response_relative_divergence"] >= freeze["strict_match_near_raw_control_gap_min"])
    material_h1 = bool(h1["material_h1_raw_context_gate_passed"])
    material_horizon = [row["horizon"] for row in horizons["horizons"] if row["material_gate_passed"]]
    compact_authorized = material_h1 or bool(material_horizon) or strict_positive
    history_tests = {str(h): _history_bootstrap(root, h, "last_4_j_plus_raw") for h in (1, 2, 4, 8)}
    history_material = any(row["gain"] >= freeze["history_incremental_gain_threshold"]
                           and row["bootstrap_ci95"][0] > 0
                           and all(value >= 0 for value in row["family_gain"].values())
                           for row in history_tests.values())
    if material_h1:
        outcome = "V18-B — CURRENT_RAW_CONTEXT_REQUIRED"
    elif any(h > 1 for h in material_horizon):
        outcome = "V18-C — LONG_HORIZON_CONTEXT_REQUIRED"
    elif history_material:
        outcome = "V18-D — HISTORY_REQUIRED_WITHIN_TESTED_REPRESENTATION"
    elif strict_positive:
        outcome = "V18-STOP — STRICT_MATCH_CONTEXT_DIVERGENCE_WITHOUT_MATERIAL_MODEL_GAIN"
    elif matched["status"] == "MATCH_NOT_IDENTIFIED":
        outcome = "V18-STOP — STRICT_MATCH_NOT_IDENTIFIED_AND_NO_MATERIAL_RAW_CEILING"
    else:
        outcome = "V18-G — NO_MATERIAL_RAW_CONTEXT_FOUND_WITHIN_TESTED_PROTOCOL"
    result = {"decision_freeze_digest": freeze["freeze_digest"], "formal_outcome": outcome,
              "material_h1_raw_context": material_h1, "material_horizons": material_horizon,
              "strict_match_status": matched["status"], "strict_match_positive": strict_positive,
              "history_incremental_tests": history_tests, "history_material": history_material,
              "compact_search_authorized": compact_authorized,
              "compact_search_status": "AUTHORIZED_NOT_YET_RUN" if compact_authorized else "COMPACT_CONTEXT_SEARCH_NOT_AUTHORIZED",
              "h2_remains": True, "h3_candidate_supported": False,
              "interventional_markov_state_candidate": False,
              "autonomous_state_model_authorized": False,
              "independent_confirmation": "UNOPENED_PENDING_FROZEN_FINALIST",
              "absolute_replacement_claim": False,
              "v18_A_near_J_context_sufficiency": False if matched["status"] == "MATCH_NOT_IDENTIFIED" else "NOT_ADJUDICATED_BEYOND_PRIMARY_CEILING",
              "v18_E_compact_context": "NOT_ADJUDICATED" if compact_authorized else False,
              "v18_F_dynamical_state": False}
    write_json_atomic(root / OUT / "v18_adjudication.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "run": run}[args.stage](root)
    print(json.dumps(result, indent=2))
