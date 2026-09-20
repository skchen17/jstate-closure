#!/usr/bin/env python3
"""Append-only correction: distinguish action-specific fit from the primary V20-D result."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    root = Path.cwd()
    verify(root)
    old_path = root / bank.OUT / "v20_adjudication.json"
    oracle_path = root / bank.OUT / "oracle_operator_search_v20.json"
    old = json.loads(old_path.read_text())
    oracle = json.loads(oracle_path.read_text())
    limit = 0.30
    action_specific = [
        {"model": row["model"], "k": row["k"],
         "seen_direction_new_state_stack_relative_l2": row["metrics"]["seen_direction_new_state"]["stack_relative_l2"],
         "unseen_direction_stack_relative_l2": row["metrics"]["unseen_direction_positive"]["stack_relative_l2"]}
        for row in oracle["model_sweep"]
        if row["metrics"]["seen_direction_new_state"]["stack_relative_l2"] <= limit
        and row["metrics"]["unseen_direction_positive"]["stack_relative_l2"] > limit
    ]
    if not action_specific or old["operator_compactness_gate"]:
        raise RuntimeError("Adjudication amendment evidence absent")
    freeze = stage_freeze(root, "adjudication_amendment_1",
                          ["scripts/amend_v20_adjudication.py",
                           "results/v20/processed/v20_adjudication.json",
                           "results/v20/processed/oracle_operator_search_v20.json"],
                          {"reason": "Original adjudication checked seen-action fit only in the model selected for unseen-direction diagnostics, overlooking a different frozen nonlinear model that fit seen actions but failed unseen directions.",
                           "original_adjudication_sha256": sha256_file(old_path),
                           "oracle_search_sha256": sha256_file(oracle_path),
                           "frozen_seen_relative_L2_limit": limit,
                           "thresholds_or_models_changed": False,
                           "responses_added_after_original_adjudication": 0,
                           "primary_formal_outcome_unchanged": old["formal_V20_outcome"]})
    result = {
        "amendment_freeze_digest": freeze["freeze_digest"],
        "original_adjudication_sha256": sha256_file(old_path),
        "primary_formal_outcome": old["formal_V20_outcome"],
        "secondary_observation": "V20-C_ACTION_SPECIFIC_OPERATOR_ENCODING_ONLY",
        "evidence": action_specific,
        "corrected_action_specific_only_descriptive": True,
        "k_operator_min": None,
        "raw_encoder_status": old["raw_P_to_operator_encoder_status"],
        "independent_V20_final_status": old["independent_V20_final_status"],
        "V21_DYNAMIC_STATE_SEARCH_AUTHORIZED": False,
        "H3_authorized": False,
    }
    write_json_atomic(root / bank.OUT / "v20_adjudication_amendment_1.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
