#!/usr/bin/env python3
"""Apply the frozen V20 operator gates without promoting an oracle to a state."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import verify, verify_stage
from jclosure.provenance import write_json_atomic


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    for name in ("splits", "actions", "operator_design", "operator_analysis",
                 "geometry_design", "operator_diagnostics_design", "operator_aliasing_design"):
        verify_stage(root, name)
    processed = root / bank.OUT
    read = lambda name: json.loads((processed / name).read_text())
    confirmation = read("independent_v19_confirmation_v20.json")
    model = read("oracle_operator_search_v20.json")
    diagnostic = read("operator_action_diagnostics_v20.json")
    geometry = read("operator_geometry_v20.json")
    aliasing = read("operator_aliasing_v20.json")
    if not confirmation["V19_B_independently_confirmed"]:
        raise RuntimeError("V19-B independent confirmation failed; V20 operator claim is ineligible")
    if diagnostic["final_operator_gate"]:
        raise RuntimeError("Oracle operator gate passed: execute frozen raw-P encoder and conditional-sufficiency stages before V20 adjudication")
    chosen = next(x for x in model["model_sweep"] if x["model"] == model["selected_model_for_diagnostics"]
                  and x["k"] == model["selected_k_for_diagnostics"])
    seen_gate = chosen["metrics"]["seen_direction_new_state"]["stack_relative_l2"] <= base["config"]["operator_analysis"]["heldout_relative_l2_max"]
    result = {
        "protocol_freeze_digest": base["freeze_digest"],
        "V19_B_independently_confirmed": True,
        "oracle_model_selected_for_diagnostics": model["selected_model_for_diagnostics"],
        "oracle_k_selected_for_diagnostics": model["selected_k_for_diagnostics"],
        "oracle_seen_action_relative_l2_within_threshold": bool(seen_gate),
        "preliminary_operator_gate": bool(model["preliminary_operator_compactness_gate"]),
        "sign_scale_composition_gate": bool(diagnostic["unseen_scale_composition_gate"]),
        "operator_compactness_gate": False,
        "k_operator_min": None,
        "formal_V20_outcome": "V20-D_NO_COMPACT_OPERATOR_DIMENSION_IDENTIFIED",
        "interpretation": "None of the frozen tested oracle models and k values established a compact cross-action response operator under all gates. This does not prove mathematical nonexistence or global high dimension.",
        "action_specific_only_descriptive": bool(seen_gate),
        "raw_P_to_operator_encoder_status": "NOT_ELIGIBLE_ORACLE_OPERATOR_GATE_FAILED",
        "conditional_raw_sufficiency_status": "NOT_ELIGIBLE_NO_COMPACT_ENCODER",
        "channel_operator_state_status": "NOT_ELIGIBLE_NO_COMPACT_ENCODER",
        "same_J_encoder_status": "NOT_ELIGIBLE_NO_COMPACT_ENCODER",
        "independent_V20_final_status": "UNOPENED_NO_FROZEN_ELIGIBLE_ENCODER_FINALIST",
        "final_heldout_action_direction_status": "SEALED_UNOPENED",
        "independent_V20_final_hash": None,
        "V21_DYNAMIC_STATE_SEARCH_AUTHORIZED": False,
        "H2_remains": True,
        "H3_authorized": False,
        "AUTONOMOUS_STATE_MODEL_AUTHORIZED": False,
        "physical_cache_replacement_authorized": False,
        "workspace_state_aliasing_observed": bool(aliasing["workspace_state_aliasing_observed"]),
        "operator_geometry_recorded": bool(geometry["pair_count"]),
    }
    write_json_atomic(processed / "v20_adjudication.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
