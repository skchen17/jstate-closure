"""Conservative V36 adjudication after sealed local and control records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/adjudicate_v36.py"


def run(root: Path):
    for stage in ("local_analysis_development", "local_analysis_validation", "final_opening"):
        verify_stage(root, stage)
    for key in ("Q", "F"):
        for role in ("development", "validation"):
            verify_stage(root, f"control_analysis_{key}_{role}")
    read = lambda name: json.loads((root / OUT / name).read_text())
    dev, val = (read(f"local_analysis_{role}_v36.json") for role in ("development", "validation"))
    opening = read("final_opening_v36.json")
    both = dev["both_models_same_class_operator_and_causal_pass"] and val["both_models_same_class_operator_and_causal_pass"]
    if both or opening["opened"]:
        raise RuntimeError("V36 negative adjudication cannot be used for a qualified finalist")
    control = {(key, role): read(f"control_analysis_{key}_{role}_v36.json") for key in ("Q", "F")
               for role in ("development", "validation")}
    outcome = {
        "V36-A_READ_OPERATOR_MATCHING_CONFIRMED": False,
        "V36-B_LOCAL_REC_CONV_INTERACTION_CONFIRMED": False,
        "V36-C_UPSTREAM_CONTEXT_DOMINATES": False,
        "V36-D_OLD_STATE_READ_DOMINATES": False,
        "V36-E_CURRENT_UPDATE_DOMINATES": False,
        "V36-F_STATE_UPDATE_READ_INTERACTION": False,
        "V36-G_POST_READ_NONLINEARITY_MEDIATES": False,
        "V36-H_CROSS_MODEL_OPERATOR_LAW": False,
        "V36-I_NO_SIMPLE_COMPUTATIONAL_LAW_IDENTIFIED": True,
    }
    record = {"formal_outcomes": outcome,
              "V36_I_scope": "tested four fixed representative layers and frozen same-input candidate; not a universal no-law result",
              "development_candidate_pass": dev["both_models_same_class_operator_and_causal_pass"],
              "validation_candidate_pass": val["both_models_same_class_operator_and_causal_pass"],
              "independent_final_opened": False,
              "independent_final_responses_seen": False,
              "composition_authorized": False,
              "task_grounded_authorized": False,
              "upper_bound_single_layer_medians": {f"{key}_{role}": control[key, role]["median_upper_fraction"]
                                                  for key in ("Q", "F") for role in ("development", "validation")},
              "descriptive_upstream_context_evidence": {f"{key}_{role}": (dev if role == "development" else val)["models"][key]["fixed_to_natural_interaction_median"]
                                                        for key in ("Q", "F") for role in ("development", "validation")},
              "outcomes_C_to_G_not_confirmed_because_no_frozen_sufficient_downstream_test": True,
              "historical_V1_V35_modified": False,
              "thresholds_retuned": False}
    path = root / OUT / "v36_adjudication.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "adjudication", [SOURCE, str(path.relative_to(root)),
                        "artifacts/computational_origin_v36_final_opening.freeze.json",
                        *(f"artifacts/computational_origin_v36_control_analysis_{key}_{role}.freeze.json"
                          for key in ("Q", "F") for role in ("development", "validation"))],
                        {"outcome_I": True, "summary_sha256": sha256_file(path), "final_opened": False})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
