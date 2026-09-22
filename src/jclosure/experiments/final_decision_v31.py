"""Apply frozen V31 finalist gate without opening historical or V31 final panels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.primitive_matrix_v31 import OUT
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/final_decision_v31.py"


def run(root: Path):
    for stage in ("composition_development", "mechanism_development", "mechanism_validation", "primitive_causal_development", "train_depth_energy"):
        verify_stage(root, stage)
    cfg = verify(root)["config"]
    composition = json.loads((root / OUT / "heldout_composition_development_v31.json").read_text())
    primitive = json.loads((root / OUT / "primitive_causal_development_v31.json").read_text())
    mechanism_dev = json.loads((root / OUT / "mechanism_development_v31.json").read_text())
    mechanism_val = json.loads((root / OUT / "mechanism_validation_v31.json").read_text())
    comp_pass = [name for name in cfg["composition_rules"] if composition["profiles"][name]["pass"]]
    dictionary_pass = [name for name, value in primitive["profiles"].items() if value["pass"] and name.startswith(("SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES", "FUNCTION_CONDITIONED", "RESPONSE_FACTOR"))]
    depth = mechanism_dev["route_selection"]
    depth_pass = bool(not depth["fallback_full24"] and mechanism_dev["depth_profiles"][depth["name"]]["pass"] and mechanism_val["depth_profiles"][depth["name"]]["pass"])
    exact_ceiling = composition["profiles"]["EXACT_REC_CONV"]
    # The frozen strict full-donor gate cannot be replaced by a post-hoc sign-only REC gate.
    rec_strict_pass = bool(exact_ceiling["pass"] and mechanism_dev["rec_improves_count"] == 20 and mechanism_val["rec_improves_count"] == 20)
    all_candidates = {"heldout_composition": comp_pass, "causal_dictionary": dictionary_pass, "REC_correction_strict": rec_strict_pass, "token_conditioned_depth_route": depth_pass}
    if comp_pass or dictionary_pass or depth_pass or rec_strict_pass:
        raise RuntimeError("A V31 development candidate qualifies; validation/final selection must be adjudicated before sealing final")
    decision = {"final_opened": False, "reason": "No mechanistic finalist passed the frozen development causal/family gate; exact REC+Conv ceiling failed the 4/5-family development gate despite replicated paired REC improvement.", "candidate_screen": all_candidates, "composition_validation_opened": False, "primitive_validation_opened": False, "independent_final_opened": False, "TOKEN_FINAL_write_or_future_response_observed": False, "historical_independent_finals_reopened": False, "REC_residual_correction_descriptive_replication": {"development_positive_pairs": mechanism_dev["rec_improves_count"], "validation_positive_pairs": mechanism_val["rec_improves_count"], "development_median_alignment": mechanism_dev["median_residual_alignment_cosine"], "validation_median_alignment": mechanism_val["median_residual_alignment_cosine"], "not_a_predeclared_F_specific_threshold": True}, "authorization": cfg["authorization"]}
    path = root / OUT / "final_opening_v31.json"
    write_json_atomic(path, decision)
    freeze = stage_freeze(root, "final_opening", [SOURCE, str(path.relative_to(root)), str(OUT / "heldout_composition_development_v31.json"), str(OUT / "primitive_causal_development_v31.json"), str(OUT / "mechanism_development_v31.json"), str(OUT / "mechanism_validation_v31.json"), "artifacts/compositional_natural_writes_v31_execution_plan.freeze.json"], {"decision_sha256": sha256_file(path), "final_opened": False, "cross_model_replication_authorized": False})
    return {"freeze_digest": freeze["freeze_digest"], "final_opened": False, "candidate_screen": all_candidates}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
