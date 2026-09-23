"""Freeze V37 finalist choice before observing any independent-final response."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/final_opening_v37.py"


def _read(root: Path, name: str):
    return json.loads((root / OUT / name).read_text(encoding="utf-8"))


def run(root: Path):
    cfg = verify(root)["config"]
    verify_stage(root, "sample_pools")
    verify_stage(root, "pool_independent_final")
    for role in ("development", "validation"):
        for stage in (f"high_level_{role}", f"read_gate_{role}",
                      f"local_analysis_{role}"):
            verify_stage(root, stage)
    high = {r: _read(root, f"high_level_{r}_v37.json")
            for r in ("development", "validation")}
    read = {r: _read(root, f"read_gate_{r}_v37.json")
            for r in ("development", "validation")}
    local = {r: _read(root, f"local_analysis_{r}_v37.json")
             for r in ("development", "validation")}
    common = all(high[r]["both_pass"] and read[r]["both_pass"] for r in high)
    eligible = {name: False for name in cfg["finalist_priority"]}
    reasons = {name: "not implemented/tested as a bilateral development+validation mechanism"
               for name in eligible}
    cross_dev = _read(root, "cross_layer_analysis_development_v37.json")
    if cross_dev["both_models_cross_layer_pass"]:
        verify_stage(root, "cross_layer_analysis_validation")
        cross_val = _read(root, "cross_layer_analysis_validation_v37.json")
        eligible["CROSS_LAYER_OPERATOR_CONDITIONING"] = bool(common and cross_val["both_models_cross_layer_pass"])
        reasons["CROSS_LAYER_OPERATOR_CONDITIONING"] = "both-role gate" if eligible["CROSS_LAYER_OPERATOR_CONDITIONING"] else "validation gate failed"
    else:
        reasons["CROSS_LAYER_OPERATOR_CONDITIONING"] = "development prospective downstream ordering gate failed; validation not opened"
    eligible["READ_OPERATOR_MATCHING"] = bool(common and all(
        local[r]["both_models_local_prediction_pass"] for r in local))
    reasons["READ_OPERATOR_MATCHING"] = (
        "prospective native local read law passed both models in development and validation"
        if eligible["READ_OPERATOR_MATCHING"] else "local read-law gate failed in at least one role/model")
    selected = next((name for name in cfg["finalist_priority"] if eligible[name]), None)
    panel = _read(root, "panel_v37.json")
    result = {
        "version": "V37", "selected_class": selected, "opened": selected is not None,
        "finalist_priority": cfg["finalist_priority"],
        "eligibility": eligible, "reasons": reasons,
        "independent_final_states_per_model": len(panel["independent_final"]),
        "independent_final_pool_sealed_before_all_interventions": True,
        "independent_final_outcomes_seen_at_opening": False,
        "v18_training_source_in_final": False,
        "v36_results_reused": False,
        "full_output_copy_not_eligible": True,
        "level_two_local_computation_not_claimed_as_level_three_or_four": True,
    }
    path = root / OUT / "final_opening_v37.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "final_opening",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/computational_origin_v37_pool_independent_final.freeze.json",
                         "artifacts/computational_origin_v37_local_analysis_development.freeze.json",
                         "artifacts/computational_origin_v37_local_analysis_validation.freeze.json"],
                        {"selected_class": selected, "opened": result["opened"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
