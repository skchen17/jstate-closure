"""Open the paired V35 independent finals only on a development-frozen validated mechanism."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.development_plan_v35 import candidates
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/final_opening_v35.py"


def conditions(name, cfg):
    if name == "FULL_DEPTH_ONLY":
        return ["FULL"]
    if name == "LATE_Q3_Q4":
        return ["Q1_Q2", "Q3_Q4", "FULL"]
    if name == "PROGRESSIVE_CUMULATIVE":
        return ["Q1", "Q2", "Q3", "Q4", "Q1_Q2", "Q1_Q2_Q3", "FULL"]
    if name.startswith("SERIAL_"):
        early, late = name.removeprefix("SERIAL_").split("_TO_")
        return [early, late, f"{early}_{late}", "FULL"]
    if name.startswith(("COMPLEMENTARY_", "REDUNDANT_")):
        pair = name.split("_", 1)[1]
        a, b = pair.split("_")
        return [a, b, pair, "FULL"]
    if name.startswith("LOCALIZED_"):
        return [name.removeprefix("LOCALIZED_"), "FULL"]
    raise RuntimeError(f"V35 unknown finalist {name}")


def run(root: Path):
    verify_stage(root, "development_plan")
    gate = verify_stage(root, "full_gate_validation")
    verify_stage(root, "depth_analysis_validation")
    if not gate["formal_depth_authorized"]:
        raise RuntimeError("V35 full-depth validation gate failed; independent final remains sealed")
    plan = json.loads((root / OUT / "development_plan_v35.json").read_text())
    validation = json.loads((root / OUT / "depth_analysis_validation_v35.json").read_text())
    cfg = verify(root)["config"]
    validated = set(candidates(validation["shared_qualitative_both_roles"]))
    predeclared = plan["shared_development_candidates"]
    selected = next((name for name in predeclared if name in validated), None)
    if selected is None:
        raise RuntimeError("V35 no same-model mechanism qualified, including full depth")
    record = {"opened": True, "both_models_simultaneously": True,
              "selected_mechanism": selected, "selected_conditions": conditions(selected, cfg),
              "development_frozen_priority": plan["candidate_priority_frozen_before_validation"],
              "development_candidates": predeclared, "validated_candidates": sorted(validated),
              "thresholds_retuned": False, "final_responses_seen": False,
              "historical_independent_finals_reopened": False,
              "model_specific_relative_depth_layers": {
                  key: {condition: json.loads((root / OUT / f"design_{key}_v35.json").read_text())["condition_layers"][condition]
                        for condition in conditions(selected, cfg)} for key in ("Q", "F")},
              "endpoints": cfg["shared_endpoint"], "family_rule": cfg["quartile_gate"]["families_required"],
              "independent_final_states_per_model": 40,
              "h2_h4_authorized": selected != "FULL_DEPTH_ONLY"}
    path = root / OUT / "final_opening_v35.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "final_opening", [SOURCE, str(path.relative_to(root)),
                                              "artifacts/hierarchical_read_v35_development_plan.freeze.json",
                                              "artifacts/hierarchical_read_v35_depth_analysis_validation.freeze.json",
                                              "artifacts/hierarchical_read_v35_full_gate_validation.freeze.json"],
                        {"opened": True, "selected_mechanism": selected,
                         "finalist_sha256": sha256_file(path), "final_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
