"""Conservative V35 causal-hierarchy adjudication from sealed role and final records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/adjudicate_v35.py"


def patterns(model):
    found = set()
    if model["late_consolidation"]["pass"]:
        found.add("LATE_Q3_Q4")
    if model["progressive_accumulation"]["pass"]:
        found.add("PROGRESSIVE_CUMULATIVE")
    found.update("COMPLEMENTARY_" + name for name in model["complementary_pair_passes"])
    found.update("REDUNDANT_" + name for name in model["redundant_pair_passes"])
    found.update("SERIAL_" + name.replace("_", "_TO_") for name in model["serial_pair_passes"])
    found.update("LOCALIZED_" + name for name in model["single_quartile_passes"] + model["strong_pair_passes"])
    return found


def run(root: Path):
    stages = ["high_level_development", "high_level_validation", "high_level_independent_final",
              "full_gate_development", "full_gate_validation", "full_gate_independent_final",
              "depth_analysis_development", "depth_analysis_validation",
              "development_plan", "final_opening", "final_analysis", "control_plan"]
    for name in stages:
        verify_stage(root, name)
    for key in ("Q", "F"):
        verify_stage(root, f"controls_{key}")
        verify_stage(root, f"flow_{key}")
    def read(name):
        return json.loads((root / OUT / name).read_text())
    dev = read("depth_analysis_development_v35.json")
    val = read("depth_analysis_validation_v35.json")
    final = read("final_analysis_v35.json")
    opening = read("final_opening_v35.json")
    selected = opening["selected_mechanism"]
    confirmed = bool(final["both_pass"])
    model_patterns = {key: sorted(patterns(dev["models"][key]) & patterns(val["models"][key]))
                      for key in ("Q", "F")}
    common = sorted(set(model_patterns["Q"]) & set(model_patterns["F"]))
    divergent_gate_patterns = (bool(model_patterns["Q"]) and bool(model_patterns["F"]) and not common)
    # Non-overlapping pass/fail labels, especially near frozen cutoffs, are not
    # by themselves a predeclared material-difference test. Keep G unconfirmed.
    specific = False
    broad_only = (confirmed and selected == "FULL_DEPTH_ONLY" and
                  not model_patterns["Q"] and not model_patterns["F"])
    outcomes = {
        "V35-A_LATE_READ_CONSOLIDATION_CONFIRMED": confirmed and selected == "LATE_Q3_Q4",
        "V35-B_PROGRESSIVE_READ_ACCUMULATION_CONFIRMED": confirmed and selected == "PROGRESSIVE_CUMULATIVE",
        "V35-C_COMPLEMENTARY_MULTI_STAGE_MEDIATION": confirmed and selected.startswith("COMPLEMENTARY_"),
        "V35-D_REDUNDANT_READ_MEDIATION": confirmed and selected.startswith("REDUNDANT_"),
        "V35-E_SERIAL_HIERARCHICAL_CORRECTION": confirmed and selected.startswith("SERIAL_"),
        "V35-F_SHARED_HIERARCHICAL_ORGANIZATION": confirmed and selected != "FULL_DEPTH_ONLY",
        "V35-G_ARCHITECTURE_SPECIFIC_HIERARCHY": bool(specific),
        "V35-H_LOCALIZED_READ_REGION": confirmed and selected.startswith("LOCALIZED_"),
        "V35-I_FULL_DEPTH_READ_CUT_ONLY": bool(broad_only),
    }
    full_gates = {role: read(f"full_gate_{role}_v35.json")["both_pass"]
                  for role in ("development", "validation", "independent_final")}
    record = {"formal_outcomes": outcomes, "selected_finalist": selected,
              "independent_final_confirmed": confirmed,
              "full_depth_reconfirmed_all_roles": all(full_gates.values()),
              "full_depth_role_gates": full_gates,
              "development_validation_qualified_by_model": model_patterns,
              "shared_development_validation_patterns": common,
              "no_shared_smaller_organization_passed": not bool(common),
              "divergent_gate_patterns_descriptive": bool(divergent_gate_patterns),
              "cross_model_profile_descriptive": val["cross_model_profile"],
              "architecture_specific_not_confirmed_without_frozen_material_difference_test": True,
              "broad_only_requires_no_validated_smaller_pattern_in_either_model": True,
              "application_experiments_authorized": bool(confirmed and selected != "FULL_DEPTH_ONLY"),
              "application_benchmark_performed": False,
              "third_model_replication_freeze": (selected if outcomes["V35-F_SHARED_HIERARCHICAL_ORGANIZATION"]
                                                  else "BROAD_RECURRENT_READ_ONLY"),
              "historical_independent_finals_reopened": False,
              "V34_exploratory_quartiles_used_as_formal_evidence": False,
              "thresholds_retuned": False}
    path = root / OUT / "v35_adjudication.json"
    write_json_atomic(path, record)
    inputs = [SOURCE, str(path.relative_to(root)),
              *(f"artifacts/hierarchical_read_v35_{name}.freeze.json" for name in stages),
              *(f"artifacts/hierarchical_read_v35_{mode}_{key}.freeze.json"
                for mode in ("controls", "flow") for key in ("Q", "F"))]
    seal = stage_freeze(root, "adjudication", inputs,
                        {"selected_finalist": selected,
                         "independent_final_confirmed": confirmed,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
