"""Freeze development-qualified refinement regions and final-candidate order before validation."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/development_plan_v35.py"
REFINE_PRIORITY = ("Q4", "Q3", "Q2", "Q1", "Q3_Q4", "Q2_Q3", "Q1_Q2")
FINAL_PRIORITY = ("LATE_Q3_Q4", "PROGRESSIVE_CUMULATIVE", "SERIAL_Q3_TO_Q4",
                  "SERIAL_Q2_TO_Q3", "SERIAL_Q2_TO_Q4",
                  "COMPLEMENTARY_Q3_Q4", "COMPLEMENTARY_Q2_Q3", "COMPLEMENTARY_Q2_Q4",
                  "COMPLEMENTARY_Q1_Q4", "COMPLEMENTARY_Q1_Q3", "COMPLEMENTARY_Q1_Q2",
                  "REDUNDANT_Q3_Q4", "REDUNDANT_Q2_Q3", "REDUNDANT_Q2_Q4",
                  "REDUNDANT_Q1_Q4", "REDUNDANT_Q1_Q3", "REDUNDANT_Q1_Q2",
                  "LOCALIZED_Q4", "LOCALIZED_Q3", "LOCALIZED_Q2", "LOCALIZED_Q1",
                  "LOCALIZED_Q3_Q4", "LOCALIZED_Q2_Q3", "LOCALIZED_Q1_Q2",
                  "LOCALIZED_Q2_Q4", "LOCALIZED_Q1_Q4", "LOCALIZED_Q1_Q3",
                  "LOCALIZED_Q1_Q2_Q3", "LOCALIZED_Q2_Q3_Q4", "FULL_DEPTH_ONLY")


def candidates(shared):
    found = {"FULL_DEPTH_ONLY"}
    if shared["late"]:
        found.add("LATE_Q3_Q4")
    if shared["progressive"]:
        found.add("PROGRESSIVE_CUMULATIVE")
    for pair in shared["serial_pairs"]:
        found.add(f"SERIAL_{pair.replace('_', '_TO_')}")
    for pair in shared["complementary_pairs"]:
        found.add(f"COMPLEMENTARY_{pair}")
    for pair in shared.get("redundant_pairs", []):
        found.add(f"REDUNDANT_{pair}")
    for name in shared["strong_quartiles"] + shared["strong_pairs"]:
        found.add(f"LOCALIZED_{name}")
    return [name for name in FINAL_PRIORITY if name in found]


def run(root: Path):
    verify_stage(root, "depth_analysis_development")
    analysis = json.loads((root / OUT / "depth_analysis_development_v35.json").read_text())
    record = {"role": "development", "candidate_priority_frozen_before_validation": list(FINAL_PRIORITY),
              "shared_development_candidates": candidates(analysis["shared_qualitative_this_role"]),
              "model_refinement": {}, "validation_responses_seen": False,
              "global_single_layer_search_authorized": False}
    for key in ("Q", "F"):
        model = analysis["models"][key]
        qualified = set(model["refinement_candidates"])
        selected = next((region for region in REFINE_PRIORITY if region in qualified), None)
        design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
        if selected:
            layers = design["condition_layers"][selected]
            if len(layers) not in (6, 12) or len(layers) % 3:
                raise RuntimeError(f"V35 invalid refinement region {key}:{selected}")
            windows = {f"{selected}_W{i+1}": layers[i:i+3] for i in range(0, len(layers), 3)}
        else:
            windows = {}
        record["model_refinement"][key] = {"qualified_coarse_regions": sorted(qualified),
                                            "selected_region": selected, "three_layer_windows": windows,
                                            "selection_rule": "first qualified region in frozen REFINE_PRIORITY"}
    path = root / OUT / "development_plan_v35.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "development_plan", [SOURCE, str(path.relative_to(root)),
                                                 "artifacts/hierarchical_read_v35_depth_analysis_development.freeze.json"],
                        {"candidate_priority": list(FINAL_PRIORITY),
                         "summary_sha256": sha256_file(path),
                         "validation_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
