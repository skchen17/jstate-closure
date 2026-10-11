"""Audit V40 development/validation and seal independent-final opening."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.protocol_v40_a5 import verify_stage as verify_a5_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "scripts/open_v40_final.py"


def premature_final_artifacts(root: Path) -> list[str]:
    patterns = (
        "results/v40/raw/formal_*_independent_final_*.jsonl*",
        "results/v40/raw/trajectory_Q_independent_final_*.jsonl*",
        "results/v40/processed/*_independent_final_v40.json",
    )
    return sorted(str(path.relative_to(root))
                  for pattern in patterns for path in root.glob(pattern))


def run(root: Path) -> dict:
    parent = verify(root)
    if (root / "artifacts/channel_function_v40_a4_final_opening.freeze.json").exists():
        raise RuntimeError("V40 final already opened")
    premature = premature_final_artifacts(root)
    if premature:
        raise RuntimeError(f"V40 independent-final outcome exists before opening: {premature}")
    analysis_paths = []
    reviews = {}
    for role in ("development", "validation"):
        reviews[role] = {}
        for key in ("Q", "F"):
            verify_stage(root, f"analysis_functional_{key}_{role}")
            verify_stage(root, f"split_sensitivity_{key}_{role}")
            path = f"results/v40/processed/analysis_functional_{key}_{role}_v40.json"
            data = json.loads((root / path).read_text(encoding="utf-8"))
            analysis_paths.append(path)
            sensitivity_path = f"results/v40/processed/split_sensitivity_{key}_{role}_v40.json"
            sensitivity = json.loads((root / sensitivity_path).read_text(encoding="utf-8"))
            analysis_paths.append(sensitivity_path)
            reviews[role][f"functional_{key}"] = {
                "sealed_states": data["sealed_states"],
                "complete_states": data["complete_states"],
                "qualified_states": data["qualified_states"],
                "baseline_qualification_gate": data["baseline_qualification_gate"],
                "split_mismatch_states": len(sensitivity["mismatched_state_ids"]),
                "strict_qualified_states": sensitivity["strict_qualified_states"],
                "strict_baseline_gate": sensitivity["strict_qwen_interpretability_gate"]}
        verify_stage(root, f"analysis_temporal_Q_{role}")
        tpath = f"results/v40/processed/analysis_temporal_Q_{role}_v40.json"
        tdata = json.loads((root / tpath).read_text(encoding="utf-8"))
        analysis_paths.append(tpath)
        reviews[role]["temporal_Q"] = {
            "sealed_states": tdata["sealed_states"],
            "common_qualified_states": tdata["common_qualified_states"]}
        verify_a5_stage(root, f"trajectory_analysis_Q_{role}")
        xpath = f"results/v40/processed/trajectory_analysis_Q_{role}_v40.json"
        xdata = json.loads((root / xpath).read_text(encoding="utf-8"))
        analysis_paths.append(xpath)
        reviews[role]["trajectory_Q"] = {
            "sealed_states": xdata["sealed_states"],
            "delay0_qualified_states": xdata["delays"]["0"]["qualified_states"],
            "delay0_three_way_gate": xdata["delays"]["0"]["task_grounded_three_way_gate"]}
    result = {
        "status": "OPENED_AFTER_DEVELOPMENT_AND_VALIDATION_REVIEW",
        "parent_a4_digest": parent["freeze_digest"],
        "reviews": reviews,
        "qwen_interpretability_gates_both_pass": all(
            reviews[role]["functional_Q"]["baseline_qualification_gate"]
            and reviews[role]["functional_Q"]["strict_baseline_gate"]
            for role in ("development", "validation")),
        "validation_outcomes_used_to_change_design_or_thresholds": False,
        "independent_final_outcomes_observed_before_opening": False,
        "final_interpretation_if_gate_fails":
            "execute the same sealed readout/conditions, report descriptive results only",
        "analysis_sha256": {path: sha256_file(root / path) for path in analysis_paths},
    }
    output = root / "results/v40/processed/final_opening_v40.json"
    if output.exists():
        raise RuntimeError("V40 final opening record already exists")
    write_json_atomic(output, result)
    stage_freeze(root, "final_opening", [SOURCE, str(output.relative_to(root)), *analysis_paths],
                 {"opened": True, "record_sha256": sha256_file(output),
                  "final_unobserved_at_opening": True})
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
