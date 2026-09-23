"""Seal the V36 independent-final opening decision without inspecting its responses."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/final_opening_v36.py"


def run(root: Path):
    verify_stage(root, "local_analysis_development")
    verify_stage(root, "local_analysis_validation")
    cfg = verify(root)["config"]
    dev = json.loads((root / OUT / "local_analysis_development_v36.json").read_text())
    val = json.loads((root / OUT / "local_analysis_validation_v36.json").read_text())
    common = (dev["both_models_same_class_operator_and_causal_pass"] and
              val["both_models_same_class_operator_and_causal_pass"])
    selected = "READ_OPERATOR_MATCHING" if common else None
    if selected is not None and selected not in cfg["finalists"]:
        raise RuntimeError("V36 finalist not frozen")
    record = {"opened": bool(common), "selected_mechanism": selected,
              "development_both_pass": dev["both_models_same_class_operator_and_causal_pass"],
              "validation_both_pass": val["both_models_same_class_operator_and_causal_pass"],
              "same_mechanism_class_required": True,
              "independent_final_states_per_model": 40,
              "independent_final_responses_seen": False,
              "composition_authorized": bool(common),
              "task_grounded_authorized": bool(common),
              "historical_finals_reopened": False}
    path = root / OUT / "final_opening_v36.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "final_opening", [SOURCE, str(path.relative_to(root)),
                        "artifacts/computational_origin_v36_local_analysis_development.freeze.json",
                        "artifacts/computational_origin_v36_local_analysis_validation.freeze.json"],
                        {"opened": bool(common), "selected_mechanism": selected,
                         "summary_sha256": sha256_file(path),
                         "independent_final_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
