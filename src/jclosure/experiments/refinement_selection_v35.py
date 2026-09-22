"""Freeze one development-qualified three-layer finalist per authorized model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.depth_analysis_v35 import stage_gate
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/refinement_selection_v35.py"


def run(root: Path, key: str):
    verify_stage(root, "development_plan")
    verify_stage(root, f"refine_{key}_development")
    cfg = verify(root)["config"]
    plan = json.loads((root / OUT / "development_plan_v35.json").read_text())
    windows = plan["model_refinement"][key]["three_layer_windows"]
    frame = pd.read_parquet(root / OUT / f"refine_{key}_development_v35.parquet")
    scores = {name: stage_gate(frame, name, cfg["quartile_gate"]) for name in windows}
    selected = next((name for name in windows if scores[name]["pass"]), None)
    result = {"model_key": key, "selected_window": selected,
              "development_window_priority": list(windows), "scores": scores,
              "validation_responses_seen": False,
              "validation_tests_only_selected_window": True}
    path = root / OUT / f"refinement_selection_{key}_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"refinement_selection_{key}",
                        [SOURCE, str(path.relative_to(root)),
                         f"artifacts/hierarchical_read_v35_refine_{key}_development.freeze.json"],
                        {"model_key": key, "selected_window": selected,
                         "summary_sha256": sha256_file(path), "validation_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
