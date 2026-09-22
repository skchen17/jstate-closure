"""Confirm the preselected V34 mediator on two untouched 40-state final panels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.stage_analysis_v34 import _stats
from jclosure.protocol_v34 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/final_analysis_v34.py"


def run(root: Path) -> dict:
    opening = verify_stage(root, "final_opening")
    verify_stage(root, "high_level_independent_final")
    high = json.loads((root / OUT / "high_level_independent_final_v34.json").read_text())
    selected = opening["selected_stage"]
    gate = verify(root)["config"]["stage_gate"]
    result = {"selected_stage": selected, "thresholds_retuned": False,
              "final_panel_opened_after_both_models_passed_development_and_validation": True,
              "high_level_final": high, "models": {}}
    inputs = [SOURCE, "artifacts/functional_mediation_v34_final_opening.freeze.json",
              "artifacts/functional_mediation_v34_high_level_independent_final.freeze.json"]
    for model in ("Q", "F"):
        verify_stage(root, f"mediation_{model}_independent_final")
        inputs.append(f"artifacts/functional_mediation_v34_mediation_{model}_independent_final.freeze.json")
        frame = pd.read_parquet(root / OUT / f"mediation_{model}_independent_final_v34.parquet")
        if frame.state_id.nunique() != 40:
            raise RuntimeError("V34 incomplete independent final")
        result["models"][model] = _stats(frame[frame.stage == selected], gate)
    result["shared_final_confirmed"] = bool(high["both_pass"] and all(x["pass"] for x in result["models"].values()))
    path = root / OUT / "final_analysis_v34.json"
    write_json_atomic(path, result)
    inputs.append(str(path.relative_to(root)))
    seal = stage_freeze(root, "final_analysis", inputs,
                        {"selected_stage": selected, "shared_final_confirmed": result["shared_final_confirmed"],
                         "result_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
