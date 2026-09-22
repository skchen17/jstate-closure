"""Open both V34 independent finals only after the shared frozen mediator passes."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v34 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/final_opening_v34.py"


def run(root: Path) -> dict:
    verify_stage(root, "stage_analysis_development")
    verify_stage(root, "stage_analysis_validation")
    validation = json.loads((root / OUT / "stage_analysis_validation_v34.json").read_text())
    stages = validation["shared_stage_passes_development_and_validation"]
    if not stages:
        raise RuntimeError("V34 same stage did not pass both models and both roles; final remains sealed")
    # Stage order was frozen before any mediation responses. Never re-select on final.
    chosen = stages[0]
    record = {"opened": True, "both_models_simultaneously": True,
              "selected_stage": chosen, "qualifying_stages": stages,
              "reason": "same frozen stage passed development and validation in both models",
              "models": ["Q", "F"], "final_states_per_model": 40,
              "validation_outcomes_used_to_retune_stage_or_threshold": False,
              "historical_independent_finals_reopened": False}
    path = root / OUT / "final_opening_v34.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "final_opening",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/functional_mediation_v34_stage_analysis_development.freeze.json",
                         "artifacts/functional_mediation_v34_stage_analysis_validation.freeze.json"],
                        {"opened": True, "selected_stage": chosen,
                         "record_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
