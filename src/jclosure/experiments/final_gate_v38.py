"""Record the preregistered decision to keep independent final unopened."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v38 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT=Path("results/v38/processed")
SOURCE="src/jclosure/experiments/final_gate_v38.py"


def run(root:Path)->dict:
    verify_stage(root,"pool_independent_final")
    verify_stage(root,"composition_development")
    verdict=verify_stage(root,"composition_validation")
    if verdict["final_opening_eligible"]:
        raise RuntimeError("V38 final eligible; closed decision invalid")
    if (root/"artifacts/trajectory_composition_v38_final_opening.freeze.json").exists():
        raise RuntimeError("V38 independent final was opened")
    for key in ("Q","F"):
        for name in (f"high_level_{key}_independent_final_v38.json",
                     f"trajectory_full_{key}_independent_final_v38.npz"):
            if (root/OUT/name).exists():raise RuntimeError(f"V38 final outcome file exists: {name}")
    development=json.loads((root/OUT/"composition_development_v38.json").read_text())
    validation=json.loads((root/OUT/"composition_validation_v38.json").read_text())
    result={"independent_final_opened":False,"states_per_model_remain_sealed":40,
            "reason":"No single composition class passed development and validation in both models",
            "development_selected_class":development["selected_class"],
            "validation_selected_class":validation["selected_class"],
            "validation_final_opening_eligible":False}
    path=root/OUT/"final_gate_v38.json"
    write_json_atomic(path,result)
    seal=stage_freeze(root,"final_closed",[SOURCE,str(path.relative_to(root)),
                      "artifacts/trajectory_composition_v38_pool_independent_final.freeze.json",
                      "artifacts/trajectory_composition_v38_composition_validation.freeze.json"],
                      {"opened":False,"summary_sha256":sha256_file(path)})
    return {"freeze_digest":seal["freeze_digest"],**result}


if __name__=="__main__":
    print(json.dumps(run(Path.cwd()),indent=2))
