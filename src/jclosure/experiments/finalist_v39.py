"""Freeze one Falcon primitive mediator before any formal removal outcome."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/finalist_v39.py"


def run(root: Path) -> dict:
    verify_stage(root, "primitive_pilot_F")
    cfg = verify(root)["config"]
    pilot = json.loads((root / OUT / "primitive_pilot_F_v39.json").read_text())
    qualified = [name for name in cfg["candidate_families_order"]
                 if pilot["screening"].get(name, {}).get("screen_pass", False)]
    finalist = qualified[0] if qualified else None
    result = {"primary_finalist": finalist,
              "layer": pilot["screening"][finalist]["layer"] if finalist else None,
              "qualified_calibration_pilot_candidates": qualified,
              "selection_priority": cfg["candidate_families_order"],
              "calibration_pilot_only": True,
              "development_removal_outcomes_seen": False,
              "validation_outcomes_seen": False,
              "independent_final_outcomes_seen": False,
              "no_backup_candidate": True}
    path = root / OUT / "primitive_finalist_F_v39.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "primitive_finalist_F",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/interaction_genesis_v39_primitive_pilot_F.freeze.json"],
                        {"primary_finalist": finalist,
                         "development_removal_outcomes_seen": False,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
