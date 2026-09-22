"""Machine-readable state/token/probe/depth/intervention and cumulative-profile hashes."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/profile_manifest_v35.py"


def run(root: Path):
    for name in ("depth_analysis_development", "depth_analysis_validation", "final_opening", "final_analysis"):
        verify_stage(root, name)
    cfg = verify(root)["config"]
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    opening = json.loads((root / OUT / "final_opening_v35.json").read_text())
    result = {"protocol": "hierarchical_read_v35", "profiles": {}, "models": {},
              "finalist_sha256": sha256_file(root / OUT / "final_opening_v35.json"),
              "final_opening_freeze_digest": verify_stage(root, "final_opening")["freeze_digest"],
              "final_analysis_freeze_digest": verify_stage(root, "final_analysis")["freeze_digest"]}
    for role in ("development", "validation"):
        analysis = json.loads((root / OUT / f"depth_analysis_{role}_v35.json").read_text())
        result["profiles"][role] = {}
        for key in ("Q", "F"):
            condition = analysis["models"][key]["conditions"]
            profile = {"REM_quartiles": [condition[q]["removed_fraction"] for q in ("Q1", "Q2", "Q3", "Q4")],
                       "REST_quartiles": [condition[q]["restored_fraction"] for q in ("Q1", "Q2", "Q3", "Q4")],
                       "REM_early": [condition[q]["removed_fraction"] for q in cfg["early_cumulative"]],
                       "REST_early": [condition[q]["restored_fraction"] for q in cfg["early_cumulative"]],
                       "REM_late": [condition[q]["removed_fraction"] for q in cfg["late_cumulative"]],
                       "REST_late": [condition[q]["restored_fraction"] for q in cfg["late_cumulative"]]}
            result["profiles"][role][key] = {"values": profile, "profile_sha256": hd(profile),
                                                "depth_stage_freeze_digest":
                                                verify_stage(root, f"depth_{key}_{role}")["freeze_digest"]}
    for key in ("Q", "F"):
        design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
        role_hashes = {}
        for role in ("calibration", "development", "validation", "independent_final"):
            entries = design[role]
            role_hashes[role] = {"state_ids_sha256": hd([x["base_trial_id"] for x in entries]),
                                 "token_pairs_sha256": hd([x["token_pair_hash"] for x in entries]),
                                 "future_probes_sha256": hd([x["future_probe_hash"] for x in entries])}
        result["models"][key] = {"role_hashes": role_hashes,
                                  "relative_depth_group_hash": design["relative_depth_group_hash"],
                                  "condition_layer_hash": design["condition_layer_hash"],
                                  "interface_freeze_digest": verify_stage(root, f"interface_{key}")["freeze_digest"],
                                  "model_revision": verify(root)["model_specs"][key]["revision"]}
    result["semantic_panel_state_hash"] = hd({role: [x["base_trial_id"] for x in panel[role]]
                                               for role in ("calibration", "development", "validation",
                                                            "independent_final")})
    path = root / OUT / "profile_manifest_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "profile_manifest",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/hierarchical_read_v35_final_analysis.freeze.json",
                         "artifacts/hierarchical_read_v35_depth_analysis_validation.freeze.json"],
                        {"manifest_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "manifest_sha256": sha256_file(path)}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
