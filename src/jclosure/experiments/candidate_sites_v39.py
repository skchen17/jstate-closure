"""Preregister factor-family sites from descriptive dev trace before pilot outcomes."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/candidate_sites_v39.py"
MAPPINGS = {
    "F1_POSTCONV_READ_FACTOR_C": ("C",),
    "F2_POSTCONV_UPDATE_FACTORS_X_B": ("X", "B"),
    "F3_DECAY_CONTROL_DT_DA": ("TRANSFORMED_CONTROL",),
    "F4_RECURRENT_STATE_INPUT_S": ("S",),
    "F5_RECURRENT_UPDATE_DBX": ("TRUE_UPDATE",),
    "F6_OUTPUT_GATE": ("OUTPUT_GATE",),
    "F7_COMBINED_RECURRENCE_INPUTS": ("S_PRIME",),
    "F8_RESIDUAL_INPUT_CONTEXT": ("MIXER_INPUT",),
}


def run(root: Path) -> dict:
    verify_stage(root, "internal_trace_F_development")
    cfg = verify(root)["config"]
    trace = json.loads((root / OUT / "internal_trace_F_development_v39.json").read_text())
    groups = json.loads((root / OUT / "design_F_v39.json").read_text())["relative_depth_layers"]
    eligible_layers = set(groups["Q2"] + groups["Q3"] + groups["Q4"])
    stable = {(int(row["layer"]), row["stage"])
              for row in trace["stable_next_layer_interactions"]}
    sites = {}
    for family in cfg["candidate_families_order"]:
        stages = MAPPINGS[family]
        matches = [layer for layer in sorted(eligible_layers)
                   if all((layer, stage) in stable for stage in stages)]
        sites[family] = {"layer": matches[0] if matches else None,
                         "trace_stages": stages,
                         "eligible": bool(matches),
                         "rule": "earliest Q2/Q3/Q4 layer with >=4/5 family prevalence >=0.75 at ratio >=0.20 and next-layer persistence"}
    result = {"candidate_order": cfg["candidate_families_order"],
              "sites": sites, "selection_uses_validation": False,
              "selection_uses_formal_removal_outcomes": False,
              "pilot_pool": "calibration", "pilot_probe_index": 0,
              "pilot_gate": cfg["necessity_gate"],
              "quantization": "float32 factor prediction then cast to native factor dtype; requested/realized hashes audited",
              "primary_finalist_rule": "earliest listed eligible factor passing pilot removal gate; freeze before development removal"}
    path = root / OUT / "candidate_site_plan_v39.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "candidate_site_plan",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/interaction_genesis_v39_internal_trace_F_development.freeze.json"],
                        {"summary_sha256": sha256_file(path),
                         "selection_uses_validation": False})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
