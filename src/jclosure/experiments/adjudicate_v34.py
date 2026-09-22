"""Append-only V34 formal outcomes from frozen development, validation and final records."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v34 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/adjudicate_v34.py"


def run(root: Path) -> dict:
    for name in ("stage_analysis_development", "stage_analysis_validation", "final_opening", "final_analysis"):
        verify_stage(root, name)
    dev = json.loads((root / OUT / "stage_analysis_development_v34.json").read_text())
    val = json.loads((root / OUT / "stage_analysis_validation_v34.json").read_text())
    final = json.loads((root / OUT / "final_analysis_v34.json").read_text())
    stage = val["shared_stage_passes_development_and_validation"]
    shared = bool(stage and final["shared_final_confirmed"])
    selected = final["selected_stage"] if shared else None
    different = (not shared and all(val["model_stage_passes_both_roles"].values()) and
                 not set(val["model_stage_passes_both_roles"]["Q"]) & set(val["model_stage_passes_both_roles"]["F"]))
    outcomes = {
        "V34-A_SHARED_FUNCTIONAL_MEDIATOR_IDENTIFIED": shared,
        "V34-B_SHARED_UPDATE_STAGE_MEDIATION": shared and selected == "TRUE_UPDATE",
        "V34-C_SHARED_GATE_DECAY_MEDIATION": shared and selected == "TRANSFORMED_CONTROL",
        "V34-D_SHARED_POSTCONV_INPUT_MEDIATION": shared and selected == "POSTCONV_INPUT",
        "V34-E_SHARED_RECURRENT_READ_MEDIATION": shared and selected == "RECURRENT_READ",
        "V34-F_SHARED_RESIDUAL_INTEGRATION_MEDIATION": False,
        "V34-G_SHARED_DISTRIBUTED_PIPELINE_MEDIATION": False,
        "V34-H_DIFFERENT_MICRO_MEDIATORS_SAME_HIGH_LEVEL_EFFECT": bool(different),
        "V34-I_NO_TESTED_MICRO_MEDIATOR_IDENTIFIED": bool(not shared and not different),
    }
    if not dev["high_level_reconfirmed"] or not val["high_level_reconfirmed"]:
        raise RuntimeError("V34 high-level reconfirmation absent")
    if shared and selected != "RECURRENT_READ":
        raise RuntimeError("V34 selected stage not reviewed for current adjudication")
    if val["pipeline_testing_authorized"]:
        raise RuntimeError("V34 predeclared pipelines authorized but not executed")
    record = {"formal_outcomes": outcomes, "selected_shared_stage": selected,
              "shared_stage_or_pipeline_passes_both_roles": bool(stage),
              "shared_final_confirmed": final["shared_final_confirmed"],
              "independent_final_opened": True, "independent_final_states_per_model": 40,
              "historical_independent_finals_reopened": False,
              "pipeline_status": "NOT_RUN_SINGLE_STAGE_ALREADY_PASSED",
              "overlap_status": "NOT_RUN_ONLY_ONE_INDEPENDENT_STAGE_QUALIFIED",
              "residual_integration_status": "ALIASED_WITH_MIXER_OUTPUT_NOT_SEPARATELY_IDENTIFIABLE",
              "stage5_independent_mediator_claim": False,
              "application_experiments_authorized": bool(outcomes["V34-A_SHARED_FUNCTIONAL_MEDIATOR_IDENTIFIED"] or outcomes["V34-G_SHARED_DISTRIBUTED_PIPELINE_MEDIATION"]),
              "third_model_replication_authorized": bool(outcomes["V34-A_SHARED_FUNCTIONAL_MEDIATOR_IDENTIFIED"] or outcomes["V34-G_SHARED_DISTRIBUTED_PIPELINE_MEDIATION"] or outcomes["V34-H_DIFFERENT_MICRO_MEDIATORS_SAME_HIGH_LEVEL_EFFECT"]),
              "practical_application_benefit_established": False,
              "universal_mechanism_claim": False,
              "read_cut_qualification": "all-layer read-output replacement is a broad causal cut; it does not uniquely identify the upstream algorithm or independent residual site",
              "base_freeze_digest": verify(root)["freeze_digest"],
              "development_stage_freeze_digest": verify_stage(root, "stage_analysis_development")["freeze_digest"],
              "validation_stage_freeze_digest": verify_stage(root, "stage_analysis_validation")["freeze_digest"],
              "final_analysis_freeze_digest": verify_stage(root, "final_analysis")["freeze_digest"]}
    path = root / OUT / "v34_adjudication.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "adjudication",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/functional_mediation_v34_stage_analysis_development.freeze.json",
                         "artifacts/functional_mediation_v34_stage_analysis_validation.freeze.json",
                         "artifacts/functional_mediation_v34_final_analysis.freeze.json"],
                        {"selected_stage": selected, "formal_outcomes": outcomes,
                         "record_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **record}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
