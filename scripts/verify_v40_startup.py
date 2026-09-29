"""Machine-auditable V40 startup seal and cross-model pairing check."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v40_a1 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

ROLES = ("calibration", "development", "validation", "independent_final")


def main(root: Path) -> dict:
    verify_stage(root, "sample_pools")
    for key in ("Q", "F"):
        verify_stage(root, f"design_{key}")
    designs = {key: json.loads((root / f"results/v40/processed/design_{key}_v40.json").read_text())
               for key in ("Q", "F")}
    manifest = json.loads((root / "data/v40/sample_pool_manifest_v40.json").read_text())
    if not manifest["models_not_queried"] or not manifest["all_role_programs_and_prompts_disjoint"]:
        raise RuntimeError("V40 startup manifest failure")
    all_ids, all_programs, all_prompts = set(), set(), set()
    roles = {}
    for role in ROLES:
        verify_stage(root, f"pool_{role}")
        pool = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())
        rows = pool["items"]
        ids = [x["state_id"] for x in rows]
        if ids != [x["state_id"] for x in designs["Q"]["roles"][role]] or \
           ids != [x["state_id"] for x in designs["F"]["roles"][role]]:
            raise RuntimeError(f"V40 cross-model state pairing drift: {role}")
        if any(not x["labels_generated_before_model_execution"] or
               x["recipient_answer"] == x["donor_answer"] for x in rows):
            raise RuntimeError(f"V40 external label failure: {role}")
        for x in rows:
            if x["state_id"] in all_ids or x["program_hash"] in all_programs:
                raise RuntimeError("V40 duplicate state/program")
            all_ids.add(x["state_id"])
            all_programs.add(x["program_hash"])
            for field in ("recipient_prompt_sha256", "donor_prompt_sha256"):
                if x[field] in all_prompts:
                    raise RuntimeError("V40 duplicate prompt")
                all_prompts.add(x[field])
        for key in ("Q", "F"):
            if any(not x["eligible"] or x["intervention_outcome_observed"]
                   for x in designs[key]["roles"][role]):
                raise RuntimeError(f"V40 design eligibility/outcome flag: {key}/{role}")
        roles[role] = {"states": len(rows), "family_counts": pool["family_counts"],
                       "pool_sha256": sha256_file(root / f"data/v40/{role}_pool_v40.json")}
    output = {"status": "STARTUP_SEALED_NO_INTERVENTION_OUTCOMES",
              "roles": roles, "total_distinct_states": len(all_ids),
              "total_distinct_prompts": len(all_prompts),
              "models": ["Q", "F"], "all_model_specific_forks_eligible": True,
              "validation_outcomes_observed": False,
              "independent_final_outcomes_observed": False}
    path = root / "results/v40/processed/startup_audit_v40.json"
    write_json_atomic(path, output)
    stage_freeze(root, "startup_audit", ["scripts/verify_v40_startup.py",
                                        str(path.relative_to(root)),
                                        *[f"artifacts/channel_function_v40_a1_design_{key}.freeze.json"
                                          for key in ("Q", "F")]],
                 {"audit_sha256": sha256_file(path),
                  "formal_intervention_outcomes_observed": False})
    return output


if __name__ == "__main__":
    print(json.dumps(main(Path.cwd()), indent=2))
