"""Assemble four presealed V38 pools without observing model responses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.experiments.runtime_v34 import hd
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic


OUT = Path("results/v38/processed")
SOURCE = "src/jclosure/experiments/panel_v38.py"
ROLES = ("calibration", "development", "validation", "independent_final")


def run(root: Path) -> dict:
    verify_stage(root, "sample_pools")
    cfg = verify(root)["config"]
    roles = {}
    ids, programs, prompts = set(), set(), set()
    for role in ROLES:
        verify_stage(root, f"pool_{role}")
        pool = json.loads((root / cfg["panel_source"].format(role=role)).read_text())
        rows = pool["items"]
        expected = len(cfg["families"]) * int(cfg["roles_per_family"][role])
        if len(rows) != expected:
            raise RuntimeError(f"V38 {role} pool count mismatch")
        for family in cfg["families"]:
            if sum(row["family"] == family for row in rows) != cfg["roles_per_family"][role]:
                raise RuntimeError(f"V38 {role}/{family} count mismatch")
        for row in rows:
            prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
            if row["prompt_sha256"] != prompt_hash or row["role"] != role:
                raise RuntimeError(f"V38 {role} prompt/role drift")
            for value, bucket in ((row["base_trial_id"], ids),
                                  (row["program_hash"], programs), (prompt_hash, prompts)):
                if value in bucket:
                    raise RuntimeError(f"V38 role-pool overlap: {role}:{value}")
                bucket.add(value)
        roles[role] = rows
    if len(ids) != 180:
        raise RuntimeError("V38 panel must contain 180 distinct states")
    result = {**roles, "source": "new_v38_generated_task_pools",
              "v18_train_states_in_formal_roles": 0,
              "historical_v1_v37_states_in_formal_roles": 0,
              "modular_validation_final_horizon": 3,
              "modular_calibration_development_horizon": 2,
              "role_hashes": {role: hd([row["base_trial_id"] for row in roles[role]]) for role in ROLES},
              "both_models_share_semantic_state_ids": True,
              "all_model_responses_unobserved_at_panel_freeze": True}
    path = root / OUT / "panel_v38.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, result)
    seal = stage_freeze(root, "panel", [SOURCE, str(path.relative_to(root)),
                                        "artifacts/trajectory_composition_v38_sample_pools.freeze.json"],
                        {"panel_hash": hd(result), "role_hashes": result["role_hashes"],
                         "all_model_responses_unobserved_at_panel_freeze": True})
    return {"freeze_digest": seal["freeze_digest"],
            "roles": {role: len(roles[role]) for role in ROLES}}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
