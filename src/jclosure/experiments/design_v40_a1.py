"""Freeze amended V40 task-grounded token-pair design without model forwards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from jclosure.experiments.design_v40 import encode, single_token_fork, tokenizer_for
from jclosure.protocol_v40 import verify as verify_initial
from jclosure.protocol_v40_a1 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v40/processed")
ROLES = ("calibration", "development", "validation", "independent_final")
SOURCE = "src/jclosure/experiments/design_v40_a1.py"


def run(root: Path, key: str) -> dict:
    verify_stage(root, "sample_pools")
    cfg = verify_initial(root)["config"]
    tokenizer = tokenizer_for(root, key)
    roles = {}
    for role in ROLES:
        verify_stage(root, f"pool_{role}")
        pool = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())
        rows = []
        for task in pool["items"]:
            recipient = encode(tokenizer, key, task["recipient_prompt"])
            donor = encode(tokenizer, key, task["donor_prompt"])
            check = single_token_fork(recipient, donor)
            if not check["eligible"]:
                raise RuntimeError(f"V40 fork ineligible {key}/{role}/{task['state_id']}: {check}")
            if task["recipient_answer"] == task["donor_answer"]:
                raise RuntimeError("V40 ground-truth contrast missing")
            rows.append({"state_id": task["state_id"], "role": role,
                         "family": task["family"], **check,
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["donor_answer"],
                         "labels_from_generator": True,
                         "intervention_outcome_observed": False})
        roles[role] = rows
    path = root / OUT / f"design_{key}_v40.json"
    spec = yaml.safe_load((root / cfg["model_spec_source"]).read_text(encoding="utf-8"))["models"][key]
    write_json_atomic(path, {"model_key": key, "model_id": spec["id"],
                             "roles": roles, "conditions": cfg["conditions"],
                             "delays": cfg["delays"],
                             "intervention_outcomes_observed_before_seal": False})
    seal = stage_freeze(root, f"design_{key}",
                        [SOURCE, str(path.relative_to(root)),
                         "artifacts/channel_function_v40_a1_sample_pools.freeze.json"],
                        {"model": key, "role_counts": {r: len(x) for r, x in roles.items()},
                         "design_sha256": sha256_file(path),
                         "intervention_outcomes_observed_before_seal": False})
    return {"model": key, "role_counts": {r: len(x) for r, x in roles.items()},
            "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
