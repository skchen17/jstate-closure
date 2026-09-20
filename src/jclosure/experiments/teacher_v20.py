"""Freeze one clean greedy teacher token for each V20 opened base state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as bank
from jclosure.protocol_v20 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file

SOURCE = "src/jclosure/experiments/teacher_v20.py"
TARGET = bank.OUT / "teacher_tokens_v20.parquet"
ROLES = ("independent_v19_confirmation", "q_calibration", "operator_train", "operator_validation")


def prepare(root: Path) -> dict:
    split = verify_stage(root, "splits")
    prompts = bank.prompt_index(root)
    bundle, dense, _, _, _, _, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    rows = []
    for role in ROLES:
        for number, item in enumerate(split[role], 1):
            prompt = str(prompts[item["base_trial_id"]]["prompt"])
            clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
            tokens, _ = v19._trajectory(bundle, dense, clean["cache"], clean["logits"], None,
                                        1, clean["prompt_length"], jids, lids, ws_layers,
                                        ws_count, max(measured))
            if len(tokens) != 1:
                raise RuntimeError(f"V20 clean teacher length mismatch: {item['base_trial_id']}")
            rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                         "role": role, "teacher_tokens": [int(tokens[0])],
                         "teacher_sha256": hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest(),
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            if number % 50 == 0 or number == len(split[role]):
                print(f"V20 teacher {role} {number}/{len(split[role])}", flush=True)
    path = root / TARGET
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False, compression="zstd")
    return stage_freeze(root, "teacher", [SOURCE, str(TARGET),
                                           "artifacts/compact_causal_response_operator_v20_splits.freeze.json"],
                        {"teacher_path": str(TARGET), "teacher_sha256": sha256_file(path),
                         "state_count": len(frame), "role_counts": frame.role.value_counts().to_dict(),
                         "teacher_policy": "one_new_V20_clean_greedy_token_per_base_frozen_before_any_q_or_a_response",
                         "responses_already_observed": 0,
                         "reserved_independent_V20_final_opened": False})


if __name__ == "__main__":
    result = prepare(Path.cwd())
    print(json.dumps({"freeze_digest": result["freeze_digest"], "state_count": result["state_count"]}, indent=2))
