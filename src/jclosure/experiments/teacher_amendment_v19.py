"""Freeze missing clean h8 teacher sequences before V19 factorial responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.protocol_v19 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/teacher_amendment_v19.py"
TEACHERS = bank.OUT / "teacher_tokens_v19.parquet"


def run(root: Path) -> dict:
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    bundle, dense, _, _, rec, att, measured, state_layer, ws_layers, ws_count = bank._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    rows = []
    for role in ("train", "validation"):
        for number, item in enumerate(split[role], 1):
            old = bank._state_metadata(root, item)
            needed = 8 if item["horizon_panel"] else 1
            existing = [int(x) for x in old["teacher_tokens_h8_or_h1"]]
            if len(existing) >= needed:
                tokens = existing[:needed]
                source = "frozen_V18_clean_teacher"
            else:
                clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
                tokens, _ = bank._trajectory(bundle, dense, clean["cache"], clean["logits"], None,
                                              needed, clean["prompt_length"], jids, lids,
                                              ws_layers, ws_count, max(measured))
                if existing and tokens[:len(existing)] != existing:
                    raise RuntimeError(f"V19 clean teacher prefix differs from V18: {item['base_trial_id']}")
                source = "new_V19_clean_greedy_same_state_prefix_verified"
            rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"], "role": role,
                         "horizon_panel": bool(item["horizon_panel"]), "teacher_tokens": tokens,
                         "teacher_source": source,
                         "teacher_sha256": hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()})
            if number % 50 == 0 or number == len(split[role]):
                print(f"V19 teacher {role} {number}/{len(split[role])}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / TEACHERS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    result = stage_freeze(root, "teacher_amendment_3",
                          [SOURCE, str(TEACHERS), "artifacts/counterfactual_workspace_v19_splits.freeze.json",
                           "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json"],
                          {"why": "V19 horizon subset was chosen independently of V18 horizon subset, so some V18 clean teacher records contain only h1. Generate missing h8 clean-greedy tokens and verify their V18 h1 prefix before any V19 factorial response.",
                           "when": "after first pilot failed before any factorial response was written",
                           "responses_already_observed": {"q_calibration_initial": 480, "q_amendment_1_calibration": 120,
                                                          "q_amendment_2_calibration": 60, "factorial_train": 0,
                                                          "factorial_validation": 0},
                           "prior_q_amendment_digest": q["freeze_digest"],
                           "teacher_path": str(TEACHERS), "teacher_sha256": sha256_file(path),
                           "state_count": len(frame), "new_h8_count": int((frame.teacher_source != "frozen_V18_clean_teacher").sum()),
                           "h1_prefix_exact_all": True})
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
