#!/usr/bin/env python3
"""Read-only V13/V18 clean-greedy teacher-token comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.crossed_bank_v18 import _prefill_history, _split, _trajectory, load_context_v18


def main() -> None:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    root = Path.cwd()
    split = _split(root)
    bundle, dense_map, v13, metadata, values = load_context_v18(root)
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    rows = []
    for item in split["validation"][:5]:
        key = str(item["base_trial_id"])
        pair = metadata[key]
        task = values["tasks"][str(pair["prompt_id"])]
        clean = _prefill_history(bundle, task.prompt, measured, dense_map, int(v8["intervention"]["layer"]))
        tokens, _ = _trajectory(bundle, dense_map, clean["cache"], clean["logits"], None, 8,
                                clean["prompt_length"], np.asarray(split["selected_j"], dtype=int),
                                np.asarray(split["selected_logits"], dtype=int),
                                [int(x) for x in jvp["workspace_layers"]],
                                int(jvp["selected_workspace_count"]), max(measured))
        prior = [int(x) for x in pair.get("teacher_tokens", [])]
        first = next((i for i, (a, b) in enumerate(zip(tokens, prior)) if a != b), None)
        rows.append({"base_trial_id": key, "v13_teacher_tokens": prior, "v18_clean_greedy_tokens": tokens,
                     "first_mismatch_zero_based": first, "matched_first_token": tokens[0] == prior[0]})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
