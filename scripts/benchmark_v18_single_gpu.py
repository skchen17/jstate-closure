#!/usr/bin/env python3
"""Read-only benchmark of one complete V18 state on GPU 0."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import actuation_v15 as act
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.crossed_bank_v18 import SCRATCH, _prefill_history, _split, _trajectory
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.model import load_model_bundle


def single_loader(config):
    modified = copy.deepcopy(config)
    modified["model"]["device_map"] = {"": 0}
    modified["model"].pop("max_memory", None)
    modified["model"].pop("offload_folder", None)
    return load_model_bundle(modified)


def main():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    root = Path.cwd()
    split = _split(root)
    item = next(row for row in split["train"] if not row["horizon_panel"])
    act._load_model = single_loader
    start = time.monotonic()
    bundle, dense_map, v13, metadata, values = act.load_context(root)
    loaded = time.monotonic()
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    rec = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    workspace_layers = [int(x) for x in jvp["workspace_layers"]]
    workspace_count = int(jvp["selected_workspace_count"])
    state_layer, response_layer = int(v8["intervention"]["layer"]), max(measured)
    task = values["tasks"][str(metadata[str(item["base_trial_id"])]["prompt_id"])]
    clean = _prefill_history(bundle, task.prompt, measured, dense_map, state_layer)
    selected_j = np.asarray(split["selected_j"], dtype=int)
    selected_logits = np.asarray(split["selected_logits"], dtype=int)
    tokens, baseline = _trajectory(bundle, dense_map, clean["cache"], clean["logits"], None,
                                   1, clean["prompt_length"], selected_j, selected_logits,
                                   workspace_layers, workspace_count, response_layer)
    pref = time.monotonic()
    responses = []
    for action in split["actions"]:
        for sign in (-1, 1):
            direction = int(action["direction_index"])
            vector = {name: values["directions"][name][direction].float() * float(0.5 * sign)
                      for name in ("recurrent", "conv", "kv")}
            edited = apply(clean["cache"], 1.0, vector, rec, att, "native_fp32_add_bf16_writeback")
            _readback(clean["cache"], edited, vector, rec, att)
            _, endpoint = _trajectory(bundle, dense_map, edited, None, tokens, 1, clean["prompt_length"],
                                      selected_j, selected_logits, workspace_layers, workspace_count, response_layer)
            delta = {target: (endpoint[1][target] - baseline[1][target]).astype(np.float32)
                     for target in TARGETS}
            responses.append((int(action["coordinate_index"]), sign,
                              stack(delta, split["target_scales"]).astype(np.float32)))
    finished = time.monotonic()
    prior_path = SCRATCH / "train" / f'response_{item["base_trial_id"]}.parquet'
    comparison = {}
    if prior_path.exists():
        prior = pd.read_parquet(prior_path)
        prior = prior[(prior.horizon == 1) & (prior.alpha == 0.5)]
        old = {(int(row.coordinate_index), int(row.sign)): np.asarray(row.response_stacked_normalized, dtype=np.float32)
               for row in prior.itertuples()}
        drifts = [np.linalg.norm(value - old[(coordinate, sign)]) / max(np.linalg.norm(old[(coordinate, sign)]), 1e-12)
                  for coordinate, sign, value in responses]
        comparison = {"pilot_response_median_rel_l2": float(np.median(drifts)),
                      "pilot_response_max_rel_l2": float(np.max(drifts))}
    print(json.dumps({"model_load_sec": loaded-start, "prefill_baseline_sec": pref-loaded,
                      "sixteen_actions_sec": finished-pref, "total_state_sec": finished-loaded,
                      "gpu0_peak_allocated_gib": torch.cuda.max_memory_allocated(0)/2**30,
                      **comparison}, indent=2))


if __name__ == "__main__":
    main()
