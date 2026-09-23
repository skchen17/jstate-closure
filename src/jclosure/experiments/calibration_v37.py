"""Fresh V37 calibration: exact native replay and 3×3 equation equality.

Only the 20 presealed calibration states per model are observed here. Formal
development, validation and independent-final outcomes remain unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.local_v36 import _capture, _local_cache, _module
from jclosure.experiments.native_operator_v37 import predict_cell
from jclosure.experiments.operator_v36 import native_local
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, prefix, step
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/calibration_v37.py"
LETTERS = ("A", "B", "C")
STAGES = ("raw", "normalized", "mixer")


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, "design")
    design = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    panel = json.loads((root / OUT / "panel_v37.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for role in
               ("calibration", "development", "validation", "independent_final") for row in panel[role]}
    model, tokenizer = load(root, key)
    layers = list(design["local_layers"].values())
    rows, replay = [], []
    for n, item in enumerate(design["calibration"], 1):
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, prompts[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V37 calibration prefix/cache drift {key}:{sid}")
        caches = {letter: step(model, incoming, token, length)["cache"]
                  for letter, token in zip(LETTERS, (item["recipient_token_id"],
                                             item["donor_token_id"], item["third_token_id"]), strict=True)}
        probe = item["future_probe_tokens"][0]
        native = step(model, caches["A"], probe, length + 1, design["target_bundle"])
        with FunctionalIntervention(model, key) as hook:
            instrumented = step(model, caches["A"], probe, length + 1, design["target_bundle"])
        if not torch.equal(native["logits"], instrumented["logits"]):
            raise RuntimeError(f"V37 instrumental logits not bitwise {key}:{sid}")
        if field_hashes(native["cache"]) != field_hashes(instrumented["cache"]):
            raise RuntimeError(f"V37 instrumental cache not bitwise {key}:{sid}")
        for block in ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace"):
            if not np.array_equal(native["targets"][block], instrumented["targets"][block]):
                raise RuntimeError(f"V37 instrumental endpoint not bitwise {key}:{sid}:{block}")
        if len(hook.capture) != 24:
            raise RuntimeError(f"V37 incomplete recurrence capture {key}:{sid}")
        replay.append({"state_id": sid, "model": key, "bitwise_logits": True,
                       "bitwise_cache": True, "bitwise_endpoint_blocks": True,
                       "captured_recurrent_layers": len(hook.capture)})
        _, hidden, _, _ = _capture(model, key, caches["A"], probe, length,
                                   design["target_bundle"], layers)
        for position, layer in design["local_layers"].items():
            module = _module(model, key, layer)
            h = hidden[layer]
            for state in LETTERS:
                for operator in LETTERS:
                    work, proof = _local_cache(caches["A"], caches[state], caches[operator], layer)
                    predicted = predict_cell(key, module, h, work,
                                             caches[state].layers[layer].recurrent_states)
                    observed = native_local(module, h, work)
                    for stage in STAGES:
                        x, y = predicted[stage].float(), observed[stage].float()
                        if x.shape != y.shape:
                            raise RuntimeError(f"V37 equation shape mismatch {key}:{stage}:{x.shape}/{y.shape}")
                        max_error = float((x - y).abs().max().item())
                        tolerance = 0.02 + 0.03 * float(y.square().mean().sqrt().item())
                        if max_error > tolerance:
                            raise RuntimeError(f"V37 equation mismatch {key}:{sid}:{position}:{state}{operator}:"
                                               f"{stage}:{max_error}>{tolerance}")
                        rows.append({"state_id": sid, "model": key, "family": item["family"],
                                     "position": position, "layer": layer, "probe_index": 0,
                                     "state": state, "operator": operator, "stage": stage,
                                     "max_abs_error": max_error, "tolerance": tolerance,
                                     "predicted_shape": str(tuple(x.shape)),
                                     "exact_cache_writeback": proof["REC_requested"] == proof["REC_realized"]
                                                              and proof["Conv_requested"] == proof["Conv_realized"]})
        if n % 5 == 0 or n == len(design["calibration"]):
            print(f"V37 calibration {key} {n}/{len(design['calibration'])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"equality": root / OUT / f"calibration_equality_{key}_v37.parquet",
             "replay": root / OUT / f"calibration_replay_{key}_v37.parquet"}
    frame.to_parquet(files["equality"], index=False, compression="zstd")
    pd.DataFrame(replay).to_parquet(files["replay"], index=False, compression="zstd")
    summary = {"model": key, "states": len(replay), "positions": len(layers),
               "cells_per_position": 9, "stages": STAGES, "equality_rows": len(frame),
               "max_abs_error": float(frame.max_abs_error.max()),
               "all_within_tolerance": bool((frame.max_abs_error <= frame.tolerance).all()),
               "all_bitwise_instrumental_replays": True,
               "development_validation_final_outcomes_seen": False,
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    path = root / OUT / f"calibration_{key}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"calibration_{key}",
                        [SOURCE, "src/jclosure/experiments/native_operator_v37.py",
                         "src/jclosure/experiments/operator_v36.py",
                         str(path.relative_to(root)),
                         *[str(file.relative_to(root)) for file in files.values()]],
                        {"model": key, "summary_sha256": sha256_file(path),
                         "all_within_tolerance": summary["all_within_tolerance"],
                         "all_bitwise": True})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=("Q", "F"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model), indent=2))
