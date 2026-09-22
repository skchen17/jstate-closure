"""Conditional h1-only qualified-depth patches with natural h2/h4 propagation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import capture, patch, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import hd, load, signature, step
from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/horizons_v35.py"


def cosine(a, b):
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-12 else None


def trajectory(model, key, cache, tokens, length, bundle, first_patch=None):
    outputs = []
    current = cache
    proof = None
    for k, token in enumerate(tokens, 1):
        if k == 1 and first_patch is not None:
            references, source_by_layer = first_patch
            out, proof = patch(model, key, current, token, length, bundle,
                               references, source_by_layer)
        else:
            out = step(model, current, token, length + k, bundle)
        current = out["cache"]
        outputs.append({"targets": out["targets"]})
    return outputs, proof


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "final_opening")
    verify_stage(root, "control_plan")
    opening = json.loads((root / OUT / "final_opening_v35.json").read_text())
    if not opening["h2_h4_authorized"]:
        raise RuntimeError("V35 h2/h4 not authorized: no validated smaller/structured depth mechanism")
    plan = json.loads((root / OUT / "control_plan_v35.json").read_text())
    selected = set(plan["KV_states"])
    conditions = [name for name in opening["selected_conditions"] if name != "FULL"]
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    rows, audits = [], []
    for n, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in selected), 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        tokens = item["future_probe_tokens"][:4]
        _, conv_ref = capture(model, key, caches["conv"], tokens[0], length, bundle)
        _, joint_ref = capture(model, key, caches["joint"], tokens[0], length, bundle)
        refs = {"CONV": conv_ref, "JOINT": joint_ref}
        natural = {name: trajectory(model, key, caches[name.lower()], tokens, length, bundle)[0]
                   for name in ("CONV", "JOINT", "DONOR")}
        for condition in conditions:
            layers = design["condition_layers"][condition]
            paths = {}
            for direction, base, source in (("REMOVE", "joint", "CONV"),
                                            ("RESTORE", "conv", "JOINT")):
                outputs, proof = trajectory(model, key, caches[base], tokens, length, bundle,
                                            (refs, {layer: source for layer in layers}))
                paths[direction] = outputs
                audits.append({"model_key": key, "state_id": sid, "condition": condition,
                               "direction": direction, "horizon_patched": 1,
                               "requested_hash": proof["requested_hash"],
                               "realized_hash": proof["realized_hash"],
                               "source_map_hash": proof["source_map_hash"],
                               "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                               "exact_writeback": proof["exact_writeback"]})
            for horizon in (2, 4):
                ys = {name: signature([outputs[horizon - 1]], scales)
                      for name, outputs in {**natural, **paths}.items()}
                yd, yc, yj = ys["DONOR"], ys["CONV"], ys["JOINT"]
                ec, ej = float(np.linalg.norm(yd - yc)), float(np.linalg.norm(yd - yj))
                benefit = ec - ej
                rows.append({"model_key": key, "state_id": sid, "family": item["family"],
                             "condition": condition, "horizon": horizon,
                             "token_sequence_hash": hd(tokens), "benefit_positive": benefit > 1e-12,
                             "benefit_absolute": benefit,
                             "removed_fraction": (float(np.linalg.norm(yd - ys["REMOVE"])) - ej) / benefit
                             if benefit > 1e-12 else None,
                             "restored_fraction": (ec - float(np.linalg.norm(yd - ys["RESTORE"]))) / benefit
                             if benefit > 1e-12 else None,
                             "reverse_correction_cosine": cosine(ys["RESTORE"] - yc, yj - yc),
                             "h1_only_patch": True, "later_steps_unpatched": True,
                             "recipient_native_KV": True, "exact_writeback": True})
        print(f"V35 horizons {key} {n}/{len(selected)}", flush=True)
    frame, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    files = {"rows": root / OUT / f"horizons_{key}_validation_v35.parquet",
             "audit": root / OUT / f"horizons_audit_{key}_validation_v35.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    audit.to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model_key": key, "states": 5, "conditions": conditions,
               "horizons": [2, 4], "h1_only_patch": True, "later_steps_unpatched": True,
               "small_subset_secondary": True,
               "median": {f"{condition}:h{horizon}": {
                   "removed": float(part.removed_fraction.median()),
                   "restored": float(part.restored_fraction.median()),
                   "cosine": float(part.reverse_correction_cosine.median())}
                          for (condition, horizon), part in frame.groupby(["condition", "horizon"])},
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    summary_path = root / OUT / f"horizons_{key}_validation_v35.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"horizons_{key}", [SOURCE, str(summary_path.relative_to(root)),
                                                  *(str(path.relative_to(root)) for path in files.values()),
                                                  "artifacts/hierarchical_read_v35_final_opening.freeze.json",
                                                  "artifacts/hierarchical_read_v35_control_plan.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
