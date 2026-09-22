"""Reconfirm V34 full-depth read mediation on each fresh V35 panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import benefit_metrics, capture, patch, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import hd, load, signature
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/full_read_v35.py"


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    verify_stage(root, f"factorial_{key}_{role}")
    verify_stage(root, f"high_level_{'development' if role == 'development' else role}")
    if role == "validation":
        verify_stage(root, "full_gate_development")
    if role == "independent_final":
        verify_stage(root, "final_opening")
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v35.parquet")
    factual = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v35.npz")["vectors"].astype(np.float64)
    by_state = {row.state_id: row for row in factorial.itertuples()}
    model, tokenizer = load(root, key)
    full = design["condition_layers"]["FULL"]
    rows, arrays, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        natural = {"CONV": [], "JOINT": []}
        patched = {"REMOVE": [], "RESTORE": []}
        for i, probe in enumerate(item["future_probe_tokens"]):
            for name, cache in (("CONV", caches["conv"]), ("JOINT", caches["joint"])):
                out, ref = capture(model, key, cache, probe, length, bundle)
                natural[name].append(out)
                if name == "CONV": conv_ref = ref
                else: joint_ref = ref
            refs = {"CONV": conv_ref, "JOINT": joint_ref}
            for direction, base, source in (("REMOVE", caches["joint"], "CONV"),
                                            ("RESTORE", caches["conv"], "JOINT")):
                out, proof = patch(model, key, base, probe, length, bundle, refs,
                                   {layer: source for layer in full})
                patched[direction].append(out)
                audits.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                               "probe_index": i, "probe_token_id": int(probe), "direction": direction,
                               "recipient_KV_hash": caches["recipient_KV_hash"],
                               "requested_hash": proof["requested_hash"],
                               "realized_hash": proof["realized_hash"],
                               "native_hash": proof["native_hash"],
                               "source_map_hash": proof["source_map_hash"],
                               "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                               "layers": proof["layers"], "exact_writeback": True})
        yconv, yjoint = (signature(natural[name], scales) for name in ("CONV", "JOINT"))
        previous = factual[int(by_state[sid].vector_index)]
        if not np.allclose(yconv, previous[2], rtol=1e-5, atol=1e-5) or not np.allclose(yjoint, previous[3], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"V35 factual replay mismatch {key}:{sid}")
        remove, restore = (signature(patched[direction], scales) for direction in ("REMOVE", "RESTORE"))
        metrics = benefit_metrics(previous[4], yconv, yjoint, remove, restore,
                                  by_state[sid].donor_norm)
        rows.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                     "condition": "FULL", "vector_index": len(arrays),
                     "token_pair_hash": item["token_pair_hash"], "probe_hash": item["future_probe_hash"],
                     "depth_group_hash": design["relative_depth_group_hash"],
                     "condition_layer_hash": design["condition_layer_hash"],
                     "recipient_native_KV": True, "six_frozen_probes": True,
                     "exact_writeback": True, **metrics})
        arrays.append(np.stack([remove, restore]).astype(np.float32))
        if n % 5 == 0 or n == len(design[role]):
            print(f"V35 full read {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"full_read_{key}_{role}_v35.parquet",
             "vectors": root / OUT / f"full_read_vectors_{key}_{role}_v35.npz",
             "audit": root / OUT / f"full_read_audit_{key}_{role}_v35.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(arrays),
                        state_ids=frame.state_id.to_numpy(str), directions=np.asarray(["REMOVE", "RESTORE"]))
    pd.DataFrame(audits).to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model_key": key, "role": role, "states": len(frame),
               "median_removed": float(frame.removed_fraction.median()),
               "median_restored": float(frame.restored_fraction.median()),
               "median_cosine": float(frame.reverse_correction_cosine.median()),
               "exact_writeback": True,
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    path = root / OUT / f"full_read_{key}_{role}_v35.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"full_read_{key}_{role}",
                        [SOURCE, "src/jclosure/experiments/depth_common_v35.py",
                         "src/jclosure/experiments/read_hooks_v35.py",
                         *(str(p.relative_to(root)) for p in files.values()), str(path.relative_to(root)),
                         f"artifacts/hierarchical_read_v35_factorial_{key}_{role}.freeze.json"],
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(path),
                         "all_exact_writeback": True})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
