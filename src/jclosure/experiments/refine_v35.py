"""Prospectively authorized three-recurrent-layer windows within one coarse region."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import benefit_metrics, capture, patch, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import load, signature
from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/refine_v35.py"


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, "development_plan")
    verify_stage(root, f"full_gate_{role}")
    plan = json.loads((root / OUT / "development_plan_v35.json").read_text())
    all_windows = plan["model_refinement"][key]["three_layer_windows"]
    if not all_windows:
        raise RuntimeError(f"V35 refinement not authorized for {key}")
    if role == "validation":
        verify_stage(root, f"refinement_selection_{key}")
        selection = json.loads((root / OUT / f"refinement_selection_{key}_v35.json").read_text())
        chosen = selection["selected_window"]
        if chosen is None:
            raise RuntimeError(f"V35 no development-qualified refined window for {key}")
        windows = {chosen: all_windows[chosen]}
    else:
        windows = all_windows
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v35.parquet")
    facts = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v35.npz")["vectors"].astype(np.float64)
    by_state = {row.state_id: row for row in factorial.itertuples()}
    model, tokenizer = load(root, key)
    rows, audits, vectors = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        natural = {"CONV": [], "JOINT": []}
        outputs = {(name, direction): [] for name in windows for direction in ("REMOVE", "RESTORE")}
        for probe_index, probe in enumerate(item["future_probe_tokens"]):
            refs = {}
            for branch in ("CONV", "JOINT"):
                out, refs[branch] = capture(model, key, caches[branch.lower()], probe, length, bundle)
                natural[branch].append({"targets": out["targets"]})
            for name, layers in windows.items():
                for direction, base, source in (("REMOVE", "joint", "CONV"),
                                                ("RESTORE", "conv", "JOINT")):
                    out, proof = patch(model, key, caches[base], probe, length, bundle, refs,
                                       {layer: source for layer in layers})
                    outputs[(name, direction)].append({"targets": out["targets"]})
                    audits.append({"model_key": key, "role": role, "state_id": sid,
                                   "window": name, "direction": direction,
                                   "probe_index": probe_index, "probe_token_id": int(probe),
                                   "requested_hash": proof["requested_hash"],
                                   "realized_hash": proof["realized_hash"],
                                   "source_map_hash": proof["source_map_hash"],
                                   "recipient_KV_hash": caches["recipient_KV_hash"],
                                   "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                                   "exact_writeback": proof["exact_writeback"]})
        yconv, yjoint = [signature(natural[name], scales) for name in ("CONV", "JOINT")]
        fact = facts[int(by_state[sid].vector_index)]
        if not np.allclose(yconv, fact[2], atol=1e-5, rtol=1e-5) or not np.allclose(yjoint, fact[3], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"V35 refined factual replay mismatch {key}:{sid}")
        state_vectors = np.empty((len(windows), 2, len(yconv)), dtype=np.float32)
        for j, name in enumerate(windows):
            remove, restore = [signature(outputs[(name, direction)], scales)
                               for direction in ("REMOVE", "RESTORE")]
            state_vectors[j] = np.stack([remove, restore]).astype(np.float32)
            rows.append({"model_key": key, "role": role, "state_id": sid,
                         "family": item["family"], "condition": name,
                         "native_layers": json.dumps(windows[name]),
                         "window_index": j, "vector_index": len(vectors),
                         "recipient_native_KV": True, "six_frozen_probes": True,
                         "exact_writeback": True,
                         **benefit_metrics(fact[4], yconv, yjoint, remove, restore, by_state[sid].donor_norm)})
        vectors.append(state_vectors)
        if n % 5 == 0:
            print(f"V35 refine {key} {role} {n}/{len(design[role])}", flush=True)
    frame, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    files = {"rows": root / OUT / f"refine_{key}_{role}_v35.parquet",
             "vectors": root / OUT / f"refine_vectors_{key}_{role}_v35.npz",
             "audit": root / OUT / f"refine_audit_{key}_{role}_v35.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        state_ids=np.asarray([x["base_trial_id"] for x in design[role]], dtype=str),
                        windows=np.asarray(list(windows), dtype=str))
    audit.to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model_key": key, "role": role, "states": len(design[role]),
               "windows": windows, "all_exact_writeback": bool(audit.exact_writeback.all()),
               "median_removed": frame.groupby("condition").removed_fraction.median().to_dict(),
               "median_restored": frame.groupby("condition").restored_fraction.median().to_dict(),
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    summary_path = root / OUT / f"refine_{key}_{role}_v35.json"
    write_json_atomic(summary_path, summary)
    inputs = [SOURCE, "src/jclosure/experiments/read_hooks_v35.py",
              "src/jclosure/experiments/depth_common_v35.py",
              str(summary_path.relative_to(root)), *(str(path.relative_to(root)) for path in files.values()),
              "artifacts/hierarchical_read_v35_development_plan.freeze.json"]
    if role == "validation":
        inputs.append(f"artifacts/hierarchical_read_v35_refinement_selection_{key}.freeze.json")
    seal = stage_freeze(root, f"refine_{key}_{role}", inputs,
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
