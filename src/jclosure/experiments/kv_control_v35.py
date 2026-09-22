"""Frozen five-state donor-KV sensitivity check for the V35 selected read-depth profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import benefit_metrics, capture, patch, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import load, native_swap, signature
from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/kv_control_v35.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "final_opening")
    verify_stage(root, "control_plan")
    opening = json.loads((root / OUT / "final_opening_v35.json").read_text())
    conditions = opening["selected_conditions"]
    plan = json.loads((root / OUT / "control_plan_v35.json").read_text())
    selected = set(plan["KV_states"])
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_validation_v35.parquet")
    facts = np.load(root / OUT / f"factorial_vectors_{key}_validation_v35.npz")["vectors"].astype(np.float64)
    by_state = {row.state_id: row for row in factorial.itertuples()}
    model, tokenizer = load(root, key)
    rows, audits = [], []
    for n, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in selected), 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        conv_kv, _ = native_swap(caches["recipient"], caches["donor"], ["Conv", "KV"], key)
        joint_kv, _ = native_swap(caches["recipient"], caches["donor"], ["REC", "Conv", "KV"], key)
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        natural = {"CONV": [], "JOINT": []}
        outputs = {(condition, direction): [] for condition in conditions
                   for direction in ("REMOVE", "RESTORE")}
        for probe_index, probe in enumerate(item["future_probe_tokens"]):
            refs = {}
            for name, cache in (("CONV", conv_kv), ("JOINT", joint_kv)):
                out, refs[name] = capture(model, key, cache, probe, length, bundle)
                natural[name].append({"targets": out["targets"]})
            for condition in conditions:
                for direction, base, source in (("REMOVE", joint_kv, "CONV"),
                                                ("RESTORE", conv_kv, "JOINT")):
                    out, proof = patch(model, key, base, probe, length, bundle, refs,
                                       {layer: source for layer in design["condition_layers"][condition]})
                    outputs[(condition, direction)].append({"targets": out["targets"]})
                    audits.append({"model_key": key, "state_id": sid, "condition": condition,
                                   "direction": direction, "probe_index": probe_index,
                                   "probe_token_id": int(probe), "donor_KV": True,
                                   "requested_hash": proof["requested_hash"],
                                   "realized_hash": proof["realized_hash"],
                                   "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                                   "exact_writeback": proof["exact_writeback"]})
        yconv, yjoint = [signature(natural[name], scales) for name in ("CONV", "JOINT")]
        fact = facts[int(by_state[sid].vector_index)]
        for condition in conditions:
            remove, restore = [signature(outputs[(condition, direction)], scales)
                               for direction in ("REMOVE", "RESTORE")]
            rows.append({"model_key": key, "state_id": sid, "family": item["family"],
                         "condition": condition, "donor_KV": True, "recipient_native_KV": False,
                         "six_frozen_probes": True, "exact_writeback": True,
                         **benefit_metrics(fact[4], yconv, yjoint, remove, restore,
                                           by_state[sid].donor_norm)})
        print(f"V35 KV control {key} {n}/{len(selected)}", flush=True)
    frame, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    files = {"rows": root / OUT / f"kv_control_{key}_validation_v35.parquet",
             "audit": root / OUT / f"kv_control_audit_{key}_validation_v35.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    audit.to_parquet(files["audit"], index=False, compression="zstd")
    result = {"model_key": key, "states": 5, "conditions": conditions,
              "donor_KV_secondary": True, "primary_inference_remains_recipient_native_KV": True,
              "median": {name: {"removed": float(part.removed_fraction.median()),
                                "restored": float(part.restored_fraction.median()),
                                "cosine": float(part.reverse_correction_cosine.median())}
                         for name, part in frame.groupby("condition")},
              "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    path = root / OUT / f"kv_control_{key}_validation_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"kv_control_{key}", [SOURCE, str(path.relative_to(root)),
                                                    *(str(path.relative_to(root)) for path in files.values()),
                                                    "artifacts/hierarchical_read_v35_final_opening.freeze.json",
                                                    "artifacts/hierarchical_read_v35_control_plan.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
