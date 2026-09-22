"""Execute only the V35 development-frozen finalist conditions on independent final states."""
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
SOURCE = "src/jclosure/experiments/finalist_v35.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "final_opening")
    verify_stage(root, "full_gate_independent_final")
    opening = json.loads((root / OUT / "final_opening_v35.json").read_text())
    selected = opening["selected_mechanism"]
    if selected == "FULL_DEPTH_ONLY":
        raise RuntimeError("V35 full-depth-only finalist is already measured in full_read; no smaller patch is authorized")
    conditions = opening["selected_conditions"]
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_independent_final_v35.parquet")
    facts = np.load(root / OUT / f"factorial_vectors_{key}_independent_final_v35.npz")["vectors"].astype(np.float64)
    full_rows = pd.read_parquet(root / OUT / f"full_read_{key}_independent_final_v35.parquet")
    full_vectors = np.load(root / OUT / f"full_read_vectors_{key}_independent_final_v35.npz")["vectors"].astype(np.float64)
    idx = {row.state_id: int(row.vector_index) for row in factorial.itertuples()}
    by_state = {row.state_id: row for row in factorial.itertuples()}
    full_idx = {row.state_id: int(row.vector_index) for row in full_rows.itertuples()}
    model, tokenizer = load(root, key)
    rows, audits, vectors, serial_rows, serial_vectors = [], [], [], [], []
    serial = selected.startswith("SERIAL_")
    if serial:
        early, late = selected.removeprefix("SERIAL_").split("_TO_")
    for n, item in enumerate(design["independent_final"], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        natural = {"CONV": [], "JOINT": []}
        outputs = {(condition, direction): [] for condition in conditions if condition != "FULL"
                   for direction in ("REMOVE", "RESTORE")}
        hybrids = {label: [] for label in ("C_C", "C_J", "J_C", "J_J")} if serial else {}
        def record(condition, direction, probe_index, probe, proof):
            audits.append({"model_key": key, "state_id": sid, "condition": condition,
                           "direction": direction, "probe_index": probe_index,
                           "probe_token_id": int(probe), "recipient_KV_hash": caches["recipient_KV_hash"],
                           "requested_hash": proof["requested_hash"],
                           "realized_hash": proof["realized_hash"],
                           "source_map_hash": proof["source_map_hash"],
                           "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                           "exact_writeback": proof["exact_writeback"]})
        for probe_index, probe in enumerate(item["future_probe_tokens"]):
            refs = {}
            for branch in ("CONV", "JOINT"):
                out, refs[branch] = capture(model, key, caches[branch.lower()], probe, length, bundle)
                natural[branch].append({"targets": out["targets"]})
            for condition in conditions:
                if condition == "FULL":
                    continue
                for direction, base, source in (("REMOVE", "joint", "CONV"),
                                                ("RESTORE", "conv", "JOINT")):
                    out, proof = patch(model, key, caches[base], probe, length, bundle, refs,
                                       {layer: source for layer in design["condition_layers"][condition]})
                    outputs[(condition, direction)].append({"targets": out["targets"]})
                    record(condition, direction, probe_index, probe, proof)
            if serial:
                for label in hybrids:
                    source_early = "CONV" if label[0] == "C" else "JOINT"
                    source_late = "CONV" if label[2] == "C" else "JOINT"
                    mapping = {**{layer: source_early for layer in design["relative_depth_layers"][early]},
                               **{layer: source_late for layer in design["relative_depth_layers"][late]}}
                    out, proof = patch(model, key, caches["conv"], probe, length, bundle, refs, mapping)
                    hybrids[label].append({"targets": out["targets"]})
                    record(f"SERIAL_{early}_{late}", label, probe_index, probe, proof)
        yconv, yjoint = [signature(natural[name], scales) for name in ("CONV", "JOINT")]
        factual = facts[idx[sid]]
        if not np.allclose(yconv, factual[2], atol=1e-5, rtol=1e-5) or not np.allclose(yjoint, factual[3], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"V35 final factual replay mismatch {key}:{sid}")
        ys = np.empty((len(conditions), 2, len(yconv)), dtype=np.float32)
        for j, condition in enumerate(conditions):
            if condition == "FULL":
                remove, restore = full_vectors[full_idx[sid]]
            else:
                remove = signature(outputs[(condition, "REMOVE")], scales)
                restore = signature(outputs[(condition, "RESTORE")], scales)
            ys[j, 0], ys[j, 1] = remove, restore
            metrics = benefit_metrics(factual[4], yconv, yjoint, remove, restore,
                                      by_state[sid].donor_norm)
            rows.append({"model_key": key, "role": "independent_final", "state_id": sid,
                         "family": item["family"], "condition": condition,
                         "condition_index": j, "vector_index": len(vectors),
                         "recipient_native_KV": True, "six_frozen_probes": True,
                         "exact_writeback": True, **metrics})
        vectors.append(ys)
        if serial:
            hv = {label: signature(hybrids[label], scales) for label in hybrids}
            serial_vectors.append(np.stack([hv[label] for label in hybrids]).astype(np.float32))
            errors = {label: float(np.linalg.norm(factual[4] - value)) for label, value in hv.items()}
            benefit = float(np.linalg.norm(factual[4] - yconv) - np.linalg.norm(factual[4] - yjoint))
            interaction = hv["J_J"] - hv["J_C"] - hv["C_J"] + hv["C_C"]
            correction = yjoint - yconv
            serial_rows.append({"model_key": key, "state_id": sid, "family": item["family"],
                                "early": early, "late": late, "benefit_positive": benefit > 1e-12,
                                "conditional_late_gain": ((errors["J_C"] - errors["J_J"])
                                                          - (errors["C_C"] - errors["C_J"])) / benefit
                                if benefit > 1e-12 else None,
                                "interaction_projection_to_correction":
                                float(np.dot(interaction, correction)) / max(float(np.dot(correction, correction)), 1e-12),
                                "exact_writeback": True})
        if n % 5 == 0:
            print(f"V35 finalist {key} {n}/40", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"finalist_{key}_v35.parquet",
             "vectors": root / OUT / f"finalist_vectors_{key}_v35.npz",
             "audit": root / OUT / f"finalist_audit_{key}_v35.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        state_ids=np.asarray([x["base_trial_id"] for x in design["independent_final"]], dtype=str),
                        conditions=np.asarray(conditions, dtype=str),
                        serial=np.stack(serial_vectors) if serial else np.empty((0,)))
    pd.DataFrame(audits).to_parquet(files["audit"], index=False, compression="zstd")
    if serial:
        files["serial"] = root / OUT / f"finalist_serial_{key}_v35.parquet"
        pd.DataFrame(serial_rows).to_parquet(files["serial"], index=False, compression="zstd")
    summary = {"model_key": key, "selected_mechanism": selected, "states": 40,
               "conditions": conditions, "all_exact_writeback": bool(pd.DataFrame(audits).exact_writeback.all()),
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    summary_path = root / OUT / f"finalist_{key}_v35.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"finalist_{key}", [SOURCE, str(summary_path.relative_to(root)),
                                                  *(str(path.relative_to(root)) for path in files.values()),
                                                  "artifacts/hierarchical_read_v35_final_opening.freeze.json",
                                                  f"artifacts/hierarchical_read_v35_full_read_{key}_independent_final.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(summary_path),
                         "selected_mechanism": selected})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
