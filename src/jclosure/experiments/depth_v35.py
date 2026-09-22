"""Prospective quartile, cumulative, pair, leave-out and serial read patches."""
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
SOURCE = "src/jclosure/experiments/depth_v35.py"


def targets_only(out):
    return {"targets": out["targets"]}


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    gate = verify_stage(root, f"full_gate_{role}")
    if not gate["formal_depth_authorized"]:
        raise RuntimeError(f"V35 full-depth read gate failed {role}")
    if role == "validation":
        verify_stage(root, "depth_analysis_development")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v35.parquet")
    factual = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v35.npz")["vectors"].astype(np.float64)
    full_vectors = np.load(root / OUT / f"full_read_vectors_{key}_{role}_v35.npz")["vectors"].astype(np.float64)
    full_rows = pd.read_parquet(root / OUT / f"full_read_{key}_{role}_v35.parquet")
    by_state = {row.state_id: row for row in factorial.itertuples()}
    full_index = {row.state_id: row.vector_index for row in full_rows.itertuples()}
    model, tokenizer = load(root, key)
    groups, conditions = design["relative_depth_layers"], design["condition_layers"]
    subset_names = list(cfg["subset_conditions"])
    leave_names = list(cfg["leave_out_conditions"])
    serial_pairs = cfg["serial_pairs"]
    serial_labels = ("C_C", "C_J", "J_C", "J_J")
    subset_rows, leave_rows, serial_rows = [], [], []
    subset_arrays, leave_arrays, serial_arrays, audits = [], [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        natural = {"CONV": [], "JOINT": []}
        subset_outputs = {(name, direction): [] for name in subset_names if name != "FULL"
                          for direction in ("REMOVE", "RESTORE")}
        leave_outputs = {name: [] for name in leave_names}
        serial_outputs = {(a, b, label): [] for a, b in serial_pairs for label in serial_labels}
        for probe_index, probe in enumerate(item["future_probe_tokens"]):
            refs = {}
            for label, cache in (("CONV", caches["conv"]), ("JOINT", caches["joint"])):
                out, ref = capture(model, key, cache, probe, length, bundle)
                natural[label].append(targets_only(out))
                refs[label] = ref
            def record(condition, direction, proof):
                audits.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                               "probe_index": probe_index, "probe_token_id": int(probe),
                               "condition": condition, "direction": direction,
                               "recipient_KV_hash": caches["recipient_KV_hash"],
                               "source_map_hash": proof["source_map_hash"],
                               "requested_hash": proof["requested_hash"],
                               "realized_hash": proof["realized_hash"],
                               "native_hash": proof["native_hash"], "layers": proof["layers"],
                               "per_layer_json": json.dumps(proof["per_layer"], sort_keys=True),
                               "exact_writeback": proof["exact_writeback"]})
            for name in subset_names:
                if name == "FULL":
                    continue
                layers = conditions[name]
                for direction, base, source in (("REMOVE", caches["joint"], "CONV"),
                                                ("RESTORE", caches["conv"], "JOINT")):
                    out, proof = patch(model, key, base, probe, length, bundle, refs,
                                       {layer: source for layer in layers})
                    subset_outputs[(name, direction)].append(targets_only(out))
                    record(name, direction, proof)
            for name, missing in cfg["leave_out_conditions"].items():
                mapping = {layer: ("CONV" if layer in groups[missing] else "JOINT")
                           for layer in conditions["FULL"]}
                out, proof = patch(model, key, caches["conv"], probe, length, bundle, refs, mapping)
                leave_outputs[name].append(targets_only(out))
                record(name, "FULL_MINUS", proof)
            for early, late in serial_pairs:
                for label in serial_labels:
                    e_source = "CONV" if label[0] == "C" else "JOINT"
                    l_source = "CONV" if label[2] == "C" else "JOINT"
                    mapping = {**{layer: e_source for layer in groups[early]},
                               **{layer: l_source for layer in groups[late]}}
                    out, proof = patch(model, key, caches["conv"], probe, length, bundle, refs, mapping)
                    serial_outputs[(early, late, label)].append(targets_only(out))
                    record(f"SERIAL_{early}_{late}", label, proof)
        yconv, yjoint = [signature(natural[label], scales) for label in ("CONV", "JOINT")]
        previous = factual[int(by_state[sid].vector_index)]
        if not np.allclose(yconv, previous[2], rtol=1e-5, atol=1e-5) or not np.allclose(yjoint, previous[3], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"V35 factual replay mismatch {key}:{sid}")
        yd, donor_norm = previous[4], by_state[sid].donor_norm
        subset_vec = np.empty((len(subset_names), 2, len(yconv)), dtype=np.float32)
        for index, name in enumerate(subset_names):
            if name == "FULL":
                remove, restore = full_vectors[int(full_index[sid])]
            else:
                remove = signature(subset_outputs[(name, "REMOVE")], scales)
                restore = signature(subset_outputs[(name, "RESTORE")], scales)
            subset_vec[index, 0], subset_vec[index, 1] = remove, restore
            metrics = benefit_metrics(yd, yconv, yjoint, remove, restore, donor_norm)
            subset_rows.append({"model_key": key, "role": role, "state_id": sid,
                                "family": item["family"], "donor_token_category": item["donor_token_category"],
                                "condition": name, "condition_index": index, "vector_index": len(subset_arrays),
                                "quartile_count": len(cfg["subset_conditions"][name]),
                                "relative_depth_groups": json.dumps(cfg["subset_conditions"][name]),
                                "token_pair_hash": item["token_pair_hash"], "probe_hash": item["future_probe_hash"],
                                "depth_group_hash": design["relative_depth_group_hash"],
                                "condition_layer_hash": design["condition_layer_hash"],
                                "recipient_native_KV": True, "six_frozen_probes": True,
                                "exact_writeback": True, **metrics})
        subset_arrays.append(subset_vec)
        leave_vec = np.empty((len(leave_names), len(yconv)), dtype=np.float32)
        full_restore = subset_vec[subset_names.index("FULL"), 1].astype(np.float64)
        full_error = float(np.linalg.norm(yd - full_restore))
        benefit = float(np.linalg.norm(yd - yconv) - np.linalg.norm(yd - yjoint))
        for index, name in enumerate(leave_names):
            vector = signature(leave_outputs[name], scales)
            leave_vec[index] = vector
            error = float(np.linalg.norm(yd - vector))
            leave_rows.append({"model_key": key, "role": role, "state_id": sid,
                               "family": item["family"], "condition": name,
                               "missing_quartile": cfg["leave_out_conditions"][name],
                               "condition_index": index, "vector_index": len(leave_arrays),
                               "benefit_positive": benefit > 1e-12,
                               "loss_fraction": (error - full_error) / benefit if benefit > 1e-12 else None,
                               "full_error": full_error, "without_error": error,
                               "recipient_native_KV": True, "exact_writeback": True})
        leave_arrays.append(leave_vec)
        serial_vec = np.empty((len(serial_pairs), len(serial_labels), len(yconv)), dtype=np.float32)
        for pair_index, (early, late) in enumerate(serial_pairs):
            vectors = {}
            for label_index, label in enumerate(serial_labels):
                vectors[label] = signature(serial_outputs[(early, late, label)], scales)
                serial_vec[pair_index, label_index] = vectors[label]
            errors = {label: float(np.linalg.norm(yd - vector)) for label, vector in vectors.items()}
            late_gain_given_early_joint = errors["J_C"] - errors["J_J"]
            late_gain_given_early_conv = errors["C_C"] - errors["C_J"]
            interaction_vector = vectors["J_J"] - vectors["J_C"] - vectors["C_J"] + vectors["C_C"]
            correction = yjoint - yconv
            serial_rows.append({"model_key": key, "role": role, "state_id": sid,
                                "family": item["family"], "early": early, "late": late,
                                "pair_index": pair_index, "vector_index": len(serial_arrays),
                                "benefit_positive": benefit > 1e-12,
                                "late_gain_if_early_joint": late_gain_given_early_joint / benefit if benefit > 1e-12 else None,
                                "late_gain_if_early_conv": late_gain_given_early_conv / benefit if benefit > 1e-12 else None,
                                "conditional_late_gain": (late_gain_given_early_joint - late_gain_given_early_conv) / benefit if benefit > 1e-12 else None,
                                "interaction_norm_to_correction": float(np.linalg.norm(interaction_vector)) / max(float(np.linalg.norm(correction)), 1e-12),
                                "interaction_projection_to_correction": float(np.dot(interaction_vector, correction)) / max(float(np.dot(correction, correction)), 1e-12),
                                "recipient_native_KV": True, "exact_writeback": True})
        serial_arrays.append(serial_vec)
        if n % 2 == 0 or n == len(design[role]):
            print(f"V35 depth {key} {role} {n}/{len(design[role])}", flush=True)
    frames = {"subset": pd.DataFrame(subset_rows), "leaveout": pd.DataFrame(leave_rows),
              "serial": pd.DataFrame(serial_rows), "audit": pd.DataFrame(audits)}
    files = {name: root / OUT / f"depth_{name}_{key}_{role}_v35.parquet" for name in frames}
    for name, frame in frames.items():
        frame.to_parquet(files[name], index=False, compression="zstd")
    vectors_path = root / OUT / f"depth_vectors_{key}_{role}_v35.npz"
    np.savez_compressed(vectors_path, subset=np.stack(subset_arrays), leaveout=np.stack(leave_arrays),
                        serial=np.stack(serial_arrays), state_ids=np.asarray([x["base_trial_id"] for x in design[role]], dtype=str),
                        subset_conditions=np.asarray(subset_names, dtype=str), leaveout_conditions=np.asarray(leave_names, dtype=str),
                        serial_pairs=np.asarray(["_".join(pair) for pair in serial_pairs], dtype=str),
                        serial_labels=np.asarray(serial_labels, dtype=str))
    summary = {"model_key": key, "role": role, "states": len(design[role]),
               "subset_conditions": subset_names, "leaveout_conditions": leave_names,
               "serial_pairs": serial_pairs, "audit_rows": len(audits),
               "all_exact_writeback": bool(frames["audit"].exact_writeback.all()),
               "median_removed": frames["subset"].groupby("condition").removed_fraction.median().to_dict(),
               "median_restored": frames["subset"].groupby("condition").restored_fraction.median().to_dict(),
               "files_sha256": {name: sha256_file(path) for name, path in {**files, "vectors": vectors_path}.items()}}
    summary_path = root / OUT / f"depth_{key}_{role}_v35.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"depth_{key}_{role}",
                        [SOURCE, "src/jclosure/experiments/depth_common_v35.py",
                         "src/jclosure/experiments/read_hooks_v35.py",
                         *(str(path.relative_to(root)) for path in files.values()),
                         str(vectors_path.relative_to(root)), str(summary_path.relative_to(root)),
                         f"artifacts/hierarchical_read_v35_full_gate_{role}.freeze.json"],
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(summary_path),
                         "all_exact_writeback": summary["all_exact_writeback"]})
    return {"freeze_digest": seal["freeze_digest"], "model": key, "role": role,
            "states": len(design[role]), "audit_rows": len(audits)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
