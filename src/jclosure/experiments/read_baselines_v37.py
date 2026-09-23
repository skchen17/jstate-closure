"""Full read causal-cut replication and Q2/Q3/Q4 interface baseline.

Output copying here measures a causal interface only, never a mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import benefit_metrics, capture, patch, prepare
from jclosure.experiments.runtime_v34 import load, signature
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/read_baselines_v37.py"
CONDITIONS = ("CC", "JC", "CJ", "JJ")


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    high = verify_stage(root, f"high_level_{role}")
    if not high["both_pass"]:
        raise RuntimeError("High-level conditional REC gate failed")
    if role == "validation":
        verify_stage(root, "read_gate_development")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening["opened"]:
            raise RuntimeError("Independent final remains sealed")
    design = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    panel = json.loads((root / OUT / "panel_v37.json").read_text())
    lookup = {row["base_trial_id"]: row["prompt"] for role_name in
              ("calibration", "development", "validation", "independent_final")
              for row in panel[role_name]}
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v37.parquet")
    old_vectors = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v37.npz")["vectors"].astype(np.float64)
    by_state = {row.state_id: row for row in factorial.itertuples()}
    model, tokenizer = load(root, key)
    groups = design["relative_depth_layers"]
    q2 = groups["Q2"]
    late = groups["Q3"] + groups["Q4"]
    all_layers = design["recurrent_layers"]
    rows, vectors, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        length, bundle, scales = caches["length"], design["target_bundle"], design["calibration_clean_scales"]
        outputs = {name: [] for name in (*CONDITIONS, "TRIPLE_REMOVE", "FULL_REMOVE", "FULL_RESTORE", "NATIVE_JOINT")}
        for pi, probe in enumerate(item["future_probe_tokens"]):
            conv, conv_ref = capture(model, key, caches["conv"], probe, length, bundle)
            joint, joint_ref = capture(model, key, caches["joint"], probe, length, bundle)
            refs = {"CONV": conv_ref, "JOINT": joint_ref}
            outputs["CC"].append(conv)
            outputs["NATIVE_JOINT"].append(joint)
            mappings = {
                "JC": {layer: "JOINT" for layer in q2},
                "CJ": {layer: "JOINT" for layer in late},
                "JJ": {layer: "JOINT" for layer in q2 + late},
            }
            for name, mapping in mappings.items():
                out, proof = patch(model, key, caches["conv"], probe, length, bundle, refs, mapping)
                outputs[name].append(out)
                audits.append({"state_id": sid, "model": key, "role": role, "probe_index": pi,
                               "condition": name, "requested_hash": proof["requested_hash"],
                               "realized_hash": proof["realized_hash"],
                               "source_map_hash": proof["source_map_hash"],
                               "exact_writeback": proof["exact_writeback"]})
            for name, base, source in (("FULL_REMOVE", caches["joint"], "CONV"),
                                       ("FULL_RESTORE", caches["conv"], "JOINT"),
                                       ("TRIPLE_REMOVE", caches["joint"], "CONV")):
                out, proof = patch(model, key, base, probe, length, bundle, refs,
                                   {layer: source for layer in (q2 + late if name == "TRIPLE_REMOVE" else all_layers)})
                outputs[name].append(out)
                audits.append({"state_id": sid, "model": key, "role": role, "probe_index": pi,
                               "condition": name, "requested_hash": proof["requested_hash"],
                               "realized_hash": proof["realized_hash"],
                               "source_map_hash": proof["source_map_hash"],
                               "exact_writeback": proof["exact_writeback"]})
        vec = {name: signature(values, scales) for name, values in outputs.items()}
        previous = old_vectors[int(by_state[sid].vector_index)]
        if not np.allclose(vec["CC"], previous[2], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"Conv factual replay drift {key}:{sid}")
        if not np.allclose(vec["NATIVE_JOINT"], previous[3], rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"Joint factual replay drift {key}:{sid}")
        yd, yconv, yjoint = previous[4], previous[2], previous[3]
        metrics = benefit_metrics(yd, yconv, yjoint, vec["FULL_REMOVE"], vec["FULL_RESTORE"],
                                  by_state[sid].donor_norm)
        triple_metrics = benefit_metrics(yd, yconv, yjoint, vec["TRIPLE_REMOVE"], vec["JJ"],
                                         by_state[sid].donor_norm)
        benefit = float(np.linalg.norm(yd - yconv) - np.linalg.norm(yd - yjoint))
        floor = 1.0
        q2_gain = float(np.linalg.norm(yd - vec["CJ"]) - np.linalg.norm(yd - vec["JJ"]))
        rows.append({"state_id": sid, "model": key, "role": role, "family": item["family"],
                     "vector_index": len(vectors), "six_probes": True,
                     "full_removed_fraction": metrics["removed_fraction"],
                     "full_restored_fraction": metrics["restored_fraction"],
                     "full_reverse_cosine": metrics["reverse_correction_cosine"],
                     "triple_removed_fraction": triple_metrics["removed_fraction"],
                     "triple_restored_fraction": triple_metrics["restored_fraction"],
                     "triple_cosine": triple_metrics["reverse_correction_cosine"],
                     "q2_enablement_absolute": q2_gain,
                     "q2_enablement_ratio": q2_gain / benefit if benefit > floor else None,
                     "benefit_absolute": benefit, "eligible_ratio": benefit > floor,
                     "exact_interface_writeback": True})
        vectors.append(np.stack([vec[name] for name in (*CONDITIONS, "TRIPLE_REMOVE", "FULL_REMOVE", "FULL_RESTORE", "NATIVE_JOINT")]).astype(np.float32))
        if n % 5 == 0 or n == len(design[role]):
            print(f"V37 read baselines {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"read_baselines_{key}_{role}_v37.parquet",
             "vectors": root / OUT / f"read_baselines_vectors_{key}_{role}_v37.npz",
             "audit": root / OUT / f"read_baselines_audit_{key}_{role}_v37.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        conditions=np.asarray((*CONDITIONS, "TRIPLE_REMOVE", "FULL_REMOVE", "FULL_RESTORE", "NATIVE_JOINT"), dtype=str),
                        state_ids=frame.state_id.to_numpy(str), model=key, role=role)
    pd.DataFrame(audits).to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model": key, "role": role, "states": len(frame),
               "median_full_removed": float(frame.full_removed_fraction.median()),
               "median_full_restored": float(frame.full_restored_fraction.median()),
               "median_full_cosine": float(frame.full_reverse_cosine.median()),
               "median_triple_restored": float(frame.triple_restored_fraction.median()),
               "median_triple_cosine": float(frame.triple_cosine.median()),
               "all_exact_writeback": bool(frame.exact_interface_writeback.all()),
               "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    path = root / OUT / f"read_baselines_{key}_{role}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"read_baselines_{key}_{role}",
                        [SOURCE, "src/jclosure/experiments/depth_common_v35.py",
                         "src/jclosure/experiments/read_hooks_v35.py",
                         str(path.relative_to(root)),
                         *[str(file.relative_to(root)) for file in files.values()]],
                        {"model": key, "role": role, "summary_sha256": sha256_file(path),
                         "all_exact_writeback": summary["all_exact_writeback"]})
    return {"freeze_digest": seal["freeze_digest"], **summary}


def _cos(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation", "independent_final"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
