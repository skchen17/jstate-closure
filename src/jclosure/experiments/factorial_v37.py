"""Fresh high-level REC×Conv factorial; validation/final remain stage-gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.runtime_v34 import load, native_swap, signature, step
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/factorial_v37.py"
CONDITIONS = ("Y00", "Y10", "Y01", "Y11", "Ydonor")


def _cosine(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-12 else None


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    verify_stage(root, "design")
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    for model_key in ("Q", "F"):
        calibration = verify_stage(root, f"calibration_{model_key}")
        if not calibration["all_within_tolerance"] or not calibration["all_bitwise"]:
            raise RuntimeError(f"V37 calibration failed for {model_key}")
    if role == "validation":
        prior = verify_stage(root, "high_level_development")
        if not prior["both_pass"]:
            raise RuntimeError("High-level development gate failed")
    if role == "independent_final":
        prior = verify_stage(root, "final_opening")
        if not prior["opened"]:
            raise RuntimeError("Independent final remains sealed")
    design = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    panel = json.loads((root / OUT / "panel_v37.json").read_text())
    lookup = {row["base_trial_id"]: row["prompt"] for role_name in
              ("calibration", "development", "validation", "independent_final")
              for row in panel[role_name]}
    model, tokenizer = load(root, key)
    rows, vectors, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        recipient, donor, conv, joint = (caches[k] for k in ("recipient", "donor", "conv", "joint"))
        rec, rec_proof = native_swap(recipient, donor, ["REC"], key)
        branch = {"Y00": recipient, "Y10": rec, "Y01": conv, "Y11": joint, "Ydonor": donor}
        responses = {}
        for label in CONDITIONS:
            outputs = [step(model, branch[label], probe, caches["length"] + 1, design["target_bundle"])
                       for probe in item["future_probe_tokens"]]
            responses[label] = signature(outputs, design["calibration_clean_scales"])
        y00, y10, y01, y11, yd = (responses[label] for label in CONDITIONS)
        donor_effect = yd - y00
        e_conv = float(np.linalg.norm(yd - y01))
        e_joint = float(np.linalg.norm(yd - y11))
        rows.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                     "donor_token_category": item["donor_token_category"],
                     "token_pair_hash": item["token_pair_hash"], "probe_hash": item["future_probe_hash"],
                     "vector_index": len(vectors), "donor_norm": float(np.linalg.norm(donor_effect)),
                     "conv_error": e_conv, "joint_error": e_joint,
                     "relative_conv_error_reduction": (e_conv - e_joint) / max(e_conv, 1e-12),
                     "improvement_positive": bool(e_joint < e_conv),
                     "residual_alignment_cosine": _cosine(y11 - y01, yd - y01),
                     "rec_only_donor_error": float(np.linalg.norm(yd - y10)),
                     "benefit_absolute": e_conv - e_joint,
                     "recipient_native_KV": True, "writeback_exact": True})
        vectors.append(np.stack([responses[label] for label in CONDITIONS]).astype(np.float32))
        audits.append({"state_id": sid, "model_key": key, "role": role,
                       "recipient_KV_hash": caches["recipient_KV_hash"],
                       "REC_swap_proof": json.dumps(rec_proof, sort_keys=True),
                       "all_six_probes": True, "native_five_branch_replay": True})
        if n % 10 == 0 or n == len(design[role]):
            print(f"V37 factorial {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"factorial_{key}_{role}_v37.parquet",
             "vectors": root / OUT / f"factorial_vectors_{key}_{role}_v37.npz",
             "audit": root / OUT / f"factorial_audit_{key}_{role}_v37.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        state_ids=frame.state_id.to_numpy(str), families=frame.family.to_numpy(str),
                        role=role, model_key=key, conditions=np.asarray(CONDITIONS, dtype=str))
    pd.DataFrame(audits).to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model_key": key, "role": role, "states": len(frame),
               "positive_fraction": float(frame.improvement_positive.mean()),
               "median_reduction": float(frame.relative_conv_error_reduction.median()),
               "files_sha256": {name: sha256_file(path) for name, path in files.items()},
               "independent_final_opened": role == "independent_final"}
    path = root / OUT / f"factorial_{key}_{role}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"factorial_{key}_{role}", [SOURCE, str(path.relative_to(root)),
                        *[str(file.relative_to(root)) for file in files.values()],
                        "artifacts/computational_origin_v37_design.freeze.json"],
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation", "independent_final"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
