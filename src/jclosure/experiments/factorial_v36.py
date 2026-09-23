"""Fresh V36 native REC×Conv factorial before any hierarchical depth inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, signature, step
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/factorial_v36.py"


def cosine(a, b):
    x, y = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (x * y)) if x * y > 1e-12 else None


@torch.no_grad()
def run(root: Path, key: str, role: str):
    verify_stage(root, "design")
    for model_key in ("Q", "F"):
        verify_stage(root, f"interface_{model_key}")
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "high_level_development")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening["opened"]:
            raise RuntimeError("V36 independent final sealed")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    rows, vectors, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V36 incoming drift {key}:{sid}")
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        donor = step(model, incoming, item["donor_token_id"], length)["cache"]
        recipient_hashes, donor_hashes = field_hashes(recipient), field_hashes(donor)
        caches = {"Y00": recipient, "Ydonor": donor}
        proofs = {}
        for condition, channels in (("Y10", ["REC"]), ("Y01", ["Conv"]), ("Y11", ["REC", "Conv"])):
            caches[condition], proofs[condition] = native_swap(recipient, donor, channels, key)
            if field_hashes(caches[condition])["KV"] != recipient_hashes["KV"]:
                raise RuntimeError("V36 primary KV changed")
        responses = {}
        for condition in cfg["factorial"]:
            values = [step(model, caches[condition], probe, length + 1, design["target_bundle"])
                      for probe in item["future_probe_tokens"]]
            responses[condition] = signature(values, design["calibration_clean_scales"])
        y00, y10, y01, y11, yd = (responses[name] for name in cfg["factorial"])
        D, C, R, RC = yd - y00, y01 - y00, y10 - y00, y11 - y00
        E, G, I = D - C, y11 - y01, y11 - y10 - y01 + y00
        dn = max(float(np.linalg.norm(D)), 1e-12)
        ec, ej, er = float(np.linalg.norm(E)), float(np.linalg.norm(D - RC)), float(np.linalg.norm(D - R))
        rows.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                     "recipient_token_id": item["recipient_token_id"], "donor_token_id": item["donor_token_id"],
                     "donor_token_category": item["donor_token_category"],
                     "token_pair_hash": item["token_pair_hash"], "probe_hash": item["future_probe_hash"],
                     "vector_index": len(vectors), "donor_norm": dn, "conv_relative_l2": ec / dn,
                     "rec_only_relative_l2": er / dn, "joint_relative_l2": ej / dn,
                     "relative_conv_error_reduction": (ec - ej) / max(ec, 1e-12),
                     "improvement_positive": ej < ec, "residual_alignment_cosine": cosine(G, E),
                     "interaction_ratio": float(np.linalg.norm(I)) / dn, "benefit_absolute": ec - ej,
                     "recipient_native_KV": True, "writeback_exact": True})
        vectors.append(np.stack([y00, y10, y01, y11, yd]).astype(np.float32))
        audits.append({"model_key": key, "role": role, "state_id": sid,
                       "incoming_hashes": json.dumps(item["incoming_state_hashes"], sort_keys=True),
                       "recipient_hashes": json.dumps(recipient_hashes, sort_keys=True),
                       "donor_hashes": json.dumps(donor_hashes, sort_keys=True),
                       "condition_hashes": json.dumps({name: field_hashes(caches[name]) for name in cfg["factorial"]}, sort_keys=True),
                       "write_proofs": json.dumps(proofs, sort_keys=True),
                       "cache_length": length, "exact_native": True})
        if n % 10 == 0 or n == len(design[role]):
            print(f"V36 factorial {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"factorial_{key}_{role}_v36.parquet",
             "vectors": root / OUT / f"factorial_vectors_{key}_{role}_v36.npz",
             "audit": root / OUT / f"factorial_audit_{key}_{role}_v36.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors), state_ids=frame.state_id.to_numpy(str),
                        families=frame.family.to_numpy(str), role=role, model_key=key)
    pd.DataFrame(audits).to_parquet(files["audit"], index=False, compression="zstd")
    summary = {"model_key": key, "role": role, "rows": len(frame),
               "positive_rows": int(frame.improvement_positive.sum()),
               "median_reduction": float(frame.relative_conv_error_reduction.median()),
               "median_alignment": float(frame.residual_alignment_cosine.median()),
               "response_sha256": {name: sha256_file(path) for name, path in files.items()},
               "independent_final_opened": role == "independent_final"}
    path = root / OUT / f"factorial_{key}_{role}_v36.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"factorial_{key}_{role}",
                        [SOURCE, *(str(p.relative_to(root)) for p in files.values()), str(path.relative_to(root)),
                         "artifacts/computational_origin_v36_design.freeze.json"],
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(path),
                         "recipient_native_KV": True})
    return {"freeze_digest": seal["freeze_digest"], "rows": len(frame),
            "positive": summary["positive_rows"], "median_reduction": summary["median_reduction"],
            "median_alignment": summary["median_alignment"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
