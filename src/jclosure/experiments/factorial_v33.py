"""Locked REC2 x CONV2 natural-token factorial, preserving recipient KV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v33 import prompts
from jclosure.experiments.runtime_v33 import context, field_hashes, hd, native_swap, prefix, signature, step
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v33/processed")
SOURCE = "src/jclosure/experiments/factorial_v33.py"


def cosine(a, b):
    x, y = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (x * y)) if x * y > 1e-12 else None


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "design")
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "factorial_development")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening.get("opened"):
            raise RuntimeError("V33 independent final remains sealed")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v33.json").read_text())
    mapping = json.loads((root / OUT / "architecture_audit_v33.json").read_text())
    if not mapping["functional_structural_comparable"]:
        raise RuntimeError("V33 interface not comparable")
    prompt_lookup = prompts(root, design)
    model, tokenizer = context(root)
    scales = design["calibration_clean_scales"]
    bundle = design["target_bundle"]
    anchor = design["anchor_token_id"]
    rows, vectors, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, _, length, ids = prefix(model, tokenizer, prompt_lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V33 incoming drift {sid}")
        recipient = step(model, incoming, anchor, length)["cache"]
        donor = step(model, incoming, item["primary_token_id"], length)["cache"]
        recipient_hashes, donor_hashes = field_hashes(recipient), field_hashes(donor)
        if recipient_hashes == donor_hashes:
            raise RuntimeError(f"V33 natural fork did not write {sid}")
        caches = {"Y00": recipient, "Ydonor": donor}
        proofs = {}
        for condition, fields in (("Y10", ["REC2"]), ("Y01", ["CONV2"]), ("Y11", ["REC2", "CONV2"])):
            caches[condition], proofs[condition] = native_swap(recipient, donor, fields)
            if field_hashes(caches[condition])["KV"] != recipient_hashes["KV"]:
                raise RuntimeError(f"V33 recipient KV violated {sid}")
        outputs = {}
        for condition in cfg["factorial"]:
            response = [step(model, caches[condition], probe, length + 1, bundle) for probe in item["future_probe_tokens"]]
            outputs[condition] = signature(response, scales)
        y00, y10, y01, y11, yd = (outputs[k] for k in cfg["factorial"])
        D, C, R, RC = yd-y00, y01-y00, y10-y00, y11-y00
        E, G, I = D-C, y11-y01, y11-y10-y01+y00
        dn = max(float(np.linalg.norm(D)), 1e-12)
        ce, je, re = float(np.linalg.norm(E)), float(np.linalg.norm(D-RC)), float(np.linalg.norm(D-R))
        cn = max(float(np.linalg.norm(C)), 1e-12)
        alpha = float(np.dot(D, C) / (cn*cn))
        gain_err = float(np.linalg.norm(D-alpha*C))
        row = {"role": role, "state_id": sid, "family": item["family"], "token_id": item["primary_token_id"], "token_category": item["primary_token_category"], "primary_pair_hash": item["primary_pair_hash"], "future_probe_hash": item["future_probe_hash"], "vector_index": len(vectors), "donor_norm": dn, "conv_relative_l2": ce/dn, "joint_relative_l2": je/dn, "rec_only_relative_l2": re/dn, "conv_to_rec_error_ratio": ce/max(re, 1e-12), "relative_conv_error_reduction": (ce-je)/max(ce, 1e-12), "improvement_positive": je < ce, "residual_alignment_cosine": cosine(G,E), "conv_donor_cosine": cosine(C,D), "joint_donor_cosine": cosine(RC,D), "rec_only_donor_cosine": cosine(R,D), "rec_to_conv_effect_norm_ratio": float(np.linalg.norm(R))/cn, "interaction_ratio": float(np.linalg.norm(I))/dn, "oracle_scalar_alpha": alpha, "scalar_gain_error_relative": gain_err/dn, "joint_beats_scalar_gain": je < gain_err, "recipient_native_KV": True, "writeback_exact": True}
        rows.append(row)
        vectors.append(np.stack([y00,y10,y01,y11,yd]).astype(np.float32))
        audits.append({"role": role, "state_id": sid, "incoming_state_hashes": json.dumps(item["incoming_state_hashes"], sort_keys=True), "recipient_hashes": json.dumps(recipient_hashes, sort_keys=True), "donor_hashes": json.dumps(donor_hashes, sort_keys=True), "condition_hashes": json.dumps({k: field_hashes(caches[k]) for k in cfg["factorial"]}, sort_keys=True), "write_proofs": json.dumps(proofs, sort_keys=True), "fork_total_length": length, "probe_hash": item["future_probe_hash"], "exact_native": True})
        if n % 10 == 0 or n == len(design[role]):
            print(f"V33 factorial {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    paths = {"factorial": root / OUT / f"factorial_{role}_v33.parquet", "vectors": root / OUT / f"factorial_vectors_{role}_v33.npz", "audit": root / OUT / f"factorial_audit_{role}_v33.parquet"}
    frame.to_parquet(paths["factorial"], index=False, compression="zstd")
    np.savez_compressed(paths["vectors"], vectors=np.stack(vectors), state_ids=frame.state_id.to_numpy(str), roles=frame.role.to_numpy(str), families=frame.family.to_numpy(str))
    pd.DataFrame(audits).to_parquet(paths["audit"], index=False, compression="zstd")
    summary = {"role": role, "states": len(frame), "rows": len(frame), "positive_rows": int(frame.improvement_positive.sum()), "median_l2_reduction": float(frame.relative_conv_error_reduction.median()), "median_alignment": float(frame.residual_alignment_cosine.median()), "response_sha256": {k: sha256_file(v) for k,v in paths.items()}, "independent_final_opened": role == "independent_final"}
    jpath = root / OUT / f"factorial_{role}_v33.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"factorial_{role}", [SOURCE, *(str(p.relative_to(root)) for p in paths.values()), str(jpath.relative_to(root)), "artifacts/cross_model_rec_conv_v33_design.freeze.json"], {"summary_sha256": sha256_file(jpath), "response_hash": hd(summary["response_sha256"]), "rows": len(frame), "recipient_native_KV": True})
    return {"freeze_digest": stage["freeze_digest"], **{k:summary[k] for k in ("states", "positive_rows", "median_l2_reduction", "median_alignment")}}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args=parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
