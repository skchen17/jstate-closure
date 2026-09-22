"""Prospective exact-native REC×Conv factorial on fresh V32 states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import signature
from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.design_v32 import hd
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/factorial_v32.py"


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na * nb > 1e-12 else None


def audit_cache(cache, recipient, donor, rec, att, rec_donor, conv_donor):
    if cache.get_seq_length() != recipient.get_seq_length() or donor.get_seq_length() != recipient.get_seq_length():
        raise RuntimeError("V32 cache length mismatch")
    for layer in rec:
        for field, use_donor in (("recurrent_states", rec_donor), ("conv_states", conv_donor)):
            expected = getattr(donor if use_donor else recipient, "layers")[layer]
            if not torch.equal(getattr(cache.layers[layer], field), getattr(expected, field)):
                raise RuntimeError(f"V32 {field} writeback mismatch at {layer}")
    for layer in att:
        for field in ("keys", "values"):
            if not torch.equal(getattr(cache.layers[layer], field), getattr(recipient.layers[layer], field)):
                raise RuntimeError(f"V32 recipient KV changed at {layer}")
    return state_hashes(cache, rec, att)


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "execution_plan")
    if role == "validation":
        verify_stage(root, "factorial_development")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    library = {x["token_id"]: x for x in design["token_library"]}
    bundle, dense, *_ = context(root)
    scale = scales(root)
    states = {x["base_trial_id"]: x for x in design[role]}
    selected = {sid for family in cfg["families"] for sid in plan["roles"][role][family]["cross_token_state_ids"]}
    rows, vectors, audits = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError(f"V32 incoming state drift: {sid}")
        recipient = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, design, cfg["readout_layer"])
        pair_tokens = [item["primary_token_id"]] + ([item["secondary_token_id"]] if sid in selected else [])
        probe_tokens = [token(tid) for tid in item["future_probe_tokens"]]
        def response(cache):
            return signature([step(bundle, dense, cache, z, length + 1, design, cfg["readout_layer"]) for z in probe_tokens], scale)
        y00 = response(recipient["cache"])
        recipient_hashes = state_hashes(recipient["cache"], rec, att)
        for pair_index, donor_id in enumerate(pair_tokens):
            donor = step(bundle, dense, incoming, token(donor_id), length, design, cfg["readout_layer"])
            donor_hashes = state_hashes(donor["cache"], rec, att)
            if donor_hashes == recipient_hashes:
                raise RuntimeError("V32 natural token fork did not write")
            caches = {
                "Y10": partial(recipient["cache"], donor["cache"], rec, (), rec),
                "Y01": partial(recipient["cache"], donor["cache"], rec, rec, ()),
                "Y11": partial(recipient["cache"], donor["cache"], rec, rec, rec),
            }
            hashes = {}
            for condition, cache in caches.items():
                hashes[condition] = audit_cache(cache, recipient["cache"], donor["cache"], rec, att, condition in ("Y10", "Y11"), condition in ("Y01", "Y11"))
            y10, y01, y11 = (response(caches[c]) for c in ("Y10", "Y01", "Y11"))
            yd = response(donor["cache"])
            D, C, R, RC = yd - y00, y01 - y00, y10 - y00, y11 - y00
            E, G, I = D - C, y11 - y01, y11 - y10 - y01 + y00
            dnorm = max(float(np.linalg.norm(D)), 1e-12)
            conv_error = float(np.linalg.norm(E))
            joint_error = float(np.linalg.norm(D - RC))
            cnorm = float(np.linalg.norm(C))
            alpha = float(np.dot(RC, C) / max(cnorm * cnorm, 1e-12))
            gain_error = float(np.linalg.norm(D - alpha * C))
            # Sequential orthogonal decomposition: gain along C, residual target in
            # C-orthogonal space, then unexplained orthogonal remainder.
            gain_component = (float(np.dot(G, C)) / max(cnorm * cnorm, 1e-12)) * C
            e_orth = E - (float(np.dot(E, C)) / max(cnorm * cnorm, 1e-12)) * C
            correction_component = (float(np.dot(G - gain_component, e_orth)) / max(float(np.dot(e_orth, e_orth)), 1e-12)) * e_orth
            remainder = G - gain_component - correction_component
            gnorm = max(float(np.linalg.norm(G)), 1e-12)
            base = {"role": role, "state_id": sid, "family": item["family"], "pair_index": pair_index, "primary": pair_index == 0, "token_id": donor_id, "token_category": library[donor_id]["category"], "pair_hash": library[donor_id]["pair_hash"], "incoming_state_hash": item["incoming_state_hash"], "recipient_state_hash": total_hash(recipient_hashes), "donor_state_hash": total_hash(donor_hashes), "probe_hash": item["future_probe_hash"], "six_frozen_probes": True, "recipient_native_KV": True, "writeback_exact": True}
            rows.append({**base, "vector_index": len(vectors), "donor_norm": dnorm, "conv_relative_l2": conv_error / dnorm, "joint_relative_l2": joint_error / dnorm, "rec_only_relative_l2": float(np.linalg.norm(D - R)) / dnorm, "paired_l2_improvement": (conv_error - joint_error) / dnorm, "relative_conv_error_reduction": (conv_error - joint_error) / max(conv_error, 1e-12), "improvement_positive": joint_error < conv_error, "residual_alignment_cosine": cosine(G, E), "conv_donor_cosine": cosine(C, D), "joint_donor_cosine": cosine(RC, D), "rec_only_donor_cosine": cosine(R, D), "interaction_ratio": float(np.linalg.norm(I)) / dnorm, "rec_gain_ratio": float(np.linalg.norm(RC)) / max(cnorm, 1e-12), "oracle_scalar_alpha": alpha, "scalar_gain_error_relative": gain_error / dnorm, "joint_beats_scalar_gain": joint_error < gain_error, "gain_fraction_of_conditional_norm": float(np.linalg.norm(gain_component)) / gnorm, "residual_correction_fraction_of_conditional_norm": float(np.linalg.norm(correction_component)) / gnorm, "orthogonal_remainder_fraction_of_conditional_norm": float(np.linalg.norm(remainder)) / gnorm, "donor_direction_component": float(np.dot(G, D)) / max(dnorm * gnorm, 1e-12), "residual_projection_fraction": float(np.dot(G, E)) / max(float(np.dot(E, E)), 1e-12)})
            vectors.append(np.stack([y00, y10, y01, y11, yd]).astype(np.float32))
            audits.append({**base, "recipient_channel_hashes": json.dumps(recipient_hashes, sort_keys=True), "donor_channel_hashes": json.dumps(donor_hashes, sort_keys=True), "condition_channel_hashes": json.dumps(hashes, sort_keys=True), "field_count": 24 * (2 + 2 + 1), "cache_length": int(recipient["cache"].get_seq_length())})
        print(f"V32 factorial {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    paths = {"factorial": root / OUT / f"factorial_{role}_v32.parquet", "vectors": root / OUT / f"factorial_vectors_{role}_v32.npz", "audit": root / OUT / f"factorial_audit_{role}_v32.parquet"}
    frame.to_parquet(paths["factorial"], index=False, compression="zstd")
    np.savez_compressed(paths["vectors"], vectors=np.stack(vectors), state_ids=frame.state_id.to_numpy(str), pair_hashes=frame.pair_hash.to_numpy(str), roles=frame.role.to_numpy(str), families=frame.family.to_numpy(str))
    pd.DataFrame(audits).to_parquet(paths["audit"], index=False, compression="zstd")
    primary = frame[frame.primary]
    summary = {"role": role, "states": len(design[role]), "rows": len(frame), "primary_rows": len(primary), "positive_primary_rows": int(primary.improvement_positive.sum()), "median_primary_l2_reduction": float(primary.relative_conv_error_reduction.median()), "median_primary_alignment": float(primary.residual_alignment_cosine.median()), "response_sha256": {k: sha256_file(v) for k, v in paths.items()}, "independent_final_opened": False}
    jpath = root / OUT / f"factorial_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"factorial_{role}", [SOURCE, *(str(p.relative_to(root)) for p in paths.values()), str(jpath.relative_to(root)), "artifacts/rec_conv_mechanism_v32_execution_plan.freeze.json"], {"summary_sha256": sha256_file(jpath), "primary_rows": len(primary), "response_hash": hd(summary["response_sha256"])})
    return {"freeze_digest": stage["freeze_digest"], **{k: summary[k] for k in ("states", "rows", "positive_primary_rows", "median_primary_l2_reduction", "median_primary_alignment")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
