"""Frozen-panel natural versus artificial compact-coordinate controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, transplant
from jclosure.experiments.global_basis_v30 import SCRATCH, load, token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.realization_v29 import contrast, inject, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/coordinate_controls_v30.py"
OUT = Path("results/v30/processed")


def seed(*parts):
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16], 16)


@torch.no_grad()
def run(root, role):
    verify_stage(root, "global_basis_fit")
    verify_stage(root, "transport_source_amendment")
    if role == "validation":
        verify_stage(root, "coordinate_controls_development")
    d, p, _, pairs = load(root)
    amendment = json.loads((root / OUT / "transport_source_amendment_v30.json").read_text())
    basis = np.load(SCRATCH / "basis_GLOBAL_v30.npy", mmap_mode="r")[:64]
    mean = np.load(SCRATCH / "basis_GLOBAL_v30_mean.npy", mmap_mode="r")
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    by_id = {x["base_trial_id"]: x for x in d[role]}
    fit_by_id = {x["base_trial_id"]: x for x in d["development"]}
    source_cache = {}
    rows = []
    for n, trial in enumerate(p["conv_profile_states"][role], 1):
        sid, pair_id = trial["base_trial_id"], trial["pair_id"]
        item, pair = by_id[sid], pairs[pair_id]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError("control target incoming drift")
        anchor = step(bundle, dense, incoming, token(pair["anchor_token_id"]), length, d, 30)
        donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
        spec = layout(anchor["cache"], rec, att)
        x = contrast(anchor["cache"], donor["cache"], spec)
        c = basis @ (x - mean)
        other_pair_id = next(x for x in p["eval_pairs"][role][sid]["heldout_token_pair_ids"] if x != pair_id)
        other_donor = step(bundle, dense, incoming, token(pairs[other_pair_id]["candidate_token_id"]), length, d, 30)
        other = contrast(anchor["cache"], other_donor["cache"], spec)
        c_wrong_token = basis @ (other - mean)
        family = item["family"]
        if family not in source_cache:
            source_id = amendment["source_by_family"][family]
            source_item = fit_by_id[source_id]
            sm = metadata(root, source_item)
            sx, _, sl = prefix(bundle, str(sm["prompt"]))
            sr, sa = native_layers(sx)
            if state_hashes(sx, sr, sa) != source_item["incoming_state_hashes"]:
                raise RuntimeError("control source incoming drift")
            anchor_s = step(bundle, dense, sx, token(pair["anchor_token_id"]), sl, d, 30)
            source_cache[family] = (sx, sl, anchor_s, layout(anchor_s["cache"], sr, sa))
        sx, sl, anchor_s, spec_s = source_cache[family]
        source_donor = step(bundle, dense, sx, token(pair["candidate_token_id"]), sl, d, 30)
        c_wrong_state = basis @ (contrast(anchor_s["cache"], source_donor["cache"], spec_s) - mean)
        rng = np.random.default_rng(seed("V30_CONTROLS", sid, pair_id))
        shuffled = np.array(c, copy=True)
        rng.shuffle(shuffled)
        random = rng.normal(0, 1, size=len(x)).astype(np.float32)
        random *= np.linalg.norm(x) / max(np.linalg.norm(random), 1e-12)
        probes = [token(t) for t in item["future_probe_tokens"]]
        future = lambda cache: signature([step(bundle, dense, cache, z, length + 1, d, 30) for z in probes], scale)
        native_sig, donor_sig = future(anchor["cache"]), future(donor["cache"])
        variants = {"GLOBAL_k64": mean + c @ basis, "RANDOM_SAME_NORM": random, "SHUFFLED_COORDINATE": mean + shuffled @ basis, "SIGN_FLIPPED": mean - c @ basis, "WRONG_STATE_COORDINATE": mean + c_wrong_state @ basis, "WRONG_TOKEN_COORDINATE": mean + c_wrong_token @ basis,
                    "AMPLITUDE_0_5": mean + .5 * (c @ basis), "AMPLITUDE_1_5": mean + 1.5 * (c @ basis)}
        for name, approx in variants.items():
            cache = inject(anchor["cache"], approx, 1, spec, rec, att)
            sig = future(cache)
            rows.append({"role": role, "state_id": sid, "family": family, "pair_id": pair_id, "condition": name, "token_pair_hash": pair["token_pair_hash"], "other_token_pair_hash": pairs[other_pair_id]["token_pair_hash"], "source_state_id": amendment["source_by_family"][family], "random_seed": seed("V30_CONTROLS", sid, pair_id), "write_relative_l2": float(np.linalg.norm(x - approx) / max(np.linalg.norm(x), 1e-12)), "same_incoming_state_target": True, "basis_TRAIN_only": True, **metrics(sig, native_sig, donor_sig)})
        for name, channels in (("EXACT_REC_CONV", "REC+Conv"), ("EXACT_CONV", "Conv")):
            cache, audit = transplant(anchor["cache"], donor["cache"], channels, rec, att)
            if not audit["writeback_pass"]:
                raise RuntimeError("exact control transplant failed")
            rows.append({"role": role, "state_id": sid, "family": family, "pair_id": pair_id, "condition": name, "token_pair_hash": pair["token_pair_hash"], "same_incoming_state_target": True, "basis_TRAIN_only": True, "exact_native_writeback": True, **metrics(future(cache), native_sig, donor_sig)})
        print(f"V30 coordinate controls {role} {n}/{len(p['conv_profile_states'][role])}", flush=True)
    frame = pd.DataFrame(rows)
    fp = root / OUT / f"coordinate_controls_{role}_v30.parquet"
    frame.to_parquet(fp, index=False, compression="zstd")
    jp = root / OUT / f"coordinate_controls_{role}_v30.json"
    summary = {"role": role, "states": len(p["conv_profile_states"][role]), "rows": len(frame), "profiles": profile(frame, p["gates"]), "response_sha256": sha256_file(fp), "off_manifold_controls_not_evidence_for_natural_write_failure": True, "independent_final_opened": False}
    write_json_atomic(jp, summary)
    prior = "artifacts/transferable_natural_writes_v30_global_basis_fit.freeze.json" if role == "development" else "artifacts/transferable_natural_writes_v30_coordinate_controls_development.freeze.json"
    fr = stage_freeze(root, f"coordinate_controls_{role}", [SOURCE, str(fp.relative_to(root)), str(jp.relative_to(root)), prior], {"response_sha256": sha256_file(fp), "role": role})
    return {"freeze_digest": fr["freeze_digest"], "states": summary["states"], "rows": len(frame)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("development", "validation"))
    args = ap.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
