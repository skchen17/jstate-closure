"""Frozen four-token h1/h2/h4 causal write persistence diagnostic."""
from __future__ import annotations

import argparse
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

SOURCE = "src/jclosure/experiments/horizon_v30.py"
OUT = Path("results/v30/processed")


@torch.no_grad()
def run(root, role):
    verify_stage(root, "global_basis_fit")
    if role == "validation":
        verify_stage(root, "horizon_development")
    d, p, _, pairs = load(root)
    basis = np.load(SCRATCH / "basis_GLOBAL_v30.npy", mmap_mode="r")
    mean = np.load(SCRATCH / "basis_GLOBAL_v30_mean.npy", mmap_mode="r")
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    by_id = {x["base_trial_id"]: x for x in d[role]}
    rows = []
    for n, trial in enumerate(p["conv_profile_states"][role], 1):
        sid, pair_id = trial["base_trial_id"], trial["pair_id"]
        item, pair = by_id[sid], pairs[pair_id]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError("horizon incoming drift")
        anchor = step(bundle, dense, incoming, token(pair["anchor_token_id"]), length, d, 30)
        donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
        spec = layout(anchor["cache"], rec, att)
        x = contrast(anchor["cache"], donor["cache"], spec)
        coeff = basis[:64] @ (x - mean)
        caches = {"NATIVE_ANCHOR": anchor["cache"], "NATIVE_DONOR": donor["cache"]}
        caches["EXACT_REC_CONV"] = transplant(anchor["cache"], donor["cache"], "REC+Conv", rec, att)[0]
        caches["EXACT_CONV"] = transplant(anchor["cache"], donor["cache"], "Conv", rec, att)[0]
        for k in (32, 64):
            approx = mean + coeff[:k] @ basis[:k]
            caches[f"GLOBAL_k{k}"] = inject(anchor["cache"], approx, 1, spec, rec, att)
        signatures = {}
        for name, cache in caches.items():
            state = cache
            checkpoints = {}
            for j, probe in enumerate(item["future_probe_tokens"], 1):
                out = step(bundle, dense, state, token(probe), length + j, d, 30)
                state = out["cache"]
                if j in (1, 2, 4):
                    checkpoints[j] = signature([out], scale)
            signatures[name] = checkpoints
        for h in (1, 2, 4):
            native_sig = signatures["NATIVE_ANCHOR"][h]
            donor_sig = signatures["NATIVE_DONOR"][h]
            for name in ("EXACT_REC_CONV", "EXACT_CONV", "GLOBAL_k32", "GLOBAL_k64"):
                rows.append({"role": role, "state_id": sid, "family": item["family"], "pair_id": pair_id, "condition": name, "horizon": h, "probe_hash": item["future_probe_hash"], "same_frozen_future_token_sequence": True, "TRAIN_only_global_basis": True, **metrics(signatures[name][h], native_sig, donor_sig)})
        print(f"V30 horizon {role} {n}/{len(p['conv_profile_states'][role])}", flush=True)
    frame = pd.DataFrame(rows)
    fp = root / OUT / f"write_horizon_{role}_v30.parquet"
    frame.to_parquet(fp, index=False, compression="zstd")
    profiles = {str(h): profile(g, p["gates"]) for h, g in frame.groupby("horizon")}
    jp = root / OUT / f"write_horizon_{role}_v30.json"
    summary = {"role": role, "states": len(p["conv_profile_states"][role]), "rows": len(frame), "profiles": profiles, "response_sha256": sha256_file(fp), "horizon_tokens_frozen_prewrite": True, "independent_final_opened": False}
    write_json_atomic(jp, summary)
    prior = "artifacts/transferable_natural_writes_v30_global_basis_fit.freeze.json" if role == "development" else "artifacts/transferable_natural_writes_v30_horizon_development.freeze.json"
    fr = stage_freeze(root, f"horizon_{role}", [SOURCE, str(fp.relative_to(root)), str(jp.relative_to(root)), prior], {"response_sha256": sha256_file(fp), "role": role})
    return {"freeze_digest": fr["freeze_digest"], "states": summary["states"], "rows": len(frame)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("development", "validation"))
    args = ap.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
