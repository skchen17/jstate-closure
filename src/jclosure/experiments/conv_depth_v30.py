"""Prospective exact-native Conv depth profile and development-only carrier choice."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.global_basis_v30 import load, token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/conv_depth_v30.py"
OUT = Path("results/v30/processed")


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def partial(recipient, donor, rec, conv_layers=(), rec_layers=()):
    result = clone_hybrid_cache(recipient)
    touched = []
    for channel, layers, field in (("Conv", conv_layers, "conv_states"), ("REC", rec_layers, "recurrent_states")):
        for layer in layers:
            a = getattr(recipient.layers[layer], field)
            b = getattr(donor.layers[layer], field)
            if a.shape != b.shape:
                raise RuntimeError("native partial transplant shape mismatch")
            setattr(result.layers[layer], field, b.detach().clone())
            touched.append((layer, field))
    if not all(torch.equal(getattr(result.layers[i], f), getattr(donor.layers[i], f)) for i, f in touched):
        raise RuntimeError("partial native writeback not exact")
    for layer in range(len(result.layers)):
        for field in ("conv_states", "recurrent_states", "keys", "values"):
            x = getattr(recipient.layers[layer], field, None)
            if isinstance(x, torch.Tensor) and (layer, field) not in touched and not torch.equal(x, getattr(result.layers[layer], field)):
                raise RuntimeError("untouched native field changed")
    return result


def conditions(p):
    rec = p["conv_layers"]
    result = {"FULL_CONV": rec}
    for kind, groups in p["conv_groups"].items():
        for j, layers in enumerate(groups):
            result[f"{kind.upper()}_{j:02d}"] = layers
    for i, layer in enumerate(rec):
        result[f"LEAVE_SINGLE_{i:02d}"] = [x for x in rec if x != layer]
    return result


def choose(frame, p):
    candidates = [(name, layers) for name, layers in conditions(p).items() if name == "FULL_CONV" or name.startswith(("QUARTILE_", "HALF_", "PREFIX_", "SUFFIX_"))]
    candidates.sort(key=lambda x: (len(x[1]), h(x[0])))
    prof = profile(frame, p["gates"])
    passing = [(name, layers) for name, layers in candidates if prof[name]["pass"]]
    name, layers = passing[0] if passing else ("FULL_CONV", p["conv_layers"])
    return {"name": name, "conv_layers": layers, "conv_layer_set_hash": h(layers), "selection_role": "development", "candidate_count": len(candidates), "candidate_profiles": {n: prof[n] for n, _ in candidates}, "fallback_full24": not bool(passing)}


@torch.no_grad()
def run(root, role):
    verify_stage(root, "transport_source_amendment")
    if role == "validation":
        verify_stage(root, "conv_depth_development")
    d, p, _, pairs = load(root)
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    by_id = {x["base_trial_id"]: x for x in d[role]}
    c = conditions(p)
    rows, conditional = [], []
    selected = None if role == "development" else json.loads((root / OUT / "conv_minimal_selection_v30.json").read_text())
    frozen = p["conv_profile_states"][role]
    for n, trial in enumerate(frozen, 1):
        sid, pair_id = trial["base_trial_id"], trial["pair_id"]
        item, pair = by_id[sid], pairs[pair_id]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError("Conv profile incoming drift")
        anchor = step(bundle, dense, incoming, token(pair["anchor_token_id"]), length, d, 30)
        donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
        probes = [token(x) for x in item["future_probe_tokens"]]
        future = lambda cache: signature([step(bundle, dense, cache, z, length + 1, d, 30) for z in probes], scale)
        native_sig, donor_sig = future(anchor["cache"]), future(donor["cache"])
        for name, layers in c.items():
            cache = partial(anchor["cache"], donor["cache"], rec, layers, ())
            sig = future(cache)
            rows.append({"role": role, "state_id": sid, "family": item["family"], "pair_id": pair_id, "condition": name, "conv_layers": json.dumps(layers), "conv_layer_set_hash": h(layers), "incoming_state_hash": item["incoming_state_hash"], "token_pair_hash": pair["token_pair_hash"], "probe_hash": item["future_probe_hash"], "probe_count": len(probes), "exact_native_conv_writeback": True, **metrics(sig, native_sig, donor_sig)})
        print(f"V30 Conv profile {role} {n}/{len(frozen)}", flush=True)
    frame = pd.DataFrame(rows)
    if role == "development":
        selected = choose(frame, p)
        write_json_atomic(root / OUT / "conv_minimal_selection_v30.json", selected)
    for n, trial in enumerate(frozen, 1):
        sid, pair_id = trial["base_trial_id"], trial["pair_id"]
        item, pair = by_id[sid], pairs[pair_id]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        anchor = step(bundle, dense, incoming, token(pair["anchor_token_id"]), length, d, 30)
        donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
        probes = [token(x) for x in item["future_probe_tokens"]]
        future = lambda cache: signature([step(bundle, dense, cache, z, length + 1, d, 30) for z in probes], scale)
        native_sig, donor_sig = future(anchor["cache"]), future(donor["cache"])
        variants = {"Y00_RECIPIENT": ([], []), "Y10_REC_ONLY": ([], rec), "Y01_CONV_ONLY": (rec, []), "Y11_REC_CONV": (rec, rec), "MIN_CONV": (selected["conv_layers"], []), "MIN_CONV_MATCHED_REC": (selected["conv_layers"], selected["conv_layers"]), "MIN_CONV_ALL_REC": (selected["conv_layers"], rec)}
        vectors = {}
        for name, (conv, recurrence) in variants.items():
            cache = partial(anchor["cache"], donor["cache"], rec, conv, recurrence)
            sig = future(cache)
            vectors[name] = sig
            conditional.append({"role": role, "state_id": sid, "family": item["family"], "pair_id": pair_id, "condition": name, "minimal_conv_layer_set_hash": selected["conv_layer_set_hash"], **metrics(sig, native_sig, donor_sig)})
        interaction = vectors["Y11_REC_CONV"] - vectors["Y10_REC_ONLY"] - vectors["Y01_CONV_ONLY"] + vectors["Y00_RECIPIENT"]
        denom = max(float(np.linalg.norm(donor_sig - native_sig)), 1e-12)
        conditional.append({"role": role, "state_id": sid, "family": item["family"], "pair_id": pair_id, "condition": "REC_CONV_INTERACTION", "minimal_conv_layer_set_hash": selected["conv_layer_set_hash"], "interaction_ratio": float(np.linalg.norm(interaction)) / denom})
    paths = {"profile": root / OUT / f"conv_layer_profile_{role}_v30.parquet", "conditional": root / OUT / f"rec_conv_conditional_{role}_v30.parquet"}
    frame.to_parquet(paths["profile"], index=False, compression="zstd")
    pd.DataFrame(conditional).to_parquet(paths["conditional"], index=False, compression="zstd")
    jp = root / OUT / f"conv_depth_{role}_v30.json"
    summary = {"role": role, "profile_states": len(frozen), "profile_rows": len(frame), "profile_conditions": len(c), "profiles": profile(frame, p["gates"]), "selection": selected, "response_sha256": {k: sha256_file(v) for k, v in paths.items()}, "validation_not_used_to_select_layer_set": True, "independent_final_opened": False}
    write_json_atomic(jp, summary)
    prior = "artifacts/transferable_natural_writes_v30_transport_source_amendment.freeze.json" if role == "development" else "artifacts/transferable_natural_writes_v30_conv_depth_development.freeze.json"
    inputs = [SOURCE, *(str(v.relative_to(root)) for v in paths.values()), str(jp.relative_to(root)), prior]
    if role == "development":
        inputs.append(str((root / OUT / "conv_minimal_selection_v30.json").relative_to(root)))
    fr = stage_freeze(root, f"conv_depth_{role}", inputs, {"summary_sha256": sha256_file(jp), "minimal_set_hash": selected["conv_layer_set_hash"], "profile_states": len(frozen)})
    return {"freeze_digest": fr["freeze_digest"], "selection": {k: selected[k] for k in ("name", "conv_layers", "conv_layer_set_hash")}, "profile_states": len(frozen), "conditions": len(c)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("development", "validation"))
    args = ap.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
