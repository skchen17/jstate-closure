"""Same-incoming-state natural token forks and exact native partial write transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.pre_readout_v27 import TARGETS, delta, metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import ATTR, allfields, channel_hash, context, fields, thash
from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/forks_v29.py"
OUT = Path("results/v29/processed")
CHANNELS = ("REC", "Conv", "KV", "REC+Conv", "REC+KV", "Conv+KV", "REC+Conv+KV")


def design(root):
    return json.loads((root / OUT / "design_v29.json").read_text())


def state_hashes(cache, rec, att):
    return {ch: channel_hash(cache, ch, rec, att) for ch in ATTR}


def total_hash(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def normvec(targets, scale):
    return np.concatenate([np.asarray(targets[k], np.float64).ravel() / max(float(scale[k]), 1e-12) for k in TARGETS])


def vector_metrics(out, native, donor, scale):
    movement = normvec(delta(out["targets"], native["targets"]), scale)
    target = normvec(delta(donor["targets"], native["targets"]), scale)
    mn, dn = float(np.linalg.norm(movement)), float(np.linalg.norm(target))
    cosine = float(np.dot(movement, target) / (mn * dn)) if mn * dn > 1e-12 else None
    return {
        "donor_cosine": cosine,
        "magnitude_ratio": mn / max(dn, 1e-12),
        "relative_l2_to_donor": float(np.linalg.norm(movement - target)) / max(dn, 1e-12),
        "donor_norm": dn,
        "movement_norm": mn,
        "movement": movement.astype(np.float32),
        "target": target.astype(np.float32),
    }


def transplant(recipient, donor, channels, rec, att):
    """Copy native REC/Conv fields, but only the newly appended KV slot."""
    chosen = set(channels.split("+"))
    result = clone_hybrid_cache(recipient)
    touched = []
    for ch in chosen:
        if ch not in ATTR:
            raise ValueError(ch)
        for layer, field, source in fields(donor, ch, rec, att):
            base = getattr(recipient.layers[layer], field)
            if source.shape != base.shape:
                raise RuntimeError(f"shape mismatch {layer}:{field}")
            if ch == "KV":
                # The same prefix must make all earlier KV slots bitwise equal.
                if not torch.equal(source[..., :-1, :], base[..., :-1, :]):
                    raise RuntimeError("different incoming KV background")
                edited = base.detach().clone()
                edited[..., -1:, :] = source[..., -1:, :]
            else:
                edited = source.detach().clone()
            setattr(result.layers[layer], field, edited)
            touched.append((layer, field))
    exact = all(torch.equal(getattr(result.layers[i], n), getattr(donor.layers[i], n)) for i, n in touched)
    untouched = all(torch.equal(x, getattr(result.layers[i], n)) for _, i, n, x in allfields(recipient, rec, att) if (i, n) not in touched)
    all_channels = state_hashes(result, rec, att)
    return result, {"requested_exact": bool(exact), "untouched_exact": bool(untouched), "writeback_pass": bool(exact and untouched), "field_count": len(touched), "state_hashes": all_channels, "state_hash": total_hash(all_channels)}


def write_geometry(incoming, a, b, rec, att):
    rows = []
    for ch, layer, name, x in allfields(incoming, rec, att):
        aa = getattr(a.layers[layer], name)
        bb = getattr(b.layers[layer], name)
        if ch == "KV":
            if aa.shape[-2] != x.shape[-2] + 1 or bb.shape[-2] != x.shape[-2] + 1:
                raise RuntimeError("KV append length invalid")
            if not torch.equal(aa[..., :-1, :], bb[..., :-1, :]) or not torch.equal(aa[..., :-1, :], x):
                raise RuntimeError("KV prefix not identical")
            av, bv = aa[..., -1:, :].float(), bb[..., -1:, :].float()
            kind = "new_KV_slot_A_minus_B"
        else:
            if aa.shape != x.shape or bb.shape != x.shape:
                raise RuntimeError("REC/Conv shape invalid")
            av, bv = aa.float() - x.float(), bb.float() - x.float()
            kind = "out_minus_in_write_A_minus_B"
        contrast = av - bv
        an = float(torch.linalg.vector_norm(av))
        bn = float(torch.linalg.vector_norm(bv))
        cn = float(torch.linalg.vector_norm(contrast))
        cosine = float(torch.sum(av * bv) / (an * bn)) if an * bn > 1e-12 else None
        rows.append({"channel": ch, "layer": layer, "field": name, "kind": kind, "A_norm": an, "B_norm": bn, "contrast_norm": cn, "A_B_cosine": cosine, "incoming_hash": thash(x), "out_A_hash": thash(aa), "out_B_hash": thash(bb), "contrast_hash": thash(contrast)})
    return rows


def _token(t):
    return torch.tensor([[int(t)]], dtype=torch.long)


def run(root: Path, role: str):
    verify_stage(root, "design")
    if role == "validation":
        verify_stage(root, "development_analysis")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening["final_opened"]:
            return {"role": role, "final_opened": False, "reason": opening["reason"]}
    d = design(root)
    cfg = verify(root)["config"]
    bundle, dense, _, rec, att, _ = context(root)
    scale = scales(root)
    items = d[role]
    if role == "independent_final":
        channels = ("REC+Conv",)
    else:
        channels = CHANNELS
    fork_rows, geo_rows, move_rows, audit_rows = [], [], [], []
    move_vectors, target_vectors = [], []
    for index, item in enumerate(items, 1):
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        ih = state_hashes(incoming, rec, att)
        if ih != item["incoming_state_hashes"]:
            raise RuntimeError(f"frozen incoming state changed: {item['base_trial_id']}")
        a0 = step(bundle, dense, incoming, _token(item["token_A"]), length, d, 30)
        b0 = step(bundle, dense, incoming, _token(item["token_B"]), length, d, 30)
        ah, bh = state_hashes(a0["cache"], rec, att), state_hashes(b0["cache"], rec, att)
        if ah == bh:
            raise RuntimeError("fork outgoing caches identical")
        z = _token(item["continuation_token"])
        a1 = step(bundle, dense, a0["cache"], z, length + 1, d, 30)
        b1 = step(bundle, dense, b0["cache"], z, length + 1, d, 30)
        future = normvec(delta(b1["targets"], a1["targets"]), scale)
        future_norm = float(np.linalg.norm(future))
        future_q = float(np.mean([float(np.linalg.norm(b1["targets"][k] - a1["targets"][k])) / max(float(scale[k]), 1e-12) for k in TARGETS]))
        current_q = float(np.mean([float(np.linalg.norm(b0["targets"][k] - a0["targets"][k])) / max(float(scale[k]), 1e-12) for k in TARGETS]))
        replay = step(bundle, dense, incoming, _token(item["token_A"]), length, d, 30) if role == "calibration" else None
        replay_exact = bool(replay is None or (state_hashes(replay["cache"], rec, att) == ah and all(np.array_equal(replay["targets"][k], a0["targets"][k]) for k in TARGETS)))
        fork_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "token_A": item["token_A"], "token_B": item["token_B"], "next_token": item["continuation_token"], "fork_token_hash": item["fork_token_hash"], "probe_hash": item["probe_hash"], "incoming_state_hash": total_hash(ih), "incoming_channel_hashes": json.dumps(ih, sort_keys=True), "out_A_state_hash": total_hash(ah), "out_B_state_hash": total_hash(bh), "out_A_channel_hashes": json.dumps(ah, sort_keys=True), "out_B_channel_hashes": json.dumps(bh, sort_keys=True), "current_q": current_q, "future_q": future_q, "future_norm": future_norm, "same_token_replay_exact": replay_exact, "incoming_identical": True, "prefix_identical": True, "current_token_only_differs": True, "natural_current_forward": True, "same_next_token": True})
        for g in write_geometry(incoming, a0["cache"], b0["cache"], rec, att):
            geo_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], **g})
        for condition in channels:
            for direction, recipient, donor, current in (("A_from_B", a0, b0, a1), ("B_from_A", b0, a0, b1)):
                goal = b1 if direction == "A_from_B" else a1
                edited, fidelity = transplant(recipient["cache"], donor["cache"], condition, rec, att)
                future_edited = step(bundle, dense, edited, z, length + 1, d, 30)
                metric = vector_metrics(future_edited, current, goal, scale)
                idx = len(move_vectors)
                move_vectors.append(metric.pop("movement"))
                target_vectors.append(metric.pop("target"))
                row = {"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "condition": condition, "direction": direction, "token_A": item["token_A"], "token_B": item["token_B"], "next_token": item["continuation_token"], "vector_index": idx, "future_q": future_q, "current_preserved": True, "same_next_token": True, **metric, **{k: float(np.linalg.norm(future_edited["targets"][k] - current["targets"][k])) / max(float(scale[k]), 1e-12) for k in TARGETS}}
                move_rows.append(row)
                audit_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "condition": condition, "direction": direction, "requested_exact": fidelity["requested_exact"], "untouched_exact": fidelity["untouched_exact"], "writeback_pass": fidelity["writeback_pass"], "field_count": fidelity["field_count"], "transplant_hash": fidelity["state_hash"], "transplant_channel_hashes": json.dumps(fidelity["state_hashes"], sort_keys=True), "recipient_current_unchanged": True, "next_token_identical": True})
        print(f"V29 forks {role} {index}/{len(items)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, rows in (("forks", fork_rows), ("geometry", geo_rows), ("transfer", move_rows), ("audit", audit_rows)):
        p = root / OUT / f"{key}_{role}_v29.parquet"
        pd.DataFrame(rows).to_parquet(p, index=False, compression="zstd")
        paths[key] = str(p.relative_to(root))
    vp = root / OUT / f"transfer_vectors_{role}_v29.npz"
    np.savez_compressed(vp, movement=np.stack(move_vectors).astype(np.float32), donor_target=np.stack(target_vectors).astype(np.float32))
    summary = {"role": role, "states": len(items), "transfer_rows": len(move_rows), "all_writeback_pass": all(x["writeback_pass"] for x in audit_rows), "all_replay_exact": all(x["same_token_replay_exact"] for x in fork_rows), "all_incoming_identical": True, "future_responses_observed_after_design_freeze": True, "role_response_hashes": {k: sha256_file(root / p) for k, p in paths.items()}, "vectors_sha256": sha256_file(vp), "historical_final_opened": False, "independent_final_opened": role == "independent_final"}
    jp = root / OUT / f"forks_{role}_v29.json"
    write_json_atomic(jp, summary)
    prior = "artifacts/natural_write_content_v29_design.freeze.json" if role in ("calibration", "development") else "artifacts/natural_write_content_v29_development_analysis.freeze.json" if role == "validation" else "artifacts/natural_write_content_v29_final_opening.freeze.json"
    fr = stage_freeze(root, f"forks_{role}", [SOURCE, str(jp.relative_to(root)), str(vp.relative_to(root)), prior, *paths.values()], summary)
    return {"freeze_digest": fr["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("calibration", "development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
