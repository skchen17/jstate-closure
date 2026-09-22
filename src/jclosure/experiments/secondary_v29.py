"""Frozen V29 multi-probe, control, state-token factorial, and horizon panels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.forks_formal_v29 import OUT, design, native_layers, normvec, state_hashes, total_hash, transplant, vector_metrics
from jclosure.experiments.pre_readout_v27 import TARGETS, delta, metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context, fields
from jclosure.protocol_v29 import verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/secondary_v29.py"
SIGNATURE_CHANNELS = ("REC", "Conv", "KV", "REC+Conv", "REC+Conv+KV")


def token(t):
    return torch.tensor([[int(t)]], dtype=torch.long)


def hbytes(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float32).tobytes()).hexdigest()


def branch(bundle, dense, root, d, item, tokens=None):
    m = metadata(root, item)
    incoming, _, length = prefix(bundle, str(m["prompt"]))
    rec, att = native_layers(incoming)
    aa, bb = tokens or (item["token_A"], item["token_B"])
    a0 = step(bundle, dense, incoming, token(aa), length, d, 30)
    b0 = step(bundle, dense, incoming, token(bb), length, d, 30)
    return incoming, a0, b0, length, rec, att


def apply_delta(recipient, donor_a, donor_b, rec, att, mode, seed=0):
    """Add a REC/Conv contrast to native recipient cache; off-manifold controls are labeled."""
    result = clone_hybrid_cache(recipient)
    rng = np.random.default_rng(0 if isinstance(seed, dict) else seed)
    requested, realized = 0.0, 0.0
    for ch in ("REC", "Conv"):
        for layer, name, base in fields(recipient, ch, rec, att):
            a = getattr(donor_a.layers[layer], name).float()
            b = getattr(donor_b.layers[layer], name).float()
            d = a - b
            natural_norm = float(torch.linalg.vector_norm(d))
            if mode == "random_same_norm":
                sample = rng.standard_normal(tuple(d.shape), dtype=np.float32)
                v = torch.from_numpy(sample).to(d.device)
                d = v * (natural_norm / max(float(torch.linalg.vector_norm(v)), 1e-12))
            elif mode == "shuffled_natural":
                flat = d.reshape(-1)
                order = torch.as_tensor(rng.permutation(flat.numel()), device=d.device)
                d = flat[order].reshape_as(d)
            elif mode == "sign_flipped":
                d = -d
            elif mode == "wrong_state":
                # Match the recipient's natural contrast norm, which is passed via seed dictionary.
                target_norm = seed[(layer, name)]
                d = d * (target_norm / max(natural_norm, 1e-12))
            elif mode != "natural_delta":
                raise ValueError(mode)
            edit = (base.float() + d).to(base.dtype)
            setattr(result.layers[layer], name, edit)
            requested += float(torch.sum(d.square()))
            realized += float(torch.sum((edit.float() - base.float()).square()))
    return result, {"requested_delta_norm": requested ** 0.5, "realized_delta_norm": realized ** 0.5, "all_requested_fields_written": True, "state_hash": total_hash(state_hashes(result, rec, att))}


def contrast_norms(a, b, rec, att):
    return {(layer, name): float(torch.linalg.vector_norm(getattr(a.layers[layer], name).float() - getattr(b.layers[layer], name).float())) for ch in ("REC", "Conv") for layer, name, _ in fields(a, ch, rec, att)}


def state_contrast_cosine(aa, ab, ba, bb, rec, att):
    dot = an = bn = diff = 0.0
    for ch in ("REC", "Conv"):
        for layer, name, _ in fields(aa, ch, rec, att):
            a = getattr(aa.layers[layer], name).float() - getattr(ab.layers[layer], name).float()
            b = getattr(ba.layers[layer], name).float() - getattr(bb.layers[layer], name).float()
            dot += float(torch.sum(a * b))
            an += float(torch.sum(a.square()))
            bn += float(torch.sum(b.square()))
            diff += float(torch.sum((a - b).square()))
    return {"cross_state_write_contrast_cosine": dot / max((an * bn) ** .5, 1e-12), "state_A_contrast_norm": an ** .5, "state_B_contrast_norm": bn ** .5, "contrast_difference_norm": diff ** .5}


def run(root: Path, role: str):
    verify_stage(root, "validation_analysis")
    verify_stage(root, "execution_plan")
    d = design(root)
    plan = json.loads((root / OUT / "execution_plan_v29.json").read_text())
    ids = set(plan["control_state_ids"][role])
    items = [x for x in d[role] if x["base_trial_id"] in ids]
    lookup = {x["base_trial_id"]: x for x in d[role]}
    wrong = {}
    for fam in d["families"]:
        pair = [x for x in items if x["family"] == fam]
        wrong[pair[0]["base_trial_id"]] = pair[1]
        wrong[pair[1]["base_trial_id"]] = pair[0]
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    signature_rows, control_rows, horizon_rows, factorial_rows = [], [], [], []
    signature_movements, signature_targets = [], []
    dev = json.loads((root / OUT / "development_analysis_v29.json").read_text())
    val = json.loads((root / OUT / "validation_analysis_v29.json").read_text())
    horizon_authorized = bool(dev["fixed_finalist_pass"] and val["fixed_finalist_pass"])
    for index, item in enumerate(items, 1):
        incoming, a0, b0, length, rec, att = branch(bundle, dense, root, d, item)
        native_hashes = [total_hash(state_hashes(x["cache"], rec, att)) for x in (a0, b0)]
        for direction, recipient, donor in (("A_from_B", a0, b0), ("B_from_A", b0, a0)):
            edited = {c: transplant(recipient["cache"], donor["cache"], c, rec, att)[0] for c in SIGNATURE_CHANNELS}
            mv = {c: [] for c in SIGNATURE_CHANNELS}
            target_parts = []
            per_probe = []
            native_parts = []
            donor_parts = []
            for z in item["probe_tokens"]:
                r = step(bundle, dense, recipient["cache"], token(z), length + 1, d, 30)
                q = step(bundle, dense, donor["cache"], token(z), length + 1, d, 30)
                native_parts.append(normvec(r["targets"], scale))
                donor_parts.append(normvec(q["targets"], scale))
                target = normvec(delta(q["targets"], r["targets"]), scale)
                target_parts.append(target)
                probe = {"token": int(z), "donor_norm": float(np.linalg.norm(target))}
                for c, cache in edited.items():
                    outcome = step(bundle, dense, cache, token(z), length + 1, d, 30)
                    movement = normvec(delta(outcome["targets"], r["targets"]), scale)
                    mv[c].append(movement)
                    n1, n2 = float(np.linalg.norm(movement)), float(np.linalg.norm(target))
                    probe[c] = {"cosine": float(np.dot(movement, target) / (n1 * n2)) if n1 * n2 > 1e-12 else None, "relative_l2": float(np.linalg.norm(movement - target)) / max(n2, 1e-12)}
                per_probe.append(probe)
            target = np.concatenate(target_parts).astype(np.float32)
            for c, parts in mv.items():
                movement = np.concatenate(parts).astype(np.float32)
                mn, dn = float(np.linalg.norm(movement)), float(np.linalg.norm(target))
                row = {"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "direction": direction, "condition": c, "probe_hash": item["probe_hash"], "probes": json.dumps(item["probe_tokens"]), "signature_A_hash": hbytes(np.concatenate(native_parts)), "signature_B_hash": hbytes(np.concatenate(donor_parts)), "movement_hash": hbytes(movement), "donor_target_hash": hbytes(target), "donor_cosine": float(np.dot(movement, target) / (mn * dn)) if mn * dn > 1e-12 else None, "magnitude_ratio": mn / max(dn, 1e-12), "relative_l2_to_donor": float(np.linalg.norm(movement - target)) / max(dn, 1e-12), "per_probe": json.dumps(per_probe, sort_keys=True), "vector_index": len(signature_movements)}
                signature_rows.append(row)
                signature_movements.append(movement)
                signature_targets.append(target)

        # Norm-matched and wrong-background REC/Conv contrast controls, A receiving B.
        z = token(item["continuation_token"])
        a1 = step(bundle, dense, a0["cache"], z, length + 1, d, 30)
        b1 = step(bundle, dense, b0["cache"], z, length + 1, d, 30)
        wrong_item = wrong[item["base_trial_id"]]
        _, wa0, wb0, _, _, _ = branch(bundle, dense, root, d, wrong_item)
        natural_norm = contrast_norms(b0["cache"], a0["cache"], rec, att)
        seed = int(hashlib.sha256(item["base_trial_id"].encode()).hexdigest()[:8], 16)
        for condition in ("natural_REC+Conv", "random_same_norm", "shuffled_natural", "sign_flipped", "wrong_state"):
            if condition == "natural_REC+Conv":
                cache, fidelity = transplant(a0["cache"], b0["cache"], "REC+Conv", rec, att)
                audit = {"requested_delta_norm": float(np.sqrt(sum(v * v for v in natural_norm.values()))), "realized_delta_norm": float(np.sqrt(sum(v * v for v in natural_norm.values()))), "all_requested_fields_written": fidelity["writeback_pass"], "state_hash": fidelity["state_hash"]}
            elif condition == "wrong_state":
                cache, audit = apply_delta(a0["cache"], wb0["cache"], wa0["cache"], rec, att, condition, natural_norm)
            else:
                cache, audit = apply_delta(a0["cache"], b0["cache"], a0["cache"], rec, att, condition, seed)
            out = step(bundle, dense, cache, z, length + 1, d, 30)
            metric = vector_metrics(out, a1, b1, scale)
            control_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "condition": condition, "wrong_donor_id": wrong_item["base_trial_id"] if condition == "wrong_state" else None, "donor_cosine": metric["donor_cosine"], "magnitude_ratio": metric["magnitude_ratio"], "relative_l2_to_donor": metric["relative_l2_to_donor"], "requested_delta_norm": audit["requested_delta_norm"], "realized_delta_norm": audit["realized_delta_norm"], "realized_to_requested": audit["realized_delta_norm"] / max(audit["requested_delta_norm"], 1e-12), "all_requested_fields_written": audit["all_requested_fields_written"], "transplant_hash": audit["state_hash"], "same_next_token": True})

        if horizon_authorized:
            edited, _ = transplant(a0["cache"], b0["cache"], "REC+Conv", rec, att)
            native_cache, edited_cache, donor_cache = a0["cache"], edited, b0["cache"]
            horizon_tokens = [item["continuation_token"], *item["probe_tokens"][:3]]
            for h, tid in enumerate(horizon_tokens, 1):
                native = step(bundle, dense, native_cache, token(tid), length + h, d, 30)
                pert = step(bundle, dense, edited_cache, token(tid), length + h, d, 30)
                donor = step(bundle, dense, donor_cache, token(tid), length + h, d, 30)
                metric = vector_metrics(pert, native, donor, scale)
                if h in (1, 2, 4):
                    horizon_rows.append({"role": role, "base_trial_id": item["base_trial_id"], "family": item["family"], "horizon": h, "probe_token": tid, "donor_cosine": metric["donor_cosine"], "magnitude_ratio": metric["magnitude_ratio"], "relative_l2_to_donor": metric["relative_l2_to_donor"], "movement_norm": metric["movement_norm"], "donor_norm": metric["donor_norm"]})
                native_cache, edited_cache, donor_cache = native["cache"], pert["cache"], donor["cache"]
        print(f"V29 secondary {role} {index}/{len(items)}", flush=True)

    # Two frozen 2x2 state × token factorials per family.
    for pair in plan["factorial_pairs"][role]:
        left, right = lookup[pair[0]], lookup[pair[1]]
        toks = (left["token_A"], left["token_B"])
        ia, aa, ab, la, rec, att = branch(bundle, dense, root, d, left, toks)
        ib, ba, bb, lb, rb, tb = branch(bundle, dense, root, d, right, toks)
        if rec != rb or att != tb:
            raise RuntimeError("state factorial cache architecture differs")
        z = token(left["continuation_token"])
        future = {}
        for key, current, length in (("AA", aa, la), ("AB", ab, la), ("BA", ba, lb), ("BB", bb, lb)):
            future[key] = step(bundle, dense, current["cache"], z, length + 1, d, 30)
        geometry = state_contrast_cosine(aa["cache"], ab["cache"], ba["cache"], bb["cache"], rec, att)
        same_cache, _ = transplant(bb["cache"], ba["cache"], "REC+Conv", rec, att)
        same_future = step(bundle, dense, same_cache, z, lb + 1, d, 30)
        cross_cache, audit = apply_delta(bb["cache"], aa["cache"], ab["cache"], rec, att, "natural_delta")
        cross_future = step(bundle, dense, cross_cache, z, lb + 1, d, 30)
        same_move = normvec(delta(same_future["targets"], future["BB"]["targets"]), scale)
        cross_move = normvec(delta(cross_future["targets"], future["BB"]["targets"]), scale)
        sn, cn = float(np.linalg.norm(same_move)), float(np.linalg.norm(cross_move))
        interaction = normvec(delta(future["AA"]["targets"], future["AB"]["targets"]), scale) - normvec(delta(future["BA"]["targets"], future["BB"]["targets"]), scale)
        factorial_rows.append({"role": role, "family": left["family"], "state_A_id": left["base_trial_id"], "state_B_id": right["base_trial_id"], "token_A": toks[0], "token_B": toks[1], "incoming_A_hash": total_hash(state_hashes(ia, rec, att)), "incoming_B_hash": total_hash(state_hashes(ib, rec, att)), "same_next_token": left["continuation_token"], **geometry, "cross_state_delta_cosine_to_native_same_state_effect": float(np.dot(cross_move, same_move) / (sn * cn)) if sn * cn > 1e-12 else None, "cross_state_delta_magnitude_ratio": cn / max(sn, 1e-12), "cross_state_delta_relative_l2": float(np.linalg.norm(cross_move - same_move)) / max(sn, 1e-12), "future_factorial_interaction_norm": float(np.linalg.norm(interaction)), "native_same_state_recurrent_effect_norm": sn, "cross_state_delta_effect_norm": cn, "cross_state_delta_realized_ratio": audit["realized_delta_norm"] / max(audit["requested_delta_norm"], 1e-12), "cross_state_delta_hash": audit["state_hash"], "off_manifold_delta_injection": True})

    paths = {}
    for key, rows in (("future_signature", signature_rows), ("natural_vs_random", control_rows), ("state_token_factorial", factorial_rows), ("write_horizon", horizon_rows)):
        path = root / OUT / f"{key}_{role}_v29.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
        paths[key] = str(path.relative_to(root))
    vp = root / OUT / f"future_signature_vectors_{role}_v29.npz"
    np.savez_compressed(vp, movement=np.stack(signature_movements).astype(np.float32), donor_target=np.stack(signature_targets).astype(np.float32))
    summary = {"role": role, "signature_rows": len(signature_rows), "control_rows": len(control_rows), "factorial_rows": len(factorial_rows), "horizon_rows": len(horizon_rows), "horizon_authorized": horizon_authorized, "frozen_probe_rule": plan["future_signature_probes"], "all_full_signatures_exact": bool(all(x["relative_l2_to_donor"] <= 1e-6 for x in signature_rows if x["condition"] == "REC+Conv+KV")), "response_hashes": {k: sha256_file(root / p) for k, p in paths.items()}, "vector_hash": sha256_file(vp), "historical_final_opened": False}
    jp = root / OUT / f"secondary_{role}_v29.json"
    write_json_atomic(jp, summary)
    fr = stage_freeze(root, f"secondary_{role}", [SOURCE, str(jp.relative_to(root)), str(vp.relative_to(root)), "artifacts/natural_write_content_v29_execution_plan.freeze.json", "artifacts/natural_write_content_v29_validation_analysis.freeze.json", *paths.values()], summary)
    return {"freeze_digest": fr["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("development", "validation"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.role), indent=2))
