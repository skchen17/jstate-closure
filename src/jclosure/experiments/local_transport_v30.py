"""TRAIN-only local atlas and held-out cross-state coordinate transport."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.global_basis_v30 import SCRATCH, load, token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.realization_v29 import contrast, inject, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/local_transport_v30.py"
OUT = Path("results/v30/processed")


def digest(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def local_fit(bundle, dense, root, d, item, pairs, rec=None, att=None, incoming=None, length=None):
    if incoming is None:
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
    if rec is None or att is None:
        rec, att = native_layers(incoming)
    if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
        raise RuntimeError(f"incoming state hash drift: {item['base_trial_id']}")
    ids = [p["pair_id"] for p in d["token_pair_library"] if p["role"] == "TOKEN_TRAIN" and p["pair_id"] in item["eligible_pair_ids"]]
    anchor = step(bundle, dense, incoming, token(pairs[ids[0]]["anchor_token_id"]), length, d, 30)
    spec = layout(anchor["cache"], rec, att)
    values = []
    hashes = []
    for pair_id in ids:
        donor = step(bundle, dense, incoming, token(pairs[pair_id]["candidate_token_id"]), length, d, 30)
        x = contrast(anchor["cache"], donor["cache"], spec)
        values.append(x)
        hashes.append(digest(x))
    matrix = np.stack(values).astype(np.float32)
    gram = np.asarray(matrix @ matrix.T, dtype=np.float64)
    centered = gram - gram.mean(0)[None, :] - gram.mean(1)[:, None] + gram.mean()
    eig, vec = np.linalg.eigh(centered)
    order = np.argsort(eig)[::-1]
    eig, vec = np.maximum(eig[order], 0), vec[:, order]
    rank = int(np.sum(eig > max(eig[0], 1e-12) * 1e-9))
    if rank < 64:
        raise RuntimeError(f"local k64 unidentifiable: {item['base_trial_id']} rank={rank}")
    mean = np.asarray(matrix.mean(0, dtype=np.float64), dtype=np.float32)
    coeff = (vec[:, :64] / np.sqrt(np.maximum(eig[:64], 1e-12))[None, :]).T
    basis = np.asarray(coeff @ matrix, dtype=np.float32)
    coords = vec[:, :64] * np.sqrt(eig[:64])[None, :]
    total = float(eig.sum())
    ranks = {f"r{int(q*100)}": int(np.searchsorted(np.cumsum(eig) / max(total, 1e-12), q) + 1) for q in (.9, .95, .99)}
    record = {"state_id": item["base_trial_id"], "family": item["family"], "train_pair_ids": ids, "train_contrast_hashes": hashes, "incoming_state_hash": item["incoming_state_hash"], "rank": rank, "ranks": ranks, "mean_sha256": digest(mean), "basis_sha256": digest(basis), "coordinates_sha256": digest(coords), "TOKEN_TRAIN_only": True}
    return {"record": record, "basis": basis, "mean": mean, "coords": dict(zip(ids, coords)), "incoming": incoming, "anchor": anchor, "length": length, "rec": rec, "att": att, "spec": spec}


def transport(source, target, common, k, method, alpha):
    x = np.stack([source["coords"][p][:k] for p in common]).astype(np.float64)
    y = np.stack([target["coords"][p][:k] for p in common]).astype(np.float64)
    if method == "RIDGE":
        mapping = np.linalg.solve(x.T @ x + alpha * np.eye(k), x.T @ y)
    else:
        u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
        mapping = u @ vt
    return mapping.astype(np.float32)


def sources(root):
    verify_stage(root, "transport_source_amendment")
    d, p, _, pairs = load(root)
    amendment = json.loads((root / OUT / "transport_source_amendment_v30.json").read_text())
    fit_items = {x["base_trial_id"]: x for x in d["development"]}
    bundle, dense, _, _, _, _ = context(root)
    records = {}
    for family, sid in amendment["source_by_family"].items():
        z = local_fit(bundle, dense, root, d, fit_items[sid], pairs)
        path = SCRATCH / f"local_source_{family}_v30.npz"
        np.savez(path, basis=z["basis"], mean=z["mean"], pair_ids=np.array(list(z["coords"])), coords=np.stack(list(z["coords"].values())).astype(np.float32))
        records[family] = {**z["record"], "scratch_npz_path": str(path), "scratch_sha256": sha256_file(path)}
        print(f"V30 canonical source {family} TRAIN-pair basis fitted", flush=True)
    jp = root / OUT / "local_source_bases_v30.json"
    write_json_atomic(jp, {"sources": records, "source_count": len(records), "heldout_token_write_not_used": True, "future_response_not_observed": True})
    fr = stage_freeze(root, "local_source_bases", [SOURCE, str(jp.relative_to(root)), "artifacts/transferable_natural_writes_v30_transport_source_amendment.freeze.json"], {"source_record_sha256": sha256_file(jp), "source_count": len(records)})
    return {"freeze_digest": fr["freeze_digest"], "source_count": len(records)}


@torch.no_grad()
def evaluate(root, role):
    verify_stage(root, "local_source_bases")
    if role == "validation":
        verify_stage(root, "local_transport_development")
    d, p, _, pairs = load(root)
    amendment = json.loads((root / OUT / "transport_source_amendment_v30.json").read_text())
    source_info = json.loads((root / OUT / "local_source_bases_v30.json").read_text())["sources"]
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    ids = p["development_holdout_ids"] if role == "development" else p["validation_ids"]
    by_id = {x["base_trial_id"]: x for x in d[role]}
    fit_items = {x["base_trial_id"]: x for x in d["development"]}
    rows, geometry, local_records = [], [], []
    current_family, source = None, None
    for n, sid in enumerate(ids, 1):
        item = by_id[sid]
        family = item["family"]
        if family != current_family:
            current_family = family
            source_id = amendment["source_by_family"][family]
            source_item = fit_items[source_id]
            sx = np.load(SCRATCH / f"local_source_{family}_v30.npz")
            source = {"basis": sx["basis"], "mean": sx["mean"], "coords": dict(zip(sx["pair_ids"].tolist(), sx["coords"]))}
            sm = metadata(root, source_item)
            source["incoming"], _, source["length"] = prefix(bundle, str(sm["prompt"]))
            source["anchor"] = step(bundle, dense, source["incoming"], token(pairs[next(iter(source["coords"]))]["anchor_token_id"]), source["length"], d, 30)
            source["rec"], source["att"] = native_layers(source["incoming"])
            source["spec"] = layout(source["anchor"]["cache"], source["rec"], source["att"])
            if state_hashes(source["incoming"], source["rec"], source["att"]) != source_item["incoming_state_hashes"]:
                raise RuntimeError("source incoming drift")
        target = local_fit(bundle, dense, root, d, item, pairs)
        local_records.append(target["record"])
        overlap = np.asarray(source["basis"][:32] @ target["basis"][:32].T, dtype=np.float64)
        singular = np.linalg.svd(overlap, compute_uv=False)
        geometry.append({"role": role, "state_id": sid, "family": family, "source_state_id": source_id, "k": 32, "principal_cosine_min": float(singular.min()), "principal_cosine_median": float(np.median(singular)), "principal_angle_max_degrees": float(np.degrees(np.arccos(np.clip(singular.min(), -1, 1)))), "chordal_distance": float(np.sqrt(np.maximum(0, 32 - np.sum(singular**2)))), "source_basis_sha256": source_info[family]["basis_sha256"], "target_basis_sha256": target["record"]["basis_sha256"]})
        common = [pair_id for pair_id in target["coords"] if pair_id in source["coords"]]
        if len(common) < 65:
            raise RuntimeError(f"TRAIN transport pairs <65: {sid}")
        maps = {(method, k): transport(source, target, common, k, method, p["transport_ridge_alpha"]) for method in ("RIDGE", "PROCRUSTES") for k in (32, 64)}
        map_hashes = {f"{method}_k{k}": digest(mat) for (method, k), mat in maps.items()}
        for pair_id in p["eval_pairs"][role][sid]["heldout_token_pair_ids"]:
            if amendment["transport_source_state"][f"{sid}:{pair_id}"] != source_id:
                raise RuntimeError("transport map pairing drift")
            pair = pairs[pair_id]
            donor = step(bundle, dense, target["incoming"], token(pair["candidate_token_id"]), target["length"], d, 30)
            x_target = contrast(target["anchor"]["cache"], donor["cache"], target["spec"])
            sdonor = step(bundle, dense, source["incoming"], token(pair["candidate_token_id"]), source["length"], d, 30)
            x_source = contrast(source["anchor"]["cache"], sdonor["cache"], source["spec"])
            cs = source["basis"] @ (x_source - source["mean"])
            ct = target["basis"] @ (x_target - target["mean"])
            probes = [token(t) for t in item["future_probe_tokens"]]
            future = lambda cache: signature([step(bundle, dense, cache, z, target["length"] + 1, d, 30) for z in probes], scale)
            native_sig = future(target["anchor"]["cache"])
            donor_sig = future(donor["cache"])
            base = {"role": role, "state_id": sid, "family": family, "pair_id": pair_id, "source_state_id": source_id, "token_pair_hash": pair["token_pair_hash"], "source_contrast_sha256": digest(x_source), "target_contrast_sha256": digest(x_target), "target_local_basis_sha256": target["record"]["basis_sha256"], "source_local_basis_sha256": source_info[family]["basis_sha256"], "target_train_pairs": len(target["coords"]), "transport_fit_pairs": len(common), "probe_hash": item["future_probe_hash"], "probe_count": len(probes), "TRAIN_only_fit": True, "heldout_token_write_not_used_in_fit": True}
            def append(name, approx, transport_hash=None):
                cache = inject(target["anchor"]["cache"], approx, 1, target["spec"], target["rec"], target["att"])
                sig = future(cache)
                rows.append({**base, "condition": name, **metrics(sig, native_sig, donor_sig), "write_relative_l2": float(np.linalg.norm(x_target - approx) / max(np.linalg.norm(x_target), 1e-12)), "transport_hash": transport_hash, "realized_cache_hashes": json.dumps(state_hashes(cache, target["rec"], target["att"]), sort_keys=True)})
            for k in (8, 16, 32, 64):
                approx = target["mean"] + ct[:k] @ target["basis"][:k]
                append(f"LOCAL_ORACLE_k{k}", approx)
            for (method, k), mat in maps.items():
                transported = cs[:k] @ mat
                approx = target["mean"] + transported @ target["basis"][:k]
                append(f"TRANSPORTED_LOCAL_{method}_k{k}", approx, map_hashes[f"{method}_k{k}"])
        print(f"V30 local transport {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    paths = {"causal": root / OUT / f"local_transport_{role}_v30.parquet", "geometry": root / OUT / f"local_geometry_{role}_v30.parquet", "bases": root / OUT / f"local_basis_records_{role}_v30.json"}
    frame.to_parquet(paths["causal"], index=False, compression="zstd")
    pd.DataFrame(geometry).to_parquet(paths["geometry"], index=False, compression="zstd")
    write_json_atomic(paths["bases"], {"records": local_records, "target_local_basis_uses_only_TOKEN_TRAIN": True})
    jp = root / OUT / f"local_transport_{role}_v30.json"
    summary = {"role": role, "states": len(ids), "rows": len(frame), "profiles": profile(frame, p["gates"]), "response_sha256": sha256_file(paths["causal"]), "geometry_sha256": sha256_file(paths["geometry"]), "basis_records_sha256": sha256_file(paths["bases"]), "source_bases_sha256": sha256_file(root / OUT / "local_source_bases_v30.json"), "independent_final_opened": False}
    write_json_atomic(jp, summary)
    prior = "artifacts/transferable_natural_writes_v30_local_source_bases.freeze.json" if role == "development" else "artifacts/transferable_natural_writes_v30_local_transport_development.freeze.json"
    fr = stage_freeze(root, f"local_transport_{role}", [SOURCE, *(str(x.relative_to(root)) for x in paths.values()), str(jp.relative_to(root)), prior], {"summary_sha256": sha256_file(jp), "response_sha256": summary["response_sha256"], "role": role})
    return {"freeze_digest": fr["freeze_digest"], "states": len(ids), "rows": len(frame), "passes": {k: v["pass"] for k, v in summary["profiles"].items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("sources", "development", "validation"))
    args = ap.parse_args()
    print(json.dumps(sources(Path.cwd()) if args.command == "sources" else evaluate(Path.cwd(), args.command), indent=2))
