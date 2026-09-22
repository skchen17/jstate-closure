"""Causal global/family/LOFO natural-write OOD evaluation on frozen pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, transplant
from jclosure.experiments.global_basis_v30 import SCRATCH, load, token
from jclosure.experiments.pre_readout_v27 import TARGETS, metadata, prefix, scales, step
from jclosure.experiments.realization_v29 import contrast, inject, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/causal_global_v30.py"
OUT = Path("results/v30/processed")


def sha(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def signature(outputs, scale):
    return np.concatenate([np.concatenate([np.asarray(o["targets"][k], np.float64).ravel() / max(float(scale[k]), 1e-12) for k in TARGETS]) for o in outputs])


def metrics(actual, native, donor):
    move, target = actual - native, donor - native
    mn, dn = float(np.linalg.norm(move)), float(np.linalg.norm(target))
    return {"donor_cosine": float(np.dot(move, target) / (mn * dn)) if mn * dn > 1e-12 else None,
            "magnitude_ratio": mn / max(dn, 1e-12),
            "relative_l2_to_donor": float(np.linalg.norm(move - target)) / max(dn, 1e-12),
            "donor_norm": dn, "movement_norm": mn}


def profile(frame, gate):
    result = {}
    for name, group in frame.groupby("condition"):
        x = group.dropna(subset=["donor_cosine", "magnitude_ratio", "relative_l2_to_donor"])
        ok = ((x.donor_cosine >= gate["cosine_min"]) & (x.magnitude_ratio.between(gate["magnitude_min"], gate["magnitude_max"])) & (x.relative_l2_to_donor <= gate["relative_l2_max"]))
        family = {fam: float(ok.loc[g.index].mean()) for fam, g in x.groupby("family")}
        result[name] = {"rows": len(group), "valid_rows": len(x), "median_cosine": float(x.donor_cosine.median()) if len(x) else None, "median_magnitude": float(x.magnitude_ratio.median()) if len(x) else None, "median_relative_l2": float(x.relative_l2_to_donor.median()) if len(x) else None, "success_fraction": float(ok.mean()) if len(x) else 0, "family_success_fraction": family, "pass": bool(len(x) == len(group) and len(x) > 0 and x.donor_cosine.median() >= gate["cosine_min"] and gate["magnitude_min"] <= x.magnitude_ratio.median() <= gate["magnitude_max"] and x.relative_l2_to_donor.median() <= gate["relative_l2_max"] and ok.mean() >= gate["success_fraction_min"] and sum(v >= gate["success_fraction_min"] for v in family.values()) >= gate["families_required"])}
    return result


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "global_basis_fit")
    if role == "validation":
        verify_stage(root, "causal_global_development")
    d, p, by_id, pairs = load(root)
    fit = json.loads((root / OUT / "global_basis_fit_v30.json").read_text())
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    global_basis = np.load(SCRATCH / "basis_GLOBAL_v30.npy", mmap_mode="r")
    global_mean = np.load(SCRATCH / "basis_GLOBAL_v30_mean.npy", mmap_mode="r")
    ids = (p["development_holdout_ids"] + list(p["seen_state_token_OOD_pairs"])) if role == "development" else p["validation_ids"]
    by_id = {x["base_trial_id"]: x for x in d[role]}
    rows = []
    active_family, family_bases = None, None
    for n, sid in enumerate(ids, 1):
        item = by_id[sid]
        family = item["family"]
        if family != active_family:
            active_family = family
            family_bases = {kind: (np.load(SCRATCH / f"basis_{kind}_{family}_v30.npy", mmap_mode="r"), np.load(SCRATCH / f"basis_{kind}_{family}_v30_mean.npy", mmap_mode="r")) for kind in ("FAMILY", "LOFO")}
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming state drift: {sid}")
        test_pairs = p["eval_pairs"][role][sid] if sid in p["eval_pairs"][role] else {"seen_state_token_OOD_pairs": p["seen_state_token_OOD_pairs"][sid]}
        for axis, pair_ids in test_pairs.items():
            axis = "STATE_OOD" if axis == "state_OOD_pair_ids" else "TOKEN_OOD" if axis == "seen_state_token_OOD_pairs" else "JOINT_OOD"
            for pair_id in pair_ids:
                pair = pairs[pair_id]
                anchor = step(bundle, dense, incoming, token(pair["anchor_token_id"]), length, d, 30)
                donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
                spec = layout(anchor["cache"], rec, att)
                x = contrast(anchor["cache"], donor["cache"], spec)
                probes = [token(t) for t in item["future_probe_tokens"]]
                future = lambda cache: signature([step(bundle, dense, cache, z, length + 1, d, 30) for z in probes], scale)
                native_sig, donor_sig = future(anchor["cache"]), future(donor["cache"])
                base_row = {"role": role, "state_id": sid, "family": family, "axis": axis, "pair_id": pair_id, "token_pair_hash": pair["token_pair_hash"], "incoming_state_hash": item["incoming_state_hash"], "natural_contrast_sha256": sha(x), "natural_write_norm": float(np.linalg.norm(x)), "probe_hash": item["future_probe_hash"], "probe_count": len(probes), "source_train_only": True, "same_incoming_state": True}
                def append(name, cache, approx=None):
                    z = future(cache)
                    row = {**base_row, "condition": name, **metrics(z, native_sig, donor_sig), "write_relative_l2": float(np.linalg.norm(x - approx) / max(np.linalg.norm(x), 1e-12)) if approx is not None else None, "realized_cache_hashes": json.dumps(state_hashes(cache, rec, att), sort_keys=True)}
                    rows.append(row)
                append("EXACT_REC_CONV", transplant(anchor["cache"], donor["cache"], "REC+Conv", rec, att)[0])
                append("EXACT_CONV", transplant(anchor["cache"], donor["cache"], "Conv", rec, att)[0])
                append("EXACT_KV", transplant(anchor["cache"], donor["cache"], "KV", rec, att)[0])
                models = {"GLOBAL": (global_basis, global_mean), "FAMILY": family_bases["FAMILY"], "LOFO": family_bases["LOFO"]}
                for model, (basis, mean) in models.items():
                    ks = p["k_grid"] if model == "GLOBAL" else (32, 64)
                    centered = x - mean
                    coeff = basis[:max(ks)] @ centered
                    approx = np.array(mean, copy=True)
                    for j in range(max(ks)):
                        approx += coeff[j] * basis[j]
                        if j + 1 in ks:
                            cache = inject(anchor["cache"], approx, 1, spec, rec, att)
                            append(f"{model}_k{j+1}", cache, approx)
        print(f"V30 global causal {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"causal_global_{role}_v30.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {"role": role, "states": len(ids), "rows": len(frame), "pair_tests": int(frame[["state_id", "pair_id"]].drop_duplicates().shape[0]), "profiles": profile(frame, p["gates"]), "basis_fit_hash": sha256_file(root / OUT / "global_basis_fit_v30.json"), "response_sha256": sha256_file(path), "four_frozen_probes": True, "independent_final_opened": False}
    jp = root / OUT / f"causal_global_{role}_v30.json"
    write_json_atomic(jp, summary)
    prior = "artifacts/transferable_natural_writes_v30_global_basis_fit.freeze.json" if role == "development" else "artifacts/transferable_natural_writes_v30_causal_global_development.freeze.json"
    fr = stage_freeze(root, f"causal_global_{role}", [SOURCE, str(path.relative_to(root)), str(jp.relative_to(root)), prior], {"summary_sha256": sha256_file(jp), "response_sha256": sha256_file(path), "role": role})
    return {"freeze_digest": fr["freeze_digest"], "states": len(ids), "rows": len(frame), "passes": {k: v["pass"] for k, v in summary["profiles"].items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=("development", "validation"))
    args = ap.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
