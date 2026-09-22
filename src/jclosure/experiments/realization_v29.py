"""Train-only PCA of natural REC/Conv write contrasts with held-out causal realization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.forks_formal_v29 import OUT, design, native_layers, state_hashes, total_hash, vector_metrics
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context, fields
from jclosure.protocol_v29 import verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/realization_v29.py"


def token(t):
    return torch.tensor([[int(t)]], dtype=torch.long)


def branch(bundle, dense, root, d, item):
    m = metadata(root, item)
    incoming, _, length = prefix(bundle, str(m["prompt"]))
    rec, att = native_layers(incoming)
    a = step(bundle, dense, incoming, token(item["token_A"]), length, d, 30)
    b = step(bundle, dense, incoming, token(item["token_B"]), length, d, 30)
    return a, b, length, rec, att


def layout(cache, rec, att):
    spec = []
    for channel in ("REC", "Conv"):
        for layer, name, t in fields(cache, channel, rec, att):
            spec.append((layer, name, tuple(t.shape), int(t.numel())))
    return spec


def contrast(a, b, spec):
    pieces = []
    for layer, name, _, _ in spec:
        aa = getattr(a.layers[layer], name)
        bb = getattr(b.layers[layer], name)
        pieces.append((bb.float() - aa.float()).reshape(-1).cpu().numpy())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def inject(base, approx, sign, spec, rec, att):
    result = clone_hybrid_cache(base)
    vec = torch.from_numpy(np.ascontiguousarray(approx)).to(next(iter(fields(base, "REC", rec, att)))[2].device)
    start = 0
    for layer, name, shape, count in spec:
        old = getattr(base.layers[layer], name)
        replacement = (old.float() + float(sign) * vec[start:start + count].view(shape)).to(old.dtype)
        setattr(result.layers[layer], name, replacement)
        start += count
    if start != len(approx):
        raise RuntimeError("write realization layout mismatch")
    return result


def gate(rows, cfg):
    profiles = {}
    for (role, k, direction), x in rows.groupby(["role", "k", "direction"]):
        success = (x.donor_cosine >= cfg["cosine_min"]) & (x.magnitude_ratio >= cfg["magnitude_min"]) & (x.magnitude_ratio <= cfg["magnitude_max"]) & (x.relative_l2_to_donor <= cfg["relative_l2_max"])
        family = {name: float(success.loc[g.index].mean()) for name, g in x.groupby("family")}
        accepted = bool(float(x.donor_cosine.median()) >= cfg["cosine_min"] and cfg["magnitude_min"] <= float(x.magnitude_ratio.median()) <= cfg["magnitude_max"] and float(x.relative_l2_to_donor.median()) <= cfg["relative_l2_max"] and float(success.mean()) >= .5 and sum(v >= .5 for v in family.values()) >= cfg["families_required"])
        profiles[f"{role}:{k}:{direction}"] = {"rows": len(x), "median_cosine": float(x.donor_cosine.median()), "median_magnitude": float(x.magnitude_ratio.median()), "median_relative_l2": float(x.relative_l2_to_donor.median()), "success_fraction": float(success.mean()), "family_success_fraction": family, "pass": accepted}
    qualified = {str(k): all(profiles[f"{role}:{k}:{direction}"]["pass"] for role in ("development", "validation") for direction in ("A_from_B", "B_from_A")) for k in sorted(rows.k.unique())}
    return profiles, qualified


def run(root: Path):
    verify_stage(root, "realization_plan")
    verify_stage(root, "validation_analysis")
    d = design(root)
    plan = json.loads((root / OUT / "realization_plan_v29.json").read_text())
    lookup = {x["base_trial_id"]: x for role in ("calibration", "development", "validation") for x in d[role]}
    bundle, dense, _, _, _, _ = context(root)
    scale = scales(root)
    train_rows = []
    spec = None
    for n, sid in enumerate(plan["train_ids"], 1):
        a, b, length, rec, att = branch(bundle, dense, root, d, lookup[sid])
        current_spec = layout(a["cache"], rec, att)
        if spec is None:
            spec = current_spec
        elif spec != current_spec:
            raise RuntimeError("natural write layout changes across states")
        train_rows.append(contrast(a["cache"], b["cache"], spec))
        if n % 10 == 0 or n == len(plan["train_ids"]):
            print(f"V29 realization train {n}/{len(plan['train_ids'])}", flush=True)
    matrix = np.stack(train_rows).astype(np.float32)
    del train_rows
    training_hash = hashlib.sha256(memoryview(matrix)).hexdigest()
    mean = matrix.mean(0, dtype=np.float64).astype(np.float32)
    matrix -= mean
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    available = int(np.sum(eigenvalues > max(eigenvalues[0], 1e-12) * 1e-9))
    ks = [k for k in plan["k_grid"] if k <= min(available, 64)]
    if not ks:
        raise RuntimeError("no estimable k")
    basis = ((eigenvectors[:, :max(ks)].T @ matrix) / np.sqrt(np.maximum(eigenvalues[:max(ks)], 1e-12))[:, None]).astype(np.float32)
    del matrix, gram
    rows = []
    for role, ids in (("development", plan["development_holdout_ids"]), ("validation", plan["validation_ids"])):
        for n, sid in enumerate(ids, 1):
            item = lookup[sid]
            a, b, length, rec, att = branch(bundle, dense, root, d, item)
            if layout(a["cache"], rec, att) != spec:
                raise RuntimeError("held-out layout mismatch")
            x = contrast(a["cache"], b["cache"], spec)
            coefficients = basis @ (x - mean)
            approx = mean.copy()
            z = token(item["continuation_token"])
            a1 = step(bundle, dense, a["cache"], z, length + 1, d, 30)
            b1 = step(bundle, dense, b["cache"], z, length + 1, d, 30)
            next_k = set(ks)
            for j in range(max(ks)):
                approx += coefficients[j] * basis[j]
                k = j + 1
                if k not in next_k:
                    continue
                approx_hash = hashlib.sha256(memoryview(approx)).hexdigest()
                fidelity = float(np.linalg.norm(x - approx) / max(np.linalg.norm(x), 1e-12))
                for direction, recipient, donor, sign in (("A_from_B", a, b, 1), ("B_from_A", b, a, -1)):
                    cache = inject(recipient["cache"], approx, sign, spec, rec, att)
                    out = step(bundle, dense, cache, z, length + 1, d, 30)
                    native = a1 if direction == "A_from_B" else b1
                    target = b1 if direction == "A_from_B" else a1
                    metric = vector_metrics(out, native, target, scale)
                    rows.append({"role": role, "base_trial_id": sid, "family": item["family"], "k": k, "direction": direction, "write_relative_l2": fidelity, "approx_hash": approx_hash, "transplant_hash": total_hash(state_hashes(cache, rec, att)), "donor_cosine": metric["donor_cosine"], "magnitude_ratio": metric["magnitude_ratio"], "relative_l2_to_donor": metric["relative_l2_to_donor"], "same_next_token": True, "off_manifold_lowrank_approximation": True})
            if n % 5 == 0 or n == len(ids):
                print(f"V29 realization {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    profiles, qualified = gate(frame, plan["gates"])
    path = root / OUT / "write_effect_realization_v29.parquet"
    spectrum = root / OUT / "write_effect_spectrum_v29.npz"
    summary_path = root / OUT / "write_effect_realization_v29.json"
    frame.to_parquet(path, index=False, compression="zstd")
    np.savez_compressed(spectrum, singular_values=np.sqrt(eigenvalues).astype(np.float32), training_ids=np.array(plan["train_ids"]), development_holdout_ids=np.array(plan["development_holdout_ids"]), validation_ids=np.array(plan["validation_ids"]))
    summary = {"train_states": len(plan["train_ids"]), "development_holdout_states": len(plan["development_holdout_ids"]), "validation_states": len(plan["validation_ids"]), "write_dimension": int(len(mean)), "train_span_rank": available, "tested_k": ks, "not_estimable_k": [k for k in plan["k_grid"] if k not in ks], "training_write_matrix_sha256": training_hash, "mean_write_hash": hashlib.sha256(memoryview(mean)).hexdigest(), "basis_hash": hashlib.sha256(memoryview(basis)).hexdigest(), "profiles": profiles, "qualified_k": qualified, "compact_write_effect_dimension_identified": any(qualified.values()), "not_a_dynamical_state_dimension": True, "off_manifold_approximation": True, "historical_final_opened": False, "response_sha256": sha256_file(path), "spectrum_sha256": sha256_file(spectrum)}
    write_json_atomic(summary_path, summary)
    fr = stage_freeze(root, "realization", [SOURCE, str(path.relative_to(root)), str(spectrum.relative_to(root)), str(summary_path.relative_to(root)), "artifacts/natural_write_content_v29_realization_plan.freeze.json"], {"summary_sha256": sha256_file(summary_path), "qualified_k": qualified, "not_estimable_k": summary["not_estimable_k"]})
    return {"freeze_digest": fr["freeze_digest"], "write_dimension": summary["write_dimension"], "train_span_rank": available, "qualified_k": qualified}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
