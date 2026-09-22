"""Fit response-blind V30 natural REC+Conv bases from frozen TRAIN contrasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.pre_readout_v27 import metadata, prefix, step
from jclosure.experiments.realization_v29 import contrast, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/global_basis_v30.py"
OUT = Path("results/v30/processed")
SCRATCH = Path("/data/CSK/J-space-project/v30-write-work")


def token(i):
    return torch.tensor([[int(i)]], dtype=torch.long)


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def load(root):
    verify_stage(root, "transport_source_amendment")
    d = json.loads((root / OUT / "design_v30.json").read_text())
    p = json.loads((root / OUT / "execution_plan_v30.json").read_text())
    by_id = {x["base_trial_id"]: x for x in d["development"]}
    pairs = {x["pair_id"]: x for x in d["token_pair_library"]}
    return d, p, by_id, pairs


@torch.no_grad()
def collect(root):
    d, p, by_id, pairs = load(root)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bundle, dense, _, _, _, _ = context(root)
    ids = p["fit_state_ids"]
    nrows = len(ids) * p["basis_fit_contrasts_per_state"]
    spec = None
    matrix = None
    records = []
    for n, sid in enumerate(ids, 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming cache hash mismatch: {sid}")
        anchor = step(bundle, dense, incoming, token(pairs[p["basis_fit_pair_ids_per_state"][sid][0]]["anchor_token_id"]), length, d, 30)
        current_spec = layout(anchor["cache"], rec, att)
        if spec is None:
            spec = current_spec
            dim = sum(row[3] for row in spec)
            matrix = np.memmap(SCRATCH / "train_write_matrix_v30.f32", dtype=np.float32, mode="w+", shape=(nrows, dim))
        elif current_spec != spec:
            raise RuntimeError("natural write layout drift")
        for j, pair_id in enumerate(p["basis_fit_pair_ids_per_state"][sid]):
            pair = pairs[pair_id]
            out = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, d, 30)
            row = (n - 1) * p["basis_fit_contrasts_per_state"] + j
            delta = contrast(anchor["cache"], out["cache"], spec)
            matrix[row] = delta
            records.append({"row": row, "state_id": sid, "family": item["family"], "pair_id": pair_id, "token_pair_hash": pair["token_pair_hash"], "contrast_sha256": sha_array(delta), "incoming_state_hash": item["incoming_state_hash"]})
        matrix.flush()
        if n % 5 == 0 or n == len(ids):
            print(f"V30 global TRAIN writes {n}/{len(ids)}", flush=True)
    table = root / OUT / "global_basis_training_rows_v30.json"
    write_json_atomic(table, {"rows": records, "layout": spec, "nrows": nrows, "dimension": dim, "scratch_path": str(SCRATCH / "train_write_matrix_v30.f32"), "no_future_response_observed": True, "heldout_token_used_in_fit": False})
    return {"rows": nrows, "dimension": dim, "row_manifest_sha256": sha256_file(table), "scratch_matrix_sha256": sha256_file(SCRATCH / "train_write_matrix_v30.f32")}


def fit(root):
    d, p, by_id, pairs = load(root)
    rowfile = root / OUT / "global_basis_training_rows_v30.json"
    manifest = json.loads(rowfile.read_text())
    records = manifest["rows"]
    n, dim = manifest["nrows"], manifest["dimension"]
    matrix = np.memmap(SCRATCH / "train_write_matrix_v30.f32", dtype=np.float32, mode="r", shape=(n, dim))
    gram_path = SCRATCH / "train_write_gram_v30.npy"
    if gram_path.exists():
        gram = np.load(gram_path)
        if gram.shape != (n, n):
            raise RuntimeError("cached Gram matrix shape mismatch")
    else:
        gram = np.asarray(matrix @ matrix.T, dtype=np.float64)
        np.save(gram_path, gram)
    results = {}
    subsets = {"GLOBAL": np.arange(n)}
    for fam in d["families"]:
        mask = np.array([x["family"] == fam for x in records])
        subsets[f"FAMILY_{fam}"] = np.where(mask)[0]
        subsets[f"LOFO_{fam}"] = np.where(~mask)[0]
    for name, indices in subsets.items():
        sub = gram[np.ix_(indices, indices)]
        centered = sub - sub.mean(0)[None, :] - sub.mean(1)[:, None] + sub.mean()
        vals, vecs = np.linalg.eigh(centered)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 0), vecs[:, order]
        rank = int(np.sum(vals > max(vals[0], 1e-12) * 1e-9))
        maxk = min(64, rank)
        mean_sum = np.zeros(dim, dtype=np.float64)
        for start in range(0, len(indices), 8):
            mean_sum += np.asarray(matrix[indices[start:start + 8]].sum(0, dtype=np.float64))
        mean = np.asarray(mean_sum / len(indices), dtype=np.float32)
        coeff = np.asarray((vecs[:, :maxk] / np.sqrt(np.maximum(vals[:maxk], 1e-12))[None, :]).T, dtype=np.float32)
        if not torch.cuda.is_available():
            raise RuntimeError("V30 streamed basis fitting requires CUDA")
        basis_gpu = torch.zeros((maxk, dim), dtype=torch.float32, device="cuda")
        for start in range(0, len(indices), 8):
            block = torch.from_numpy(np.asarray(matrix[indices[start:start + 8]], dtype=np.float32)).to("cuda")
            weights = torch.from_numpy(np.ascontiguousarray(coeff[:, start:start + 8])).to("cuda")
            basis_gpu.addmm_(weights, block)
            del block, weights
        basis = basis_gpu.cpu().numpy()
        del basis_gpu
        # Centering is implicit in eigenvector coefficients (orthogonal to ones).
        prefix = SCRATCH / f"basis_{name}_v30"
        np.save(str(prefix) + ".npy", basis)
        np.save(str(prefix) + "_mean.npy", mean)
        total = float(np.sum(vals))
        ranks = {f"r{int(q*100)}": int(np.searchsorted(np.cumsum(vals) / max(total, 1e-12), q) + 1) for q in (.9, .95, .99)}
        results[name] = {"training_rows": len(indices), "rank": rank, "tested_k": [k for k in p["k_grid"] if k <= maxk], "ranks": ranks, "basis_sha256": sha_array(basis), "mean_sha256": sha_array(mean), "eigenvalue_sha256": sha_array(vals.astype(np.float32)), "basis_path": str(prefix) + ".npy"}
        print(f"V30 basis {name} rank={rank} r95={ranks['r95']}", flush=True)
    out = root / OUT / "global_basis_fit_v30.json"
    write_json_atomic(out, {"models": results, "train_rows": n, "write_dimension": dim, "training_rows_sha256": sha256_file(rowfile), "matrix_sha256": sha256_file(SCRATCH / "train_write_matrix_v30.f32"), "gram_sha256": sha256_file(gram_path), "basis_fit_uses_ONLY_TOKEN_TRAIN": True, "no_future_response_observed_in_fit": True, "scratch_not_committed": True})
    fr = stage_freeze(root, "global_basis_fit", [SOURCE, str(rowfile.relative_to(root)), str(out.relative_to(root)), "artifacts/transferable_natural_writes_v30_transport_source_amendment.freeze.json"], {"result_sha256": sha256_file(out), "model_count": len(results), "write_dimension": dim})
    return {"freeze_digest": fr["freeze_digest"], "models": {k: {"rank": v["rank"], "ranks": v["ranks"]} for k, v in results.items()}}


if __name__ == "__main__":
    import sys
    root = Path.cwd()
    print(json.dumps(collect(root) if sys.argv[1:] == ["collect"] else fit(root), indent=2))
