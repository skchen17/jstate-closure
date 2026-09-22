"""Collect frozen TRAIN natural write contrasts and descriptive write geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, step
from jclosure.experiments.realization_v29 import contrast, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v31 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/primitive_matrix_v31.py"
OUT = Path("results/v31/processed")
SCRATCH = Path("/data/CSK/J-space-project/v31-write-work")


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


@torch.no_grad()
def collect(root: Path):
    verify_stage(root, "execution_plan")
    design = json.loads((root / OUT / "design_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    by_id = {x["base_trial_id"]: x for x in design["development"]}
    pairs = {x["pair_id"]: x for x in design["token_pair_library"]}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    bundle, dense, *_ = context(root)
    matrix, spec = None, None
    records = []
    for n, sid in enumerate(plan["fit_state_ids"], 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming cache drift: {sid}")
        pids = plan["fit_pair_ids_per_state"][sid]
        anchor = step(bundle, dense, incoming, token(pairs[pids[0]]["anchor_token_id"]), length, design, 30)
        current = layout(anchor["cache"], rec, att)
        if spec is None:
            spec = current
            dim = sum(x[3] for x in spec)
            nrows = len(plan["fit_state_ids"]) * len(pids)
            matrix = np.memmap(SCRATCH / "train_write_matrix_v31.f32", dtype=np.float32, mode="w+", shape=(nrows, dim))
        elif current != spec:
            raise RuntimeError("V31 write layout drift")
        for j, pid in enumerate(pids):
            pair = pairs[pid]
            donor = step(bundle, dense, incoming, token(pair["candidate_token_id"]), length, design, 30)
            delta = contrast(anchor["cache"], donor["cache"], spec)
            index = (n - 1) * len(pids) + j
            matrix[index] = delta
            records.append({"row": index, "state_id": sid, "family": item["family"], "pair_id": pid, "candidate_token_id": pair["candidate_token_id"], "category": pair["category"], "token_pair_hash": pair["token_pair_hash"], "incoming_state_hash": item["incoming_state_hash"], "natural_write_sha256": sha_array(delta), "write_norm": float(np.linalg.norm(delta))})
            del donor, delta
        matrix.flush()
        if n % 5 == 0 or n == len(plan["fit_state_ids"]):
            print(f"V31 TRAIN write collection {n}/{len(plan['fit_state_ids'])}", flush=True)
    manifest_path = root / OUT / "primitive_training_rows_v31.json"
    write_json_atomic(manifest_path, {"rows": records, "layout": spec, "nrows": len(records), "dimension": dim, "scratch_path": str(SCRATCH / "train_write_matrix_v31.f32"), "TRAIN_only": True, "no_heldout_current_token_write_observed": True, "no_future_response_observed": True})
    freeze = stage_freeze(root, "primitive_collect", [SOURCE, str(manifest_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_execution_plan.freeze.json"], {"rows": len(records), "dimension": dim, "matrix_sha256": sha256_file(SCRATCH / "train_write_matrix_v31.f32"), "manifest_sha256": sha256_file(manifest_path)})
    return {"freeze_digest": freeze["freeze_digest"], "rows": len(records), "dimension": dim}


def geometry(root: Path):
    verify_stage(root, "primitive_collect")
    manifest = json.loads((root / OUT / "primitive_training_rows_v31.json").read_text())
    rows = manifest["rows"]
    n, dim = manifest["nrows"], manifest["dimension"]
    matrix = np.memmap(SCRATCH / "train_write_matrix_v31.f32", dtype=np.float32, mode="r", shape=(n, dim))
    gram_path = SCRATCH / "train_write_gram_v31.npy"
    gram = np.asarray(matrix @ matrix.T, dtype=np.float64)
    np.save(gram_path, gram)
    center = gram - gram.mean(0)[None, :] - gram.mean(1)[:, None] + gram.mean()
    eig, vec = np.linalg.eigh(center)
    order = np.argsort(eig)[::-1]
    eig, vec = np.maximum(eig[order], 0), vec[:, order]
    np.save(SCRATCH / "train_write_eigenvectors_v31.npy", vec.astype(np.float32))
    np.save(SCRATCH / "train_write_eigenvalues_v31.npy", eig.astype(np.float32))
    total = max(float(eig.sum()), 1e-12)
    ranks = {f"r{int(q*100)}": int(np.searchsorted(np.cumsum(eig) / total, q) + 1) for q in (.9, .95, .99)}
    spectrum_path = root / OUT / "primitive_write_geometry_v31.npz"
    np.savez_compressed(spectrum_path, singular_values=np.sqrt(eig).astype(np.float32), train_state_ids=np.array([x["state_id"] for x in rows]), token_ids=np.array([x["candidate_token_id"] for x in rows]))
    # This subset statistic is descriptive only; validation/final are not used.
    train_tokens = [x["candidate_token_id"] for x in json.loads((root / OUT / "design_v31.json").read_text())["token_pair_library"] if x["role"] == "TOKEN_TRAIN"]
    scaling = {}
    for width in (16, 32, 64, 128, 160):
        allowed = set(train_tokens[:width])
        idx = np.asarray([i for i, row in enumerate(rows) if row["candidate_token_id"] in allowed])
        if len(idx) < 8:
            continue
        sub = gram[np.ix_(idx, idx)]
        sub -= sub.mean(0)[None, :] + sub.mean(1)[:, None] - sub.mean()
        values = np.maximum(np.linalg.eigvalsh(sub), 0)[::-1]
        mass = max(float(values.sum()), 1e-12)
        scaling[str(width)] = {"token_library_size": width, "observed_fit_rows": len(idx), "rank_90": int(np.searchsorted(np.cumsum(values) / mass, .9) + 1), "rank_95": int(np.searchsorted(np.cumsum(values) / mass, .95) + 1), "rank_99": int(np.searchsorted(np.cumsum(values) / mass, .99) + 1)}
    summary = {"train_rows": n, "write_dimension": dim, "training_span_rank": int(np.sum(eig > max(eig[0], 1e-12) * 1e-9)), "ranks": ranks, "token_library_scaling": scaling, "primitive_count_not_established_by_PCA_rank": True, "heldout_response_not_used": True, "gram_sha256": sha_array(gram), "spectrum_sha256": sha256_file(spectrum_path), "matrix_sha256": sha256_file(SCRATCH / "train_write_matrix_v31.f32"), "eigenvector_sha256": sha256_file(SCRATCH / "train_write_eigenvectors_v31.npy"), "eigenvalue_sha256": sha256_file(SCRATCH / "train_write_eigenvalues_v31.npy")}
    summary_path = root / OUT / "primitive_write_geometry_v31.json"
    write_json_atomic(summary_path, summary)
    freeze = stage_freeze(root, "primitive_geometry", [SOURCE, str(summary_path.relative_to(root)), str(spectrum_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_collect.freeze.json"], {"geometry_sha256": sha256_file(summary_path), "gram_sha256": sha_array(gram)})
    return {"freeze_digest": freeze["freeze_digest"], "ranks": ranks, "scaling": scaling}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "geometry"))
    args = parser.parse_args()
    print(json.dumps(collect(Path.cwd()) if args.action == "collect" else geometry(Path.cwd()), indent=2))
