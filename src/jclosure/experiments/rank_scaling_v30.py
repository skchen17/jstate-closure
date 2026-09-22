"""Descriptive V30 pooled write-rank scaling from frozen TRAIN contrasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jclosure.experiments.global_basis_v30 import SCRATCH
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/rank_scaling_v30.py"
OUT = Path("results/v30/processed")


def ranks(gram, indices):
    g = gram[np.ix_(indices, indices)]
    centered = g - g.mean(0)[None, :] - g.mean(1)[:, None] + g.mean()
    eig = np.maximum(np.linalg.eigvalsh(centered), 0)[::-1]
    total = max(float(eig.sum()), 1e-12)
    cum = np.cumsum(eig) / total
    return {f"r{int(q*100)}": int(np.searchsorted(cum, q) + 1) for q in (.9, .95, .99)} | {"rank": int(np.sum(eig > max(float(eig[0]), 1e-12) * 1e-9)), "rows": len(indices)}


def run(root):
    verify_stage(root, "global_basis_fit")
    rows = json.loads((root / OUT / "global_basis_training_rows_v30.json").read_text())["rows"]
    design = json.loads((root / OUT / "design_v30.json").read_text())
    gram = np.load(SCRATCH / "train_write_gram_v30.npy")
    by_family = {f: [] for f in design["families"]}
    for row in rows:
        if row["state_id"] not in by_family[row["family"]]:
            by_family[row["family"]].append(row["state_id"])
    balanced = [by_family[f][j] for j in range(18) for f in design["families"]]
    state_scaling = {}
    for n in (10, 20, 40, 80, 90):
        ids = set(balanced[:n])
        indices = [r["row"] for r in rows if r["state_id"] in ids]
        state_scaling[str(n)] = ranks(gram, indices)
    token_scaling = {}
    for k in (1, 2, 4, 8):
        chosen = set()
        counts = {}
        for row in rows:
            sid = row["state_id"]
            counts[sid] = counts.get(sid, 0) + 1
            if counts[sid] <= k:
                chosen.add(row["row"])
        token_scaling[str(k)] = ranks(gram, sorted(chosen))
    jp = root / OUT / "write_rank_scaling_v30.json"
    result = {"state_count_scaling": state_scaling, "train_token_pairs_per_state_scaling": token_scaling, "balanced_state_order": balanced, "gram_sha256": sha256_file(SCRATCH / "train_write_gram_v30.npy"), "training_rows_sha256": sha256_file(root / OUT / "global_basis_training_rows_v30.json"), "descriptive_only": True, "max_state_count_in_frozen_fit": 90, "requested_120_state_point_not_in_global_training_fit": True, "not_a_causal_dimension": True}
    write_json_atomic(jp, result)
    fr = stage_freeze(root, "rank_scaling", [SOURCE, str(jp.relative_to(root)), "artifacts/transferable_natural_writes_v30_global_basis_fit.freeze.json"], {"result_sha256": sha256_file(jp), "max_states": 90})
    return {"freeze_digest": fr["freeze_digest"], "state_r95": {n: x["r95"] for n, x in state_scaling.items()}, "pair_r95": {n: x["r95"] for n, x in token_scaling.items()}}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
