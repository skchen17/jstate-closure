"""Supplemental descriptive 120-state write-spectrum point; never used to fit/test V30 decoders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.global_basis_v30 import SCRATCH, load, token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, step
from jclosure.experiments.realization_v29 import contrast, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/rank_120_v30.py"
OUT = Path("results/v30/processed")


def hash_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_hash(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def plan(root):
    verify_stage(root, "execution_plan")
    d, p, _, pairs = load(root)
    by_id = {x["base_trial_id"]: x for x in d["development"]}
    chosen = {}
    for sid in p["development_holdout_ids"]:
        item = by_id[sid]
        mandatory = p["eval_pairs"]["development"][sid]["state_OOD_pair_ids"]
        eligible = [q for q in pairs.values() if q["role"] == "TOKEN_TRAIN" and q["pair_id"] in item["eligible_pair_ids"] and q["pair_id"] not in mandatory]
        eligible.sort(key=lambda q: hash_json(["V30_RANK120", sid, q["pair_id"]]))
        chosen[sid] = mandatory + [q["pair_id"] for q in eligible[:6]]
        if len(chosen[sid]) != 8:
            raise RuntimeError("120-rank state lacks eight TRAIN pairs")
    record = {"holdout_state_pair_ids": chosen, "states": len(chosen), "pairs_per_state": 8, "selection_input": "frozen state IDs, frozen STATE_OOD pairs, prewrite eligibility, deterministic hash", "post_development_response_descriptive_only": True, "not_used_for_any_basis_fit_transport_fit_or_causal_gate": True, "independent_final_opened": False}
    jp = root / OUT / "rank120_plan_v30.json"
    write_json_atomic(jp, record)
    fr = stage_freeze(root, "rank120_plan", [SOURCE, str(jp.relative_to(root)), "artifacts/transferable_natural_writes_v30_execution_plan.freeze.json"], {"plan_sha256": sha256_file(jp), "descriptive_only": True})
    return {"freeze_digest": fr["freeze_digest"], "states": len(chosen)}


@torch.no_grad()
def collect(root):
    verify_stage(root, "rank120_plan")
    d, p, _, pairs = load(root)
    chosen = json.loads((root / OUT / "rank120_plan_v30.json").read_text())["holdout_state_pair_ids"]
    by_id = {x["base_trial_id"]: x for x in d["development"]}
    bundle, dense, _, _, _, _ = context(root)
    original = json.loads((root / OUT / "global_basis_training_rows_v30.json").read_text())
    spec = [(int(layer), name, tuple(shape), int(count)) for layer, name, shape, count in original["layout"]]
    dim = original["dimension"]
    matrix = np.memmap(SCRATCH / "rank120_extra_matrix_v30.f32", dtype=np.float32, mode="w+", shape=(240, dim))
    rows = []
    for n, sid in enumerate(p["development_holdout_ids"], 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError("rank120 incoming drift")
        anchor = step(bundle, dense, incoming, token(pairs[chosen[sid][0]]["anchor_token_id"]), length, d, 30)
        if layout(anchor["cache"], rec, att) != spec:
            raise RuntimeError("rank120 write layout drift")
        for j, pair_id in enumerate(chosen[sid]):
            donor = step(bundle, dense, incoming, token(pairs[pair_id]["candidate_token_id"]), length, d, 30)
            x = contrast(anchor["cache"], donor["cache"], spec)
            row = (n - 1) * 8 + j
            matrix[row] = x
            rows.append({"row": row, "state_id": sid, "family": item["family"], "pair_id": pair_id, "contrast_sha256": array_hash(x)})
        matrix.flush()
        if n % 5 == 0:
            print(f"V30 supplemental rank120 writes {n}/30", flush=True)
    jp = root / OUT / "rank120_extra_rows_v30.json"
    write_json_atomic(jp, {"rows": rows, "dimension": dim, "scratch_matrix_sha256": sha256_file(SCRATCH / "rank120_extra_matrix_v30.f32"), "descriptive_only": True})
    fr = stage_freeze(root, "rank120_collect", [SOURCE, str(jp.relative_to(root)), "artifacts/transferable_natural_writes_v30_rank120_plan.freeze.json"], {"row_manifest_sha256": sha256_file(jp), "states": 30})
    return {"freeze_digest": fr["freeze_digest"], "rows": len(rows)}


def spectrum(g):
    centered = g - g.mean(0)[None, :] - g.mean(1)[:, None] + g.mean()
    eig = np.maximum(np.linalg.eigvalsh(centered), 0)[::-1]
    c = np.cumsum(eig) / max(float(eig.sum()), 1e-12)
    return {f"r{int(q*100)}": int(np.searchsorted(c, q) + 1) for q in (.9, .95, .99)} | {"rows": len(g)}


def fit(root):
    verify_stage(root, "rank120_collect")
    original = json.loads((root / OUT / "global_basis_training_rows_v30.json").read_text())
    extra = json.loads((root / OUT / "rank120_extra_rows_v30.json").read_text())
    dim = original["dimension"]
    a = np.memmap(SCRATCH / "train_write_matrix_v30.f32", dtype=np.float32, mode="r", shape=(720, dim))
    b = np.memmap(SCRATCH / "rank120_extra_matrix_v30.f32", dtype=np.float32, mode="r", shape=(240, dim))
    g0 = np.load(SCRATCH / "train_write_gram_v30.npy")
    cross = np.asarray(a @ b.T, dtype=np.float64)
    g1 = np.asarray(b @ b.T, dtype=np.float64)
    g = np.block([[g0, cross], [cross.T, g1]])
    np.save(SCRATCH / "rank120_gram_v30.npy", g)
    rows = original["rows"] + [{**r, "row": r["row"] + 720} for r in extra["rows"]]
    families = json.loads((root / OUT / "design_v30.json").read_text())["families"]
    by_fam = {f: [] for f in families}
    for r in rows:
        if r["state_id"] not in by_fam[r["family"]]:
            by_fam[r["family"]].append(r["state_id"])
    balanced = [by_fam[f][j] for j in range(24) for f in families]
    scaling = {}
    for n in (10, 20, 40, 80, 120):
        selected = set(balanced[:n])
        idx = [r["row"] for r in rows if r["state_id"] in selected]
        scaling[str(n)] = spectrum(g[np.ix_(idx, idx)])
    jp = root / OUT / "write_rank_scaling_120_v30.json"
    result = {"state_count_scaling": scaling, "balanced_state_order": balanced, "gram_sha256": sha256_file(SCRATCH / "rank120_gram_v30.npy"), "supplemental_state_pairs_selected_after_development_response": True, "descriptive_only": True, "never_used_to_refit_any_causal_decoder": True, "not_a_state_dimension": True}
    write_json_atomic(jp, result)
    fr = stage_freeze(root, "rank120_fit", [SOURCE, str(jp.relative_to(root)), "artifacts/transferable_natural_writes_v30_rank120_collect.freeze.json"], {"result_sha256": sha256_file(jp), "max_states": 120})
    return {"freeze_digest": fr["freeze_digest"], "state_r95": {n: x["r95"] for n, x in scaling.items()}}


if __name__ == "__main__":
    import sys
    command = sys.argv[1]
    print(json.dumps({"plan": plan, "collect": collect, "fit": fit}[command](Path.cwd()), indent=2))
