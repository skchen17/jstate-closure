"""TRAIN-only function-conditioned and future-response-factor dictionaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.causal_global_v30 import signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, transplant
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.primitive_matrix_v31 import OUT
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v31 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/primitive_extra_v31.py"


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def function_fit(root: Path):
    verify_stage(root, "primitive_dictionary_fit")
    manifest = json.loads((root / OUT / "primitive_training_rows_v31.json").read_text())
    score = np.load(root / OUT / "primitive_dictionary_coordinates_v31.npz")["pca_training_scores"].astype(np.float64)
    categories = sorted({x["category"] for x in manifest["rows"]})
    models = {}
    for category in categories:
        idx = [i for i, x in enumerate(manifest["rows"]) if x["category"] == category]
        x = score[idx]
        mu = x.mean(0)
        u, s, v = np.linalg.svd(x - mu, full_matrices=False)
        rank = int(np.sum(s > max(float(s[0]), 1e-12) * 1e-8)) if len(s) else 0
        models[category] = {"train_rows": len(idx), "estimable_rank": rank, "global_PCA_coordinate_mean": mu.tolist(), "local_directions": v[:min(128, rank)].tolist(), "source_row_indices_sha256": sha_array(np.asarray(idx, np.int32)), "category_is_response_blind": True}
    path = root / OUT / "function_conditioned_fit_v31.json"
    write_json_atomic(path, {"models": models, "source_train_rows_sha256": sha256_file(root / OUT / "primitive_training_rows_v31.json"), "heldout_write_used_in_fit": False, "future_response_used_in_fit": False})
    freeze = stage_freeze(root, "function_conditioned_fit", [SOURCE, str(path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json"], {"result_sha256": sha256_file(path), "models": {k: v["estimable_rank"] for k, v in models.items()}})
    return {"freeze_digest": freeze["freeze_digest"], "models": {k: v["estimable_rank"] for k, v in models.items()}}


@torch.no_grad()
def response_fit(root: Path):
    verify_stage(root, "primitive_dictionary_fit")
    design = json.loads((root / OUT / "design_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    manifest = json.loads((root / OUT / "primitive_training_rows_v31.json").read_text())
    scores = np.load(root / OUT / "primitive_dictionary_coordinates_v31.npz")["pca_training_scores"].astype(np.float32)
    by_id = {x["base_trial_id"]: x for x in design["development"]}
    pairs = {x["pair_id"]: x for x in design["token_pair_library"]}
    lookup = {(x["state_id"], x["pair_id"]): x["row"] for x in manifest["rows"]}
    bundle, dense, *_ = context(root)
    scale = scales(root)
    xs, ys, records = [], [], []
    for n, sid in enumerate(plan["fit_state_ids"], 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming state drift: {sid}")
        pids = plan["fit_pair_ids_per_state"][sid][:2]
        anchor = step(bundle, dense, incoming, token(pairs[pids[0]]["anchor_token_id"]), length, design, 30)
        probes = [token(tid) for tid in item["future_probe_tokens"]]
        def response(cache):
            return signature([step(bundle, dense, cache, z, length + 1, design, 30) for z in probes], scale)
        native = response(anchor["cache"])
        for pid in pids:
            donor = step(bundle, dense, incoming, token(pairs[pid]["candidate_token_id"]), length, design, 30)
            exact = transplant(anchor["cache"], donor["cache"], "REC+Conv", rec, att)[0]
            effect = response(exact) - native
            idx = lookup[(sid, pid)]
            xs.append(scores[idx])
            ys.append(effect.astype(np.float32))
            records.append({"state_id": sid, "pair_id": pid, "training_row": idx, "incoming_state_hash": item["incoming_state_hash"], "future_probe_hash": item["future_probe_hash"], "causal_effect_sha256": sha_array(effect.astype(np.float32))})
        if n % 5 == 0 or n == len(plan["fit_state_ids"]):
            print(f"V31 TRAIN response-factor fit {n}/{len(plan['fit_state_ids'])}", flush=True)
    x = np.stack(xs).astype(np.float64)
    y = np.stack(ys).astype(np.float64)
    xc, yc = x - x.mean(0), y - y.mean(0)
    covariance = xc.T @ yc / len(x)
    u, singular, _ = np.linalg.svd(covariance, full_matrices=False)
    available = int(np.sum(singular > max(float(singular[0]), 1e-12) * 1e-8))
    coordinate_path = root / OUT / "response_factor_coordinates_v31.npz"
    np.savez_compressed(coordinate_path, latent_directions=u[:, :min(128, available)].astype(np.float32), singular_values=singular.astype(np.float32), selected_train_scores=x.astype(np.float32), train_effects=y.astype(np.float32))
    result = {"training_rows": len(records), "training_states": len(plan["fit_state_ids"]), "causal_response_signature_length": y.shape[1], "estimable_factor_rank": available, "tested_M_grid": [m for m in (8, 16, 32, 64, 128) if m <= available], "source_train_records": records, "cross_covariance_sha256": sha_array(covariance), "coordinate_npz_sha256": sha256_file(coordinate_path), "TRAIN_only": True, "heldout_token_or_state_used_in_fit": False, "independent_final_opened": False}
    path = root / OUT / "response_factor_fit_v31.json"
    write_json_atomic(path, result)
    freeze = stage_freeze(root, "response_factor_fit", [SOURCE, str(path.relative_to(root)), str(coordinate_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json"], {"result_sha256": sha256_file(path), "estimable_factor_rank": available, "heldout_response_seen": False})
    return {"freeze_digest": freeze["freeze_digest"], "training_rows": len(records), "rank": available}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("function", "response"))
    args = parser.parse_args()
    print(json.dumps(function_fit(Path.cwd()) if args.action == "function" else response_fit(Path.cwd()), indent=2))
