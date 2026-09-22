"""TRAIN-only surface-composition fit and held-out causal writeback tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, transplant
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.realization_v29 import contrast, inject, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/composition_v31.py"
OUT = Path("results/v31/processed")


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def load(root):
    verify_stage(root, "execution_plan")
    design = json.loads((root / OUT / "design_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    cfg = verify(root)["config"]
    pairs = {x["candidate_token_id"]: x for x in design["token_pair_library"]}
    compositions = {x["composition_id"]: x for group in design["surface_composition_splits"].values() for x in group}
    return design, plan, cfg, pairs, compositions


def interaction(a, b):
    na = max(float(np.linalg.norm(a)), 1e-12)
    nb = max(float(np.linalg.norm(b)), 1e-12)
    return (math.sqrt(len(a)) / (na * nb) * (a * b)).astype(np.float32)


def feature_stats(design, plan, pairs, comps):
    by_id = {x["base_trial_id"]: x for x in design["development"]}
    values = []
    for sid in plan["fit_state_ids"]:
        item = by_id[sid]
        for cid in plan["fit_composition_ids_per_state"][sid]:
            x = comps[cid]
            values.append((float(item["fork_total_length"]), float(pairs[x["A"]]["calibration_mean_rank"]), float(pairs[x["B"]]["calibration_mean_rank"])))
    array = np.asarray(values, dtype=np.float64)
    return {"length_mean": float(array[:, 0].mean()), "length_std": float(max(array[:, 0].std(), 1)), "rank_mean": float(array[:, 1:].mean()), "rank_std": float(max(array[:, 1:].std(), 1))}


def columns(a, b, length, rank_a, rank_b, stats):
    l = (length - stats["length_mean"]) / stats["length_std"]
    ra = (rank_a - stats["rank_mean"]) / stats["rank_std"]
    rb = (rank_b - stats["rank_mean"]) / stats["rank_std"]
    return [a, b, interaction(a, b)], [a, a * l, a * ra, b, b * l, b * rb]


def gram_update(g, rhs, xs, y):
    for i, xi in enumerate(xs):
        rhs[i] += float(np.dot(xi, y))
        for j in range(i + 1):
            dot = float(np.dot(xi, xs[j]))
            g[i, j] += dot
            if i != j:
                g[j, i] += dot


@torch.no_grad()
def fit(root: Path):
    design, plan, cfg, pairs, comps = load(root)
    bundle, dense, *_ = context(root)
    by_id = {x["base_trial_id"]: x for x in design["development"]}
    stats = feature_stats(design, plan, pairs, comps)
    gram3, rhs3 = np.zeros((3, 3), np.float64), np.zeros(3, np.float64)
    gram6, rhs6 = np.zeros((6, 6), np.float64), np.zeros(6, np.float64)
    records = []
    for n, sid in enumerate(plan["fit_state_ids"], 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming state drift: {sid}")
        anchor_id = cfg["token_library"]["anchor_token_id"]
        anchor = step(bundle, dense, incoming, token(anchor_id), length, design, 30)
        spec = layout(anchor["cache"], rec, att)
        for cid in plan["fit_composition_ids_per_state"][sid]:
            comp = comps[cid]
            vectors = {}
            for key in ("A", "B", "AB"):
                branch = step(bundle, dense, incoming, token(comp[key]), length, design, 30)
                vectors[key] = contrast(anchor["cache"], branch["cache"], spec)
                del branch
            a, b, y = vectors["A"], vectors["B"], vectors["AB"]
            length_feature = (float(item["fork_total_length"]) - stats["length_mean"]) / stats["length_std"]
            rank_a_feature = (float(pairs[comp["A"]]["calibration_mean_rank"]) - stats["rank_mean"]) / stats["rank_std"]
            rank_b_feature = (float(pairs[comp["B"]]["calibration_mean_rank"]) - stats["rank_mean"]) / stats["rank_std"]
            aa, bb, ab = float(np.dot(a, a)), float(np.dot(b, b)), float(np.dot(a, b))
            ay, by = float(np.dot(a, y)), float(np.dot(b, y))
            inter = interaction(a, b)
            ai, bi, ii, iy = float(np.dot(a, inter)), float(np.dot(b, inter)), float(np.dot(inter, inter)), float(np.dot(inter, y))
            gram3 += np.asarray([[aa, ab, ai], [ab, bb, bi], [ai, bi, ii]])
            rhs3 += np.asarray([ay, by, iy])
            factors = np.asarray([1, length_feature, rank_a_feature, 1, length_feature, rank_b_feature])
            kinds = [0, 0, 0, 1, 1, 1]
            base = np.asarray([[aa, ab], [ab, bb]])
            targets = np.asarray([ay, by])
            gram6 += np.asarray([[factors[i] * factors[j] * base[kinds[i], kinds[j]] for j in range(6)] for i in range(6)])
            rhs6 += np.asarray([factors[i] * targets[kinds[i]] for i in range(6)])
            additive = a + b
            records.append({"state_id": sid, "family": item["family"], "composition_id": cid, "natural_write_sha256": sha_array(y), "additive_write_relative_l2": float(np.linalg.norm(y - additive) / max(np.linalg.norm(y), 1e-12)), "natural_write_norm": float(np.linalg.norm(y))})
            del vectors, inter
        if n % 5 == 0 or n == len(plan["fit_state_ids"]):
            print(f"V31 TRAIN composition fit {n}/{len(plan['fit_state_ids'])}", flush=True)
    beta2 = np.linalg.solve(gram3[:2, :2] + np.eye(2) * 1e-6 * max(gram3[:2, :2].trace(), 1), rhs3[:2])
    beta3 = np.linalg.solve(gram3 + np.eye(3) * 1e-6 * max(gram3.trace(), 1), rhs3)
    beta6 = np.linalg.solve(gram6 + np.eye(6) * 1.0, rhs6)
    frame = pd.DataFrame(records)
    data_path = root / OUT / "composition_training_rows_v31.parquet"
    frame.to_parquet(data_path, index=False, compression="zstd")
    result = {"training_rows": len(frame), "training_states": len(plan["fit_state_ids"]), "three_scalar_coefficients": beta3.tolist(), "two_scalar_coefficients": beta2.tolist(), "state_conditioned_six_coefficients": beta6.tolist(), "feature_stats": stats, "gram3_sha256": sha_array(gram3), "rhs3_sha256": sha_array(rhs3), "gram6_sha256": sha_array(gram6), "rhs6_sha256": sha_array(rhs6), "TRAIN_only": True, "future_response_used_in_fit": False, "heldout_token_used_in_fit": False, "independent_final_opened": False, "training_rows_sha256": sha256_file(data_path)}
    result_path = root / OUT / "composition_fit_v31.json"
    write_json_atomic(result_path, result)
    frozen = stage_freeze(root, "composition_fit", [SOURCE, str(data_path.relative_to(root)), str(result_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_execution_plan.freeze.json"], {"fit_result_sha256": sha256_file(result_path), "fit_rows": len(frame), "heldout_response_seen": False})
    return {"freeze_digest": frozen["freeze_digest"], "training_rows": len(frame), "coefficients": beta3.tolist()}


@torch.no_grad()
def evaluate(root: Path, role: str):
    verify_stage(root, "composition_fit")
    if role == "validation":
        verify_stage(root, "composition_development")
    design, plan, cfg, pairs, comps = load(root)
    fit_result = json.loads((root / OUT / "composition_fit_v31.json").read_text())
    coef = np.asarray(fit_result["three_scalar_coefficients"], np.float32)
    coef2 = np.asarray(fit_result["two_scalar_coefficients"], np.float32)
    c6 = np.asarray(fit_result["state_conditioned_six_coefficients"], np.float32)
    stats = fit_result["feature_stats"]
    bundle, dense, *_ = context(root)
    scale = scales(root)
    by_id = {x["base_trial_id"]: x for x in design[role]}
    rows = []
    for n, (sid, cids) in enumerate(plan["heldout_composition_eval"][role].items(), 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming state drift: {sid}")
        anchor = step(bundle, dense, incoming, token(cfg["token_library"]["anchor_token_id"]), length, design, 30)
        spec = layout(anchor["cache"], rec, att)
        probes = [token(tid) for tid in item["future_probe_tokens"]]
        def response(cache):
            return signature([step(bundle, dense, cache, z, length + 1, design, 30) for z in probes], scale)
        native_sig = response(anchor["cache"])
        for cid in cids:
            comp = comps[cid]
            branch = {key: step(bundle, dense, incoming, token(comp[key]), length, design, 30) for key in ("A", "B", "AB")}
            a, b, target = [contrast(anchor["cache"], branch[key]["cache"], spec) for key in ("A", "B", "AB")]
            donor_sig = response(branch["AB"]["cache"])
            xs3, xs6 = columns(a, b, float(item["fork_total_length"]), float(pairs[comp["A"]]["calibration_mean_rank"]), float(pairs[comp["B"]]["calibration_mean_rank"]), stats)
            models = {"EXACT_REC_CONV": None, "UNIT_ADDITIVE": a + b, "GLOBAL_SCALAR_GATED": coef2[0] * a + coef2[1] * b, "LOW_ORDER_INTERACTION": coef[0] * a + coef[1] * b + coef[2] * xs3[2], "STATE_CONDITIONED_SCALAR_GATED": sum(c6[i] * xs6[i] for i in range(6)), "A_ONLY": a, "B_ONLY": b, "SIGN_FLIPPED_B": a - b}
            for condition, estimate in models.items():
                if condition == "EXACT_REC_CONV":
                    cache, _ = transplant(anchor["cache"], branch["AB"]["cache"], "REC+Conv", rec, att)
                else:
                    cache = inject(anchor["cache"], np.asarray(estimate, np.float32), 1, spec, rec, att)
                result = response(cache)
                row = {"role": role, "state_id": sid, "family": item["family"], "composition_id": cid, "surface_A": comp["surface_A"], "surface_B": comp["surface_B"], "surface_AB": comp["surface_AB"], "condition": condition, "incoming_state_hash": item["incoming_state_hash"], "probe_hash": item["future_probe_hash"], "natural_write_sha256": sha_array(target), "reconstruction_sha256": sha_array(estimate) if estimate is not None else None, "natural_donor_norm": float(np.linalg.norm(target)), "write_relative_l2": float(np.linalg.norm(target - estimate) / max(np.linalg.norm(target), 1e-12)) if estimate is not None else 0.0, **metrics(result, native_sig, donor_sig)}
                rows.append(row)
            del branch, a, b, target, donor_sig
        if n % 5 == 0 or n == len(plan["heldout_composition_eval"][role]):
            print(f"V31 causal composition {role} {n}/{len(plan['heldout_composition_eval'][role])}", flush=True)
    frame = pd.DataFrame(rows)
    data_path = root / OUT / f"heldout_composition_{role}_v31.parquet"
    frame.to_parquet(data_path, index=False, compression="zstd")
    profiles = profile(frame, cfg["causal_gate"])
    summary = {"role": role, "states": len(plan["heldout_composition_eval"][role]), "composition_tests": int(frame[["state_id", "composition_id"]].drop_duplicates().shape[0]), "rows": len(frame), "profiles": profiles, "response_sha256": sha256_file(data_path), "independent_final_opened": False, "surface_composition_not_semantic_proof": True, "causal_primary": True}
    result_path = root / OUT / f"heldout_composition_{role}_v31.json"
    write_json_atomic(result_path, summary)
    frozen = stage_freeze(root, f"composition_{role}", [SOURCE, str(data_path.relative_to(root)), str(result_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_composition_fit.freeze.json"], {"response_sha256": sha256_file(data_path), "profiles": {k: x["pass"] for k, x in profiles.items()}, "independent_final_opened": False})
    return {"freeze_digest": frozen["freeze_digest"], "states": summary["states"], "tests": summary["composition_tests"], "profiles": {k: {"pass": v["pass"], "median_relative_l2": v["median_relative_l2"]} for k, v in profiles.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "development", "validation"))
    args = parser.parse_args()
    print(json.dumps(fit(Path.cwd()) if args.action == "fit" else evaluate(Path.cwd(), args.action), indent=2))
