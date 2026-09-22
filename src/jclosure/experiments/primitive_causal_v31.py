"""Causal writeback grid for TRAIN-only primitive dictionaries on held-out AB tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, transplant
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.primitive_matrix_v31 import OUT, SCRATCH
from jclosure.experiments.realization_v29 import contrast, inject, layout
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/primitive_causal_v31.py"


def sha_array(x):
    return hashlib.sha256(memoryview(np.ascontiguousarray(x))).hexdigest()


def omp(gram, correlation, m, active):
    chosen = []
    coeff = np.zeros(m, np.float64)
    for _ in range(min(active, m)):
        residual_correlation = correlation[:m] - gram[:m, :m] @ coeff
        candidates = np.abs(residual_correlation)
        candidates[chosen] = -np.inf
        j = int(np.argmax(candidates))
        chosen.append(j)
        sub = gram[np.ix_(chosen, chosen)]
        coeff[chosen] = np.linalg.solve(sub + np.eye(len(chosen)) * 1e-6, correlation[chosen])
    return coeff.astype(np.float32), chosen


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "primitive_dictionary_fit")
    verify_stage(root, "function_conditioned_fit")
    verify_stage(root, "response_factor_fit")
    if role == "validation":
        verify_stage(root, "primitive_causal_development")
    design = json.loads((root / OUT / "design_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    fit = json.loads((root / OUT / "primitive_dictionary_fit_v31.json").read_text())
    cfg = verify(root)["config"]
    by_id = {x["base_trial_id"]: x for x in design[role]}
    pairs = {x["candidate_token_id"]: x for x in design["token_pair_library"]}
    comps = {x["composition_id"]: x for group in design["surface_composition_splits"].values() for x in group}
    dim = fit["write_dimension"]
    mean = np.memmap(SCRATCH / "primitive_mean_v31.f32", dtype=np.float32, mode="r", shape=(dim,))
    basis = {"PCA": np.memmap(SCRATCH / "basis_PCA256_v31.f32", dtype=np.float32, mode="r", shape=(256, dim)), "SPARSE_DICTIONARY": np.memmap(SCRATCH / "basis_SPARSE128_v31.f32", dtype=np.float32, mode="r", shape=(128, dim)), "CLUSTERED_PROTOTYPES": np.memmap(SCRATCH / "basis_PROTOTYPE128_v31.f32", dtype=np.float32, mode="r", shape=(128, dim))}
    coordinate = np.load(root / OUT / "primitive_dictionary_coordinates_v31.npz")
    function_models = json.loads((root / OUT / "function_conditioned_fit_v31.json").read_text())["models"]
    response_directions = np.load(root / OUT / "response_factor_coordinates_v31.npz")["latent_directions"].astype(np.float32)
    atoms = {"SPARSE_DICTIONARY": coordinate["sparse_atoms"].astype(np.float64), "CLUSTERED_PROTOTYPES": coordinate["prototype_atoms"].astype(np.float64)}
    atoms_gram = {model: x @ x.T for model, x in atoms.items()}
    bundle, dense, *_ = context(root)
    scale = scales(root)
    rows = []
    checkpoint_dir = SCRATCH / f"primitive_causal_{role}_{sha256_file(root / SOURCE)[:12]}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for n, (sid, cids) in enumerate(plan["heldout_composition_eval"][role].items(), 1):
        shard = checkpoint_dir / f"{hashlib.sha256(str(sid).encode()).hexdigest()[:20]}.parquet"
        if shard.exists():
            saved = pd.read_parquet(shard)
            if set(saved.state_id) != {sid} or set(saved.composition_id) != set(cids):
                raise RuntimeError(f"V31 causal checkpoint drift: {sid}")
            rows.extend(saved.to_dict("records"))
            print(f"V31 causal primitive grid {role} {n}/{len(plan['heldout_composition_eval'][role])} restored", flush=True)
            continue
        row_start = len(rows)
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
            donor = step(bundle, dense, incoming, token(comp["AB"]), length, design, 30)
            x = contrast(anchor["cache"], donor["cache"], spec)
            donor_sig = response(donor["cache"])
            base = {"role": role, "state_id": sid, "family": item["family"], "composition_id": cid, "token_id": comp["AB"], "token_category": pairs[comp["AB"]]["category"], "incoming_state_hash": item["incoming_state_hash"], "probe_hash": item["future_probe_hash"], "natural_write_sha256": sha_array(x), "primitive_fit_sha256": sha256_file(root / OUT / "primitive_dictionary_fit_v31.json"), "oracle_uses_heldout_natural_write_for_coefficient_estimation": True, "not_a_predictive_composition_test": True}
            exact = response(transplant(anchor["cache"], donor["cache"], "REC+Conv", rec, att)[0])
            rows.append({**base, "condition": "EXACT_REC_CONV", "model": "EXACT", "M": None, "active": None, "active_indices": "[]", "write_relative_l2": 0.0, **metrics(exact, native_sig, donor_sig)})
            centered = np.asarray(x - mean, dtype=np.float32)
            pca_coeff = basis["PCA"] @ centered
            for size in cfg["primitive_count_grid"]:
                approximation = np.asarray(mean + pca_coeff[:size] @ basis["PCA"][:size], dtype=np.float32)
                result = response(inject(anchor["cache"], approximation, 1, spec, rec, att))
                rows.append({**base, "condition": f"PCA_M{size}", "model": "PCA", "M": size, "active": size, "active_indices": json.dumps(list(range(size))), "write_relative_l2": float(np.linalg.norm(x - approximation) / max(np.linalg.norm(x), 1e-12)), **metrics(result, native_sig, donor_sig)})
                function = function_models[base["token_category"]]
                if size <= function["estimable_rank"]:
                    local_mean = np.asarray(function["global_PCA_coordinate_mean"], np.float32)
                    directions = np.asarray(function["local_directions"][:size], np.float32)
                    latent = local_mean + (directions @ (pca_coeff - local_mean)) @ directions
                    approximation = np.asarray(mean + latent @ basis["PCA"], dtype=np.float32)
                    result = response(inject(anchor["cache"], approximation, 1, spec, rec, att))
                    rows.append({**base, "condition": f"FUNCTION_CONDITIONED_M{size}", "model": "FUNCTION_CONDITIONED", "M": size, "active": size, "active_indices": json.dumps(list(range(size))), "write_relative_l2": float(np.linalg.norm(x - approximation) / max(np.linalg.norm(x), 1e-12)), **metrics(result, native_sig, donor_sig)})
                else:
                    rows.append({**base, "condition": f"FUNCTION_CONDITIONED_M{size}", "model": "FUNCTION_CONDITIONED", "M": size, "active": None, "active_indices": "[]", "write_relative_l2": None, "donor_cosine": None, "magnitude_ratio": None, "relative_l2_to_donor": None, "donor_norm": None, "movement_norm": None, "not_estimable_for_category": True})
                if size <= len(response_directions):
                    directions = response_directions[:, :size]
                    latent = directions @ (directions.T @ pca_coeff)
                    approximation = np.asarray(mean + latent @ basis["PCA"], dtype=np.float32)
                    result = response(inject(anchor["cache"], approximation, 1, spec, rec, att))
                    rows.append({**base, "condition": f"RESPONSE_FACTOR_M{size}", "model": "RESPONSE_FACTOR", "M": size, "active": size, "active_indices": json.dumps(list(range(size))), "write_relative_l2": float(np.linalg.norm(x - approximation) / max(np.linalg.norm(x), 1e-12)), **metrics(result, native_sig, donor_sig)})
                else:
                    rows.append({**base, "condition": f"RESPONSE_FACTOR_M{size}", "model": "RESPONSE_FACTOR", "M": size, "active": None, "active_indices": "[]", "write_relative_l2": None, "donor_cosine": None, "magnitude_ratio": None, "relative_l2_to_donor": None, "donor_norm": None, "movement_norm": None, "not_estimable_for_training_rank": True})
            for model in ("SPARSE_DICTIONARY", "CLUSTERED_PROTOTYPES"):
                correlation = atoms[model] @ pca_coeff.astype(np.float64)
                for size in cfg["primitive_count_grid"]:
                    for active in cfg["active_count_grid"]:
                        if active > size:
                            continue
                        coeff, indices = omp(atoms_gram[model], correlation, size, active)
                        approximation = np.asarray(mean + coeff @ basis[model][:size], dtype=np.float32)
                        result = response(inject(anchor["cache"], approximation, 1, spec, rec, att))
                        rows.append({**base, "condition": f"{model}_M{size}_s{active}", "model": model, "M": size, "active": active, "active_indices": json.dumps(indices), "write_relative_l2": float(np.linalg.norm(x - approximation) / max(np.linalg.norm(x), 1e-12)), **metrics(result, native_sig, donor_sig)})
            del donor, x, centered, pca_coeff
        pd.DataFrame(rows[row_start:]).to_parquet(shard, index=False, compression="zstd")
        if n % 5 == 0 or n == len(plan["heldout_composition_eval"][role]):
            print(f"V31 causal primitive grid {role} {n}/{len(plan['heldout_composition_eval'][role])}", flush=True)
    frame = pd.DataFrame(rows)
    data_path = root / OUT / f"primitive_causal_{role}_v31.parquet"
    frame.to_parquet(data_path, index=False, compression="zstd")
    profiles = profile(frame, cfg["causal_gate"])
    summary = {"role": role, "states": len(plan["heldout_composition_eval"][role]), "tests": int(frame[["state_id", "composition_id"]].drop_duplicates().shape[0]), "conditions": len(profiles), "rows": len(frame), "profiles": profiles, "response_sha256": sha256_file(data_path), "heldout_write_used_to_project_dictionary": True, "not_a_predictive_composition_test": True, "independent_final_opened": False}
    summary_path = root / OUT / f"primitive_causal_{role}_v31.json"
    write_json_atomic(summary_path, summary)
    freeze = stage_freeze(root, f"primitive_causal_{role}", [SOURCE, str(data_path.relative_to(root)), str(summary_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json"], {"response_sha256": sha256_file(data_path), "passing_conditions": [k for k, v in profiles.items() if v["pass"]], "independent_final_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "states": summary["states"], "tests": summary["tests"], "conditions": summary["conditions"], "passing_conditions": [k for k, v in profiles.items() if v["pass"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
