"""Frozen four-layer, same-input state × Conv operator factorial and suffix reinsertion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.operator_v36 import factors, native_local, parts
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/local_v36.py"
LETTERS = ("A", "B", "C")


def _module(model, key, layer):
    block = model.model.layers[layer]
    return block.linear_attn if key == "Q" else block.mamba


def _cos(a, b, floor=1e-6):
    if isinstance(a, torch.Tensor): a = a.float().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.float().cpu().numpy()
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > floor else None


def _norm(x):
    if isinstance(x, torch.Tensor): x = x.float().cpu().numpy()
    return float(np.linalg.norm(np.asarray(x, np.float64).ravel()))


def _spearman(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def _local_cache(base, state_source, conv_source, layer):
    work = clone_hybrid_cache(base)
    target = work.layers[layer]
    state = state_source.layers[layer].recurrent_states.detach().clone()
    conv = conv_source.layers[layer].conv_states.detach().clone()
    target.recurrent_states = state
    target.conv_states = conv
    if not torch.equal(target.recurrent_states, state_source.layers[layer].recurrent_states):
        raise RuntimeError("V36 REC writeback mismatch")
    if not torch.equal(target.conv_states, conv_source.layers[layer].conv_states):
        raise RuntimeError("V36 Conv writeback mismatch")
    return work, {"REC_requested": thash(state), "REC_realized": thash(target.recurrent_states),
                  "Conv_requested": thash(conv), "Conv_realized": thash(target.conv_states)}


@torch.no_grad()
def _capture(model, key, cache, probe, length, bundle, layers, with_raw=False):
    inputs, mixers, raws = {}, {}, {}
    handles = []
    for layer in layers:
        module = _module(model, key, layer)
        def pre(_m, args, kwargs, layer=layer):
            value = kwargs.get("hidden_states", args[0] if args else None)
            inputs[layer] = value.detach().clone()
        def output(_m, _args, result, layer=layer):
            mixers[layer] = result.detach().clone()
        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(output))
        if with_raw:
            def raw(_m, args, layer=layer):
                raws[layer] = args[0].detach().clone()
            handles.append(module.norm.register_forward_pre_hook(raw))
    try:
        result = step(model, cache, probe, length + 1, bundle)
    finally:
        for handle in handles: handle.remove()
    if set(inputs) != set(layers) or set(mixers) != set(layers):
        raise RuntimeError("V36 incomplete local capture")
    if with_raw and set(raws) != set(layers):
        raise RuntimeError("V36 incomplete raw read capture")
    return result, inputs, mixers, raws


@torch.no_grad()
def _patch_step(model, key, cache, probe, length, bundle, layer, replacement):
    module = _module(model, key, layer)
    proof = {}
    def patch(_m, _args, original):
        if replacement.shape != original.shape or replacement.dtype != original.dtype:
            raise RuntimeError("V36 local insertion topology mismatch")
        proof["native_hash"] = thash(original)
        proof["requested_hash"] = thash(replacement)
        proof["realized_hash"] = thash(replacement)
        return replacement
    handle = module.register_forward_hook(patch)
    try:
        result = step(model, cache, probe, length + 1, bundle)
    finally:
        handle.remove()
    if not proof or proof["requested_hash"] != proof["realized_hash"]:
        raise RuntimeError("V36 local insertion not realized")
    return result, proof


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"equation_{key}")
    gate = verify_stage(root, f"high_level_{role}")
    if not gate["formal_operator_authorized"]:
        raise RuntimeError("V36 high-level cross-model gate failed")
    if role == "validation":
        verify_stage(root, "local_analysis_development")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v36.parquet")
    base_npz = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v36.npz")
    baseline_vectors = base_npz["vectors"].astype(np.float64)
    baseline_by_id = {row.state_id: baseline_vectors[int(row.vector_index)] for row in factorial.itertuples()}
    rows, state_rows, vectors, proof_rows = [], [], [], []
    floor = float(cfg["operator_gate"]["denominator_benefit_floor_absolute"])
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V36 local incoming drift {key}:{sid}")
        caches = {
            "A": step(model, incoming, item["recipient_token_id"], length)["cache"],
            "B": step(model, incoming, item["donor_token_id"], length)["cache"],
            "C": step(model, incoming, item["third_token_id"], length)["cache"],
        }
        conv, _ = native_swap(caches["A"], caches["B"], ["Conv"], key)
        rec_only, _ = native_swap(caches["A"], caches["B"], ["REC"], key)
        joint, _ = native_swap(caches["A"], caches["B"], ["REC", "Conv"], key)
        branches = {"AA": caches["A"], "BA": rec_only, "AB": conv, "BB": joint}
        patch_outputs = {layer: {"DONOR_REC": [], "INTERACTION": []} for layer in design["local_layers"]}
        native_outputs = {name: [] for name in branches}
        for pi, probe in enumerate(item["future_probe_tokens"]):
            natural = {}
            for name, branch in branches.items():
                natural[name] = _capture(model, key, branch, probe, length,
                                         design["target_bundle"], design["local_layers"], with_raw=True)
                native_outputs[name].append(natural[name][0])
            h = natural["AB"][1]
            for layer in design["local_layers"]:
                module = _module(model, key, layer)
                s = {letter: caches[letter].layers[layer].recurrent_states for letter in LETTERS}
                f = {letter: factors(key, module, h[layer], caches[letter]) for letter in LETTERS}
                # Cast each raw read before subtraction: BF16 subtraction would
                # otherwise round the finite difference a second time.
                predictions = {letter: (parts(key, f[letter], s["B"])["raw"].float() -
                                        parts(key, f[letter], s["A"])["raw"].float())
                               for letter in LETTERS}
                local = {}
                proofs = {}
                # Full 3x3 response-blind state × native Conv-factor matrix.
                for si in LETTERS:
                    for oj in LETTERS:
                        condition = si + oj
                        work, proof = _local_cache(caches["A"], caches[si], caches[oj], layer)
                        local[condition] = native_local(module, h[layer], work)
                        proofs[condition] = proof
                if not torch.equal(local["AB"]["mixer"], natural["AB"][2][layer]):
                    raise RuntimeError(f"V36 same-input native replay mismatch {key}:{sid}:{layer}:{pi}")
                observed = {letter: local["B" + letter]["raw"].float() - local["A" + letter]["raw"].float()
                            for letter in LETTERS}
                pred_norms = [_norm(predictions[x].cpu()) for x in LETTERS]
                obs_norms = [_norm(observed[x].cpu()) for x in LETTERS]
                for letter in LETTERS:
                    pred, obs = predictions[letter], observed[letter]
                    error = float((pred - obs).abs().max().item())
                    limit = .02 + .03 * float(obs.square().mean().sqrt().item())
                    if error > limit:
                        pa = parts(key, f[letter], s["A"])["raw"].float()
                        pb = parts(key, f[letter], s["B"])["raw"].float()
                        oa = local["A" + letter]["raw"].float()
                        ob = local["B" + letter]["raw"].float()
                        raise RuntimeError(f"V36 prospective equation mismatch {key}:{sid}:{layer}:{pi}:{letter} "
                                           f"error={error} limit={limit} a={float((pa-oa).abs().max())} "
                                           f"b={float((pb-ob).abs().max())}")
                raw_i = (local["BB"]["raw"] - local["BA"]["raw"] -
                         local["AB"]["raw"] + local["AA"]["raw"])
                norm_i = (local["BB"]["normalized"] - local["BA"]["normalized"] -
                          local["AB"]["normalized"] + local["AA"]["normalized"])
                mix_i = (local["BB"]["mixer"] - local["BA"]["mixer"] -
                         local["AB"]["mixer"] + local["AA"]["mixer"])
                natural_i = (natural["BB"][2][layer] - natural["BA"][2][layer] -
                             natural["AB"][2][layer] + natural["AA"][2][layer])
                raw_nat_i = (natural["BB"][3][layer] - natural["BA"][3][layer] -
                             natural["AB"][3][layer] + natural["AA"][3][layer])
                causal_out = local["BB"]["mixer"]
                interaction_out = natural["AB"][2][layer] + mix_i
                patched, proof = _patch_step(model, key, conv, probe, length,
                                             design["target_bundle"], layer, causal_out)
                patched_i, proof_i = _patch_step(model, key, conv, probe, length,
                                                 design["target_bundle"], layer, interaction_out)
                patch_outputs[layer]["DONOR_REC"].append(patched)
                patch_outputs[layer]["INTERACTION"].append(patched_i)
                proof_rows.append({"state_id": sid, "layer": layer, "probe_index": pi,
                                   "source_hashes": json.dumps(proofs, sort_keys=True),
                                   "donor_rec_patch": json.dumps(proof, sort_keys=True),
                                   "interaction_patch": json.dumps(proof_i, sort_keys=True)})
                rec_effect = local["BB"]["mixer"] - local["AB"]["mixer"]
                factors_b = f["B"]
                pa = parts(key, factors_b, s["A"])
                pb = parts(key, factors_b, s["B"])
                old_diff = pb["decayed"] - pa["decayed"]
                update_diff = pb["update"] - pa["update"]
                rows.append({"model": key, "role": role, "state_id": sid, "family": item["family"],
                             "donor_token_category": item["donor_token_category"],
                             "layer": layer, "probe_index": pi, "probe_hash": item["future_probe_hash"],
                             "local_input_hash": thash(h[layer]),
                             "local_input_norm": float(h[layer].float().norm().item()),
                             "natural_input_A_B_distance": float((natural["AA"][1][layer] - natural["BB"][1][layer]).float().norm().item()),
                             "predicted_observed_cosine_A": _cos(predictions["A"].cpu(), observed["A"].cpu()),
                             "predicted_observed_cosine_B": _cos(predictions["B"].cpu(), observed["B"].cpu()),
                             "predicted_observed_cosine_C": _cos(predictions["C"].cpu(), observed["C"].cpu()),
                             "predicted_magnitude_rank_spearman": _spearman(pred_norms, obs_norms),
                             "predicted_matched_superior": pred_norms[1] > max(pred_norms[0], pred_norms[2]),
                             "observed_matched_superior": obs_norms[1] > max(obs_norms[0], obs_norms[2]),
                             "raw_interaction_norm": float(raw_i.float().norm().item()),
                             "normalized_interaction_norm": float(norm_i.float().norm().item()),
                             "mixer_interaction_norm": float(mix_i.float().norm().item()),
                             "natural_raw_interaction_norm": float(raw_nat_i.float().norm().item()),
                             "natural_mixer_interaction_norm": float(natural_i.float().norm().item()),
                             "fixed_vs_natural_mixer_interaction_cosine": _cos(mix_i.cpu(), natural_i.cpu()),
                             "raw_to_normalized_interaction_cosine": _cos(raw_i.cpu(), norm_i.cpu()),
                             "normalization_gain": float(norm_i.float().norm().item()) / max(float(raw_i.float().norm().item()), 1e-6),
                             "old_state_term_difference_norm": float(old_diff.float().norm().item()),
                             "state_dependent_update_difference_norm": float(update_diff.float().norm().item()),
                             "full_rec_mixer_effect_norm": float(rec_effect.float().norm().item()),
                             "formula_max_abs_error": max(float((predictions[x] - observed[x]).abs().max().item()) for x in LETTERS),
                             "native_replay_bitwise": True, "requested_realized_exact": True,
                             "vector_index": len(vectors)})
                vectors.append(np.stack([predictions[x].float().cpu().numpy().ravel() for x in LETTERS]).astype(np.float32))
        natural_signatures = {name: signature(outputs, design["calibration_clean_scales"])
                              for name, outputs in native_outputs.items()}
        baseline = baseline_by_id[sid]
        for name, ix in (("AA", 0), ("BA", 1), ("AB", 2), ("BB", 3)):
            if not np.allclose(natural_signatures[name], baseline[ix], atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"V36 natural response replay drift {key}:{sid}:{name}")
        donor_y, conv_y, joint_y = baseline[4], baseline[2], baseline[3]
        e_conv = _norm(donor_y - conv_y)
        natural_benefit = e_conv - _norm(donor_y - joint_y)
        for layer in design["local_layers"]:
            donor_rec_y = signature(patch_outputs[layer]["DONOR_REC"], design["calibration_clean_scales"])
            interaction_y = signature(patch_outputs[layer]["INTERACTION"], design["calibration_clean_scales"])
            benefit = e_conv - _norm(donor_y - donor_rec_y)
            interaction_benefit = e_conv - _norm(donor_y - interaction_y)
            state_rows.append({"model": key, "role": role, "state_id": sid, "family": item["family"],
                               "donor_token_category": item["donor_token_category"], "layer": layer,
                               "natural_benefit_absolute": natural_benefit,
                               "local_donor_rec_benefit_absolute": benefit,
                               "local_interaction_benefit_absolute": interaction_benefit,
                               "local_fraction": benefit / natural_benefit if natural_benefit >= floor else None,
                               "interaction_fraction": interaction_benefit / natural_benefit if natural_benefit >= floor else None,
                               "denominator_eligible": natural_benefit >= floor,
                               "local_donor_direction_cosine": _cos(donor_rec_y - conv_y, joint_y - conv_y),
                               "local_interaction_direction_cosine": _cos(interaction_y - conv_y, joint_y - conv_y),
                               "six_probes_one_state": True, "independent_unit": "state"})
        if n % 10 == 0 or n == len(design[role]):
            print(f"V36 local {key} {role} {n}/{len(design[role])}", flush=True)
    frame, state_frame = pd.DataFrame(rows), pd.DataFrame(state_rows)
    files = {"local": root / OUT / f"local_{key}_{role}_v36.parquet",
             "state": root / OUT / f"local_state_{key}_{role}_v36.parquet",
             "predictions": root / OUT / f"local_prediction_vectors_{key}_{role}_v36.npz",
             "proof": root / OUT / f"local_proof_{key}_{role}_v36.parquet"}
    frame.to_parquet(files["local"], index=False, compression="zstd")
    state_frame.to_parquet(files["state"], index=False, compression="zstd")
    np.savez_compressed(files["predictions"], predicted=np.stack(vectors).astype(np.float16),
                        state_ids=frame.state_id.to_numpy(str), layers=frame.layer.to_numpy(np.int16),
                        probe_indices=frame.probe_index.to_numpy(np.int8))
    pd.DataFrame(proof_rows).to_parquet(files["proof"], index=False, compression="zstd")
    summary = {"model": key, "role": role, "states": len(design[role]), "local_rows": len(frame),
               "state_layer_rows": len(state_frame),
               "median_matched_prediction_cosine": float(frame.predicted_observed_cosine_B.median()),
               "median_magnitude_rank_spearman": float(frame.predicted_magnitude_rank_spearman.median()),
               "matched_superior_fraction": float(frame.observed_matched_superior.mean()),
               "median_local_fraction_best_layer_descriptive": float(state_frame.groupby("layer").local_fraction.median().max()),
               "files_sha256": {k: sha256_file(v) for k, v in files.items()},
               "final_responses_seen": False}
    summary_path = root / OUT / f"local_{key}_{role}_v36.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"local_{key}_{role}", [SOURCE,
                        "src/jclosure/experiments/operator_v36.py",
                        *(str(path.relative_to(root)) for path in files.values()),
                        str(summary_path.relative_to(root)),
                        f"artifacts/computational_origin_v36_high_level_{role}.freeze.json",
                        f"artifacts/computational_origin_v36_equation_{key}.freeze.json"],
                        {"model": key, "role": role, "summary_sha256": sha256_file(summary_path),
                         "state_unit": True, "final_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
