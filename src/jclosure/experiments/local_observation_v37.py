"""Observe frozen V37 3×3 predictions and natural suffix consequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.local_v36 import _capture, _local_cache, _module, _patch_step
from jclosure.experiments.operator_v36 import native_local
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/local_observation_v37.py"
LETTERS = ("A", "B", "C")
FOUR = ("AA", "BA", "AB", "BB")


def _norm(x):
    return float(np.linalg.norm(np.asarray(x, np.float64).ravel()))


def _cos(a, b, floor=1e-8):
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = _norm(x) * _norm(y)
    return float(np.dot(x, y) / d) if d > floor else None


def _rank(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def _ordering(x, y, floor=1e-6):
    good = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx, dy = float(x[i]-x[j]), float(y[i]-y[j])
            if abs(dx) > floor and abs(dy) > floor:
                good.append((dx > 0) == (dy > 0))
    return float(np.mean(good)) if good else None


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"local_prediction_{key}_{role}")
    if role == "validation":
        verify_stage(root, "local_analysis_development")
    design = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    panel = json.loads((root / OUT / "panel_v37.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for r in
               ("calibration", "development", "validation", "independent_final") for row in panel[r]}
    predicted = pd.read_parquet(root / OUT / f"local_prediction_{key}_{role}_v37.parquet")
    vectors = np.load(root / OUT / f"local_prediction_vectors_{key}_{role}_v37.npz")
    pred_raw, pred_mixer = vectors["raw"], vectors["mixer"]
    by_key = {(row.state_id, row.position, int(row.probe_index)): row for row in predicted.itertuples()}
    high = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v37.parquet")
    high_vectors = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v37.npz")["vectors"].astype(np.float64)
    high_by_id = {row.state_id: high_vectors[int(row.vector_index)] for row in high.itertuples()}
    model, tokenizer = load(root, key)
    layers = list(design["local_layers"].values())
    rows, causal_rows, proofs = [], [], []
    observed_raw_vectors, observed_mixer_vectors = [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, prompts[sid])
        caches = {letter: step(model, incoming, token, length)["cache"]
                  for letter, token in zip(LETTERS, (item["recipient_token_id"],
                                             item["donor_token_id"], item["third_token_id"]), strict=True)}
        conv, _ = native_swap(caches["A"], caches["B"], ["Conv"], key)
        rec, _ = native_swap(caches["A"], caches["B"], ["REC"], key)
        joint, _ = native_swap(caches["A"], caches["B"], ["REC", "Conv"], key)
        natural_caches = {"AA": caches["A"], "BA": rec, "AB": conv, "BB": joint}
        future = {position: {name: [] for name in FOUR} for position in design["local_layers"]}
        for pi, probe in enumerate(item["future_probe_tokens"]):
            natural = {name: _capture(model, key, cache, probe, length,
                                      design["target_bundle"], layers, with_raw=True)
                       for name, cache in natural_caches.items()}
            h = natural["AB"][1]
            for position, layer in design["local_layers"].items():
                p = by_key[(sid, position, pi)]
                index = int(p.vector_index)
                if thash(h[layer]) != p.hidden_input_hash:
                    raise RuntimeError(f"V37 frozen hidden-input mismatch {key}:{sid}:{position}:{pi}")
                if thash(torch.from_numpy(pred_raw[index])) != p.raw_prediction_hash or \
                   thash(torch.from_numpy(pred_mixer[index])) != p.mixer_prediction_hash:
                    raise RuntimeError(f"V37 frozen prediction hash mismatch {key}:{sid}:{position}:{pi}")
                module = _module(model, key, layer)
                local = {}
                for si, state in enumerate(LETTERS):
                    for oi, operator in enumerate(LETTERS):
                        name = state + operator
                        work, proof = _local_cache(caches["A"], caches[state], caches[operator], layer)
                        local[name] = native_local(module, h[layer], work)
                        proofs.append({"state_id": sid, "model": key, "role": role,
                                       "position": position, "layer": layer, "probe_index": pi,
                                       "cell": name, "REC_requested": proof["REC_requested"],
                                       "REC_realized": proof["REC_realized"],
                                       "Conv_requested": proof["Conv_requested"],
                                       "Conv_realized": proof["Conv_realized"]})
                if not torch.equal(local["AB"]["mixer"], natural["AB"][2][layer]):
                    raise RuntimeError(f"V37 same-input Conv replay mismatch {key}:{sid}:{position}:{pi}")
                observed_raw = np.stack([np.stack([local[s+o]["raw"].float().cpu().numpy().ravel()
                                                    for o in LETTERS]) for s in LETTERS])
                observed_mix = np.stack([np.stack([local[s+o]["mixer"].float().cpu().numpy().ravel()
                                                    for o in LETTERS]) for s in LETTERS])
                observed_raw_vectors.append(observed_raw.astype(np.float32))
                observed_mixer_vectors.append(observed_mix.astype(np.float32))
                pr, pm = pred_raw[index].astype(np.float64), pred_mixer[index].astype(np.float64)
                if observed_raw.shape != pr.shape or observed_mix.shape != pm.shape:
                    raise RuntimeError("V37 3×3 predicted/observed topology mismatch")
                raw_error = float(np.max(np.abs(observed_raw-pr)))
                mixer_error = float(np.max(np.abs(observed_mix-pm)))
                if raw_error > .05 or mixer_error > .05:
                    raise RuntimeError(f"V37 prediction mismatch {key}:{sid}:{position}:{pi}:"
                                       f"raw={raw_error} mixer={mixer_error}")
                raw_delta_pred = pr.reshape(9,-1) - pr[0,0]
                raw_delta_obs = observed_raw.reshape(9,-1) - observed_raw[0,0]
                mix_delta_pred = pm.reshape(9,-1) - pm[0,0]
                mix_delta_obs = observed_mix.reshape(9,-1) - observed_mix[0,0]
                raw_norm_pred = np.linalg.norm(raw_delta_pred,axis=1)
                raw_norm_obs = np.linalg.norm(raw_delta_obs,axis=1)
                mix_norm_pred = np.linalg.norm(mix_delta_pred,axis=1)
                mix_norm_obs = np.linalg.norm(mix_delta_obs,axis=1)
                raw_cosines = [_cos(a,b) for a,b in zip(raw_delta_pred[1:],raw_delta_obs[1:],strict=True)]
                mixer_cosines = [_cos(a,b) for a,b in zip(mix_delta_pred[1:],mix_delta_obs[1:],strict=True)]
                def interaction(stage):
                    return (local["BB"][stage].float() - local["BA"][stage].float()
                            - local["AB"][stage].float() + local["AA"][stage].float()).cpu().numpy().ravel()
                i_raw, i_norm, i_mix = (interaction(stage) for stage in ("raw","normalized","mixer"))
                natural_i = (natural["BB"][2][layer].float() - natural["BA"][2][layer].float()
                             - natural["AB"][2][layer].float() + natural["AA"][2][layer].float()).cpu().numpy().ravel()
                rows.append({"state_id": sid, "model": key, "role": role, "family": item["family"],
                             "position": position, "layer": layer, "probe_index": pi,
                             "raw_effect_cosine": float(np.median([v for v in raw_cosines if v is not None])),
                             "mixer_effect_cosine": float(np.median([v for v in mixer_cosines if v is not None])),
                             "raw_magnitude_rank": _rank(raw_norm_pred, raw_norm_obs),
                             "mixer_magnitude_rank": _rank(mix_norm_pred, mix_norm_obs),
                             "raw_pairwise_order_accuracy": _ordering(raw_norm_pred, raw_norm_obs),
                             "mixer_pairwise_order_accuracy": _ordering(mix_norm_pred, mix_norm_obs),
                             "max_raw_error": raw_error, "max_mixer_error": mixer_error,
                             "raw_interaction_norm": _norm(i_raw),
                             "normalized_interaction_norm": _norm(i_norm),
                             "mixer_interaction_norm": _norm(i_mix),
                             "natural_interaction_norm": _norm(natural_i),
                             "fixed_natural_interaction_cosine": _cos(i_mix,natural_i),
                             "raw_to_normalized_cosine": _cos(i_raw,i_norm),
                             "normalization_gain": _norm(i_norm)/max(_norm(i_raw),1e-6),
                             "local_input_hash": thash(h[layer]),
                             "prediction_seal_verified": True})
                for name in FOUR:
                    out, proof = _patch_step(model, key, conv, probe, length,
                                             design["target_bundle"], layer, local[name]["mixer"])
                    future[position][name].append(out)
                    proofs.append({"state_id": sid, "model": key, "role": role,
                                   "position": position, "layer": layer, "probe_index": pi,
                                   "cell": "FUTURE_"+name, "requested_hash": proof["requested_hash"],
                                   "realized_hash": proof["realized_hash"],
                                   "native_hash": proof["native_hash"]})
        base = high_by_id[sid]
        yd, yconv, yjoint = base[4], base[2], base[3]
        natural_benefit = _norm(yd-yconv)-_norm(yd-yjoint)
        for position, conditions in future.items():
            ys = {name: signature(output, design["calibration_clean_scales"])
                  for name, output in conditions.items()}
            if not np.allclose(ys["AB"],yconv,rtol=1e-5,atol=1e-5):
                raise RuntimeError(f"V37 local Conv future replay drift {key}:{sid}:{position}")
            local_benefit = _norm(yd-yconv)-_norm(yd-ys["BB"])
            i_future = ys["BB"]-ys["BA"]-ys["AB"]+ys["AA"]
            causal_rows.append({"state_id": sid, "model": key, "role": role,
                                "family": item["family"], "position": position,
                                "layer": design["local_layers"][position],
                                "natural_benefit_absolute": natural_benefit,
                                "local_benefit_absolute": local_benefit,
                                "local_benefit_fraction": local_benefit/natural_benefit
                                                          if natural_benefit>1.0 else None,
                                "local_direction_cosine": _cos(ys["BB"]-yconv,yjoint-yconv),
                                "future_interaction_norm": _norm(i_future),
                                "future_interaction_cosine_to_joint": _cos(i_future,yjoint-yconv),
                                "six_probes_one_state": True,
                                "natural_downstream_propagation": True})
        if n%5 == 0 or n == len(design[role]):
            print(f"V37 local observation {key} {role} {n}/{len(design[role])}",flush=True)
    frames = {"local": pd.DataFrame(rows), "causal": pd.DataFrame(causal_rows),
              "proof": pd.DataFrame(proofs)}
    files = {name: root/OUT/f"{name}_observation_{key}_{role}_v37.parquet" for name in frames}
    files["observed_vectors"] = root/OUT/f"local_observed_vectors_{key}_{role}_v37.npz"
    for name,frame in frames.items():
        frame.to_parquet(files[name],index=False,compression="zstd")
    np.savez_compressed(files["observed_vectors"],raw=np.stack(observed_raw_vectors),
                        mixer=np.stack(observed_mixer_vectors),
                        state_ids=frames["local"].state_id.to_numpy(str),
                        positions=frames["local"].position.to_numpy(str),
                        probe_indices=frames["local"].probe_index.to_numpy(np.int8))
    summary = {"model":key,"role":role,"states":len(design[role]),
               "local_rows":len(frames["local"]),"state_layer_rows":len(frames["causal"]),
               "median_raw_cosine":float(frames["local"].raw_effect_cosine.median()),
               "median_mixer_cosine":float(frames["local"].mixer_effect_cosine.median()),
               "median_mixer_pairwise_order":float(frames["local"].mixer_pairwise_order_accuracy.median()),
               "files_sha256":{name:sha256_file(path) for name,path in files.items()},
               "prediction_seal_verified":True,"independent_final_seen":False}
    path=root/OUT/f"local_observation_{key}_{role}_v37.json"
    write_json_atomic(path,summary)
    seal=stage_freeze(root,f"local_observation_{key}_{role}",
                      [SOURCE,str(path.relative_to(root)),*[str(p.relative_to(root)) for p in files.values()],
                       f"artifacts/computational_origin_v37_local_prediction_{key}_{role}.freeze.json"],
                      {"model":key,"role":role,"summary_sha256":sha256_file(path),
                       "prediction_seal_verified":True})
    return {"freeze_digest":seal["freeze_digest"],**summary}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("model",choices=("Q","F"))
    p.add_argument("role",choices=("development","validation"))
    a=p.parse_args();print(json.dumps(run(Path.cwd(),a.model,a.role),indent=2))
