"""Metric-corrected rank scaling and oracle local-chart diagnostics for V23."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments import action_pool_v22 as actions
from jclosure.protocol_v23 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/analyze_v23.py"
OUT = Path("results/v23/processed")
SCRATCH = Path("/data/CSK/J-space-project/v23-oracle-chart-work")
V22_EXPANDED = Path("/data/CSK/J-space-project/v22-action-manifold-work/expanded_response_bank")
V22_SCALE = Path("/data/CSK/J-space-project/v22-action-manifold-work/scale_composition")
STATE_REPS = Path("/data/CSK/J-space-project/v21-action-geometry-work/state_representations_v21.npz")
RAW_REPS = Path("/data/CSK/J-space-project/v21-action-geometry-work/raw_state_reference_v21.npz")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _metric(x: np.ndarray, relative_floor: float = 1e-8, absolute_floor: float = 1e-12):
    gram = np.asarray(x, np.float64) @ np.asarray(x, np.float64).T
    eigen, q = np.linalg.eigh(gram)
    order = np.argsort(eigen)[::-1]
    eigen, q = eigen[order], q[:, order]
    keep = eigen > max(float(eigen[0]) * relative_floor, absolute_floor)
    retained = eigen[keep]
    w = q[:, keep] / np.sqrt(retained)[None, :]
    return {"gram": gram, "eigen": eigen, "keep": keep, "retained": retained, "w": w,
            "rank": int(keep.sum()), "condition": float(retained[0] / retained[-1]),
            "min": float(retained[-1]), "max": float(retained[0])}


def _ranks(s: np.ndarray):
    e = np.square(s)
    p = e / max(float(e.sum()), 1e-30)
    c = np.cumsum(p)
    rank = [int(np.searchsorted(c, level) + 1) for level in (0.90, 0.95, 0.99)]
    stable = float(e.sum() / max(float(e[0]), 1e-30))
    entropy = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    return rank, stable, entropy


def _operator(y: np.ndarray, metric: dict):
    u, s, vh = np.linalg.svd(np.asarray(y, np.float64) @ metric["w"], full_matrices=False)
    rank, stable, entropy = _ranks(s)
    return {"u": u, "s": s, "v": vh.T, "r90": rank[0], "r95": rank[1], "r99": rank[2],
            "stable": stable, "entropy": entropy}


def _subspace(left: np.ndarray, right: np.ndarray):
    n = min(left.shape[1], right.shape[1])
    if n == 0:
        return {"median_angle": 90.0, "max_angle": 90.0, "grassmann": float(math.pi / 2), "overlap": 0.0}
    s = np.clip(np.linalg.svd(left[:, :n].T @ right[:, :n], compute_uv=False), 0, 1)
    angle = np.arccos(s)
    return {"median_angle": float(np.degrees(np.median(angle))), "max_angle": float(np.degrees(np.max(angle))),
            "grassmann": float(np.linalg.norm(angle)), "overlap": float(np.mean(s * s))}


def _metrics(target: np.ndarray, predicted: np.ndarray):
    target = np.asarray(target, np.float64).reshape(-1, 288)
    predicted = np.asarray(predicted, np.float64).reshape(-1, 288)
    l2 = float(np.linalg.norm(predicted - target) / max(np.linalg.norm(target), 1e-12))
    tn = np.linalg.norm(target, axis=1); pn = np.linalg.norm(predicted, axis=1)
    cosine = np.sum(target * predicted, axis=1) / np.maximum(tn * pn, 1e-12)
    jtn = np.linalg.norm(target[:, :128], axis=1); jpn = np.linalg.norm(predicted[:, :128], axis=1)
    jcos = np.sum(target[:, :128] * predicted[:, :128], axis=1) / np.maximum(jtn * jpn, 1e-12)
    ratio = pn / np.maximum(tn, 1e-12)
    return {"relative_l2": l2, "median_cosine": float(np.median(cosine)), "median_j_cosine": float(np.median(jcos)),
            "median_norm_ratio": float(np.median(ratio)), "sample_count": int(len(target))}


def rank_scaling(root: Path) -> dict:
    config = verify(root)["config"]
    verify_stage(root, "rank_measurement_design")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    with np.load(root / SCRATCH / "probe_metric_v23.npz") as z:
        x = np.asarray(z["action_matrix"], np.float64)
    rows = []
    gram = {}
    for m in config["probe_counts"]:
        gm = _metric(x[:m], config["geometry"]["gram_relative_eigenvalue_floor"], config["geometry"]["absolute_eigenvalue_floor"])
        gram[str(m)] = {"raw_gram_rank": int(np.linalg.matrix_rank(gm["gram"])), "effective_gram_rank": gm["rank"],
                        "retained_condition_number": gm["condition"], "retained_eigenvalue_min": gm["min"],
                        "retained_eigenvalue_max": gm["max"], "discarded_modes": int(m - gm["rank"])}
        for kind, role in (("JVP", "jvp_rank_development"), ("finite", "finite_rank_development")):
            for item in design["state_roles"][role]:
                path = root / SCRATCH / f"rank_{kind.lower()}" / f"{kind.lower()}_{item['base_trial_id']}.npz"
                with np.load(path) as z:
                    for state in ("P0", "Pq"):
                        y = np.asarray(z[f"{state}_JVP"][:, :m], np.float64) if kind == "JVP" else (
                            np.asarray(z[f"{state}_plus"][:m], np.float64) - np.asarray(z[f"{state}_minus"][:m], np.float64)).T / 2
                        op = _operator(y, gm)
                        rows.append({"kind": kind, "m": m, "base_trial_id": item["base_trial_id"], "family": item["family"],
                                     "state": state, "input_r90": op["r90"], "input_r95": op["r95"], "input_r99": op["r99"],
                                     "output_r90": op["r90"], "output_r95": op["r95"], "output_r99": op["r99"],
                                     "stable_rank": op["stable"], "entropy_effective_rank": op["entropy"]})
    frame = pd.DataFrame(rows)
    parquet = root / OUT / "input_rank_scaling_rows_v23.parquet"
    frame.to_parquet(parquet, index=False, compression="zstd")
    summary = {"gram": gram, "curves": {}, "saturation_rule": config["rank_saturation"],
               "input_output_rank_note": "Input and output energy ranks are the shared nonzero singular-spectrum ranks of the metric-corrected operator; input vectors and output vectors live in different spaces.",
               "rows_sha256": sha256_file(parquet)}
    for kind in ("JVP", "finite"):
        summary["curves"][kind] = {}
        for m in config["probe_counts"]:
            group = frame[(frame.kind == kind) & (frame.m == m)]
            summary["curves"][kind][str(m)] = {
                "input_r90_r95_r99_median": [float(group.input_r90.median()), float(group.input_r95.median()), float(group.input_r99.median())],
                "output_r90_r95_r99_median": [float(group.output_r90.median()), float(group.output_r95.median()), float(group.output_r99.median())],
                "stable_rank_median": float(group.stable_rank.median()), "entropy_effective_rank_median": float(group.entropy_effective_rank.median()),
                "family_input_r95_median": {family: float(g.input_r95.median()) for family, g in group.groupby("family")},
                "operator_state_count": int(len(group)),
            }
    def saturated(kind: str):
        counts = config["probe_counts"]
        checks = []
        for a, b in zip(counts[-3:-1], counts[-2:]):
            old = summary["curves"][kind][str(a)]; new = summary["curves"][kind][str(b)]
            values = [(old["input_r90_r95_r99_median"][1], new["input_r90_r95_r99_median"][1])]
            values += [(old["family_input_r95_median"][family], new["family_input_r95_median"][family])
                       for family in old["family_input_r95_median"] if family in new["family_input_r95_median"]]
            checks.append({"from": a, "to": b, "pass": all(abs(y-x) <= 2 and abs(y-x)/max(x,1) <= .15 for x,y in values),
                           "values": values})
        return bool(all(x["pass"] for x in checks)), checks
    for kind in ("JVP", "finite"):
        value, checks = saturated(kind)
        summary[f"{kind}_INPUT_RANK_SATURATED"] = value
        summary[f"{kind}_saturation_checks"] = checks
    summary["INPUT_RANK_SATURATED"] = bool(summary["JVP_INPUT_RANK_SATURATED"] and summary["finite_INPUT_RANK_SATURATED"])
    write_json_atomic(root / OUT / "input_rank_scaling_v23.json", summary)
    return summary


def _chart(x_train: np.ndarray, odd: np.ndarray):
    metric = _metric(x_train)
    op = _operator(odd.T, metric)
    return {"metric": metric, **op}


def _coordinates(chart: dict, x_train: np.ndarray, raw_actions: np.ndarray, k: int):
    physical = chart["metric"]["w"].T @ (x_train @ np.asarray(raw_actions, np.float64).T)
    return (chart["v"][:, :k].T @ physical).T


def _features(z: np.ndarray, degree: int):
    blocks = [np.ones((len(z), 1)), z]
    if degree >= 2: blocks.append(z * z)
    if degree >= 3: blocks.append(z * z * z)
    return np.concatenate(blocks, axis=1)


def _ridge(features: np.ndarray, targets: np.ndarray):
    scale = max(float(np.trace(features.T @ features) / features.shape[1]) * 1e-4, 1e-8)
    penalty = np.eye(features.shape[1]) * scale; penalty[0, 0] = 0
    return np.linalg.solve(features.T @ features + penalty, features.T @ targets)


def _decoder(chart: dict, x_train: np.ndarray, plus: np.ndarray, minus: np.ndarray, k: int, model: str):
    z = _coordinates(chart, x_train, x_train, k)
    train_z = np.concatenate([z, -z])
    train_y = np.concatenate([plus, minus])
    if model == "M0_nearest":
        def predict(raw):
            query = _coordinates(chart, x_train, raw, k)
            distance = ((query[:, None] - train_z[None]) ** 2).sum(2)
            return train_y[np.argmin(distance, axis=1)]
        return predict
    if model == "M1_local_linear":
        def predict(raw):
            query = _coordinates(chart, x_train, raw, k)
            return (chart["u"][:, :k] * chart["s"][:k]) @ query.T
        return lambda raw: predict(raw).T
    degree = {"M2_ridge": 1, "M3_quadratic": 2, "M4_cubic_nonlinear": 3}[model]
    beta = _ridge(_features(train_z, degree), train_y)
    return lambda raw: _features(_coordinates(chart, x_train, raw, k), degree) @ beta


def _expanded_state(root: Path, role: str, item: dict, train_ids: list[str], held_ids: list[str]):
    with np.load(root / V22_EXPANDED / role / f"expanded_{item['base_trial_id']}.npz") as z:
        ids = [str(x) for x in z["action_ids"]]
        train = [ids.index(x) for x in train_ids]
        held = [ids.index(x) for x in held_ids]
        return {state: {"train_plus": np.asarray(z[f"{state}_plus"][train], np.float64),
                        "train_minus": np.asarray(z[f"{state}_minus"][train], np.float64),
                        "held_plus": np.asarray(z[f"{state}_plus"][held], np.float64),
                        "held_minus": np.asarray(z[f"{state}_minus"][held], np.float64)} for state in ("P0", "Pq")}


def _raw_lookup(root: Path):
    pool = json.loads((root / "results/v22/processed/action_pool_v22.json").read_text())
    selection = json.loads((root / "results/v22/processed/action_selection_v22.json").read_text())
    calibration = json.loads((root / "results/v22/processed/action_pool_calibration_v22.json").read_text())
    specs = {x["action_id"]: x for x in pool["candidate_pool"]}
    alpha = {x["action_id"]: float(x["selected_alpha"]) for x in calibration["by_action"] if x["selected_alpha"] is not None}
    alpha.update({k: float(v) for k,v in selection["action_alphas"].items()})
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    score = np.asarray(directions["score_directions"], np.float64); del directions
    return specs, alpha, score


def oracle(root: Path) -> dict:
    config = verify(root)["config"]
    verify_stage(root, "rank_measurement_design")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    selection = json.loads((root / OUT / "probe_selection_v23.json").read_text())
    v22_selection = json.loads((root / "results/v22/processed/action_selection_v22.json").read_text())
    scale_design = json.loads((root / "results/v22/processed/scale_composition_design_v22.json").read_text())
    scale_law = json.loads((root / "results/v22/processed/action_scale_law_v22.json").read_text())
    with np.load(root / SCRATCH / "probe_metric_v23.npz") as z:
        x_all = np.asarray(z["action_matrix"], np.float64); x_hold = np.asarray(z["heldout_action_matrix"], np.float64)
    x_train = x_all[:128]; train_ids = selection["train_action_ids"][:128]; held_ids = selection["heldout_validation_action_ids"]
    specs, alpha, score = _raw_lookup(root)
    def raw(action_id): return actions._score(specs[action_id], score) * alpha[action_id]
    accumulator = {kind: defaultdict(lambda: defaultdict(list)) for kind in ("finite", "JVP")}
    chart_records = {"finite": {}, "JVP": {}}
    state_models = {"finite": {}, "JVP": {}}
    role_items = [("development", x) for x in design["state_roles"]["oracle_development"]] + [("validation", x) for x in design["state_roles"]["oracle_validation"]]
    jvp_ids = {x["base_trial_id"] for x in design["state_roles"]["jvp_rank_development"]}
    for role, item in role_items:
        expanded = _expanded_state(root, role, item, train_ids, held_ids)
        jvp_path = root / SCRATCH / "rank_jvp" / f"jvp_{item['base_trial_id']}.npz"
        jvp_file = np.load(jvp_path) if item["base_trial_id"] in jvp_ids else None
        for state in ("P0", "Pq"):
            finite = expanded[state]
            odd = (finite["train_plus"] - finite["train_minus"]) / 2
            chart_sources = {"finite": odd}
            if jvp_file is not None:
                chart_sources["JVP"] = np.asarray(jvp_file[f"{state}_JVP"][:, :128], np.float64).T
            for kind, chart_source in chart_sources.items():
                chart = _chart(x_train, chart_source)
                key = f"{item['base_trial_id']}::{state}"
                chart_records[kind][key] = {"base_trial_id": item["base_trial_id"], "role": role, "family": item["family"],
                                            "q_name": item["q_name"], "state": state, "r90": chart["r90"], "r95": chart["r95"],
                                            "r99": chart["r99"], "stable_rank": chart["stable"], "entropy_rank": chart["entropy"],
                                            "v": chart["v"], "u": chart["u"], "s": chart["s"]}
                state_models[(kind, key)] = (chart, finite)
                for k0 in config["chart_dimensions"]:
                    k = min(k0, chart["v"].shape[1])
                    for model in config["models"]["names"]:
                        predict = _decoder(chart, x_train, finite["train_plus"], finite["train_minus"], k, model)
                        token = f"k{k0}:{model}"
                        for category, target, raw_actions in (("unseen_direction", finite["held_plus"], x_hold),
                                                              ("unseen_sign", finite["held_minus"], -x_hold)):
                            accumulator[kind][token][category].append((target, predict(raw_actions), item["family"]))
        if jvp_file is not None: jvp_file.close()
    # Scale and composition use the ten frozen V22 panels and no final labels.
    panel_items = [("development", x) for x in scale_design["development_states"]] + [("validation", x) for x in scale_design["validation_states"]]
    scale_ids = scale_design["scale_action_ids"][scale_design["scale_train_count"]:]
    for role, item in panel_items:
        panel_path = root / V22_SCALE / role / f"panel_{item['base_trial_id']}.npz"
        with np.load(panel_path) as panel:
            for state in ("P0", "Pq"):
                key = f"{item['base_trial_id']}::{state}"
                for kind in ("finite", "JVP"):
                    if (kind, key) not in state_models: continue
                    chart, finite = state_models[(kind, key)]
                    for k0 in config["chart_dimensions"]:
                        k = min(k0, chart["v"].shape[1])
                        for model in config["models"]["names"]:
                            predict = _decoder(chart, x_train, finite["train_plus"], finite["train_minus"], k, model)
                            token = f"k{k0}:{model}"
                            scale_pred, scale_target = [], []
                            for local_index, action_id in enumerate(scale_ids):
                                global_index = scale_design["scale_action_ids"].index(action_id)
                                base_raw = raw(action_id)[None]
                                for amp_index, amp in enumerate(scale_design["amplitudes"]):
                                    plus_gain = scale_law["results"]["plus"]["monotone_gain_network"]["fitted_gains"][amp_index]
                                    minus_gain = scale_law["results"]["minus"]["monotone_gain_network"]["fitted_gains"][amp_index]
                                    scale_pred.extend([predict(base_raw)[0] * plus_gain, predict(-base_raw)[0] * minus_gain])
                                    scale_target.extend([panel[f"{state}_scale_plus"][global_index, amp_index],
                                                         panel[f"{state}_scale_minus"][global_index, amp_index]])
                            accumulator[kind][token]["unseen_amplitude"].append((np.asarray(scale_target), np.asarray(scale_pred), item["family"]))
                            for comp_kind in ("pair", "dense"):
                                pred, target = [], []
                                for index, comp in enumerate(scale_design["compositions"]):
                                    if comp["kind"] != comp_kind: continue
                                    combined = sum(float(weight) * raw(action_id) for action_id, weight in comp["components"])[None]
                                    pred.extend([predict(combined)[0], predict(-combined)[0]])
                                    target.extend([panel[f"{state}_composition_plus"][index], panel[f"{state}_composition_minus"][index]])
                                accumulator[kind][token][f"unseen_{comp_kind}"].append((np.asarray(target), np.asarray(pred), item["family"]))
    def reduce(kind):
        answer = {}
        for token, categories in accumulator[kind].items():
            answer[token] = {}
            for category, blocks in categories.items():
                target = np.concatenate([x[0] for x in blocks]); predicted = np.concatenate([x[1] for x in blocks])
                metric = _metrics(target, predicted)
                family = {}
                for name in sorted({x[2] for x in blocks}):
                    t = np.concatenate([x[0] for x in blocks if x[2] == name]); p = np.concatenate([x[1] for x in blocks if x[2] == name])
                    family[name] = _metrics(t, p)["relative_l2"]
                metric["family_relative_l2"] = family
                answer[token][category] = metric
        return answer
    results = {kind: reduce(kind) for kind in ("finite", "JVP")}
    required = config["oracle_gate"]["required_categories"]
    def passes(categories):
        gate = config["oracle_gate"]
        return all(name in categories and categories[name]["relative_l2"] <= gate["relative_l2_max"]
                   and categories[name]["median_cosine"] >= gate["stack_cosine_min"]
                   and categories[name]["median_j_cosine"] >= gate["j_cosine_min"]
                   and gate["norm_ratio_min"] <= categories[name]["median_norm_ratio"] <= gate["norm_ratio_max"]
                   and max(categories[name]["family_relative_l2"].values(), default=99) <= gate["relative_l2_max"] for name in required)
    best = {}
    for kind in ("finite", "JVP"):
        ranked = sorted(results[kind], key=lambda token: np.mean([results[kind][token].get(name,{"relative_l2":99})["relative_l2"] for name in required]))
        best[kind] = {"token": ranked[0], "categories": results[kind][ranked[0]], "gate_pass": passes(results[kind][ranked[0]])}
        passing = [token for token in ranked if passes(results[kind][token])]
        best[kind]["passing_tokens"] = passing
    finite_pass = bool(best["finite"]["gate_pass"])
    jvp_pass = bool(best["JVP"]["gate_pass"])
    passing_ks = [int(token.split(":")[0][1:]) for token in best["finite"]["passing_tokens"] + best["JVP"]["passing_tokens"]]
    # JVP/finite chart comparisons and same-J aliasing.
    cross, same_j, transport = [], [], []
    best_k = int(best["finite"]["token"].split(":")[0][1:])
    for key, fin in chart_records["finite"].items():
        kf = min(best_k, fin["v"].shape[1])
        if key in chart_records["JVP"]:
            jvp = chart_records["JVP"][key]; kj = min(kf, jvp["v"].shape[1])
            cross.append(_subspace(jvp["v"][:, :kj], fin["v"][:, :kj]))
        if fin["state"] == "P0":
            other = chart_records["finite"].get(f"{fin['base_trial_id']}::Pq")
            if other:
                value = _subspace(fin["v"][:, :kf], other["v"][:, :kf])
                out = _subspace(fin["u"][:, :kf], other["u"][:, :kf])
                value.update({"family": fin["family"], "base_trial_id": fin["base_trial_id"],
                              "output_median_angle": out["median_angle"],
                              "spectrum_relative_change": float(np.linalg.norm(fin["s"]-other["s"]) / max(np.linalg.norm(fin["s"]),1e-12))})
                same_j.append(value)
                c = fin["v"][:, :kf].T @ other["v"][:, :kf]
                uu, _, vv = np.linalg.svd(c); rotation = uu @ vv
                residual = float(np.linalg.norm(fin["v"][:, :kf] @ rotation - other["v"][:, :kf]) / math.sqrt(kf))
                transport.append({"base_trial_id": fin["base_trial_id"], "family": fin["family"], "procrustes_residual": residual,
                                  "transport_fidelity": float(max(0, 1-residual/math.sqrt(2))), "projector_overlap": value["overlap"]})
    # Natural P0 smoothness on nearest-J prompt states, same-family and cross-family pairs.
    reps = np.load(STATE_REPS)
    rep_map = {}
    for split in ("train", "validation"):
        for i, name in enumerate(reps[f"{split}_ids"]): rep_map[str(name)] = np.asarray(reps[f"S0_{split}"][i], np.float64)
    p0 = [x for x in chart_records["finite"].values() if x["state"] == "P0"]
    pair_rows = []
    for i, left in enumerate(p0):
        lv = rep_map[f"{left['base_trial_id']}::P0"]
        for right in p0[i+1:]:
            rv = rep_map[f"{right['base_trial_id']}::P0"]
            distance = float(np.linalg.norm(lv-rv) / max(np.linalg.norm(lv)+np.linalg.norm(rv),1e-12))
            angle = _subspace(left["v"][:, :best_k], right["v"][:, :best_k])
            pair_rows.append({"left": left["base_trial_id"], "right": right["base_trial_id"], "same_family": left["family"]==right["family"],
                              "j_distance": distance, **angle})
    nearest = []
    for left in p0:
        candidates = [x for x in pair_rows if x["left"] == left["base_trial_id"] or x["right"] == left["base_trial_id"]]
        nearest.append(min(candidates, key=lambda x:x["j_distance"]))
    smoothness = {
        "adjacent_tokens_same_prompt": "NOT_MEASURED_NO_FROZEN_ADJACENT_TOKEN_ORACLE_PANEL",
        "nearby_J_median_angle_degrees": float(np.median([x["median_angle"] for x in nearest])),
        "nearby_J_median_grassmann": float(np.median([x["grassmann"] for x in nearest])),
        "same_family_median_angle_degrees": float(np.median([x["median_angle"] for x in pair_rows if x["same_family"]])),
        "across_family_median_angle_degrees": float(np.median([x["median_angle"] for x in pair_rows if not x["same_family"]])),
        "natural_state_count": len(p0), "claim": "Nearest-J clean prompt states only; no adjacent-token smoothness claim is made."
    }
    # Partial local-probe identification curve, finite charts, held-out direction/sign.
    sample_curve = {}
    for n in config["local_probe_counts"]:
        candidate_scores = defaultdict(list)
        xn = x_train[:n]
        for role, item in role_items:
            expanded = _expanded_state(root, role, item, train_ids, held_ids)
            for state in ("P0", "Pq"):
                finite = expanded[state]; plus=finite["train_plus"][:n]; minus=finite["train_minus"][:n]
                chart = _chart(xn, (plus-minus)/2)
                for k0 in config["chart_dimensions"]:
                    if k0 > min(n,64): continue
                    k=min(k0,chart["v"].shape[1])
                    for model in config["models"]["names"]:
                        pred=_decoder(chart,xn,plus,minus,k,model)
                        candidate_scores[f"k{k0}:{model}"].append((finite["held_plus"],pred(x_hold),finite["held_minus"],pred(-x_hold)))
        scored=[]
        for token, blocks in candidate_scores.items():
            t=np.concatenate([np.concatenate([x[0],x[2]]) for x in blocks]); p=np.concatenate([np.concatenate([x[1],x[3]]) for x in blocks])
            scored.append(( _metrics(t,p)["relative_l2"], token, _metrics(t,p)))
        scored.sort()
        sample_curve[str(n)]={"best_token":scored[0][1],**scored[0][2]}
    learned_v22 = json.loads((root / "results/v22/processed/state_conditioned_action_chart_v22.json").read_text())
    learned_l2 = float(learned_v22.get("state_conditioned_local_chart", {}).get("relative_l2", 1.0016))
    oracle_direction_l2 = min(best["finite"]["categories"]["unseen_direction"]["relative_l2"], best["JVP"]["categories"]["unseen_direction"]["relative_l2"])
    result = {
        "chart_dimensions": config["chart_dimensions"], "models": config["models"]["names"],
        "best": best, "ORACLE_FINITE_LOCAL_CHART_PASS": finite_pass, "ORACLE_JVP_LOCAL_CHART_PASS": jvp_pass,
        "ORACLE_LOCAL_CHART_PASS": bool(finite_pass or jvp_pass), "k_chart_min": min(passing_ks) if passing_ks else None,
        "jvp_finite_comparison": {"median_input_angle_degrees": float(np.median([x["median_angle"] for x in cross])),
                                  "median_input_overlap": float(np.median([x["overlap"] for x in cross])),
                                  "finite_minus_jvp_best_direction_l2": float(best["finite"]["categories"]["unseen_direction"]["relative_l2"]-best["JVP"]["categories"]["unseen_direction"]["relative_l2"]),
                                  "identical_operator_state_count": len(cross)},
        "same_J": {"finite_input_chart_angle_median_degrees": float(np.median([x["median_angle"] for x in same_j])),
                   "finite_output_chart_angle_median_degrees": float(np.median([x["output_median_angle"] for x in same_j])),
                   "spectrum_relative_change_median": float(np.median([x["spectrum_relative_change"] for x in same_j])),
                   "pair_count": len(same_j)},
        "smoothness": smoothness,
        "transport": {"procrustes_residual_median": float(np.median([x["procrustes_residual"] for x in transport])),
                      "transport_fidelity_median": float(np.median([x["transport_fidelity"] for x in transport])),
                      "projector_overlap_median": float(np.median([x["projector_overlap"] for x in transport])), "pair_count":len(transport)},
        "local_probe_curve": sample_curve,
        "LOCAL_CAUSAL_SYSTEM_IDENTIFICATION_IS_SAMPLE_EFFICIENT": bool(any(v["relative_l2"] <= .30 for k,v in sample_curve.items() if int(k)<=16)),
        "chart_predictability": {"status": "NOT_RUN_ORACLE_GATE_FAILED" if not (finite_pass or jvp_pass) else "ELIGIBLE_PENDING",
                                 "S0": None, "S1": None, "S2": None, "S4": None,
                                 "rule": "Gauge-invariant chart predictors are downstream of oracle sufficiency; raw-P predictor additionally requires oracle pass."},
        "oracle_vs_learned": {"oracle_best_direction_l2": oracle_direction_l2, "V22_learned_chart_l2": learned_l2,
                              "oracle_minus_learned_l2": float(oracle_direction_l2-learned_l2)},
        "within_chart": {model: best["finite"]["categories"]["unseen_direction"] for model in []},
        "historical_final_responses_opened": False, "v23_independent_final_responses_opened": False,
    }
    # Explicit linear/quadratic/cubic comparison at the best finite k.
    bk = int(best["finite"]["token"].split(":")[0][1:])
    for model in ("M1_local_linear","M2_ridge","M3_quadratic","M4_cubic_nonlinear"):
        token=f"k{bk}:{model}"
        if token in results["finite"]:
            result["within_chart"][model]=results["finite"][token]["unseen_direction"]
    # Save machine-readable chart summary arrays without raw responses.
    keys=sorted(chart_records["finite"])
    projectors=[]
    for key in keys:
        v=chart_records["finite"][key]["v"][:,:best_k]; projectors.append((v@v.T).astype(np.float32))
    npz=root/SCRATCH/"oracle_chart_summary_v23.npz"
    np.savez_compressed(npz, state_ids=np.asarray(keys), finite_projectors=np.stack(projectors), best_k=np.asarray(best_k))
    result["chart_summary_npz_sha256"]=sha256_file(npz)
    write_json_atomic(root / OUT / "oracle_local_action_charts_v23.json", result)
    # Compact long-form tables.
    rows=[]
    for kind in results:
        for token,categories in results[kind].items():
            for category,metric in categories.items(): rows.append({"chart":kind,"candidate":token,"category":category,**{k:v for k,v in metric.items() if not isinstance(v,dict)}})
    table=root/OUT/"oracle_chart_metrics_v23.parquet"; pd.DataFrame(rows).to_parquet(table,index=False,compression="zstd")
    result["metrics_parquet_sha256"]=sha256_file(table); write_json_atomic(root / OUT / "oracle_local_action_charts_v23.json", result)
    return result


def adjudicate(root: Path) -> dict:
    rank=json.loads((root/OUT/"input_rank_scaling_v23.json").read_text())
    oracle_result=json.loads((root/OUT/"oracle_local_action_charts_v23.json").read_text())
    oracle_pass=bool(oracle_result["ORACLE_LOCAL_CHART_PASS"])
    learned_pass=False
    finite_diff=bool(oracle_result["jvp_finite_comparison"]["finite_minus_jvp_best_direction_l2"] < -0.10 and oracle_result["jvp_finite_comparison"]["median_input_overlap"] < .8)
    outcomes=[]
    if rank["INPUT_RANK_SATURATED"]: outcomes.append("V23-D_INPUT_RANK_SATURATED")
    else: outcomes.append("V23-E_INPUT_RANK_NOT_SATURATED")
    if oracle_pass: outcomes.append("V23-A_ORACLE_LOCAL_ACTION_CHART_VALIDATED")
    else: outcomes.append("V23-F_ORACLE_CHART_INSUFFICIENT")
    if finite_diff: outcomes.append("V23-G_FINITE_CHART_DIFFERS_FROM_JVP_CHART")
    result={"formal_outcomes":outcomes,"primary_outcome":"+".join(outcomes),
            "ORACLE_LOCAL_CHART_PASS":oracle_pass,"LEARNED_LOCAL_CHART_PASS":learned_pass,
            "CHART_PREDICTION_BOTTLENECK":bool(oracle_pass and not learned_pass),
            "LOCAL_CHART_SUFFICIENCY":oracle_pass,"INPUT_RANK_SATURATED":rank["INPUT_RANK_SATURATED"],
            "FINITE_CHART_DIFFERS_FROM_JVP_CHART":finite_diff,
            "COMPACT_OPERATOR_SEARCH_REOPENED":bool(oracle_pass or learned_pass),
            "RAW_TO_OPERATOR_ENCODER_AUTHORIZED":False,"H2_REMAINS":True,"H3_AUTHORIZED":False,
            "DYNAMIC_STATE_SEARCH_AUTHORIZED":False,"historical_final_responses_opened":False,
            "v23_independent_final_responses_opened":False,
            "independent_final_status":"FROZEN_AND_UNOPENED_NO_DEVELOPMENT_VALIDATION_FINALIST",
            "negative_result_wording":"The tested low-rank subspace chart is insufficient for held-out finite-action prediction; this does not imply that local charts do not exist. Input-sensitive causal rank remains probe-limited if saturation fails."}
    write_json_atomic(root/OUT/"v23_adjudication.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("rank","oracle","adjudicate")); args=parser.parse_args()
    answer={"rank":rank_scaling,"oracle":oracle,"adjudicate":adjudicate}[args.stage](Path.cwd())
    print(json.dumps(answer,indent=2))
