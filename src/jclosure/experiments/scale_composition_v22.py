"""Frozen V22 scale-law and unseen pair/dense finite-response panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.operator_v15 import stack
from jclosure.protocol_v22 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/scale_composition_v22.py"
OUT = Path("results/v22/processed")
RAW = Path("/data/CSK/J-space-project/v22-action-manifold-work/scale_composition")
EXPANDED = Path("/data/CSK/J-space-project/v22-action-manifold-work/expanded_response_bank")


def _hash_order(values: list[str], seed: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def prepare(root: Path) -> dict:
    verify_stage(root, "expanded_bank_design")
    expanded = json.loads((root / OUT / "expanded_bank_design_v22.json").read_text())
    selection = json.loads((root / OUT / "action_selection_v22.json").read_text())
    calibration = json.loads((root / OUT / "action_pool_calibration_v22.json").read_text())
    reliable_all = []
    for item in calibration["by_action"]:
        detail = item["candidate_reliability"]
        if all(min(detail[str(alpha)].values()) >= 0.8 for alpha in (0.25, 0.5, 1.0)):
            reliable_all.append(item["action_id"])
    common = [value for value in selection["common_measured_train_action_ids"] if value in reliable_all]
    validation = [value for value in selection["validation_action_ids"] if value in reliable_all]
    scale_actions = _hash_order(common, "V22-SCALE-TRAIN")[:12] + _hash_order(validation, "V22-SCALE-VALIDATION")[:8]
    if len(scale_actions) != 20:
        raise RuntimeError("V22 all-amplitude reliable scale panel too small")
    component_ids = _hash_order(common, "V22-COMPOSITION-COMPONENTS")[:32]
    compositions = []
    for index in range(8):
        compositions.append({"composition_id": f"pair_{index:02d}", "kind": "pair",
                             "components": [[component_ids[2 * index], 1.0],
                                            [component_ids[2 * index + 1], -1.0 if index % 2 else 1.0]]})
    for index in range(8):
        ids = component_ids[16 + 2 * index:18 + 2 * index] + component_ids[2 * index:2 * index + 2]
        weights = [0.5, -0.5 if index % 2 else 0.5, 0.5, -0.5]
        compositions.append({"composition_id": f"dense_{index:02d}", "kind": "dense",
                             "components": [[value, weight] for value, weight in zip(ids, weights)]})
    states = {}
    for role in ("development", "validation"):
        source = expanded[f"{role}_states"]
        states[role] = [next(item for item in source if item["family"] == family)
                        for family in sorted({item["family"] for item in source})]
    payload = {"scale_action_ids": scale_actions, "scale_train_count": 12, "scale_validation_count": 8,
               "amplitudes": [0.25, 0.5, 1.0], "signs": [-1, 1], "compositions": compositions,
               "development_states": states["development"], "validation_states": states["validation"],
               "responses_observed_before_freeze": 0, "historical_final_six_opened": False,
               "new_independent_final_opened": False}
    target = root / OUT / "scale_composition_design_v22.json"
    write_json_atomic(target, payload)
    frozen = stage_freeze(root, "scale_composition_design", [SOURCE,
        "artifacts/causal_action_manifold_v22_expanded_bank_design.freeze.json",
        str(OUT / "action_selection_v22.json"), str(OUT / "action_pool_calibration_v22.json"), str(target)], payload)
    return {"freeze_digest": frozen["freeze_digest"], "scale_actions": len(scale_actions),
            "compositions": len(compositions)}


def _evaluate(bundle, dense, cache, token, prompt_length, jids, lids, ws_layers, ws_count, main, scales):
    _, endpoints = v19._trajectory(bundle, dense, cache, None, [token], 1, prompt_length,
                                   jids, lids, ws_layers, ws_count, main)
    return stack(endpoints[1], scales).astype(np.float32)


def _combine(directions: dict, specs: dict, selection: dict, components: list[list], sign: int = 1):
    rows = [(actions._row(directions, specs[action_id], selection["action_alphas"][action_id], 1), float(weight))
            for action_id, weight in components]
    return {channel: sum(weight * row[channel] for row, weight in rows) * int(sign)
            for channel in ("recurrent", "conv", "kv")}


def run(root: Path, role: str) -> dict:
    verify_stage(root, "scale_composition_design")
    design = json.loads((root / OUT / "scale_composition_design_v22.json").read_text())
    selection = json.loads((root / OUT / "action_selection_v22.json").read_text())
    pool = json.loads((root / OUT / "action_pool_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    split = json.loads((root / "artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts, teachers = v20_bank.prompt_index(root), v20_response._teacher_map(root)
    q_lookup = {item["name"]: item for item in operator["q"]}
    jids, lids = np.asarray(split["selected_j"], dtype=int), np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    directions = values["directions"]
    completed = 0
    for number, item in enumerate(design[f"{role}_states"], 1):
        target = root / RAW / role / f"panel_{item['base_trial_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            completed += 1; continue
        token = int(teachers[item["base_trial_id"]][0])
        clean = v19._prefill_history(bundle, str(prompts[item["base_trial_id"]]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        q = q_lookup[item["q_name"]]
        pq = v19.apply(p0, 1.0, v19._q_row(directions, q, float(q["alpha"])), rec, att,
                       "native_fp32_add_bf16_writeback")
        arrays = {}
        for state_name, state in (("P0", p0), ("Pq", pq)):
            baseline = _evaluate(bundle, dense, state, token, clean["prompt_length"], jids, lids,
                                 ws_layers, ws_count, max(measured), scales)
            scale_plus, scale_minus = [], []
            for action_id in design["scale_action_ids"]:
                positive_by_alpha, negative_by_alpha = [], []
                for alpha in design["amplitudes"]:
                    row = actions._row(directions, specs[action_id], float(alpha), 1)
                    positive = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    negative = v19.apply(state, -1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    positive_by_alpha.append(_evaluate(bundle, dense, positive, token, clean["prompt_length"],
                                                       jids, lids, ws_layers, ws_count, max(measured), scales) - baseline)
                    negative_by_alpha.append(_evaluate(bundle, dense, negative, token, clean["prompt_length"],
                                                       jids, lids, ws_layers, ws_count, max(measured), scales) - baseline)
                scale_plus.append(np.stack(positive_by_alpha)); scale_minus.append(np.stack(negative_by_alpha))
            composition_plus, composition_minus = [], []
            for composition in design["compositions"]:
                for sign, sink in ((1, composition_plus), (-1, composition_minus)):
                    row = _combine(directions, specs, selection, composition["components"], sign)
                    edited = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    sink.append(_evaluate(bundle, dense, edited, token, clean["prompt_length"], jids, lids,
                                          ws_layers, ws_count, max(measured), scales) - baseline)
            arrays[f"{state_name}_scale_plus"] = np.stack(scale_plus).astype(np.float32)
            arrays[f"{state_name}_scale_minus"] = np.stack(scale_minus).astype(np.float32)
            arrays[f"{state_name}_composition_plus"] = np.stack(composition_plus).astype(np.float32)
            arrays[f"{state_name}_composition_minus"] = np.stack(composition_minus).astype(np.float32)
            print(f"V22 scale/composition {role} base={number}/5 state={state_name}", flush=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays, base_trial_id=np.asarray(item["base_trial_id"]),
                            family=np.asarray(item["family"]), q_name=np.asarray(item["q_name"]))
        os.replace(temporary, target)
        completed += 1
    return {"role": role, "completed": completed}


def _relative(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(truth - prediction) / max(np.linalg.norm(truth), 1e-12))


def analyze(root: Path) -> dict:
    design = json.loads((root / OUT / "scale_composition_design_v22.json").read_text())
    selection = json.loads((root / OUT / "action_selection_v22.json").read_text())
    expanded_design = json.loads((root / OUT / "expanded_bank_design_v22.json").read_text())
    panels = {}
    for role in ("development", "validation"):
        values = {key: [] for key in ("scale_plus", "scale_minus", "composition_plus", "composition_minus")}
        for item in design[f"{role}_states"]:
            with np.load(root / RAW / role / f"panel_{item['base_trial_id']}.npz") as source:
                for state in ("P0", "Pq"):
                    for key in values:
                        values[key].append(np.asarray(source[f"{state}_{key}"], dtype=np.float64))
        panels[role] = {key: np.stack(value) for key, value in values.items()}
    amplitudes = np.asarray(design["amplitudes"], dtype=np.float64)
    scale_results = {}
    for sign_name in ("plus", "minus"):
        development = panels["development"][f"scale_{sign_name}"]
        validation = panels["validation"][f"scale_{sign_name}"]
        reference_dev, reference_val = development[:, :, -1, :], validation[:, :, -1, :]
        nonunit = slice(0, 2)
        scale_results[sign_name] = {}
        for name, powers in (("linear_alpha", [1]), ("alpha_plus_alpha2", [1, 2]),
                             ("alpha_plus_alpha2_plus_alpha3", [1, 2, 3])):
            columns, target = [], development[:, :, nonunit, :].reshape(-1)
            base = np.broadcast_to(reference_dev[:, :, None, :], development[:, :, nonunit, :].shape)
            for power in powers:
                multiplier = amplitudes[nonunit] ** power
                columns.append((base * multiplier[None, None, :, None]).reshape(-1))
            matrix = np.stack(columns, axis=1)
            coefficients = np.linalg.solve(matrix.T @ matrix + 1e-6 * np.eye(len(powers)), matrix.T @ target)
            val_base = np.broadcast_to(reference_val[:, :, None, :], validation[:, :, nonunit, :].shape)
            prediction = sum(coefficient * val_base * (amplitudes[nonunit] ** power)[None, None, :, None]
                             for coefficient, power in zip(coefficients, powers))
            scale_results[sign_name][name] = {"coefficients": coefficients.tolist(),
                                               "validation_nonunit_relative_l2": _relative(validation[:, :, nonunit, :], prediction)}
        gains = []
        for index in range(3):
            numerator = np.sum(development[:, :, index, :] * reference_dev)
            denominator = max(float(np.sum(reference_dev * reference_dev)), 1e-12)
            gains.append(float(numerator / denominator))
        gains = np.maximum.accumulate(np.maximum(gains, 0.0))
        monotone_prediction = reference_val[:, :, None, :] * gains[None, None, :, None]
        scale_results[sign_name]["monotone_gain_network"] = {"fitted_gains": gains.tolist(),
                                                               "validation_relative_l2": _relative(validation, monotone_prediction)}

    # Match validation panel states to the expanded bank and form additive composition predictions.
    action_positions = {value: index for index, value in enumerate(
        selection["common_measured_train_action_ids"] + selection["validation_action_ids"])}
    singles = []
    for item in design["validation_states"]:
        with np.load(root / EXPANDED / "validation" / f"expanded_{item['base_trial_id']}.npz") as source:
            for state in ("P0", "Pq"):
                singles.append((np.asarray(source[f"{state}_plus"], dtype=np.float64),
                                np.asarray(source[f"{state}_minus"], dtype=np.float64)))
    composition_results = {}
    for sign_name, sign_multiplier in (("plus", 1.0), ("minus", -1.0)):
        truth = panels["validation"][f"composition_{sign_name}"]
        prediction = []
        for state_index, (positive, negative) in enumerate(singles):
            state_rows = []
            for composition in design["compositions"]:
                row = np.zeros(288, dtype=np.float64)
                for action_id, weight in composition["components"]:
                    signed = float(weight) * sign_multiplier
                    source = positive[action_positions[action_id]] if signed >= 0 else negative[action_positions[action_id]]
                    row += abs(signed) * source
                state_rows.append(row)
            prediction.append(np.stack(state_rows))
        prediction = np.stack(prediction)
        composition_results[sign_name] = {}
        for kind in ("pair", "dense"):
            mask = np.asarray([item["kind"] == kind for item in design["compositions"]])
            composition_results[sign_name][kind] = {"additive_superposition_relative_l2":
                _relative(truth[:, mask], prediction[:, mask])}
    scale_payload = {"amplitudes": design["amplitudes"], "results": scale_results,
                     "interpretation": "Scale-only law holds the measured unit response fixed; it does not use a new-action response at deployment."}
    composition_payload = {"results": composition_results, "composition_count": len(design["compositions"]),
                           "historical_final_six_opened": False, "new_independent_final_opened": False}
    write_json_atomic(root / OUT / "action_scale_law_v22.json", scale_payload)
    write_json_atomic(root / OUT / "action_composition_v22.json", composition_payload)
    return {"scale": scale_payload, "composition": composition_payload}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--role", choices=("development", "validation"))
    args = parser.parse_args()
    if args.stage == "prepare": answer = prepare(Path.cwd())
    elif args.stage == "run": answer = run(Path.cwd(), args.role)
    else: answer = analyze(Path.cwd())
    print(json.dumps(answer, indent=2))
