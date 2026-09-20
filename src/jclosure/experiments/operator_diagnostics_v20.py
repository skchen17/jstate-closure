"""Frozen unseen-scale, composition and dense-action oracle diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import gpu1_v19, operator_bank_v20 as bank
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments import operator_model_v20 as model
from jclosure.experiments.response_operator_v20 import _action_ok, _matching_ok, _teacher_map
from jclosure.protocol_v20 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/operator_diagnostics_v20.py"
SCRATCH = bank.SCRATCH / "operator_diagnostics"
OUTPUT = bank.OUT / "operator_action_diagnostics_v20.parquet"
SUMMARY = bank.OUT / "operator_action_diagnostics_v20.json"


def _specs() -> list[dict]:
    return [
        {"name": "unseen_scale_train_0", "kind": "unseen_amplitude_train", "components": [[0, 2.0]]},
        {"name": "unseen_scale_validation_20", "kind": "unseen_amplitude_validation", "components": [[20, 2.0]]},
        {"name": "unseen_pair_20_16", "kind": "unseen_pair", "components": [[20, 1.0], [16, 1.0]]},
        {"name": "unseen_pair_21_17", "kind": "unseen_pair", "components": [[21, 1.0], [17, 1.0]]},
        {"name": "unseen_dense_0_1_2_20_21", "kind": "unseen_dense", "components": [[0, 0.5], [1, 0.5], [2, 0.5], [20, 0.5], [21, 0.5]]},
    ]


def prepare(root: Path) -> dict:
    split = verify_stage(root, "splits")
    operator = verify_stage(root, "operator_design")
    actions = verify_stage(root, "actions")
    analysis = verify_stage(root, "operator_analysis")
    by_coord = {int(x["coordinate_index"]): x for x in actions["actions"]}
    train_coords = {int(x["coordinate_index"]) for x in actions["partitions"]["train"]}
    validation_coords = {int(x["coordinate_index"]) for x in actions["partitions"]["validation"]}
    for spec in _specs():
        coords = {int(x[0]) for x in spec["components"]}
        if not coords <= train_coords | validation_coords or coords & {int(x["coordinate_index"]) for x in actions["partitions"]["final_heldout"]}:
            raise RuntimeError("V20 diagnostic action outside opened partitions")
        for coord in coords:
            if coord not in by_coord:
                raise RuntimeError("V20 diagnostic direction missing")
    ids = [x["base_trial_id"] for family in sorted({x["family"] for x in split["operator_validation"]})
           for x in [row for row in split["operator_validation"] if row["family"] == family][:2]]
    if len(ids) != 10:
        raise RuntimeError("V20 diagnostic panel must be two bases per family")
    return stage_freeze(root, "operator_diagnostics_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_operator_analysis.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_actions.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_splits.freeze.json"],
                        {"base_ids": ids, "base_id_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                         "operator_states": 4 * len(ids), "specs": _specs(),
                         "raw_action_construction": "linear_sum_of_frozen_V13_full_REC_Conv_KV_directions_at_frozen_base_alpha_and_spec_multiplier",
                         "response_target": "V16_normalized_stack_288",
                         "model_prediction_action_descriptor": "same_linear_sum_of_frozen_train_normalized_continuous_action_descriptors",
                         "response_rows_already_observed": 0, "final_heldout_directions_opened": False,
                         "independent_V20_final_opened": False})


def _action(values: dict, by_coord: dict, spec: dict) -> dict:
    rows = [(v19._action_row(values["directions"], int(by_coord[coord]["direction_index"]),
                             float(by_coord[coord]["base_alpha"]), 1), multiplier)
            for coord, multiplier in spec["components"]]
    return {key: sum((row[key] * multiplier for row, multiplier in rows)) for key in ("recurrent", "conv", "kv")}


def run(root: Path, shard: int) -> dict:
    diagnostic = verify_stage(root, "operator_diagnostics_design")
    split = verify_stage(root, "splits")
    operator = verify_stage(root, "operator_design")
    actions = verify_stage(root, "actions")
    by_coord = {int(x["coordinate_index"]): x for x in actions["actions"]}
    teacher = _teacher_map(root)
    prompts = bank.prompt_index(root)
    if shard == 1:
        v19.load_context_v18 = gpu1_v19.load_context_gpu1
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    items = {x["base_trial_id"]: x for x in split["operator_validation"]}
    selected = [items[key] for i, key in enumerate(diagnostic["base_ids"]) if i % 2 == shard]
    (SCRATCH).mkdir(parents=True, exist_ok=True)
    done = 0
    for item in selected:
        base_id = item["base_trial_id"]
        path = SCRATCH / f"diagnostic_{base_id}.parquet"
        if path.exists():
            done += 1
            continue
        prompt = str(prompts[base_id]["prompt"])
        token = teacher[base_id]
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        contexts = [("P0", p0, True)]
        for q in operator["q"]:
            qrow = v19._q_row(values["directions"], q, float(q["alpha"]))
            pq = v19.apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
            from jclosure.experiments.action_bank_v16 import _readback as readback
            contexts.append((q["name"], pq, v19._reliable(readback(p0, pq, qrow, rec, att), operator["V19_q_reliability"])))
        baselines = {}
        for qname, state, _ in contexts:
            _, endpoint = v19._trajectory(bundle, dense, state, None, token, 1, clean["prompt_length"],
                                          jids, lids, ws_layers, ws_count, max(measured))
            baselines[qname] = v19.stack(endpoint[1], scales).astype(np.float32)
        rows = []
        for spec in diagnostic["specs"]:
            action = _action(values, by_coord, spec)
            p0a = v19.apply(p0, 1.0, action, rec, att, "native_fp32_add_bf16_writeback")
            for qname, state, qok in contexts:
                edited = p0a if qname == "P0" else v19.apply(state, 1.0, action, rec, att, "native_fp32_add_bf16_writeback")
                actual = _readback(state, edited, action, rec, att)
                match = None if qname == "P0" else v19._matching(p0, p0a, state, edited, action, rec, att)
                matched = True if match is None else _matching_ok(match, operator["V19_action_matching"])
                _, endpoint = v19._trajectory(bundle, dense, edited, None, token, 1, clean["prompt_length"],
                                              jids, lids, ws_layers, ws_count, max(measured))
                response = v19.stack(endpoint[1], scales).astype(np.float32) - baselines[qname]
                rows.append({"base_trial_id": base_id, "operator_state_id": f"{base_id}::{qname}",
                             "family": item["family"], "q_name": qname, "diagnostic_name": spec["name"],
                             "diagnostic_kind": spec["kind"], "q_reliable": bool(qok),
                             "action_reliable": bool(_action_ok(actual, operator["action_reliability"])),
                             "matched_vs_P0": bool(matched), "requested_norm": actual["requested_state_norm"],
                             "realized_norm": actual["realized_state_norm"],
                             "realized_cosine": actual["realized_state_cosine"],
                             "realized_gain": actual["realized_state_gain"],
                             "channel_survival": json.dumps(actual["channel_survival"]),
                             "response_stack": response.tolist(),
                             "diagnostic_design_digest": diagnostic["freeze_digest"]})
        tmp = path.with_suffix(".tmp.parquet")
        pd.DataFrame(rows).to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
        done += 1
        print(f"V20 diagnostic shard{shard} {done}/{len(selected)}", flush=True)
    return {"shard": shard, "complete_base_states": done}


def aggregate(root: Path) -> dict:
    design = verify_stage(root, "operator_diagnostics_design")
    paths = [root / SCRATCH / f"diagnostic_{base_id}.parquet" for base_id in design["base_ids"]]
    if not all(x.exists() for x in paths):
        raise RuntimeError("V20 diagnostics incomplete")
    frame = pd.concat([pd.read_parquet(x) for x in paths], ignore_index=True)
    expected = len(design["base_ids"]) * 4 * len(design["specs"])
    if len(frame) != expected:
        raise RuntimeError(f"V20 diagnostics expected {expected}, got {len(frame)}")
    frame.to_parquet(root / OUTPUT, index=False, compression="zstd")
    return {"rows": len(frame), "output_sha256": sha256_file(root / OUTPUT),
            "q_reliable_rate": float(frame.q_reliable.mean()),
            "action_reliable_rate": float(frame.action_reliable.mean()),
            "matched_vs_P0_rate": float(frame.matched_vs_P0.mean())}


def analyze(root: Path) -> dict:
    design = verify_stage(root, "operator_diagnostics_design")
    analysis = verify_stage(root, "operator_analysis")
    operator = verify_stage(root, "operator_design")
    outcome = json.loads((root / model.SUMMARY).read_text())
    descriptor = json.loads((root / model.DESCRIPTORS).read_text())
    train = model._load(root, "operator_train", operator)
    validation = model._load(root, "operator_validation", operator)
    pca = model._pca(model._fingerprints(train), model._fingerprints(validation))
    k = int(outcome["selected_k_for_diagnostics"])
    ctrain, cval, scale = model._coordinate(pca, k)
    train_coords = [int(x["coordinate_index"]) for x in operator["shared_measured_actions"][:operator["train_action_count"]]]
    fitted = model.Fitted(outcome["selected_model_for_diagnostics"], k, pca["mean"], pca["basis"][:, :k],
                          scale, model._action_matrix(descriptor, train_coords),
                          train["Y"][("train", 1)], ctrain, analysis)
    frame = pd.read_parquet(root / OUTPUT)
    state_index = {value: i for i, value in enumerate(validation["ids"])}
    spec_results = {}
    predictions = {}
    for spec in design["specs"]:
        z = sum((model._action_matrix(descriptor, [int(coord)])[0] * float(multiplier)
                 for coord, multiplier in spec["components"]))
        sub = frame[frame.diagnostic_name == spec["name"]].sort_values("operator_state_id")
        ids = list(sub.operator_state_id)
        c = cval[[state_index[x] for x in ids]]
        pred = fitted.predict(c, z[None, :])[:, 0, :]
        truth = np.stack([np.asarray(x, dtype=np.float32) for x in sub.response_stack])
        predictions[spec["name"]] = dict(zip(ids, pred))
        metric = model._metric(truth[:, None, :], pred[:, None, :], np.asarray(sub.family), np.asarray(sub.base_trial_id))
        spec_results[spec["name"]] = {"kind": spec["kind"], "components": spec["components"],
                                      "metrics": metric, "q_reliable_rate": float(sub.q_reliable.mean()),
                                      "action_reliable_rate": float(sub.action_reliable.mean()),
                                      "matched_vs_P0_rate": float(sub.matched_vs_P0.mean())}
    interaction = {}
    operator_frame = model._load(root, "operator_validation", operator)
    val_raw = pd.concat([pd.read_parquet(path) for path in sorted((root / bank.OUT).glob("response_operator_operator_validation_*_v20.parquet"))])
    for spec in design["specs"]:
        if spec["kind"] != "unseen_pair":
            continue
        coords = [int(x[0]) for x in spec["components"]]
        rows = frame[frame.diagnostic_name == spec["name"]]
        residual = []
        for row in rows.itertuples():
            child = val_raw[(val_raw.operator_state_id == row.operator_state_id) &
                            (val_raw.action_sign == 1) & val_raw.action_coordinate.isin(coords)]
            if len(child) != 2:
                raise RuntimeError("V20 pair component responses absent")
            additive = np.sum([np.asarray(x, dtype=np.float32) for x in child.response_stack], axis=0)
            actual = np.asarray(row.response_stack, dtype=np.float32)
            residual.append(float(np.linalg.norm(actual - additive) / max(np.linalg.norm(actual), 1e-8)))
        interaction[spec["name"]] = {"relative_additivity_residual_median": float(np.median(residual))}
    threshold = analysis["heldout_gates"]
    overall_pass = all(model._gate(x["metrics"], analysis) and x["action_reliable_rate"] == 1.0
                       and x["matched_vs_P0_rate"] == 1.0 for x in spec_results.values())
    result = {"design_digest": design["freeze_digest"], "selected_model": outcome["selected_model_for_diagnostics"],
              "selected_k": k, "diagnostic_base_states": len(design["base_ids"]),
              "diagnostic_operator_states": 4 * len(design["base_ids"]),
              "by_spec": spec_results, "composition_interaction": interaction,
              "unseen_scale_composition_gate": bool(overall_pass),
              "preliminary_cross_action_gate": outcome["preliminary_operator_compactness_gate"],
              "final_operator_gate": bool(overall_pass and outcome["preliminary_operator_compactness_gate"]),
              "k_operator_min": outcome["preliminary_k_operator_min"] if overall_pass and outcome["preliminary_operator_compactness_gate"] else None,
              "final_heldout_action_directions_opened": False,
              "independent_V20_final_opened": False}
    write_json_atomic(root / SUMMARY, result)
    return {"final_operator_gate": result["final_operator_gate"], "k_operator_min": result["k_operator_min"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "aggregate", "analyze"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    result = {"prepare": prepare, "run": lambda root: run(root, args.shard),
              "aggregate": aggregate, "analyze": analyze}[args.stage](Path.cwd())
    print(json.dumps(result, indent=2))
