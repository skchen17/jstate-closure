"""Independent V19-B replication on V13 final-test prompts disjoint from V18/V19 roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import gpu1_v19, operator_bank_v20 as bank
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.protocol_v19 import verify as verify_v19, verify_stage as verify_v19_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/independent_confirm_v20.py"
DIR = bank.SCRATCH / "independent_v19_confirmation"
ROWS = bank.OUT / "independent_v19_confirmation_v20.parquet"
METRICS = bank.OUT / "independent_v19_confirmation_metrics_v20.parquet"
SUMMARY = bank.OUT / "independent_v19_confirmation_v20.json"


def prepare(root: Path) -> dict:
    split = verify_stage(root, "splits")
    teacher = verify_stage(root, "teacher")
    q = verify_v19_stage(root, "q_amendment_2")
    actions = verify_v19_stage(root, "splits")
    cfg = verify(root)["config"]["V19_confirmation"]
    q_names = [x["name"] for x in q["q"]]
    if q_names != cfg["q_names"]:
        raise RuntimeError("V20 independent q order differs from immutable V19 q bank")
    action_rows = [x for x in actions["actions"] if x["coordinate_index"] in cfg["action_coordinates"]]
    if [x["coordinate_index"] for x in action_rows] != cfg["action_coordinates"]:
        raise RuntimeError("V20 independent action order differs from V19")
    return stage_freeze(root, "independent_confirmation_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_teacher.freeze.json",
                         "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "artifacts/counterfactual_workspace_v19_splits.freeze.json"],
                        {"state_count": len(split["independent_v19_confirmation"]),
                         "state_id_sha256": split["role_id_sha256"]["independent_v19_confirmation"],
                         "teacher_sha256": teacher["teacher_sha256"],
                         "q_names": q_names, "active_q_names": cfg["active_q_names"],
                         "q_freeze_digest": q["freeze_digest"], "action_rows": action_rows,
                         "action_signs": cfg["action_signs"], "action_alpha": cfg["action_alpha"],
                         "horizon": cfg["horizon"], "bootstrap_replicates": cfg["bootstrap_replicates"],
                         "bootstrap_seed": cfg["bootstrap_seed"],
                         "dependence_median_min": cfg["dependence_median_min"],
                         "dependence_ci_lower_min": cfg["dependence_ci_lower_min"],
                         "natural_to_action_ratio_min": cfg["natural_to_action_ratio_min"],
                         "matched_action_rate_min": cfg["matched_action_rate_min"],
                         "response_rows_already_observed": 0,
                         "representation_learning_already_performed": False})


def record_path(base_id: str) -> Path:
    return DIR / f"factorial_{base_id}.parquet"


def run(root: Path, shard: int) -> dict:
    design = verify_stage(root, "independent_confirmation_design")
    split = verify_stage(root, "splits")
    teacher = verify_stage(root, "teacher")
    cfg = verify(root)["config"]
    old_q = verify_v19_stage(root, "q_amendment_2")
    old_interventions = verify_v19_stage(root, "interventions")
    old_q_threshold = verify_v19(root)["config"]["q"]
    acfg = old_interventions["action_matching_threshold"]
    if design["q_freeze_digest"] != old_q["freeze_digest"]:
        raise RuntimeError("V20 independent q freeze changed")
    prompts = bank.prompt_index(root)
    teacher_frame = pd.read_parquet(root / teacher["teacher_path"])
    teacher_map = {str(x.base_trial_id): [int(v) for v in x.teacher_tokens] for x in teacher_frame.itertuples()}
    if shard == 1:
        v19.load_context_v18 = gpu1_v19.load_context_gpu1
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    items = [x for i, x in enumerate(split["independent_v19_confirmation"]) if i % 2 == shard]
    DIR.mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, item in enumerate(items, 1):
        path = record_path(item["base_trial_id"])
        if path.exists():
            completed += 1
            continue
        prompt = str(prompts[item["base_trial_id"]]["prompt"])
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        token = teacher_map[item["base_trial_id"]]
        if len(token) != 1:
            raise RuntimeError("V20 independent teacher length mismatch")
        boundary_j = np.asarray(clean["history_j"][-1], dtype=np.float16)
        jhash = hashlib.sha256(boundary_j.tobytes()).hexdigest()
        p0 = clone_hybrid_cache(clean["cache"])
        p0hash = v19._snapshot_hash(p0, rec, att)
        _, y00 = v19._trajectory(bundle, dense, p0, None, token, 1, clean["prompt_length"],
                                 jids, lids, ws_layers, ws_count, max(measured))
        actions = []
        for action in design["action_rows"]:
            for sign in design["action_signs"]:
                arow = v19._action_row(values["directions"], int(action["direction_index"]),
                                       float(design["action_alpha"]), int(sign))
                p0a = v19.apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                _, y01 = v19._trajectory(bundle, dense, p0a, None, token, 1, clean["prompt_length"],
                                         jids, lids, ws_layers, ws_count, max(measured))
                actions.append((action, sign, arow, p0a, y01))
        records = []
        for q in old_q["q"]:
            qrow = v19._q_row(values["directions"], q, float(q["alpha"]))
            pq = v19.apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
            qreal = _readback(p0, pq, qrow, rec, att)
            pqhash = v19._snapshot_hash(pq, rec, att)
            _, y10 = v19._trajectory(bundle, dense, pq, None, token, 1, clean["prompt_length"],
                                     jids, lids, ws_layers, ws_count, max(measured))
            qok = v19._reliable(qreal, old_q_threshold)
            for action, sign, arow, p0a, y01 in actions:
                pqa = v19.apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                match = v19._matching(p0, p0a, pq, pqa, arow, rec, att)
                ratio = match["realized_norm_ratio_pq_over_p0"]
                # The exact matching rule is inherited from the immutable V19 config.
                matched = (match["realized_action_pair_cosine"] is not None
                           and match["realized_action_pair_cosine"] >= acfg["matched_realized_cosine_min"]
                           and ratio is not None and acfg["matched_norm_ratio_min"] <= ratio <= acfg["matched_norm_ratio_max"]
                           and match["realized_cosine_p0"] is not None and match["realized_cosine_pq"] is not None
                           and match["realized_cosine_p0"] >= acfg["reliability_min_cosine"]
                           and match["realized_cosine_pq"] >= acfg["reliability_min_cosine"]
                           and acfg["reliability_gain_min"] <= match["realized_gain_p0"] <= acfg["reliability_gain_max"]
                           and acfg["reliability_gain_min"] <= match["realized_gain_pq"] <= acfg["reliability_gain_max"])
                _, y11 = v19._trajectory(bundle, dense, pqa, None, token, 1, clean["prompt_length"],
                                         jids, lids, ws_layers, ws_count, max(measured))
                endpoints = [y00[1], y01[1], y10[1], y11[1]]
                stacked = [v19.stack(x, scales).astype(np.float32) for x in endpoints]
                records.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                                "role": "independent_v19_confirmation", "q_name": q["name"],
                                "q_channel": q["channel"], "q_alpha": float(q["alpha"]),
                                "q_reliable": bool(qok), "q_realized_state_norm": qreal["realized_state_norm"],
                                "q_realized_cosine": qreal["realized_state_cosine"],
                                "q_gain": qreal["realized_state_gain"],
                                "q_channel_survival": json.dumps(qreal["channel_survival"]),
                                "boundary_j_hash_p0": jhash, "boundary_j_hash_pq": jhash,
                                "boundary_j_difference": 0.0,
                                "p0_snapshot_hash": p0hash, "pq_snapshot_hash": pqhash,
                                "coordinate_index": int(action["coordinate_index"]),
                                "action_sign": int(sign), "action_alpha": float(design["action_alpha"]),
                                "action_status": "MATCHED_REALIZED_ACTION" if matched else "ACTION_REALIZATION_MISMATCH",
                                **{key: value for key, value in match.items() if key != "channel_joint_survival"},
                                "action_channel_joint_survival": json.dumps(match["channel_joint_survival"]),
                                "y00_stack": stacked[0].tolist(), "y01_stack": stacked[1].tolist(),
                                "y10_stack": stacked[2].tolist(), "y11_stack": stacked[3].tolist(),
                                "teacher_token_sha256": hashlib.sha256(np.asarray(token, dtype=np.int32).tobytes()).hexdigest(),
                                "design_freeze_digest": design["freeze_digest"],
                                "split_freeze_digest": split["freeze_digest"]})
        temp = path.with_suffix(".tmp.parquet")
        pd.DataFrame(records).to_parquet(temp, index=False, compression="zstd")
        os.replace(temp, path)
        completed += 1
        if number % 10 == 0 or number == len(items):
            print(f"V20 independent shard{shard} {number}/{len(items)}", flush=True)
    return {"shard": shard, "completed": completed, "selected": len(items)}


def aggregate(root: Path) -> dict:
    design = verify_stage(root, "independent_confirmation_design")
    split = verify_stage(root, "splits")
    paths = [record_path(x["base_trial_id"]) for x in split["independent_v19_confirmation"]]
    if not all(path.exists() for path in paths):
        raise RuntimeError("V20 independent confirmation incomplete")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = len(paths) * len(design["q_names"]) * len(design["action_rows"]) * len(design["action_signs"])
    if len(frame) != expected or frame.base_trial_id.nunique() != len(paths):
        raise RuntimeError("V20 independent confirmation row count mismatch")
    output = root / ROWS
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return {"states": len(paths), "rows": len(frame), "path": str(ROWS), "sha256": sha256_file(output)}


def _state_bootstrap(frame: pd.DataFrame, name: str, seed: int, n: int) -> dict:
    groups = {key: group[name].to_numpy(dtype=float) for key, group in frame.groupby("base_trial_id")}
    ids = sorted(groups)
    values = [groups[key] for key in ids]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        selected = rng.integers(0, len(ids), size=len(ids))
        draws.append(float(np.median(np.concatenate([values[i] for i in selected]))))
    return {"median": float(frame[name].median()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "states": len(ids), "rows": len(frame)}


def analyze(root: Path) -> dict:
    design = verify_stage(root, "independent_confirmation_design")
    frame = pd.read_parquet(root / ROWS)
    y = {name: np.stack(frame[name]).astype(np.float64) for name in
         ("y00_stack", "y01_stack", "y10_stack", "y11_stack")}
    natural = y["y10_stack"] - y["y00_stack"]
    r0 = y["y01_stack"] - y["y00_stack"]
    rq = y["y11_stack"] - y["y10_stack"]
    modulation = rq - r0
    norm = lambda x: np.linalg.norm(x, axis=1)
    n0, nr0, nrq, nm = map(norm, (natural, r0, rq, modulation))
    metrics = frame[["base_trial_id", "family", "q_name", "q_channel", "q_reliable",
                     "coordinate_index", "action_sign", "action_status"]].copy()
    metrics["natural_norm"] = n0
    metrics["clean_action_norm"] = nr0
    metrics["counterfactual_action_norm"] = nrq
    metrics["modulation_norm"] = nm
    metrics["modulation_ratio"] = nm / np.maximum(np.maximum(nr0, nrq), 1e-8)
    metrics["natural_to_action_ratio"] = n0 / np.maximum(nr0, 1e-8)
    metrics["response_cosine"] = np.sum(r0 * rq, axis=1) / np.maximum(nr0 * nrq, 1e-8)
    metrics["response_magnitude_ratio"] = nrq / np.maximum(nr0, 1e-8)
    metrics.to_parquet(root / METRICS, index=False, compression="zstd")
    active = metrics[metrics.q_name.isin(design["active_q_names"]) & metrics.q_reliable &
                     (metrics.action_status == "MATCHED_REALIZED_ACTION")]
    modulation_ci = _state_bootstrap(active, "modulation_ratio", int(design["bootstrap_seed"]),
                                     int(design["bootstrap_replicates"]))
    natural_ci = _state_bootstrap(active, "natural_to_action_ratio", int(design["bootstrap_seed"]) + 1,
                                  int(design["bootstrap_replicates"]))
    family = active.groupby("family").modulation_ratio.median().to_dict()
    action = active.groupby("coordinate_index").modulation_ratio.median().to_dict()
    matched_rate = float((metrics.action_status == "MATCHED_REALIZED_ACTION").mean())
    confirmed = (modulation_ci["median"] >= design["dependence_median_min"]
                 and modulation_ci["ci95"][0] > design["dependence_ci_lower_min"]
                 and natural_ci["median"] >= design["natural_to_action_ratio_min"]
                 and matched_rate >= design["matched_action_rate_min"]
                 and len(family) == 5 and min(family.values()) >= design["dependence_median_min"]
                 and len(action) >= 2 and min(action.values()) >= design["dependence_median_min"]
                 and (frame.boundary_j_difference == 0).all()
                 and (frame.boundary_j_hash_p0 == frame.boundary_j_hash_pq).all()
                 and (frame.p0_snapshot_hash != frame.pq_snapshot_hash).all())
    result = {"design_freeze_digest": design["freeze_digest"], "independent_states": int(frame.base_trial_id.nunique()),
              "all_rows": len(frame), "active_matched_rows": len(active),
              "boundary_j_exact_all": bool((frame.boundary_j_difference == 0).all()),
              "persistent_snapshots_distinct_all": bool((frame.p0_snapshot_hash != frame.pq_snapshot_hash).all()),
              "q_reliable_rate": float(metrics.q_reliable.mean()), "matched_action_rate": matched_rate,
              "modulation_ratio": modulation_ci, "natural_to_action_ratio": natural_ci,
              "natural_norm_median": float(active.natural_norm.median()),
              "clean_action_norm_median": float(active.clean_action_norm.median()),
              "counterfactual_action_norm_median": float(active.counterfactual_action_norm.median()),
              "modulation_norm_median": float(active.modulation_norm.median()),
              "response_cosine_median": float(active.response_cosine.median()),
              "response_magnitude_ratio_median": float(active.response_magnitude_ratio.median()),
              "family_modulation_median": family,
              "action_modulation_median": {str(k): v for k, v in action.items()},
              "V19_B_independently_confirmed": bool(confirmed),
              "outcome": "V19_B_INDEPENDENTLY_CONFIRMED" if confirmed else "V19_B_NOT_CONFIRMED_ON_INDEPENDENT_BANK",
              "metrics_path": str(METRICS), "metrics_sha256": sha256_file(root / METRICS),
              "raw_path": str(ROWS), "raw_sha256": sha256_file(root / ROWS)}
    write_json_atomic(root / SUMMARY, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "aggregate", "analyze"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    root = Path.cwd()
    result = {"prepare": prepare, "aggregate": aggregate, "analyze": analyze}.get(args.stage)
    value = run(root, args.shard) if args.stage == "run" else result(root)
    if args.stage == "prepare":
        value = {"freeze_digest": value["freeze_digest"], "state_count": value["state_count"]}
    print(json.dumps(value, indent=2))
