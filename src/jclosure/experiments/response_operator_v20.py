"""Shared, crossed local action→response fingerprints on natural and same-J states."""

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
from jclosure.protocol_v19 import verify as verify_v19, verify_stage as verify_v19_stage
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/response_operator_v20.py"
SCRATCH = bank.SCRATCH / "response_operator"


def prepare(root: Path) -> dict:
    cfg = verify(root)["config"]
    split = verify_stage(root, "splits")
    teacher = verify_stage(root, "teacher")
    actions = verify_stage(root, "actions")
    calibration = verify_stage(root, "action_calibration")
    qscale = verify_stage(root, "q_locality_scales")
    q = verify_v19_stage(root, "q_amendment_2")
    confirmation = json.loads((root / "results/v20/processed/independent_v19_confirmation_v20.json").read_text())
    if not confirmation["V19_B_independently_confirmed"]:
        raise RuntimeError("V20 operator representation requires prior independent V19-B confirmation")
    expected = cfg["operator_bank"]["q_names"]
    q_lookup = {x["name"]: x for x in q["q"]}
    selected_q = [q_lookup[name] for name in expected if name in q_lookup]
    if [x["name"] for x in selected_q] != expected:
        raise RuntimeError("V20 operator q order differs from frozen config")
    action_rows = actions["partitions"]["train"] + actions["partitions"]["validation"]
    if len(action_rows) < cfg["action_bank"]["direction_count_minimum"]:
        raise RuntimeError("V20 shared action bank below 16 reliable directions")
    by_action = {(x["coordinate_index"], x["scale_multiplier"]): x for x in calibration["by_action_scale"]}
    insufficient = [x["coordinate_index"] for x in action_rows
                    if by_action[(x["coordinate_index"], 1.0)]["reliable_fraction"] < 0.95]
    if insufficient:
        raise RuntimeError(f"V20 train-only action calibration below 0.95: {insufficient}")
    return stage_freeze(root, "operator_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_teacher.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_actions.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_action_calibration.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_q_locality_scales.freeze.json",
                         "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "results/v20/processed/independent_v19_confirmation_v20.json"],
                        {"operator_train_states": len(split["operator_train"]),
                         "operator_validation_states": len(split["operator_validation"]),
                         "train_id_sha256": split["role_id_sha256"]["operator_train"],
                         "validation_id_sha256": split["role_id_sha256"]["operator_validation"],
                         "q": selected_q, "operator_states_per_base": 1 + len(selected_q),
                         "shared_measured_actions": action_rows,
                         "train_action_count": len(actions["partitions"]["train"]),
                         "validation_action_count": len(actions["partitions"]["validation"]),
                         "sealed_final_action_count": len(actions["partitions"]["final_heldout"]),
                         "action_hashes": actions["action_hashes"],
                         "signs_measured": cfg["action_bank"]["signs_measured"],
                         "decoder_inference_actions": cfg["operator_bank"]["inference_actions"],
                         "horizon": cfg["operator_bank"]["horizon"],
                         "normalization": cfg["operator_bank"]["normalization"],
                         "target_slices": {"j": [0, 128], "logits": [128, 160],
                                           "semantic_continuous": [160, 192], "workspace": [192, 288],
                                           "stacked_normalized": [0, 288]},
                         "action_reliability": actions["reliability_thresholds"],
                         "V19_action_matching": verify_v19(root)["config"]["actions"],
                         "V19_q_reliability": verify_v19(root)["config"]["q"],
                         "frozen_direction_file_sha256": sha256_file(root / "artifacts/causal/v13/probe_directions_v13.pt"),
                         "requested_raw_direction_recording": "frozen_direction_file_sha256+direction_index+sign+alpha; raw tensor not copied to result banks",
                         "teacher_sha256": teacher["teacher_sha256"],
                         "V19_independent_confirmation_result_sha256": sha256_file(root / "results/v20/processed/independent_v19_confirmation_v20.json"),
                         "operator_responses_already_observed": 0,
                         "final_heldout_action_responses_observed": 0})


def record_path(role: str, base_id: str) -> Path:
    return SCRATCH / role / f"operator_{base_id}.parquet"


def state_record_path(role: str, base_id: str) -> Path:
    return SCRATCH / role / f"state_{base_id}.parquet"


def _teacher_map(root: Path) -> dict[str, list[int]]:
    teacher = verify_stage(root, "teacher")
    frame = pd.read_parquet(root / teacher["teacher_path"])
    return {str(row.base_trial_id): [int(x) for x in row.teacher_tokens] for row in frame.itertuples()}


def _action_ok(readback: dict, threshold: dict) -> bool:
    return (readback["realized_state_cosine"] is not None
            and readback["realized_state_cosine"] >= threshold["cosine_min"]
            and readback["realized_state_gain"] is not None
            and threshold["gain_min"] <= readback["realized_state_gain"] <= threshold["gain_max"])


def _matching_ok(match: dict, threshold: dict) -> bool:
    ratio = match["realized_norm_ratio_pq_over_p0"]
    return (match["realized_action_pair_cosine"] is not None
            and match["realized_action_pair_cosine"] >= threshold["matched_realized_cosine_min"]
            and ratio is not None and threshold["matched_norm_ratio_min"] <= ratio <= threshold["matched_norm_ratio_max"]
            and match["realized_cosine_p0"] is not None and match["realized_cosine_pq"] is not None
            and match["realized_cosine_p0"] >= threshold["reliability_min_cosine"]
            and match["realized_cosine_pq"] >= threshold["reliability_min_cosine"]
            and threshold["reliability_gain_min"] <= match["realized_gain_p0"] <= threshold["reliability_gain_max"]
            and threshold["reliability_gain_min"] <= match["realized_gain_pq"] <= threshold["reliability_gain_max"])


def run(root: Path, role: str, shard: int) -> dict:
    design = verify_stage(root, "operator_design")
    split = verify_stage(root, "splits")
    prompts = bank.prompt_index(root)
    teacher = _teacher_map(root)
    if shard == 1:
        v19.load_context_v18 = gpu1_v19.load_context_gpu1
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    items = [x for i, x in enumerate(split[role]) if i % 2 == shard]
    (SCRATCH / role).mkdir(parents=True, exist_ok=True)
    completed = 0
    for number, item in enumerate(items, 1):
        path = record_path(role, item["base_trial_id"])
        state_path = state_record_path(role, item["base_trial_id"])
        if path.exists() and state_path.exists():
            completed += 1
            continue
        prompt = str(prompts[item["base_trial_id"]]["prompt"])
        token = teacher[item["base_trial_id"]]
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        boundary_j = np.asarray(clean["history_j"][-1], dtype=np.float16)
        jhash = hashlib.sha256(boundary_j.tobytes()).hexdigest()
        p0 = clone_hybrid_cache(clean["cache"])
        p0hash = v19._snapshot_hash(p0, rec, att)
        contexts = []
        qspec = {"name": "P0", "channel": "none", "alpha": 0.0}
        contexts.append((qspec, p0, p0hash, True, 0.0, None, None, None))
        for q in design["q"]:
            qrow = v19._q_row(values["directions"], q, float(q["alpha"]))
            pq = v19.apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
            readback = _readback(p0, pq, qrow, rec, att)
            qok = v19._reliable(readback, design["V19_q_reliability"])
            contexts.append((q, pq, v19._snapshot_hash(pq, rec, att), qok,
                             float(readback["realized_state_norm"]), readback["realized_state_cosine"],
                             readback["realized_state_gain"], json.dumps(readback["channel_survival"])))
        baselines = {}
        for q, state, _, _, _, _, _, _ in contexts:
            _, endpoint = v19._trajectory(bundle, dense, state, None, token, 1, clean["prompt_length"],
                                          jids, lids, ws_layers, ws_count, max(measured))
            baselines[q["name"]] = v19.stack(endpoint[1], scales).astype(np.float32)
        rows = []
        for action in design["shared_measured_actions"]:
            partition = "train" if action["coordinate_index"] in {x["coordinate_index"] for x in design["shared_measured_actions"][:design["train_action_count"]]} else "validation"
            for sign in design["signs_measured"]:
                alpha = float(action["base_alpha"])
                arow = v19._action_row(values["directions"], int(action["direction_index"]), alpha, int(sign))
                p0a = v19.apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                for q, state, state_hash, qok, qnorm, qcos, qgain, qsurvival in contexts:
                    edited = p0a if q["name"] == "P0" else v19.apply(state, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                    actread = _readback(state, edited, arow, rec, att)
                    match = None if q["name"] == "P0" else v19._matching(p0, p0a, state, edited, arow, rec, att)
                    matched = True if match is None else _matching_ok(match, design["V19_action_matching"])
                    _, endpoint = v19._trajectory(bundle, dense, edited, None, token, 1, clean["prompt_length"],
                                                  jids, lids, ws_layers, ws_count, max(measured))
                    y = v19.stack(endpoint[1], scales).astype(np.float32)
                    response = y - baselines[q["name"]]
                    rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"], "role": role,
                                 "operator_state_id": f"{item['base_trial_id']}::{q['name']}",
                                 "state_distribution": "natural" if q["name"] == "P0" else "boundary_held_J_counterfactual",
                                 "q_name": q["name"], "q_channel": q["channel"], "q_alpha": float(q["alpha"]),
                                 "q_reliable": bool(qok), "q_realized_state_norm": qnorm,
                                 "q_realized_cosine": qcos, "q_gain": qgain, "q_channel_survival": qsurvival,
                                 "boundary_j_hash": jhash, "boundary_j_difference_from_P0": 0.0,
                                 "p0_snapshot_hash": p0hash, "persistent_snapshot_hash": state_hash,
                                 "action_coordinate": int(action["coordinate_index"]),
                                 "action_direction_index": int(action["direction_index"]),
                                 "action_proposal_family": action["proposal_family"],
                                 "action_partition": partition, "action_sign": int(sign), "action_alpha": alpha,
                                 "requested_raw_direction_source_sha256": design["frozen_direction_file_sha256"],
                                 "requested_raw_direction_index": int(action["direction_index"]),
                                 "requested_action_norm": actread["requested_state_norm"],
                                 "realized_action_norm": actread["realized_state_norm"],
                                 "realized_action_cosine": actread["realized_state_cosine"],
                                 "realized_action_gain": actread["realized_state_gain"],
                                 "action_channel_survival": json.dumps(actread["channel_survival"]),
                                 "action_reliable": bool(_action_ok(actread, design["action_reliability"])),
                                 "matched_vs_P0": bool(matched),
                                 "action_pair_cosine_vs_P0": None if match is None else match["realized_action_pair_cosine"],
                                 "no_action_stack": baselines[q["name"]].tolist(),
                                 "response_stack": response.tolist(),
                                 "teacher_token_sha256": hashlib.sha256(np.asarray(token, dtype=np.int32).tobytes()).hexdigest(),
                                 "operator_design_freeze_digest": design["freeze_digest"]})
        state_rows = [{"base_trial_id": item["base_trial_id"], "family": item["family"], "role": role,
                       "boundary_j_vector": boundary_j.astype(np.float32).tolist(),
                       "boundary_j_sha256": jhash, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "teacher_token_sha256": hashlib.sha256(np.asarray(token, dtype=np.int32).tobytes()).hexdigest(),
                       "operator_design_freeze_digest": design["freeze_digest"]}]
        state_temp = state_path.with_suffix(".tmp.parquet")
        pd.DataFrame(state_rows).to_parquet(state_temp, index=False, compression="zstd")
        os.replace(state_temp, state_path)
        temp = path.with_suffix(".tmp.parquet")
        pd.DataFrame(rows).to_parquet(temp, index=False, compression="zstd")
        os.replace(temp, path)
        completed += 1
        if number % 5 == 0 or number == len(items):
            print(f"V20 operator {role} shard{shard} {number}/{len(items)}", flush=True)
    return {"role": role, "shard": shard, "completed": completed, "selected": len(items)}


def aggregate(root: Path, role: str) -> dict:
    design = verify_stage(root, "operator_design")
    split = verify_stage(root, "splits")
    outputs = []
    for family in sorted({x["family"] for x in split[role]}):
        items = [x for x in split[role] if x["family"] == family]
        paths = [record_path(role, x["base_trial_id"]) for x in items]
        state_paths = [state_record_path(role, x["base_trial_id"]) for x in items]
        if not all(path.exists() for path in paths + state_paths):
            raise RuntimeError(f"V20 operator {role}/{family} incomplete")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        expected = len(items) * design["operator_states_per_base"] * len(design["shared_measured_actions"]) * len(design["signs_measured"])
        if len(frame) != expected or frame.operator_state_id.nunique() != len(items) * design["operator_states_per_base"]:
            raise RuntimeError(f"V20 operator {role}/{family} row count mismatch")
        target = root / bank.OUT / f"response_operator_{role}_{family}_v20.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False, compression="zstd")
        outputs.append({"path": str(target.relative_to(root)), "family": family,
                        "base_states": len(items), "operator_states": int(frame.operator_state_id.nunique()),
                        "response_rows": len(frame), "sha256": sha256_file(target),
                        "q_reliable_rate": float(frame.q_reliable.mean()),
                        "action_reliable_rate": float(frame.action_reliable.mean()),
                        "matched_vs_P0_rate": float(frame.matched_vs_P0.mean())})
    state_frame = pd.concat([pd.read_parquet(state_record_path(role, x["base_trial_id"]))
                             for x in split[role]], ignore_index=True)
    if len(state_frame) != len(split[role]) or state_frame.base_trial_id.nunique() != len(split[role]):
        raise RuntimeError("V20 boundary-J state metadata incomplete")
    state_target = root / bank.OUT / f"operator_state_metadata_{role}_v20.parquet"
    state_frame.to_parquet(state_target, index=False, compression="zstd")
    result = {"role": role, "outputs": outputs, "operator_design_freeze_digest": design["freeze_digest"],
              "base_states": sum(x["base_states"] for x in outputs),
              "operator_states": sum(x["operator_states"] for x in outputs),
              "response_rows": sum(x["response_rows"] for x in outputs),
              "state_metadata_path": str(state_target.relative_to(root)),
              "state_metadata_sha256": sha256_file(state_target),
              "final_heldout_actions_unopened": True}
    write_json_atomic(root / bank.OUT / f"response_operator_{role}_v20.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--role", choices=("operator_train", "operator_validation"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    root = Path.cwd()
    result = prepare(root) if args.stage == "prepare" else (
        run(root, args.role, args.shard) if args.stage == "run" else aggregate(root, args.role))
    if args.stage == "prepare":
        result = {"freeze_digest": result["freeze_digest"], "train_states": result["operator_train_states"],
                  "validation_states": result["operator_validation_states"]}
    print(json.dumps(result, indent=2))
