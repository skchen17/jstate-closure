"""Train-calibrated smaller-q locality test, with frozen validation scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as bank
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.protocol_v19 import verify as verify_v19, verify_stage as verify_v19_stage
from jclosure.protocol_v20 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/q_locality_v20.py"
CALIBRATION = bank.OUT / "q_locality_calibration_v20.parquet"
VALIDATION = bank.OUT / "q_locality_validation_v20.parquet"
SUMMARY = bank.OUT / "q_locality_v20.json"


def prepare(root: Path) -> dict:
    cfg = verify(root)["config"]["q_locality"]
    split = verify_stage(root, "splits")
    teacher = verify_stage(root, "teacher")
    q = verify_v19_stage(root, "q_amendment_2")
    actions = verify_v19_stage(root, "splits")
    old_action = next(x for x in actions["actions"] if x["coordinate_index"] == cfg["action_coordinate"])
    old_names = {x["name"] for x in q["q"]}
    if not set(cfg["names"]) <= old_names:
        raise RuntimeError("V20 q locality proposal absent from frozen V19 q bank")
    validation_ids = []
    for family in verify(root)["config"]["roles"]["families"]:
        validation_ids.extend([x["base_trial_id"] for x in split["operator_validation"]
                               if x["family"] == family][: int(cfg["validation_states_per_family"])])
    return stage_freeze(root, "q_locality_design",
                        [SOURCE, "artifacts/compact_causal_response_operator_v20_splits.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_teacher.freeze.json",
                         "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json"],
                        {"q_names": cfg["names"], "alpha_candidates": cfg["candidate_alpha"],
                         "train_calibration_states": len(split["q_calibration"]),
                         "validation_ids": validation_ids,
                         "validation_id_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
                         "action": old_action, "action_alpha": verify_v19(root)["config"]["actions"]["alpha"],
                         "q_reliability": verify_v19(root)["config"]["q"],
                         "action_matching": verify_v19(root)["config"]["actions"],
                         "teacher_sha256": teacher["teacher_sha256"],
                         "before_validation_response_rows": 0,
                         "before_operator_response_rows": 0})


def _teacher_map(root: Path) -> dict[str, list[int]]:
    teacher = verify_stage(root, "teacher")
    frame = pd.read_parquet(root / teacher["teacher_path"])
    return {str(row.base_trial_id): [int(x) for x in row.teacher_tokens] for row in frame.itertuples()}


def _evaluate(root: Path, items: list[dict], stage: dict, alphas: dict[str, list[float]]) -> pd.DataFrame:
    split = verify_stage(root, "splits")
    qbank = verify_v19_stage(root, "q_amendment_2")
    qmap = {x["name"]: x for x in qbank["q"]}
    prompts = bank.prompt_index(root)
    teacher = _teacher_map(root)
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    qcfg = stage["q_reliability"]
    acfg = stage["action_matching"]
    action = stage["action"]
    arow = v19._action_row(values["directions"], int(action["direction_index"]), float(stage["action_alpha"]), 1)
    rows = []
    for number, item in enumerate(items, 1):
        prompt = str(prompts[item["base_trial_id"]]["prompt"])
        clean = v19._prefill_history(bundle, prompt, measured, dense, state_layer)
        token = teacher[item["base_trial_id"]]
        p0 = clone_hybrid_cache(clean["cache"])
        p0hash = v19._snapshot_hash(p0, rec, att)
        p0a = v19.apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
        _, y00 = v19._trajectory(bundle, dense, p0, None, token, 1, clean["prompt_length"],
                                 jids, lids, ws_layers, ws_count, max(measured))
        _, y01 = v19._trajectory(bundle, dense, p0a, None, token, 1, clean["prompt_length"],
                                 jids, lids, ws_layers, ws_count, max(measured))
        base = v19.stack(y00[1], scales).astype(np.float64)
        r0 = v19.stack(y01[1], scales).astype(np.float64) - base
        r0norm = float(np.linalg.norm(r0))
        for qname in stage["q_names"]:
            q = qmap[qname]
            for alpha in alphas[qname]:
                row = v19._q_row(values["directions"], q, alpha)
                pq = v19.apply(p0, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                qreal = _readback(p0, pq, row, rec, att)
                pqa = v19.apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                matched = v19._matching(p0, p0a, pq, pqa, arow, rec, att)
                ratio = matched["realized_norm_ratio_pq_over_p0"]
                action_ok = (matched["realized_action_pair_cosine"] is not None
                             and matched["realized_action_pair_cosine"] >= acfg["matched_realized_cosine_min"]
                             and ratio is not None and acfg["matched_norm_ratio_min"] <= ratio <= acfg["matched_norm_ratio_max"]
                             and matched["realized_cosine_p0"] is not None and matched["realized_cosine_pq"] is not None
                             and matched["realized_cosine_p0"] >= acfg["reliability_min_cosine"]
                             and matched["realized_cosine_pq"] >= acfg["reliability_min_cosine"]
                             and acfg["reliability_gain_min"] <= matched["realized_gain_p0"] <= acfg["reliability_gain_max"]
                             and acfg["reliability_gain_min"] <= matched["realized_gain_pq"] <= acfg["reliability_gain_max"])
                _, y10 = v19._trajectory(bundle, dense, pq, None, token, 1, clean["prompt_length"],
                                         jids, lids, ws_layers, ws_count, max(measured))
                _, y11 = v19._trajectory(bundle, dense, pqa, None, token, 1, clean["prompt_length"],
                                         jids, lids, ws_layers, ws_count, max(measured))
                n = v19.stack(y10[1], scales).astype(np.float64) - base
                rq = v19.stack(y11[1], scales).astype(np.float64) - v19.stack(y10[1], scales).astype(np.float64)
                m = rq - r0
                qnorm = float(qreal["realized_state_norm"])
                rqnorm = float(np.linalg.norm(rq))
                cosine = float(np.dot(r0, rq) / max(r0norm * rqnorm, 1e-8))
                rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                             "q_name": qname, "q_channel": q["channel"], "q_alpha": alpha,
                             "q_reliable": bool(v19._reliable(qreal, qcfg)), "action_matched": bool(action_ok),
                             "q_realized_norm": qnorm, "q_realized_cosine": qreal["realized_state_cosine"],
                             "q_gain": qreal["realized_state_gain"], "q_channel_survival": json.dumps(qreal["channel_survival"]),
                             "action_pair_cosine": matched["realized_action_pair_cosine"],
                             "natural_norm": float(np.linalg.norm(n)),
                             "clean_action_norm": r0norm, "counterfactual_action_norm": rqnorm,
                             "modulation_norm": float(np.linalg.norm(m)),
                             "modulation_ratio": float(np.linalg.norm(m) / max(r0norm, rqnorm, 1e-8)),
                             "natural_to_action_ratio": float(np.linalg.norm(n) / max(r0norm, 1e-8)),
                             "modulation_per_q_norm": float(np.linalg.norm(m) / max(qnorm, 1e-8)),
                             "response_angle_degrees": math.degrees(math.acos(float(np.clip(cosine, -1, 1)))),
                             "boundary_j_difference": 0.0,
                             "p0_snapshot_hash": p0hash,
                             "pq_snapshot_hash": v19._snapshot_hash(pq, rec, att)})
        if number % 5 == 0 or number == len(items):
            print(f"V20 q locality {number}/{len(items)}", flush=True)
    return pd.DataFrame(rows)


def calibrate(root: Path) -> dict:
    stage = verify_stage(root, "q_locality_design")
    split = verify_stage(root, "splits")
    frame = _evaluate(root, split["q_calibration"], stage,
                      {name: [float(x) for x in stage["alpha_candidates"]] for name in stage["q_names"]})
    path = root / CALIBRATION
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    allowed = {}
    by_q_scale = []
    for qname in stage["q_names"]:
        allowed[qname] = []
        for alpha in stage["alpha_candidates"]:
            group = frame[(frame.q_name == qname) & (frame.q_alpha == alpha)]
            reliability = float((group.q_reliable & group.action_matched).mean())
            if reliability >= verify(root)["config"]["q_locality"]["reliable_fraction_min"]:
                allowed[qname].append(float(alpha))
            by_q_scale.append({"q_name": qname, "alpha": alpha, "states": int(group.base_trial_id.nunique()),
                               "reliable_matched_fraction": reliability,
                               "median_N_R": float(group.natural_to_action_ratio.median()),
                               "median_M_R": float(group.modulation_ratio.median()),
                               "median_M_per_q_norm": float(group.modulation_per_q_norm.median()),
                               "median_operator_angle_degrees": float(group.response_angle_degrees.median())})
    if not any(alpha < 1.0 for alphas in allowed.values() for alpha in alphas):
        raise RuntimeError("V20 no smaller reliable q; locality validation cannot be opened")
    return stage_freeze(root, "q_locality_scales",
                        [SOURCE, str(CALIBRATION),
                         "artifacts/compact_causal_response_operator_v20_q_locality_design.freeze.json"],
                        {"calibration_path": str(CALIBRATION), "calibration_sha256": sha256_file(path),
                         "allowed_scales": allowed, "by_q_scale": by_q_scale,
                         "calibration_states": int(frame.base_trial_id.nunique()),
                         "calibration_rows": len(frame),
                         "operator_validation_responses_already_observed": 0,
                         "independent_V20_final_opened": False})


def validate(root: Path) -> dict:
    design = verify_stage(root, "q_locality_design")
    scales = verify_stage(root, "q_locality_scales")
    split = verify_stage(root, "splits")
    ids = set(design["validation_ids"])
    items = [x for x in split["operator_validation"] if x["base_trial_id"] in ids]
    frame = _evaluate(root, items, design, scales["allowed_scales"])
    path = root / VALIDATION
    frame.to_parquet(path, index=False, compression="zstd")
    by_q_scale = []
    for (qname, alpha), group in frame.groupby(["q_name", "q_alpha"], sort=True):
        eligible = group[group.q_reliable & group.action_matched]
        by_q_scale.append({"q_name": qname, "alpha": float(alpha), "states": int(group.base_trial_id.nunique()),
                           "reliable_matched_fraction": float(len(eligible) / len(group)),
                           "median_N_R": float(eligible.natural_to_action_ratio.median()) if len(eligible) else None,
                           "median_M_R": float(eligible.modulation_ratio.median()) if len(eligible) else None,
                           "median_M_per_q_norm": float(eligible.modulation_per_q_norm.median()) if len(eligible) else None,
                           "median_operator_angle_degrees": float(eligible.response_angle_degrees.median()) if len(eligible) else None})
    result = {"design_freeze_digest": design["freeze_digest"], "scale_freeze_digest": scales["freeze_digest"],
              "validation_states": int(frame.base_trial_id.nunique()), "rows": len(frame),
              "by_q_scale": by_q_scale, "path": str(VALIDATION), "sha256": sha256_file(path),
              "small_q_effect_inference": "descriptive_validation_only; primary independent V19-B bank remains separate"}
    write_json_atomic(root / SUMMARY, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "validate"))
    args = parser.parse_args()
    result = {"prepare": prepare, "calibrate": calibrate, "validate": validate}[args.stage](Path.cwd())
    print(json.dumps({"stage": args.stage,
                      "freeze_digest": result.get("freeze_digest", result.get("scale_freeze_digest")),
                      "states": result.get("calibration_states", result.get("validation_states"))}, indent=2))
