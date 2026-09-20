"""Frozen V19 q sign/scale and writeback-order diagnostics, separate from primary bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments.actuation_v15 import apply
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.protocol_v19 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/diagnostics_v19.py"
Q_NAMES = ("rec_arch4_amended", "conv_causal1", "kv_arch14_amended",
           "rec_conv_causal1_amended", "joint_causal11")


def prepare(root: Path) -> dict:
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    teacher = verify_stage(root, "teacher_amendment_3")
    cfg = verify(root)["config"]
    selected = []
    order = []
    for family in sorted({x["family"] for x in split["validation"]}):
        items = [x for x in split["validation"] if x["family"] == family and x["horizon_panel"]]
        selected.extend(items[:int(cfg["diagnostics"]["sign_scale_states_per_family"])])
        order.extend(items[:int(cfg["diagnostics"]["writeback_order_states_per_family"])])
    return stage_freeze(root, "diagnostics",
                        [SOURCE, "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "artifacts/counterfactual_workspace_v19_teacher_amendment_3.freeze.json"],
                        {"q_freeze_digest": q["freeze_digest"], "teacher_freeze_digest": teacher["freeze_digest"],
                         "sign_scale_items": selected, "order_items": order,
                         "q_names": list(Q_NAMES), "signs": [-1, 1], "scales": [0.5, 1.0],
                         "probe_coordinate": 5, "probe_sign": 1, "probe_alpha": 0.5,
                         "horizons": [1, 2, 4, 8],
                         "diagnostic_only_not_primary_estimand": True,
                         "factorial_train_state_records_already_observed": len(list((bank.SCRATCH / "train").glob("factorial_*.parquet"))),
                         "factorial_validation_state_records_already_observed": len(list((bank.SCRATCH / "validation").glob("factorial_*.parquet")))})


def _load(root: Path):
    design = verify_stage(root, "diagnostics")
    split = verify_stage(root, "splits")
    q = verify_stage(root, "q_amendment_2")
    teachers = pd.read_parquet(root / verify_stage(root, "teacher_amendment_3")["teacher_path"])
    teacher_map = {str(row.base_trial_id): [int(x) for x in row.teacher_tokens]
                   for row in teachers.itertuples()}
    ctx = bank._setup(root)
    return design, split, {x["name"]: x for x in q["q"]}, teacher_map, ctx


def _endpoints(bundle, dense, cache, tokens, prompt_length, split, ws_layers, ws_count, measured):
    _, values = bank._trajectory(bundle, dense, cache, None, tokens, len(tokens), prompt_length,
                                 np.asarray(split["selected_j"], dtype=int),
                                 np.asarray(split["selected_logits"], dtype=int),
                                 ws_layers, ws_count, max(measured))
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    return {h: stack(data, scales).astype(np.float32) for h, data in values.items()}


def sign_scale(root: Path) -> dict:
    design, split, q_map, teacher_map, ctx = _load(root)
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = ctx
    action = next(x for x in split["actions"] if x["coordinate_index"] == design["probe_coordinate"])
    arow = bank._action_row(values["directions"], int(action["direction_index"]),
                            float(design["probe_alpha"]), int(design["probe_sign"]))
    rows = []
    for item in design["sign_scale_items"]:
        old = bank._state_metadata(root, item)
        clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        tokens = teacher_map[item["base_trial_id"]]
        p0 = clean["cache"]
        y00 = _endpoints(bundle, dense, p0, tokens, clean["prompt_length"], split, ws_layers, ws_count, measured)
        p0a = apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
        y01 = _endpoints(bundle, dense, p0a, tokens, clean["prompt_length"], split, ws_layers, ws_count, measured)
        for qname in design["q_names"]:
            q = q_map[qname]
            for alpha in design["scales"]:
                for sign in design["signs"]:
                    qrow = bank._q_row(values["directions"], q, float(alpha), int(sign))
                    pq = apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
                    readback = _readback(p0, pq, qrow, rec, att)
                    y10 = _endpoints(bundle, dense, pq, tokens, clean["prompt_length"], split, ws_layers, ws_count, measured)
                    pqa = apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                    match = bank._matching(p0, p0a, pq, pqa, arow, rec, att)
                    y11 = _endpoints(bundle, dense, pqa, tokens, clean["prompt_length"], split, ws_layers, ws_count, measured)
                    for h in y00:
                        n = y10[h] - y00[h]
                        r0 = y01[h] - y00[h]
                        rq = y11[h] - y10[h]
                        m = rq - r0
                        rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                                     "q_name": qname, "q_channel": q["channel"], "q_alpha": float(alpha),
                                     "q_sign": int(sign), "horizon": h,
                                     "q_reliable": bank._reliable(readback, verify(root)["config"]["q"]),
                                     "q_realized_cosine": readback["realized_state_cosine"],
                                     "q_gain": readback["realized_state_gain"],
                                     "action_pair_cosine": match["realized_action_pair_cosine"],
                                     "action_norm_ratio": match["realized_norm_ratio_pq_over_p0"],
                                     "natural_norm": float(np.linalg.norm(n)), "modulation_norm": float(np.linalg.norm(m)),
                                     "modulation_ratio": float(np.linalg.norm(m) / max(np.linalg.norm(r0), np.linalg.norm(rq), 1e-8)),
                                     "natural_stack": n.tolist(), "modulation_stack": m.tolist()})
    frame = pd.DataFrame(rows)
    path = root / bank.OUT / "q_sign_scale_v19.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    pairs = []
    for key, group in frame.groupby(["base_trial_id", "q_name", "q_alpha", "horizon"]):
        if set(group.q_sign) != {-1, 1}:
            continue
        plus = group[group.q_sign == 1].iloc[0]
        minus = group[group.q_sign == -1].iloc[0]
        for endpoint in ("natural", "modulation"):
            a = np.asarray(plus[f"{endpoint}_stack"]); b = np.asarray(minus[f"{endpoint}_stack"])
            odd = (a - b) / 2; even = (a + b) / 2
            pairs.append({"base_trial_id": key[0], "q_name": key[1], "q_alpha": key[2], "horizon": key[3],
                          "endpoint": endpoint, "odd_norm": float(np.linalg.norm(odd)),
                          "even_norm": float(np.linalg.norm(even)),
                          "even_to_odd_ratio": float(np.linalg.norm(even) / max(np.linalg.norm(odd), 1e-8))})
    pair_path = root / bank.OUT / "q_sign_odd_even_v19.parquet"
    pd.DataFrame(pairs).to_parquet(pair_path, index=False, compression="zstd")
    result = {"states": int(frame.base_trial_id.nunique()), "rows": len(frame), "pair_rows": len(pairs),
              "q_names": list(design["q_names"]), "horizons": list(design["horizons"]),
              "median_modulation_ratio_by_horizon": {str(h): float(group.modulation_ratio.median()) for h, group in frame.groupby("horizon")},
              "median_modulation_ratio_by_scale": {str(h): float(group.modulation_ratio.median()) for h, group in frame.groupby("q_alpha")},
              "q_reliable_rate": float(frame.q_reliable.mean()),
              "sign_scale_path": str(path.relative_to(root)), "sign_scale_sha256": sha256_file(path),
              "odd_even_path": str(pair_path.relative_to(root)), "odd_even_sha256": sha256_file(pair_path),
              "freeze_digest": design["freeze_digest"]}
    write_json_atomic(root / bank.OUT / "q_sign_scale_v19.json", result)
    return result


def order(root: Path) -> dict:
    design, split, q_map, teacher_map, ctx = _load(root)
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = ctx
    q = q_map["rec_conv_causal1_amended"]
    action = next(x for x in split["actions"] if x["coordinate_index"] == design["probe_coordinate"])
    qrow = bank._q_row(values["directions"], q, float(q["alpha"]))
    arow = bank._action_row(values["directions"], int(action["direction_index"]),
                            float(design["probe_alpha"]), int(design["probe_sign"]))
    combined = {key: qrow[key] + arow[key] for key in qrow}
    rows = []
    for item in design["order_items"]:
        old = bank._state_metadata(root, item)
        clean = bank._prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        p0 = clean["cache"]
        tokens = teacher_map[item["base_trial_id"]]
        pq = apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
        snapshot = bank.clone_hybrid_cache(pq)
        branches = {
            "A_snapshot_q_then_a": apply(snapshot, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback"),
            "B_direct_q_then_a": apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback"),
            "C_direct_a_then_q": apply(apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback"),
                                        1.0, qrow, rec, att, "native_fp32_add_bf16_writeback"),
            "D_single_composed_write": apply(p0, 1.0, combined, rec, att, "native_fp32_add_bf16_writeback"),
        }
        endpoints = {name: _endpoints(bundle, dense, cache, tokens, clean["prompt_length"], split,
                                      ws_layers, ws_count, measured) for name, cache in branches.items()}
        hashes = {name: bank._snapshot_hash(cache, rec, att) for name, cache in branches.items()}
        ref = endpoints["A_snapshot_q_then_a"]
        for name in branches:
            for h in ref:
                delta = endpoints[name][h] - ref[h]
                rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                             "branch": name, "horizon": h, "stack_difference_from_A": float(np.linalg.norm(delta)),
                             "stack_difference_relative_to_A": float(np.linalg.norm(delta) / max(np.linalg.norm(ref[h]), 1e-8)),
                             "persistent_snapshot_hash": hashes[name],
                             "snapshot_A_identical": hashes[name] == hashes["A_snapshot_q_then_a"]})
    frame = pd.DataFrame(rows)
    path = root / bank.OUT / "writeback_order_audit_v19.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    result = {"states": int(frame.base_trial_id.nunique()), "rows": len(frame),
              "branches": list(frame.branch.unique()),
              "median_stack_difference_from_A_by_branch": {name: float(group.stack_difference_from_A.median())
                                                            for name, group in frame.groupby("branch")},
              "snapshot_B_exact_fraction": float(frame[frame.branch == "B_direct_q_then_a"].snapshot_A_identical.mean()),
              "path": str(path.relative_to(root)), "sha256": sha256_file(path),
              "freeze_digest": design["freeze_digest"], "primary_estimand_uses_only_A": True}
    write_json_atomic(root / bank.OUT / "writeback_order_audit_v19.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "sign_scale", "order"))
    args = parser.parse_args()
    result = {"prepare": prepare, "sign_scale": sign_scale, "order": order}[args.stage](Path.cwd())
    print(json.dumps(result, indent=2))
