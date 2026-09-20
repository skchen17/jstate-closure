"""V19 exact-boundary-J persistent counterfactual four-way bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.actuation_v15 import apply, components
from jclosure.experiments.action_bank_v16 import _readback
from jclosure.experiments.crossed_bank_v18 import (
    _prefill_history, _split as v18_split, _trajectory, load_context_v18,
)
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.protocol_v19 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v19/processed")
SCRATCH = Path("/data/CSK/J-space-project/v19-counterfactual-work")
SOURCE = "src/jclosure/experiments/counterfactual_bank_v19.py"
V18_STATE = Path("results/v18/processed")
CHANNELS = ("recurrent", "conv", "kv")


def _sha_ids(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(str(x["base_trial_id"]) for x in rows)).encode()).hexdigest()


def prepare(root: Path) -> dict[str, Any]:
    cfg = verify(root)["config"]
    old = v18_split(root)
    selection: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation"):
        rows = []
        for family in sorted({x["family"] for x in old[role]}):
            pool = [x for x in old[role] if x["family"] == family]
            pool.sort(key=lambda x: hashlib.sha256(f"{cfg['roles']['seed']}:{role}:{x['base_trial_id']}".encode()).hexdigest())
            skip = int(cfg["roles"]["calibration_per_family"]) if role == "train" else 0
            if role == "train":
                rows.extend({"base_trial_id": x["base_trial_id"], "family": family,
                             "role": "calibration", "horizon_panel": False}
                            for x in pool[:skip])
            count = int(cfg["roles"][f"{role}_per_family"])
            nh = int(cfg["roles"][f"horizon_{role}_per_family"])
            chosen = pool[skip:skip + count]
            if len(chosen) != count:
                raise RuntimeError(f"V19 insufficient {role}/{family} states")
            rows.extend({"base_trial_id": x["base_trial_id"], "family": family,
                         "role": role, "horizon_panel": i < nh}
                        for i, x in enumerate(chosen))
        if role == "train":
            selection["calibration"] = [x for x in rows if x["role"] == "calibration"]
            rows = [x for x in rows if x["role"] == "train"]
        selection[role] = rows
    ids = [x["base_trial_id"] for group in selection.values() for x in group]
    if len(set(ids)) != len(ids):
        raise RuntimeError("V19 role overlap")
    actions = [x for x in old["actions"] if x["coordinate_index"] in cfg["actions"]["coordinate_indices"]]
    if [x["coordinate_index"] for x in actions] != cfg["actions"]["coordinate_indices"]:
        raise RuntimeError("V18 action bank differs from V19 declaration")
    detail = {**selection, "role_id_sha256": {name: _sha_ids(rows) for name, rows in selection.items()},
              "actions": actions, "selected_j": old["selected_j"],
              "selected_logits": old["selected_logits"], "target_scales": old["target_scales"],
              "v18_split_digest": old["freeze_digest"],
              "validation_is_development_not_independent": True}
    return stage_freeze(root, "splits", [SOURCE, "artifacts/strong_state_context_ceiling_v18_splits.freeze.json",
                                         "results/v13/processed/probe_directions_v13.json"], detail)


def _setup(root: Path):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    bundle, dense_map, v13, metadata, values = load_context_v18(root)
    v8 = v13["persistent_state_v8"]
    rec = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    att = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    workspace_layers = [int(x) for x in jvp["workspace_layers"]]
    workspace_count = int(jvp["selected_workspace_count"])
    return bundle, dense_map, metadata, values, rec, att, measured, int(v8["intervention"]["layer"]), workspace_layers, workspace_count


def _q_row(directions: dict[str, Any], design: dict, alpha: float, sign: int = 1) -> dict[str, torch.Tensor]:
    channel = design["channel"]
    active = {"recurrent": ("recurrent",), "conv": ("conv",), "kv": ("kv",),
              "rec_conv": ("recurrent", "conv"), "joint": CHANNELS}[channel]
    index = int(design["direction_index"])
    return {key: directions[key][index].float() * (alpha * sign if key in active else 0.0)
            for key in CHANNELS}


def _action_row(directions: dict[str, Any], index: int, alpha: float, sign: int) -> dict[str, torch.Tensor]:
    return {key: directions[key][index].float() * (alpha * sign) for key in CHANNELS}


def _endpoint_args(bundle, dense_map, split, workspace_layers, workspace_count, measured, prompt_length):
    return (bundle, dense_map, None, None, None, 1, prompt_length,
            np.asarray(split["selected_j"], dtype=int), np.asarray(split["selected_logits"], dtype=int),
            workspace_layers, workspace_count, max(measured))


def _state_metadata(root: Path, item: dict) -> dict:
    path = root / V18_STATE / f"crossed_state_{'train' if item['role'] == 'calibration' else item['role']}_{item['family']}_v18.parquet"
    frame = pd.read_parquet(path)
    row = frame[frame.base_trial_id == item["base_trial_id"]]
    if len(row) != 1:
        raise RuntimeError(f"V18 state metadata missing: {item['base_trial_id']}")
    return row.iloc[0].to_dict()


def _snapshot_hash(cache: Any, rec: list[int], att: list[int]) -> str:
    h = hashlib.sha256()
    for layer in rec:
        for key in ("recurrent_states", "conv_states"):
            tensor = getattr(cache.layers[layer], key).detach().contiguous().cpu()
            h.update(tensor.view(torch.int16).numpy().tobytes() if tensor.dtype == torch.bfloat16 else tensor.numpy().tobytes())
    for layer in att:
        for key in ("keys", "values"):
            tensor = getattr(cache.layers[layer], key).detach().contiguous().cpu()
            h.update(tensor.view(torch.int16).numpy().tobytes() if tensor.dtype == torch.bfloat16 else tensor.numpy().tobytes())
    return h.hexdigest()


def _matching(base0: Any, action0: Any, baseq: Any, actionq: Any, row: dict, rec: list[int], att: list[int]) -> dict:
    requested_sq = n0 = nq = dot0 = dotq = cross = 0.0
    survived: dict[str, list[float]] = {key: [] for key in ("recurrent", "conv", "keys", "values")}
    for (name, layer, attr, old0, desired), (_, _, _, oldq, _) in zip(
            components(base0, row, rec, att), components(baseq, row, rec, att), strict=True):
        new0 = getattr(action0.layers[layer], attr)
        newq = getattr(actionq.layers[layer], attr)
        if name in ("keys", "values"):
            new0 = new0[..., :old0.shape[-2], :]
            newq = newq[..., :oldq.shape[-2], :]
        d = desired.double().reshape(-1)
        x = (new0.float() - old0.float()).double().reshape(-1)
        y = (newq.float() - oldq.float()).double().reshape(-1)
        requested_sq += float(torch.dot(d, d))
        n0 += float(torch.dot(x, x)); nq += float(torch.dot(y, y))
        dot0 += float(torch.dot(d, x)); dotq += float(torch.dot(d, y))
        cross += float(torch.dot(x, y))
        nz = d != 0
        if bool(nz.any()):
            survived[name].append(float(((x[nz] != 0) & (y[nz] != 0)).double().mean()))
    requested = math.sqrt(requested_sq)
    norm0, normq = math.sqrt(n0), math.sqrt(nq)
    return {"requested_norm": requested, "realized_norm_p0": norm0, "realized_norm_pq": normq,
            "realized_gain_p0": norm0 / requested if requested else None,
            "realized_gain_pq": normq / requested if requested else None,
            "realized_cosine_p0": dot0 / (requested * norm0) if requested * norm0 else None,
            "realized_cosine_pq": dotq / (requested * normq) if requested * normq else None,
            "realized_action_pair_cosine": cross / (norm0 * normq) if norm0 * normq else None,
            "realized_norm_ratio_pq_over_p0": normq / norm0 if norm0 else None,
            "realized_delta_difference_norm": math.sqrt(max(n0 + nq - 2 * cross, 0.0)),
            "channel_joint_survival": {key: float(np.mean(values)) if values else None for key, values in survived.items()}}


def _reliable(m: dict, spec: dict) -> bool:
    return (m["realized_state_cosine"] is not None and m["realized_state_cosine"] >= spec["reliability_min_cosine"]
            and m["realized_state_gain"] is not None
            and spec["reliability_gain_min"] <= m["realized_state_gain"] <= spec["reliability_gain_max"])


def calibrate(root: Path) -> dict:
    split = verify_stage(root, "splits")
    cfg = verify(root)["config"]
    bundle, dense, metadata, values, rec, att, measured, state_layer, ws_layers, ws_count = _setup(root)
    jids = np.asarray(split["selected_j"], dtype=int); lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(v) for key, v in split["target_scales"].items()}
    records = []
    for item in split["calibration"]:
        old = _state_metadata(root, item)
        clean = _prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        tokens = [int(x) for x in old["teacher_tokens_h8_or_h1"][:1]]
        _, y00 = _trajectory(bundle, dense, clean["cache"], None, tokens, 1, clean["prompt_length"],
                             jids, lids, ws_layers, ws_count, max(measured))
        action = split["actions"][0]
        edited_a = apply(clean["cache"], 1.0, _action_row(values["directions"], int(action["direction_index"]),
                           float(cfg["actions"]["alpha"]), 1), rec, att, "native_fp32_add_bf16_writeback")
        _, y01 = _trajectory(bundle, dense, edited_a, None, tokens, 1, clean["prompt_length"],
                             jids, lids, ws_layers, ws_count, max(measured))
        r = stack({key: y01[1][key] - y00[1][key] for key in TARGETS}, scales)
        for design in cfg["q"]["candidates"]:
            for alpha in cfg["q"]["alpha_candidates"]:
                row = _q_row(values["directions"], design, float(alpha))
                edited_q = apply(clean["cache"], 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                readback = _readback(clean["cache"], edited_q, row, rec, att)
                _, y10 = _trajectory(bundle, dense, edited_q, None, tokens, 1, clean["prompt_length"],
                                     jids, lids, ws_layers, ws_count, max(measured))
                n = stack({key: y10[1][key] - y00[1][key] for key in TARGETS}, scales)
                records.append({"base_trial_id": item["base_trial_id"], "family": item["family"],
                                "q_name": design["name"], "q_channel": design["channel"], "alpha": float(alpha),
                                "natural_to_action_ratio": float(np.linalg.norm(n) / max(np.linalg.norm(r), 1e-8)),
                                "natural_stack_norm": float(np.linalg.norm(n)), "clean_action_stack_norm": float(np.linalg.norm(r)),
                                **readback, "reliable": _reliable(readback, cfg["q"]),
                                "finite_endpoint": bool(np.isfinite(n).all())})
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    path = root / OUT / "q_calibration_v19.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    chosen = []
    for design in cfg["q"]["candidates"]:
        possibilities = []
        for alpha in cfg["q"]["alpha_candidates"]:
            sub = frame[(frame.q_name == design["name"]) & (frame.alpha == alpha)]
            possibilities.append({"alpha": float(alpha), "reliable_fraction": float((sub.reliable & sub.finite_endpoint).mean()),
                                  "median_natural_to_action_ratio": float(sub.natural_to_action_ratio.median()),
                                  "median_raw_state_delta_norm": float(sub.realized_state_norm.median())})
        eligible = [x for x in possibilities if x["reliable_fraction"] >= 0.8
                    and x["median_raw_state_delta_norm"] >= cfg["q"]["min_raw_state_delta_norm"]]
        selected = max(eligible, key=lambda x: x["median_natural_to_action_ratio"]) if eligible else max(possibilities, key=lambda x: x["reliable_fraction"])
        chosen.append({**design, **selected, "calibration_status": "RELIABLE" if eligible else "NOT_RELIABLY_ACTUATABLE",
                       "natural_active_train_pilot": selected["median_natural_to_action_ratio"] >= cfg["q"]["natural_active_ratio_min"]})
    return stage_freeze(root, "interventions", [SOURCE, str(path.relative_to(root)),
                                                "artifacts/counterfactual_workspace_v19_splits.freeze.json"],
                        {"q": chosen, "q_calibration_rows": len(frame),
                         "q_calibration_state_count": int(frame.base_trial_id.nunique()),
                         "probe_actions": split["actions"], "action_matching_threshold": cfg["actions"],
                         "thresholds": cfg["targets"], "validation_responses_observed": 0,
                         "independent_final_opened": False})


def _record_path(role: str, item: dict) -> Path:
    return SCRATCH / role / f"factorial_{item['base_trial_id']}.parquet"


def run(root: Path, role: str, limit: int | None = None) -> dict:
    split = verify_stage(root, "splits")
    interventions = verify_stage(root, "interventions")
    cfg = verify(root)["config"]
    bundle, dense, metadata, values, rec, att, measured, state_layer, ws_layers, ws_count = _setup(root)
    jids = np.asarray(split["selected_j"], dtype=int); lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(v) for key, v in split["target_scales"].items()}
    chosen = [x for x in interventions["q"] if x["calibration_status"] == "RELIABLE"]
    if not chosen:
        raise RuntimeError("No reliable q direction; four-way bank not identifiable")
    items = split[role][:limit] if limit is not None else split[role]
    (SCRATCH / role).mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(items, 1):
        path = _record_path(role, item)
        if path.exists():
            continue
        old = _state_metadata(root, item)
        clean = _prefill_history(bundle, old["prompt"], measured, dense, state_layer)
        count = 8 if item["horizon_panel"] else 1
        old_tokens = [int(x) for x in old["teacher_tokens_h8_or_h1"]]
        if len(old_tokens) < count:
            raise RuntimeError(f"V18 teacher sequence too short for {item['base_trial_id']}")
        tokens = old_tokens[:count]
        boundary_j = np.asarray(clean["history_j"][-1], dtype=np.float16)
        jhash = hashlib.sha256(boundary_j.tobytes()).hexdigest()
        p0 = clone_hybrid_cache(clean["cache"])
        p0hash = _snapshot_hash(p0, rec, att)
        _, y00 = _trajectory(bundle, dense, p0, None, tokens, count, clean["prompt_length"],
                             jids, lids, ws_layers, ws_count, max(measured))
        action_designs = [x for x in split["actions"] if x["coordinate_index"] in cfg["actions"]["development_coordinates"]
                          or (item["horizon_panel"] and x["coordinate_index"] in cfg["actions"]["heldout_coordinates"])]
        actions = []
        for design in action_designs:
            for sign in cfg["actions"]["signs"]:
                arow = _action_row(values["directions"], int(design["direction_index"]), float(cfg["actions"]["alpha"]), int(sign))
                p0a = apply(p0, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                _, y01 = _trajectory(bundle, dense, p0a, None, tokens, count, clean["prompt_length"],
                                     jids, lids, ws_layers, ws_count, max(measured))
                actions.append((design, int(sign), arow, p0a, y01))
        rows = []
        for q in chosen:
            qrow = _q_row(values["directions"], q, float(q["alpha"]))
            pq = apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
            qreal = _readback(p0, pq, qrow, rec, att)
            pqhash = _snapshot_hash(pq, rec, att)
            _, y10 = _trajectory(bundle, dense, pq, None, tokens, count, clean["prompt_length"],
                                 jids, lids, ws_layers, ws_count, max(measured))
            qok = _reliable(qreal, cfg["q"])
            for action, sign, arow, p0a, y01 in actions:
                pqa = apply(pq, 1.0, arow, rec, att, "native_fp32_add_bf16_writeback")
                match = _matching(p0, p0a, pq, pqa, arow, rec, att)
                ratio = match["realized_norm_ratio_pq_over_p0"]
                matched = (match["realized_action_pair_cosine"] is not None
                           and match["realized_action_pair_cosine"] >= cfg["actions"]["matched_realized_cosine_min"]
                           and ratio is not None and cfg["actions"]["matched_norm_ratio_min"] <= ratio <= cfg["actions"]["matched_norm_ratio_max"]
                           and match["realized_cosine_p0"] is not None and match["realized_cosine_pq"] is not None
                           and match["realized_cosine_p0"] >= cfg["actions"]["reliability_min_cosine"]
                           and match["realized_cosine_pq"] >= cfg["actions"]["reliability_min_cosine"]
                           and cfg["actions"]["reliability_gain_min"] <= match["realized_gain_p0"] <= cfg["actions"]["reliability_gain_max"]
                           and cfg["actions"]["reliability_gain_min"] <= match["realized_gain_pq"] <= cfg["actions"]["reliability_gain_max"])
                _, y11 = _trajectory(bundle, dense, pqa, None, tokens, count, clean["prompt_length"],
                                     jids, lids, ws_layers, ws_count, max(measured))
                for h in y00:
                    endpoints = [y00[h], y01[h], y10[h], y11[h]]
                    stacked = [stack(x, scales).astype(np.float32) for x in endpoints]
                    rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"], "role": role,
                                 "horizon_panel": bool(item["horizon_panel"]), "horizon": h,
                                 "q_name": q["name"], "q_channel": q["channel"], "q_direction_index": int(q["direction_index"]),
                                 "q_alpha": float(q["alpha"]), "q_reliable": bool(qok),
                                 "q_requested_state_norm": qreal["requested_state_norm"],
                                 "q_realized_state_norm": qreal["realized_state_norm"],
                                 "q_realized_cosine": qreal["realized_state_cosine"],
                                 "q_gain": qreal["realized_state_gain"], "q_channel_survival": json.dumps(qreal["channel_survival"]),
                                 "boundary_j_hash_p0": jhash, "boundary_j_hash_pq": jhash, "boundary_j_difference": 0.0,
                                 "p0_snapshot_hash": p0hash, "pq_snapshot_hash": pqhash,
                                 "snapshot_branch_identity_verified": True,
                                 "coordinate_index": int(action["coordinate_index"]), "action_direction_index": int(action["direction_index"]),
                                 "action_sign": sign, "action_alpha": float(cfg["actions"]["alpha"]),
                                 "action_role": "development" if action["coordinate_index"] in cfg["actions"]["development_coordinates"] else "heldout",
                                 "action_status": "MATCHED_REALIZED_ACTION" if matched else "ACTION_REALIZATION_MISMATCH",
                                 **{key: value for key, value in match.items() if key != "channel_joint_survival"},
                                 "action_channel_joint_survival": json.dumps(match["channel_joint_survival"]),
                                 "y00_stack": stacked[0].tolist(), "y01_stack": stacked[1].tolist(),
                                 "y10_stack": stacked[2].tolist(), "y11_stack": stacked[3].tolist(),
                                 "teacher_tokens_sha256": hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest(),
                                 "split_freeze_digest": split["freeze_digest"],
                                 "intervention_freeze_digest": interventions["freeze_digest"]})
                del pqa
            del pq
        pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd", compression_level=9)
        if number % 5 == 0 or number == len(items):
            print(f"V19 {role} {number}/{len(items)} {item['base_trial_id']} rows={len(rows)}", flush=True)
    return {"role": role, "completed": len(items), "selected": len(items)}


def aggregate(root: Path, role: str) -> dict:
    split = verify_stage(root, "splits")
    interventions = verify_stage(root, "interventions")
    outputs = []
    for family in sorted({x["family"] for x in split[role]}):
        items = [x for x in split[role] if x["family"] == family]
        paths = [_record_path(role, x) for x in items]
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"V19 {role}/{family} incomplete")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        path = root / OUT / f"factorial_{role}_{family}_v19.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd", compression_level=9)
        outputs.append({"path": str(path.relative_to(root)), "states": int(frame.base_trial_id.nunique()),
                        "rows": len(frame), "sha256": sha256_file(path)})
    result = {"role": role, "outputs": outputs, "complete": True,
              "split_freeze_digest": split["freeze_digest"],
              "intervention_freeze_digest": interventions["freeze_digest"]}
    write_json_atomic(root / OUT / f"bank_{role}_v19.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "train", "validation", "aggregate_train", "aggregate_validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "prepare":
        result = prepare(root)
    elif args.stage == "calibrate":
        result = calibrate(root)
    elif args.stage.startswith("aggregate_"):
        result = aggregate(root, args.stage.split("_", 1)[1])
    else:
        result = run(root, args.stage, args.limit)
    print(json.dumps(result, indent=2, default=str))
