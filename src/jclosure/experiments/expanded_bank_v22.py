"""Expanded ± finite-response bank for frozen V22 action designs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import action_pool_v22 as actions
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.operator_v15 import stack
from jclosure.protocol_v22 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/expanded_bank_v22.py"
OUT = Path("results/v22/processed")
SCRATCH = Path("/data/CSK/J-space-project/v22-action-manifold-work/expanded_response_bank")


def prepare(root: Path) -> dict:
    selection = verify_stage(root, "action_selection")
    roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    design = {}
    for role, count in (("development", 3), ("validation", 2)):
        source = roles[f"jvp_{role}"]
        selected = []
        for family in sorted({item["family"] for item in source}):
            selected.extend([item for item in source if item["family"] == family][:count])
        design[role] = selected
    payload = {
        "development_states": design["development"], "validation_states": design["validation"],
        "development_base_count": len(design["development"]), "validation_base_count": len(design["validation"]),
        "persistent_states_per_base": ["P0", "Pq"],
        "train_action_count": 128, "heldout_validation_action_count": 32,
        "signs": [-1, 1], "target": "V20_normalized_stack_288_h1",
        "action_selection_digest": selection["freeze_digest"],
        "expanded_responses_observed_before_freeze": 0,
        "historical_final_responses_opened": False, "new_independent_final_responses_opened": False,
    }
    target = root / OUT / "expanded_bank_design_v22.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, payload)
    frozen = stage_freeze(root, "expanded_bank_design", [SOURCE,
        "artifacts/causal_action_manifold_v22_action_selection.freeze.json",
        "artifacts/action_coordinate_geometry_v21_roles.freeze.json", str(target)], payload)
    return {"freeze_digest": frozen["freeze_digest"],
            "development_base_count": len(design["development"]), "validation_base_count": len(design["validation"])}


def _evaluate(bundle, dense, cache, token, prompt_length, jids, lids, ws_layers, ws_count, main, scales):
    _, endpoints = v19._trajectory(bundle, dense, cache, None, [token], 1, prompt_length,
                                   jids, lids, ws_layers, ws_count, main)
    return stack(endpoints[1], scales).astype(np.float32)


def run(root: Path, role: str, limit: int | None = None) -> dict:
    if role not in ("development", "validation"):
        raise RuntimeError("V22 expanded bank role invalid")
    verify_stage(root, "expanded_bank_design")
    design = json.loads((root / OUT / "expanded_bank_design_v22.json").read_text())
    selection = json.loads((root / OUT / "action_selection_v22.json").read_text())
    pool = json.loads((root / OUT / "action_pool_v22.json").read_text())
    specs = {item["action_id"]: item for item in pool["candidate_pool"]}
    action_ids = selection["common_measured_train_action_ids"] + selection["validation_action_ids"]
    split = json.loads((root / "artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = v20_bank.prompt_index(root)
    teachers = v20_response._teacher_map(root)
    q_lookup = {item["name"]: item for item in operator["q"]}
    jids = np.asarray(split["selected_j"], dtype=int)
    lids = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    directions = values["directions"]
    items = design[f"{role}_states"][:limit]
    paths = []
    for number, item in enumerate(items, 1):
        target = root / SCRATCH / role / f"expanded_{item['base_trial_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"V22 expanded {role} {number}/{len(items)} exists", flush=True)
            paths.append(target)
            continue
        token = int(teachers[item["base_trial_id"]][0])
        clean = v19._prefill_history(bundle, str(prompts[item["base_trial_id"]]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        q = q_lookup[item["q_name"]]
        qrow = v19._q_row(directions, q, float(q["alpha"]))
        pq = v19.apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
        arrays = {}
        for state_name, state in (("P0", p0), ("Pq", pq)):
            baseline = _evaluate(bundle, dense, state, token, clean["prompt_length"], jids, lids,
                                 ws_layers, ws_count, max(measured), scales)
            plus, minus = [], []
            for action_number, action_id in enumerate(action_ids, 1):
                row = actions._row(directions, specs[action_id], selection["action_alphas"][action_id], 1)
                positive = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                negative = v19.apply(state, -1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                plus.append(_evaluate(bundle, dense, positive, token, clean["prompt_length"], jids, lids,
                                      ws_layers, ws_count, max(measured), scales) - baseline)
                minus.append(_evaluate(bundle, dense, negative, token, clean["prompt_length"], jids, lids,
                                       ws_layers, ws_count, max(measured), scales) - baseline)
                if action_number % 32 == 0:
                    print(f"V22 expanded {role} base={number}/{len(items)} state={state_name} action={action_number}/{len(action_ids)}", flush=True)
            arrays[f"{state_name}_plus"] = np.stack(plus).astype(np.float32)
            arrays[f"{state_name}_minus"] = np.stack(minus).astype(np.float32)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays, action_ids=np.asarray(action_ids),
                            base_trial_id=np.asarray(item["base_trial_id"]), family=np.asarray(item["family"]),
                            q_name=np.asarray(item["q_name"]), p0_snapshot_sha256=np.asarray(v19._snapshot_hash(p0, rec, att)),
                            pq_snapshot_sha256=np.asarray(v19._snapshot_hash(pq, rec, att)))
        os.replace(temporary, target)
        paths.append(target)
    return {"role": role, "completed": len(paths), "requested": len(items)}


def summarize(root: Path) -> dict:
    design = json.loads((root / OUT / "expanded_bank_design_v22.json").read_text())
    rows = []
    for role in ("development", "validation"):
        for item in design[f"{role}_states"]:
            path = root / SCRATCH / role / f"expanded_{item['base_trial_id']}.npz"
            if not path.exists():
                raise RuntimeError(f"missing V22 expanded response file: {path}")
            rows.append({"role": role, **item, "path": str(path), "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    target = root / OUT / "expanded_response_bank_index_v22.parquet"
    frame.to_parquet(target, index=False, compression="zstd")
    result = {"operator_base_states": len(frame), "operator_states": 2 * len(frame),
              "train_actions": 128, "validation_actions": 32, "signs": [-1, 1],
              "response_rows": 2 * len(frame) * 160 * 2,
              "index_sha256": sha256_file(target), "historical_final_responses_opened": False,
              "new_independent_final_responses_opened": False}
    write_json_atomic(root / OUT / "expanded_response_bank_v22.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--role", choices=("development", "validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "prepare": answer = prepare(Path.cwd())
    elif args.stage == "run": answer = run(Path.cwd(), args.role, args.limit)
    else: answer = summarize(Path.cwd())
    print(json.dumps(answer, indent=2))
