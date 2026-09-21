"""Exact-JVP and finite-response measurement for the frozen V23 nested probe panel."""

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
from jclosure.experiments import paired_geometry_v21 as paired
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.operator_v15 import stack
from jclosure.protocol_v23 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/rank_measurement_v23.py"
OUT = Path("results/v23/processed")
SCRATCH = Path("/data/CSK/J-space-project/v23-oracle-chart-work")
V21_RAW = Path("/data/CSK/J-space-project/v21-action-geometry-work/paired_geometry_forward_ad/development")
V22_RAW = Path("/data/CSK/J-space-project/v22-action-manifold-work/expanded_response_bank/development")


def prepare(root: Path) -> dict:
    selection = verify_stage(root, "probe_selection_and_metric")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    payload = {
        "jvp_base_count": len(design["state_roles"]["jvp_rank_development"]),
        "jvp_operator_state_count": 2 * len(design["state_roles"]["jvp_rank_development"]),
        "finite_base_count": len(design["state_roles"]["finite_rank_development"]),
        "finite_operator_state_count": 2 * len(design["state_roles"]["finite_rank_development"]),
        "probe_count": 512, "target_dimension": 288,
        "jvp_method": "torch.func.jvp exact forward AD on BF16-writeback action path",
        "finite_primary": "actual signed response at each action's smallest symmetrically reliable calibrated alpha",
        "finite_secondary": "2x calibrated alpha on frozen first-64 panel",
        "v22_common_128_finite_reused_without_remeasurement": True,
        "probe_selection_digest": selection["freeze_digest"],
        "historical_final_responses_opened": False, "v23_independent_final_responses_opened": False,
    }
    target = root / OUT / "rank_measurement_design_v23.json"
    write_json_atomic(target, payload)
    frozen = stage_freeze(root, "rank_measurement_design", [SOURCE,
        "artifacts/oracle_local_action_charts_v23_probe_selection_and_metric.freeze.json",
        "results/v23/processed/probe_candidate_design_v23.json", str(target)], payload)
    return {"freeze_digest": frozen["freeze_digest"], **payload}


def _setup(root: Path):
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    selection = json.loads((root / OUT / "probe_selection_v23.json").read_text())
    split = json.loads((root / "artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = v20_bank.prompt_index(root)
    teachers = v20_response._teacher_map(root)
    q_lookup = {item["name"]: item for item in operator["q"]}
    specs = {item["action_id"]: item for item in selection["action_specs"]}
    jids_np = np.asarray(split["selected_j"], dtype=int)
    lids_np = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    device = next(bundle.hf_model.parameters()).device
    jids = torch.as_tensor(jids_np, device=device, dtype=torch.long)
    lids = torch.as_tensor(lids_np, device=device, dtype=torch.long)
    return design, selection, prompts, teachers, q_lookup, specs, jids_np, lids_np, jids, lids, scales, bundle, dense, values, rec, att, measured, state_layer, ws_layers, ws_count


def _fast_jvp(bundle, dense, state, prompt_length, token, jids, lids, ws_layers, ws_count, main, row, rec, att, scales):
    device = next(bundle.hf_model.parameters()).device
    def target(epsilon):
        return paired._torch_stack(paired._target(bundle, dense, state, prompt_length, token, jids, lids,
                                                  ws_layers, ws_count, main, row, rec, att, epsilon), scales)
    epsilon = torch.zeros((), device=device, dtype=torch.float32)
    _, derivative = torch.func.jvp(target, (epsilon,), (torch.ones_like(epsilon),))
    return derivative.detach().cpu().numpy().astype(np.float32)


def _evaluate(bundle, dense, cache, token, prompt_length, jids, lids, ws_layers, ws_count, main, scales):
    _, endpoints = v19._trajectory(bundle, dense, cache, None, [token], 1, prompt_length,
                                   jids, lids, ws_layers, ws_count, main)
    return stack(endpoints[1], scales).astype(np.float32)


def run_jvp(root: Path, limit: int | None = None) -> dict:
    verify_stage(root, "rank_measurement_design")
    (design, selection, prompts, teachers, q_lookup, specs, jids_np, lids_np, jids, lids, scales,
     bundle, dense, values, rec, att, measured, state_layer, ws_layers, ws_count) = _setup(root)
    directions = values["directions"]
    items = design["state_roles"]["jvp_rank_development"][:limit]
    action_ids = selection["train_action_ids"]
    v21_roles = json.loads((root / "artifacts/action_coordinate_geometry_v21_roles.freeze.json").read_text())
    v21_scales = json.loads((root / "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json").read_text())
    old_indices = [int(x) for x in v21_roles["jvp_probe_direction_indices"]]
    completed = []
    for base_number, item in enumerate(items, 1):
        target = root / SCRATCH / "rank_jvp" / f"jvp_{item['base_trial_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"V23 JVP base={base_number}/{len(items)} exists", flush=True)
            completed.append(target); continue
        token = int(teachers[item["base_trial_id"]][0])
        clean = v19._prefill_history(bundle, str(prompts[item["base_trial_id"]]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        q = q_lookup[item["q_name"]]
        pq = v19.apply(p0, 1.0, v19._q_row(values["directions"], q, float(q["alpha"])), rec, att,
                       "native_fp32_add_bf16_writeback")
        old_path = root / V21_RAW / f"paired_{item['base_trial_id']}.npz"
        old = np.load(old_path)
        matrices = {}
        for state_name, state in (("P0", p0), ("Pq", pq)):
            columns = []
            reused = 0
            for action_number, action_id in enumerate(action_ids, 1):
                spec = specs[action_id]
                alpha = float(selection["action_alphas"][action_id])
                reuse_column = None
                if spec["kind"] == "direction" and int(spec["direction_index"]) in old_indices:
                    index = int(spec["direction_index"])
                    old_alpha = v21_scales["selected_alpha_by_direction"].get(str(index))
                    if old_alpha is not None and abs(float(old_alpha) - alpha) < 1e-12:
                        reuse_column = old_indices.index(index)
                if reuse_column is not None:
                    columns.append(np.asarray(old[f"{state_name}_JVP"][:, reuse_column], dtype=np.float32)); reused += 1
                else:
                    row = actions._row(directions, spec, alpha, 1)
                    columns.append(_fast_jvp(bundle, dense, state, clean["prompt_length"], token, jids, lids,
                                             ws_layers, ws_count, max(measured), row, rec, att, scales))
                if action_number % 64 == 0:
                    print(f"V23 JVP base={base_number}/{len(items)} state={state_name} action={action_number}/512 reused={reused}", flush=True)
            matrices[f"{state_name}_JVP"] = np.stack(columns, axis=1)
            matrices[f"{state_name}_reused_v21"] = np.asarray(reused)
        old.close()
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **matrices, action_ids=np.asarray(action_ids), base_trial_id=np.asarray(item["base_trial_id"]),
                            family=np.asarray(item["family"]), q_name=np.asarray(item["q_name"]),
                            p0_snapshot_sha256=np.asarray(v19._snapshot_hash(p0, rec, att)),
                            pq_snapshot_sha256=np.asarray(v19._snapshot_hash(pq, rec, att)))
        os.replace(temporary, target)
        completed.append(target)
    return {"completed": len(completed), "requested": len(items), "operator_states": 2 * len(completed)}


def run_finite(root: Path, limit: int | None = None) -> dict:
    verify_stage(root, "rank_measurement_design")
    (design, selection, prompts, teachers, q_lookup, specs, jids_np, lids_np, jids, lids, scales,
     bundle, dense, values, rec, att, measured, state_layer, ws_layers, ws_count) = _setup(root)
    directions = values["directions"]
    items = design["state_roles"]["finite_rank_development"][:limit]
    action_ids = selection["train_action_ids"]
    completed = []
    for base_number, item in enumerate(items, 1):
        target = root / SCRATCH / "rank_finite" / f"finite_{item['base_trial_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"V23 finite base={base_number}/{len(items)} exists", flush=True)
            completed.append(target); continue
        token = int(teachers[item["base_trial_id"]][0])
        clean = v19._prefill_history(bundle, str(prompts[item["base_trial_id"]]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        q = q_lookup[item["q_name"]]
        pq = v19.apply(p0, 1.0, v19._q_row(values["directions"], q, float(q["alpha"])), rec, att,
                       "native_fp32_add_bf16_writeback")
        v22 = np.load(root / V22_RAW / f"expanded_{item['base_trial_id']}.npz")
        v22_ids = [str(x) for x in v22["action_ids"]]
        arrays = {}
        for state_name, state in (("P0", p0), ("Pq", pq)):
            baseline = _evaluate(bundle, dense, state, token, clean["prompt_length"], jids_np, lids_np,
                                 ws_layers, ws_count, max(measured), scales)
            plus, minus = [], []
            for action_number, action_id in enumerate(action_ids, 1):
                if action_id in v22_ids:
                    column = v22_ids.index(action_id)
                    plus.append(np.asarray(v22[f"{state_name}_plus"][column], dtype=np.float32))
                    minus.append(np.asarray(v22[f"{state_name}_minus"][column], dtype=np.float32))
                else:
                    row = actions._row(directions, specs[action_id], float(selection["action_alphas"][action_id]), 1)
                    positive = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    negative = v19.apply(state, -1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    plus.append(_evaluate(bundle, dense, positive, token, clean["prompt_length"], jids_np, lids_np,
                                          ws_layers, ws_count, max(measured), scales) - baseline)
                    minus.append(_evaluate(bundle, dense, negative, token, clean["prompt_length"], jids_np, lids_np,
                                           ws_layers, ws_count, max(measured), scales) - baseline)
                if action_number % 64 == 0:
                    print(f"V23 finite base={base_number}/{len(items)} state={state_name} action={action_number}/512", flush=True)
            arrays[f"{state_name}_plus"] = np.stack(plus)
            arrays[f"{state_name}_minus"] = np.stack(minus)
            secondary_plus, secondary_minus = [], []
            for action_number, action_id in enumerate(action_ids[:64], 1):
                row = actions._row(directions, specs[action_id], 2.0 * float(selection["action_alphas"][action_id]), 1)
                positive = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                negative = v19.apply(state, -1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                secondary_plus.append(_evaluate(bundle, dense, positive, token, clean["prompt_length"], jids_np, lids_np,
                                                ws_layers, ws_count, max(measured), scales) - baseline)
                secondary_minus.append(_evaluate(bundle, dense, negative, token, clean["prompt_length"], jids_np, lids_np,
                                                 ws_layers, ws_count, max(measured), scales) - baseline)
            arrays[f"{state_name}_secondary_plus"] = np.stack(secondary_plus)
            arrays[f"{state_name}_secondary_minus"] = np.stack(secondary_minus)
        v22.close()
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays, action_ids=np.asarray(action_ids), secondary_action_ids=np.asarray(action_ids[:64]),
                            base_trial_id=np.asarray(item["base_trial_id"]), family=np.asarray(item["family"]), q_name=np.asarray(item["q_name"]),
                            p0_snapshot_sha256=np.asarray(v19._snapshot_hash(p0, rec, att)),
                            pq_snapshot_sha256=np.asarray(v19._snapshot_hash(pq, rec, att)))
        os.replace(temporary, target)
        completed.append(target)
    return {"completed": len(completed), "requested": len(items), "operator_states": 2 * len(completed)}


def summarize(root: Path) -> dict:
    verify_stage(root, "rank_measurement_design")
    design = json.loads((root / OUT / "probe_candidate_design_v23.json").read_text())
    rows = []
    for kind, role in (("jvp", "jvp_rank_development"), ("finite", "finite_rank_development")):
        for item in design["state_roles"][role]:
            path = root / SCRATCH / f"rank_{kind}" / f"{kind}_{item['base_trial_id']}.npz"
            if not path.exists():
                raise RuntimeError(f"missing V23 {kind} file: {path}")
            rows.append({"kind": kind, **item, "path": str(path), "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    parquet = root / OUT / "rank_measurement_index_v23.parquet"
    frame.to_parquet(parquet, index=False, compression="zstd")
    summary = {"jvp_base_count": int((frame.kind == "jvp").sum()), "jvp_operator_state_count": 2 * int((frame.kind == "jvp").sum()),
               "finite_base_count": int((frame.kind == "finite").sum()), "finite_operator_state_count": 2 * int((frame.kind == "finite").sum()),
               "probe_count": 512, "index_sha256": sha256_file(parquet),
               "historical_final_responses_opened": False, "v23_independent_final_responses_opened": False}
    write_json_atomic(root / OUT / "rank_measurement_summary_v23.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "jvp", "finite", "summarize"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    answer = {"prepare": lambda: prepare(Path.cwd()), "jvp": lambda: run_jvp(Path.cwd(), args.limit),
              "finite": lambda: run_finite(Path.cwd(), args.limit), "summarize": lambda: summarize(Path.cwd())}[args.stage]()
    print(json.dumps(answer, indent=2))
