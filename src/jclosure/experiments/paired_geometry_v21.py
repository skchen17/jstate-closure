"""Paired exact-JVP and central finite operators on the frozen V21 probe panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments import response_operator_v20 as v20_response
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.operator_v15 import TARGETS, stack
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

SOURCE = "src/jclosure/experiments/paired_geometry_v21.py"
RAW = bank.SCRATCH / "paired_geometry"
SUMMARY = bank.OUT / "paired_jvp_finite_operator_v21.json"
ROWS = bank.OUT / "paired_jvp_finite_operator_v21.parquet"


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    scales = verify_stage(root, "probe_scales")
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    q_lookup = {x["name"]: x for x in operator["q"]}
    if not {x["q_name"] for role in ("jvp_development", "jvp_validation") for x in roles[role]} <= set(q_lookup):
        raise RuntimeError("V21 Pq assignment differs from V20 frozen q bank")
    return stage_freeze(root, "paired_geometry_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/action_coordinate_geometry_v21_probe_scales.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json"],
                        {"development_base_states": 50, "validation_base_states": 25,
                         "persistent_states_per_base": ["P0", "one_V20_Pq"],
                         "probe_direction_indices": roles["jvp_probe_direction_indices"],
                         "probe_sha256": roles["jvp_probe_sha256"],
                         "probe_alpha_by_direction": scales["selected_alpha_by_direction"],
                         "target": "V20_h1_selected_J128_logits32_logsoftmax32_workspace96_normalized_stack288",
                         "teacher_source": "V20_frozen_teacher_tokens", "jvp": "exact_autograd_forward_mode_via_torch.autograd.functional.jvp_on_BF16_writeback",
                         "finite": "central_signed_(Y(P+a)-Y(P-a))/2_on_identical_probes",
                         "rank_energy_thresholds": [0.90, 0.95, 0.99],
                         "raw_matrix_storage": "scratch_outside_git_repository",
                         "final_six_action_responses_opened": False})


def _target(bundle, dense, base_cache, prompt_length, token, jids, lids, ws_layers, ws_count, main,
            row, rec, att, epsilon):
    device = next(bundle.hf_model.parameters()).device
    cache = _apply_direction(base_cache, epsilon, row, rec, att)
    with ActivationRecorder(bundle.layers, at=ws_layers, clone=False, detach=False) as recorder:
        output = bundle.hf_model(input_ids=torch.tensor([[token]], device=device),
                                 attention_mask=torch.ones((1, prompt_length + 1), device=device, dtype=torch.long),
                                 past_key_values=cache, use_cache=True)
    logits = output.logits[0, -1].float()
    hidden = recorder.activations[main][0, -1].float()
    j = dense.dense_state(hidden, main)
    workspace = torch.cat([recorder.activations[layer][0, -1].float()[:ws_count] for layer in ws_layers])
    return {"j": j[jids], "logits": logits[lids],
            "semantic_continuous": torch.log_softmax(logits, dim=-1)[lids],
            "workspace": workspace}


def _torch_stack(parts, scales):
    return torch.cat([parts[key] / (scales[key] * math.sqrt(parts[key].numel())) for key in TARGETS])


def _one_jvp(bundle, dense, state, prompt_length, token, jids, lids, ws_layers, ws_count, main,
             row, rec, att, scales):
    device = next(bundle.hf_model.parameters()).device
    def target(epsilon):
        return _torch_stack(_target(bundle, dense, state, prompt_length, token, jids, lids,
                                    ws_layers, ws_count, main, row, rec, att, epsilon), scales)
    epsilon = torch.zeros((), device=device, dtype=torch.float32)
    _, derivative = torch.autograd.functional.jvp(target, epsilon, torch.ones_like(epsilon),
                                                   create_graph=False, strict=True)
    return derivative.detach().cpu().numpy().astype(np.float32)


def _finite(bundle, dense, state, prompt_length, token, jids, lids, ws_layers, ws_count, main,
            row, rec, att, scales):
    plus = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
    minus = v19.apply(state, -1.0, row, rec, att, "native_fp32_add_bf16_writeback")
    def evaluate(cache):
        _, endpoints = v19._trajectory(bundle, dense, cache, None, [token], 1, prompt_length,
                                       jids, lids, ws_layers, ws_count, main)
        return stack(endpoints[1], scales).astype(np.float32)
    return (evaluate(plus) - evaluate(minus)) / 2.0


def _matrix_path(role: str, base_id: str) -> Path:
    return RAW / role / f"paired_{base_id}.npz"


def run(root: Path, role: str, limit: int | None = None) -> dict:
    if role not in ("development", "validation"):
        raise RuntimeError("unknown V21 paired geometry role")
    design = verify_stage(root, "paired_geometry_design")
    roles = verify_stage(root, "roles")
    split = json.loads((root / "artifacts/compact_causal_response_operator_v20_splits.freeze.json").read_text())
    operator = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = v20_bank.prompt_index(root)
    teachers = v20_response._teacher_map(root)
    q_lookup = {x["name"]: x for x in operator["q"]}
    jids_np = np.asarray(split["selected_j"], dtype=int)
    lids_np = np.asarray(split["selected_logits"], dtype=int)
    scales = {key: float(value) for key, value in split["target_scales"].items()}
    bundle, dense, _, values, rec, att, measured, state_layer, ws_layers, ws_count = v19._setup(root)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    device = next(bundle.hf_model.parameters()).device
    jids = torch.as_tensor(jids_np, device=device, dtype=torch.long)
    lids = torch.as_tensor(lids_np, device=device, dtype=torch.long)
    items = roles[f"jvp_{role}"]
    if limit is not None:
        items = items[:limit]
    results = []
    for count, item in enumerate(items, 1):
        base_id = item["base_trial_id"]
        target = _matrix_path(role, base_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"V21 paired {role} {count}/{len(items)} exists", flush=True)
            results.append(str(target))
            continue
        token = int(teachers[base_id][0])
        clean = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
        p0 = clone_hybrid_cache(clean["cache"])
        q = q_lookup[item["q_name"]]
        qrow = v19._q_row(values["directions"], q, float(q["alpha"]))
        pq = v19.apply(p0, 1.0, qrow, rec, att, "native_fp32_add_bf16_writeback")
        matrices = {}
        for name, state in (("P0", p0), ("Pq", pq)):
            jvp_cols, finite_cols = [], []
            for index in design["probe_direction_indices"]:
                alpha = design["probe_alpha_by_direction"][str(index)]
                if alpha is None:
                    raise RuntimeError("unreliable probe has no calibrated scale")
                row = v19._action_row(values["directions"], int(index), float(alpha), 1)
                jvp_cols.append(_one_jvp(bundle, dense, state, clean["prompt_length"], token,
                                         jids, lids, ws_layers, ws_count, max(measured), row, rec, att, scales))
                finite_cols.append(_finite(bundle, dense, state, clean["prompt_length"], token,
                                           jids_np, lids_np, ws_layers, ws_count, max(measured), row, rec, att, scales))
            matrices[f"{name}_JVP"] = np.stack(jvp_cols, axis=1)
            matrices[f"{name}_finite"] = np.stack(finite_cols, axis=1)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **matrices,
                            probe_indices=np.asarray(design["probe_direction_indices"], dtype=np.int32),
                            base_trial_id=np.asarray(base_id), family=np.asarray(item["family"]),
                            q_name=np.asarray(item["q_name"]), probe_sha256=np.asarray(design["probe_sha256"]),
                            p0_snapshot_sha256=np.asarray(v19._snapshot_hash(p0, rec, att)),
                            pq_snapshot_sha256=np.asarray(v19._snapshot_hash(pq, rec, att)))
        os.replace(temporary, target)
        results.append(str(target))
        print(f"V21 paired {role} {count}/{len(items)} {base_id}", flush=True)
    return {"role": role, "completed": len(results), "requested": len(items)}


def summarize(root: Path) -> dict:
    design = verify_stage(root, "paired_geometry_design")
    roles = verify_stage(root, "roles")
    rows = []
    for role in ("development", "validation"):
        for item in roles[f"jvp_{role}"]:
            path = _matrix_path(role, item["base_trial_id"])
            if not path.exists():
                raise RuntimeError(f"V21 paired geometry missing: {path}")
            rows.append({"role": role, **item, "path": str(path), "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    target = root / ROWS
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    result = {"design_digest": design["freeze_digest"], "probe_sha256": design["probe_sha256"],
              "development_base_states": int((frame.role == "development").sum()),
              "validation_base_states": int((frame.role == "validation").sum()),
              "raw_matrix_index_sha256": sha256_file(target), "final_six_action_responses_opened": False}
    write_json_atomic(root / SUMMARY, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--role", choices=("development", "validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "run" and args.role is None:
        parser.error("--role required for run")
    result = {"prepare": lambda: prepare(Path.cwd()),
              "run": lambda: run(Path.cwd(), args.role, args.limit),
              "summarize": lambda: summarize(Path.cwd())}[args.stage]()
    print(json.dumps(result, indent=2))
