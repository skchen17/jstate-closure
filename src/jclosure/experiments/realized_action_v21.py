"""Z3 state-dependent BF16 readback coordinates from raw REC/Conv/KV deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.experiments.raw_state_reference_v21 import _flat
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/realized_action_v21.py"
SCRATCH = bank.SCRATCH / "realized_action_v21.npz"
SUMMARY = bank.OUT / "realized_action_v21.json"


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    if len(roles["train_actions"]) != 12 or len(roles["validation_actions"]) != 6:
        raise RuntimeError("V21 Z3 V20 action roles drift")
    return stage_freeze(root, "realized_action_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json"],
                        {"Z3": "actual_BF16_P_after_minus_P_before_architecture_resolved_Nystrom_coordinate",
                         "anchor_base_id": roles["raw_Nystrom_anchor_base_ids"][0],
                         "anchor_action_coordinates": [x["coordinate_index"] for x in roles["train_actions"]],
                         "anchor_alpha": [x["base_alpha"] for x in roles["train_actions"]],
                         "anchor_support": "12_train_actions_on_one_train_P0_state_only",
                         "feature_rule": "per_channel_exact_raw_realized_delta_inner_products_against_12_train_anchor_deltas_plus_raw_delta_norm",
                         "feature_dimension": 39,
                         "train": "600_train_P0/Pq_states_x_12_positive_train_actions",
                         "validation": "200_validation_P0/Pq_states_x_18_open_train_or_validation_actions_x_both_signs",
                         "no_validation_response_labels_used": True,
                         "not_full_raw_action_embedding": "39D train-anchor Nyström representation of realized delta; kernel approximation caveat",
                         "final_six_action_responses_opened": False})


def _delta(base: dict, post: dict) -> dict[str, torch.Tensor]:
    return {name: post[name]-base[name] for name in ("REC", "Conv", "KV")}


def run(root: Path) -> dict:
    design = verify_stage(root, "realized_action_design")
    if SCRATCH.exists():
        raise RuntimeError("V21 Z3 scratch already exists; append-only policy")
    op = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = v20_bank.prompt_index(root)
    actions = op["shared_measured_actions"]
    if [x["coordinate_index"] for x in actions[:12]] != design["anchor_action_coordinates"]:
        raise RuntimeError("V21 Z3 action order drift")
    q_lookup = {x["name"]: x for x in op["q"]}
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    with np.load(bank.SCRATCH / "state_representations_v21.npz", allow_pickle=False) as source:
        ids = {"train": source["train_ids"], "validation": source["validation_ids"]}
    with torch.no_grad():
        anchor_id = design["anchor_base_id"]
        p0 = v19._prefill_history(bundle, str(prompts[anchor_id]["prompt"]), measured, dense, state_layer)["cache"]
        anchor_p0_hash = v19._snapshot_hash(p0, rec, att)
        raw0 = _flat(p0, rec, att)
        bases: dict[str, list[torch.Tensor]] = {name: [] for name in ("REC", "Conv", "KV")}
        anchor_hashes = []
        for action in actions[:12]:
            row = v19._action_row(values["directions"], int(action["direction_index"]), float(action["base_alpha"]), 1)
            edited = v19.apply(p0, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
            diff = _delta(raw0, _flat(edited, rec, att))
            sha = hashlib.sha256()
            for name in bases:
                bases[name].append(diff[name])
                sha.update(diff[name].cpu().numpy().tobytes())
            anchor_hashes.append(sha.hexdigest())
        global_norm = torch.median(torch.stack([torch.sqrt(sum(torch.sum(bases[name][i]**2)
                                                                     for name in bases)) for i in range(12)]))
        basis = {}
        for name in bases:
            matrix = torch.stack(bases[name])
            median = torch.median(torch.linalg.vector_norm(matrix, dim=1))
            floor = torch.maximum(median, global_norm/4).clamp_min(1e-8)
            basis[name] = (matrix, floor)
        output = {}
        state_count = {}
        for role in ("train", "validation"):
            by_id = {}
            for state_id in ids[role]:
                base_id, qname = str(state_id).split("::", 1)
                by_id.setdefault(base_id, []).append(qname)
            encoded = {}
            for number, (base_id, qnames) in enumerate(by_id.items(), 1):
                p0 = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)["cache"]
                for qname in qnames:
                    if qname == "P0":
                        state = p0
                    else:
                        q = q_lookup[qname]
                        row = v19._q_row(values["directions"], q, float(q["alpha"]))
                        state = v19.apply(p0, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    raw_state = _flat(state, rec, att)
                    cells = []
                    for action in actions[:12] if role == "train" else actions:
                        signs = (1,) if role == "train" else (1,-1)
                        sign_cells = []
                        for sign in signs:
                            row = v19._action_row(values["directions"], int(action["direction_index"]), float(action["base_alpha"]), sign)
                            edited = v19.apply(state, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                            delta = _delta(raw_state, _flat(edited, rec, att))
                            feature = []
                            for name in ("REC", "Conv", "KV"):
                                anchors, scale = basis[name]
                                feature.extend(((anchors @ delta[name]) / (scale*scale)).cpu().tolist())
                                feature.append(float(torch.linalg.vector_norm(delta[name])/scale))
                            sign_cells.append(feature)
                        cells.append(sign_cells[0] if role == "train" else sign_cells)
                    encoded[f"{base_id}::{qname}"] = cells
                if number % 5 == 0 or number == len(by_id):
                    print(f"V21 Z3 {role} base {number}/{len(by_id)}", flush=True)
            if set(encoded) != set(ids[role].astype(str)):
                raise RuntimeError("V21 Z3 state ids mismatch")
            key = f"Z3_{role}"
            output[key] = np.asarray([encoded[str(x)] for x in ids[role]], dtype=np.float32)
            state_count[role] = len(ids[role])
        SCRATCH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SCRATCH, **output, train_ids=ids["train"], validation_ids=ids["validation"])
    result = {"design_digest": design["freeze_digest"], "anchor_realized_delta_hashes": anchor_hashes,
              "anchor_P0_snapshot_sha256": anchor_p0_hash,
              "train_operator_states": state_count["train"], "validation_operator_states": state_count["validation"],
              "Z3_train_shape": list(output["Z3_train"].shape),
              "Z3_validation_shape": list(output["Z3_validation"].shape),
              "Z3_scratch_path": str(SCRATCH), "Z3_scratch_sha256": sha256_file(SCRATCH),
              "final_six_action_responses_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    freeze = stage_freeze(root, "realized_action_representations",
                          [SOURCE, "artifacts/action_coordinate_geometry_v21_realized_action_design.freeze.json", str(SUMMARY)],
                          {"Z3_scratch_sha256": result["Z3_scratch_sha256"],
                           "validation_response_labels_used_for_Z3": False,
                           "final_six_action_responses_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "Z3_train_shape": result["Z3_train_shape"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd()), indent=2))
