"""S4 architecture-resolved raw persistent-state Nyström ceiling reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jclosure.experiments import bank_v21 as bank
from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments import operator_bank_v20 as v20_bank
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/raw_state_reference_v21.py"
SCRATCH = bank.SCRATCH / "raw_state_reference_v21.npz"
SUMMARY = bank.OUT / "raw_state_reference_v21.json"


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    ids = roles["raw_Nystrom_anchor_base_ids"]
    if len(ids) != 30 or len(set(ids)) != 30:
        raise RuntimeError("V21 raw state anchors invalid")
    return stage_freeze(root, "raw_state_reference_design",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                         "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json"],
                        {"S4": "architecture_resolved_REC_Conv_KV_raw_persistent_state_train_only_Nystrom_reference",
                         "anchor_base_ids": ids, "anchor_sha256": roles["raw_anchor_id_sha256"],
                         "anchor_states": "30_V20_operator_train_P0_only; no_validation_or_final_anchor",
                         "feature_rule": "per_channel_centered_raw_state_Gram_against_30_anchors_plus_centered_raw_norm",
                         "feature_dimension": 93,
                         "KV_alignment": "first_256_cache_positions_zero_padded_or_truncated_to_frozen_V13_KV_direction_support",
                         "state_roles": "600_train_and_200_validation_V20_operator_P0/Pq_states",
                         "not_compact_interpretation": "Nyström approximation to architecture-resolved raw-state kernel; not a learned compact sufficient state",
                         "response_labels_used": False,
                         "final_six_action_responses_opened": False})


def _flat(cache, rec: list[int], att: list[int]) -> dict[str, torch.Tensor]:
    recurrent = torch.cat([cache.layers[layer].recurrent_states.detach().float().reshape(-1) for layer in rec])
    conv = torch.cat([cache.layers[layer].conv_states.detach().float().reshape(-1) for layer in rec])
    kv = []
    for layer in att:
        for name in ("keys", "values"):
            value = getattr(cache.layers[layer], name).detach().float()
            value = value[..., :256, :]
            if value.shape[-2] < 256:
                value = F.pad(value, (0, 0, 0, 256-value.shape[-2]))
            kv.append(value.reshape(-1))
    return {"REC": recurrent, "Conv": conv, "KV": torch.cat(kv)}


def run(root: Path) -> dict:
    design = verify_stage(root, "raw_state_reference_design")
    if SCRATCH.exists():
        raise RuntimeError("V21 S4 raw feature scratch already exists; append-only policy")
    op = json.loads((root / "artifacts/compact_causal_response_operator_v20_operator_design.freeze.json").read_text())
    prompts = v20_bank.prompt_index(root)
    q_lookup = {x["name"]: x for x in op["q"]}
    bundle, dense, _, values, rec, att, measured, state_layer, _, _ = v19._setup(root)
    anchor_vectors: dict[str, list[torch.Tensor]] = {name: [] for name in ("REC", "Conv", "KV")}
    anchor_hashes = []
    with torch.no_grad():
        for number, base_id in enumerate(design["anchor_base_ids"], 1):
            clean = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
            cache = clean["cache"]
            parts = _flat(cache, rec, att)
            for name in anchor_vectors:
                anchor_vectors[name].append(parts[name].cpu())
            anchor_hashes.append(v19._snapshot_hash(cache, rec, att))
            print(f"V21 S4 anchor {number}/30", flush=True)
        anchors = {}
        for name, pieces in anchor_vectors.items():
            raw = torch.stack(pieces).to(next(bundle.hf_model.parameters()).device)
            mean = raw.mean(0)
            centered = raw-mean
            scale = torch.median(torch.linalg.vector_norm(centered, dim=1)).clamp_min(1e-8)
            anchors[name] = (centered, mean, scale)
        with np.load(bank.SCRATCH / "state_representations_v21.npz", allow_pickle=False) as source:
            ids = {"train": source["train_ids"], "validation": source["validation_ids"]}
        output = {}
        state_hashes = {}
        for role in ("train", "validation"):
            by_id = {}
            for state_id in ids[role]:
                base_id, qname = str(state_id).split("::", 1)
                by_id.setdefault(base_id, []).append(qname)
            encoded, hashes = {}, {}
            for number, (base_id, qnames) in enumerate(by_id.items(), 1):
                clean = v19._prefill_history(bundle, str(prompts[base_id]["prompt"]), measured, dense, state_layer)
                p0 = clean["cache"]
                for qname in qnames:
                    if qname == "P0":
                        cache = p0
                    else:
                        q = q_lookup[qname]
                        row = v19._q_row(values["directions"], q, float(q["alpha"]))
                        cache = v19.apply(p0, 1.0, row, rec, att, "native_fp32_add_bf16_writeback")
                    raw = _flat(cache, rec, att)
                    features = []
                    for name in ("REC", "Conv", "KV"):
                        anchor, mean, scale = anchors[name]
                        centered = raw[name] - mean
                        features.extend(((anchor @ centered) / (scale*scale)).cpu().tolist())
                        features.append(float(torch.linalg.vector_norm(centered) / scale))
                    state_key = f"{base_id}::{qname}"
                    encoded[state_key] = features
                    hashes[state_key] = v19._snapshot_hash(cache, rec, att)
                if number % 10 == 0 or number == len(by_id):
                    print(f"V21 S4 {role} base {number}/{len(by_id)}", flush=True)
            if set(encoded) != set(ids[role].astype(str)):
                raise RuntimeError("V21 S4 state id/order mismatch")
            output[f"S4_{role}"] = np.asarray([encoded[str(x)] for x in ids[role]], dtype=np.float32)
            state_hashes[role] = hashes
        SCRATCH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SCRATCH, **output, train_ids=ids["train"], validation_ids=ids["validation"])
    result = {"design_digest": design["freeze_digest"], "state_feature_dimension": 93,
              "train_operator_states": len(output["S4_train"]),
              "validation_operator_states": len(output["S4_validation"]),
              "anchor_P0_snapshot_hashes": anchor_hashes,
              "per_state_snapshot_sha256": bank.digest(state_hashes),
              "S4_feature_scratch_path": str(SCRATCH), "S4_feature_sha256": sha256_file(SCRATCH),
              "not_compact": True, "response_labels_used": False,
              "final_six_action_responses_opened": False}
    target = root / SUMMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    freeze = stage_freeze(root, "raw_state_reference",
                          [SOURCE, "artifacts/action_coordinate_geometry_v21_raw_state_reference_design.freeze.json", str(SUMMARY)],
                          {"raw_state_reference_sha256": result["S4_feature_sha256"],
                           "state_snapshot_digest": result["per_state_snapshot_sha256"],
                           "validation_action_response_labels_used": False,
                           "final_six_action_responses_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "S4_dimension": 93}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd()), indent=2))
