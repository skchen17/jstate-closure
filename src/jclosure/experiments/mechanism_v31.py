"""Exact REC×Conv factorial and Conv-depth causal profiles on V31 panels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import metrics, profile, signature
from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v31 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/mechanism_v31.py"
OUT = Path("results/v31/processed")
CONV = [i for i in range(32) if i % 4 != 3]


def hd(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na * nb > 1e-12 else None


def conditions(groups):
    selected = {"FULL_CONV": CONV}
    for kind, arrays in groups.items():
        for i, layers in enumerate(arrays):
            selected[f"{kind.upper()}_{i:02d}"] = layers
    return selected


def select_route(frame, gate, groups):
    candidates = [(k, v) for k, v in conditions(groups).items() if k == "FULL_CONV" or k.startswith(("QUARTILE", "HALF", "PREFIX", "SUFFIX"))]
    candidates.sort(key=lambda x: (len(x[1]), hd(x[0])))
    profiles = profile(frame, gate)
    passing = [(name, layers) for name, layers in candidates if profiles[name]["pass"]]
    name, layers = passing[0] if passing else ("FULL_CONV", CONV)
    return {"name": name, "layers": layers, "layer_hash": hd(layers), "development_selected": True, "fallback_full24": not bool(passing), "candidate_profiles": {k: profiles[k] for k, _ in candidates}}


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "execution_plan")
    if role == "validation":
        verify_stage(root, "mechanism_development")
    design = json.loads((root / OUT / "design_v31.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v31.json").read_text())
    cfg = verify(root)["config"]
    by_id = {x["base_trial_id"]: x for x in design[role]}
    pairs = {x["candidate_token_id"]: x for x in design["token_pair_library"]}
    comps = {x["composition_id"]: x for group in design["surface_composition_splits"].values() for x in group}
    bundle, dense, *_ = context(root)
    scale = scales(root)
    depth, fact, tensor = [], [], []
    groups = conditions(plan["depth_group_layers"])
    selected = None if role == "development" else json.loads((root / OUT / "depth_route_selection_v31.json").read_text())
    for n, sid in enumerate(plan["rec_conv_factorial_state_ids"][role], 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if state_hashes(incoming, rec, att) != item["incoming_state_hashes"]:
            raise RuntimeError(f"incoming state drift: {sid}")
        anchor = step(bundle, dense, incoming, token(cfg["token_library"]["anchor_token_id"]), length, design, 30)
        probes = [token(tid) for tid in item["future_probe_tokens"]]
        def response(cache):
            return signature([step(bundle, dense, cache, z, length + 1, design, 30) for z in probes], scale)
        for cid in plan["heldout_composition_eval"][role][sid]:
            comp = comps[cid]
            donor = step(bundle, dense, incoming, token(comp["AB"]), length, design, 30)
            label = pairs[comp["AB"]]["category"]
            y00 = response(anchor["cache"])
            yd = response(donor["cache"])
            y10 = response(partial(anchor["cache"], donor["cache"], rec, (), rec))
            y01 = response(partial(anchor["cache"], donor["cache"], rec, rec, ()))
            y11 = response(partial(anchor["cache"], donor["cache"], rec, rec, rec))
            direction = yd - y00
            conv_move, rec_move, joint_move = y01 - y00, y10 - y00, y11 - y00
            residual, rec_given_conv = yd - y01, y11 - y01
            interaction = y11 - y10 - y01 + y00
            dnorm = max(float(np.linalg.norm(direction)), 1e-12)
            enorm = max(float(np.linalg.norm(residual)), 1e-12)
            base = {"role": role, "state_id": sid, "family": item["family"], "composition_id": cid, "token_category": label, "incoming_state_hash": item["incoming_state_hash"], "probe_hash": item["future_probe_hash"], "six_frozen_probes": True, "exact_native_fields": True}
            fact.append({**base, "conv_relative_l2": float(np.linalg.norm(yd - y01)) / dnorm, "joint_relative_l2": float(np.linalg.norm(yd - y11)) / dnorm, "rec_only_relative_l2": float(np.linalg.norm(yd - y10)) / dnorm, "paired_l2_improvement": float(np.linalg.norm(yd - y01) - np.linalg.norm(yd - y11)) / dnorm, "residual_alignment_cosine": cosine(rec_given_conv, residual), "residual_projection_fraction": float(np.dot(rec_given_conv, residual)) / (enorm * enorm), "rec_given_conv_norm_ratio": float(np.linalg.norm(rec_given_conv)) / dnorm, "interaction_ratio": float(np.linalg.norm(interaction)) / dnorm, "interaction_donor_cosine": cosine(interaction, direction), "conv_donor_cosine": cosine(conv_move, direction), "joint_donor_cosine": cosine(joint_move, direction), "rec_gain_ratio": float(np.linalg.norm(joint_move)) / max(float(np.linalg.norm(conv_move)), 1e-12), "rec_direction_improvement": (cosine(joint_move, direction) or 0) - (cosine(conv_move, direction) or 0), "rec_only_donor_cosine": cosine(rec_move, direction), "donor_direction_norm": dnorm})
            for layer in rec:
                a = anchor["cache"].layers[layer].conv_states.float()
                b = donor["cache"].layers[layer].conv_states.float()
                tensor.append({**base, "layer": layer, "conv_write_norm": float(torch.linalg.vector_norm(b - a).item()), "conv_native_norm": float(torch.linalg.vector_norm(a).item()), "conv_donor_norm": float(torch.linalg.vector_norm(b).item())})
            for condition, layers in groups.items():
                sig = response(partial(anchor["cache"], donor["cache"], rec, layers, ()))
                depth.append({**base, "condition": condition, "conv_layers": json.dumps(layers), "conv_layer_set_hash": hd(layers), **metrics(sig, y00, yd)})
            del donor
        print(f"V31 REC×Conv/depth {role} {n}/{len(plan['rec_conv_factorial_state_ids'][role])}", flush=True)
    depth_frame, fact_frame, tensor_frame = pd.DataFrame(depth), pd.DataFrame(fact), pd.DataFrame(tensor)
    paths = {"depth": root / OUT / f"conv_depth_{role}_v31.parquet", "factorial": root / OUT / f"rec_conv_factorial_{role}_v31.parquet", "tensor": root / OUT / f"conv_depth_tensor_{role}_v31.parquet"}
    depth_frame.to_parquet(paths["depth"], index=False, compression="zstd")
    fact_frame.to_parquet(paths["factorial"], index=False, compression="zstd")
    tensor_frame.to_parquet(paths["tensor"], index=False, compression="zstd")
    if role == "development":
        selected = select_route(depth_frame, cfg["causal_gate"], plan["depth_group_layers"])
        write_json_atomic(root / OUT / "depth_route_selection_v31.json", selected)
    summary = {"role": role, "states": len(plan["rec_conv_factorial_state_ids"][role]), "token_pairs": len(fact_frame), "depth_conditions": len(groups), "depth_profiles": profile(depth_frame, cfg["causal_gate"]), "rec_improves_count": int((fact_frame.paired_l2_improvement > 0).sum()), "median_residual_alignment_cosine": float(fact_frame.residual_alignment_cosine.median()), "median_interaction_ratio": float(fact_frame.interaction_ratio.median()), "median_rec_projection_fraction": float(fact_frame.residual_projection_fraction.median()), "route_selection": selected, "response_sha256": {k: sha256_file(v) for k, v in paths.items()}, "independent_final_opened": False}
    summary_path = root / OUT / f"mechanism_{role}_v31.json"
    write_json_atomic(summary_path, summary)
    inputs = [SOURCE, *(str(path.relative_to(root)) for path in paths.values()), str(summary_path.relative_to(root)), "artifacts/compositional_natural_writes_v31_execution_plan.freeze.json"]
    if role == "development":
        inputs.append(str((OUT / "depth_route_selection_v31.json")))
    freeze = stage_freeze(root, f"mechanism_{role}", inputs, {"summary_sha256": sha256_file(summary_path), "route_hash": selected["layer_hash"], "independent_final_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "states": summary["states"], "pairs": summary["token_pairs"], "rec_improves_count": summary["rec_improves_count"], "median_residual_alignment_cosine": summary["median_residual_alignment_cosine"], "route_selection": {k: selected[k] for k in ("name", "layers", "fallback_full24")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
