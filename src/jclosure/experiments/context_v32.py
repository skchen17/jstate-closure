"""Matched versus off-context REC with the same frozen target Conv write."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.causal_global_v30 import signature
from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.design_v32 import hd
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/context_v32.py"
CONDITIONS = ("MATCHED_REC", "RECIPIENT_REC", "WRONG_TOKEN_REC", "SAME_FAMILY_WRONG_STATE_REC", "CROSS_FAMILY_WRONG_STATE_REC", "SHUFFLED_REC", "SIGN_FLIPPED_REC", "RANDOM_SAME_NORM_REC")


def mapping(root, role):
    cfg = verify(root)["config"]
    d = json.loads((root / OUT / "design_v32.json").read_text())
    p = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    by_id = {x["base_trial_id"]: x for x in d[role]}
    groups = {family: p["roles"][role][family]["context_state_ids"] for family in cfg["families"]}
    result = {}
    for j, family in enumerate(cfg["families"]):
        ids = groups[family]
        cross_family = cfg["families"][(j + 1) % len(cfg["families"])]
        cross_ids = groups[cross_family]
        for i, sid in enumerate(ids):
            result[sid] = {"same_family_source": ids[(i + 1) % len(ids)], "cross_family_source": cross_ids[i % len(cross_ids)], "wrong_token_id": by_id[sid]["secondary_token_id"]}
    return result


def freeze_mapping(root, role):
    verify_stage(root, "execution_plan")
    payload = {"role": role, "mapping": mapping(root, role), "selection_uses_only_frozen_state_and_token_ids": True, "context_responses_observed_before_mapping": 0}
    path = root / OUT / f"context_mapping_{role}_v32.json"
    write_json_atomic(path, payload)
    stage = stage_freeze(root, f"context_mapping_{role}", [SOURCE, str(path.relative_to(root)), "results/v32/processed/execution_plan_v32.json"], {"mapping_hash": hd(payload["mapping"]), "context_responses_observed_before_mapping": 0})
    return {"freeze_digest": stage["freeze_digest"], "targets": len(payload["mapping"])}


def artificial(c01, a, b, rec, kind, seed):
    cache = clone_hybrid_cache(c01)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for layer in rec:
        x = a.layers[layer].recurrent_states
        y = b.layers[layer].recurrent_states
        if kind == "SHUFFLED_REC":
            z = y.flip(-1)
        elif kind == "SIGN_FLIPPED_REC":
            z = (2 * x.float() - y.float()).to(x.dtype)
        else:
            delta = (y.float() - x.float())
            noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta.device)
            noise = noise * (torch.linalg.vector_norm(delta) / max(float(torch.linalg.vector_norm(noise)), 1e-12))
            z = (x.float() + noise).to(x.dtype)
        cache.layers[layer].recurrent_states = z.detach().clone()
    return cache


@torch.no_grad()
def run(root, role):
    verify_stage(root, f"context_mapping_{role}")
    verify_stage(root, f"factorial_{role}")
    cfg = verify(root)["config"]
    d = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / f"context_mapping_{role}_v32.json").read_text())["mapping"]
    by_id = {x["base_trial_id"]: x for x in d[role]}
    factorial = pd.read_parquet(root / OUT / f"factorial_{role}_v32.parquet")
    vectors = np.load(root / OUT / f"factorial_vectors_{role}_v32.npz")["vectors"]
    bundle, dense, *_ = context(root)
    scale = scales(root)
    rows = []
    for n, (sid, sources) in enumerate(plan.items(), 1):
        item = by_id[sid]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError("V32 context target incoming drift")
        a = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, d, cfg["readout_layer"])
        b = step(bundle, dense, incoming, token(item["primary_token_id"]), length, d, cfg["readout_layer"])
        wrong_token = step(bundle, dense, incoming, token(item["secondary_token_id"]), length, d, cfg["readout_layer"])
        c01 = partial(a["cache"], b["cache"], rec, rec, ())
        condition_caches = {"MATCHED_REC": partial(c01, b["cache"], rec, (), rec), "RECIPIENT_REC": c01, "WRONG_TOKEN_REC": partial(c01, wrong_token["cache"], rec, (), rec)}
        source_hashes = {"WRONG_TOKEN_REC": total_hash(state_hashes(wrong_token["cache"], rec, att))}
        for condition, source_id in (("SAME_FAMILY_WRONG_STATE_REC", sources["same_family_source"]), ("CROSS_FAMILY_WRONG_STATE_REC", sources["cross_family_source"])):
            source = by_id[source_id]
            sm = metadata(root, source)
            sin, _, slength = prefix(bundle, str(sm["prompt"]))
            sb = step(bundle, dense, sin, token(source["primary_token_id"]), slength, d, cfg["readout_layer"])
            condition_caches[condition] = partial(c01, sb["cache"], rec, (), rec)
            source_hashes[condition] = total_hash(state_hashes(sb["cache"], rec, att))
        for j, condition in enumerate(("SHUFFLED_REC", "SIGN_FLIPPED_REC", "RANDOM_SAME_NORM_REC")):
            condition_caches[condition] = artificial(c01, a["cache"], b["cache"], rec, condition, cfg["seed"] + 1000*n + j)
        row = factorial[(factorial.state_id == sid) & factorial.primary].iloc[0]
        y00, _y10, y01, y11, yd = vectors[int(row.vector_index)].astype(np.float64)
        probe_tokens = [token(tid) for tid in item["future_probe_tokens"]]
        for condition in CONDITIONS:
            cache = condition_caches[condition]
            hashes = state_hashes(cache, rec, att)
            if hashes["KV"] != state_hashes(a["cache"], rec, att)["KV"] or hashes["Conv"] != state_hashes(b["cache"], rec, att)["Conv"]:
                raise RuntimeError("V32 context wrong persistent field modified")
            sig = signature([step(bundle, dense, cache, probe, length+1, d, cfg["readout_layer"]) for probe in probe_tokens], scale)
            if condition == "MATCHED_REC" and not np.allclose(sig, y11, rtol=1e-5, atol=1e-5):
                raise RuntimeError("V32 matched context replay drift")
            rows.append({"role": role, "state_id": sid, "family": item["family"], "condition": condition, "source_state_id": sources.get("same_family_source") if condition == "SAME_FAMILY_WRONG_STATE_REC" else sources.get("cross_family_source") if condition == "CROSS_FAMILY_WRONG_STATE_REC" else sid, "source_state_hash": source_hashes.get(condition), "target_incoming_hash": item["incoming_state_hash"], "target_pair_hash": item["primary_pair_hash"], "probe_hash": item["future_probe_hash"], "recipient_KV_hash": hashes["KV"], "target_Conv_hash": hashes["Conv"], "realized_REC_hash": hashes["REC"], "relative_l2_to_target_donor": float(np.linalg.norm(yd - sig)) / max(float(np.linalg.norm(yd-y00)), 1e-12), "six_probe_signature": True, "native_field_audit_pass": True})
        print(f"V32 context {role} {n}/{len(plan)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"context_{role}_v32.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    pivot = frame.pivot(index="state_id", columns="condition", values="relative_l2_to_target_donor")
    summary = {"role": role, "states": len(plan), "conditions": list(CONDITIONS), "matched_better_fraction": {condition: float((pivot.MATCHED_REC < pivot[condition]).mean()) for condition in CONDITIONS if condition != "MATCHED_REC"}, "median_relative_l2": frame.groupby("condition").relative_l2_to_target_donor.median().to_dict(), "response_sha256": sha256_file(path), "context_mapping_frozen_before_context_responses": True, "independent_final_opened": False}
    jpath = root / OUT / f"context_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"context_{role}", [SOURCE, str(path.relative_to(root)), str(jpath.relative_to(root)), f"artifacts/rec_conv_mechanism_v32_context_mapping_{role}.freeze.json"], {"summary_sha256": sha256_file(jpath), "rows": len(frame)})
    return {"freeze_digest": stage["freeze_digest"], "states": len(plan), "matched_better_fraction": summary["matched_better_fraction"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-development", "freeze-validation", "development", "validation"))
    args = parser.parse_args()
    root = Path.cwd()
    role = args.command.split("-")[-1]
    print(json.dumps(freeze_mapping(root, role) if args.command.startswith("freeze-") else run(root, role), indent=2))
