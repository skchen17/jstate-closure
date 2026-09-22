"""Response-blind model-2 state, token, probe and endpoint freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v31 import category, usable
from jclosure.experiments.runtime_v33 import context, field_hashes, hd, prefix, step
from jclosure.protocol_v33 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

OUT = Path("results/v33/processed")
ROLES = ("calibration", "development", "validation", "independent_final")
SOURCE = "src/jclosure/experiments/design_v33.py"
SEED = 330133


def prompts(root, old):
    lookup = {}
    needed = {(r["source_role"], r["family"]) for role in ROLES for r in old[role]}
    for source, family in sorted(needed):
        table = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{family}_v18.parquet", columns=["base_trial_id", "prompt"])
        lookup.update({str(r.base_trial_id): str(r.prompt) for r in table.itertuples(index=False)})
    return lookup


def ranked_tokens(model, tokenizer, cache, prompt_ids, k):
    # Prefix-only logits, before any current-token fork or future response.
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor(prompt_ids, device=device), use_cache=False).logits[0, -1].float()
    return torch.topk(logits, k=k).indices.tolist()


def common_library(rank_lists, tokenizer, cfg, anchor):
    max_rank = cfg["token_rule"]["common_rank_max"]
    common = set(rank_lists[0])
    for ranked in rank_lists[1:]:
        common.intersection_update(ranked)
    by_class = {}
    for label in cfg["token_rule"]["token_classes"]:
        options = []
        for tid in common:
            if tid == anchor:
                continue
            surface = tokenizer.decode([tid])
            if category(surface) == label:
                positions = [r.index(tid) + 1 for r in rank_lists]
                if max(positions) <= max_rank:
                    options.append({"token_id": int(tid), "surface": surface, "category": label, "calibration_mean_rank": float(np.mean(positions))})
        options.sort(key=lambda x: (x["calibration_mean_rank"], x["token_id"]))
        if len(options) < cfg["token_rule"]["candidate_count"] // 4:
            raise RuntimeError(f"V33 response-blind token-library shortage {label}: {len(options)}")
        by_class[label] = options[:cfg["token_rule"]["candidate_count"] // 4]
    return [r for label in cfg["token_rule"]["token_classes"] for r in by_class[label]]


@torch.no_grad()
def run(root: Path):
    verify_stage(root, "architecture")
    cfg = verify(root)["config"]
    old = json.loads((root / "results/v32/processed/design_v32.json").read_text())
    lookup = prompts(root, old)
    model, tokenizer = context(root)
    anchor_tokens = tokenizer.encode(cfg["token_rule"]["anchor_surface"], add_special_tokens=False)
    if len(anchor_tokens) != 1:
        raise RuntimeError("V33 anchor split")
    anchor = int(anchor_tokens[0])
    k = cfg["token_rule"]["rank_max_per_state"]
    rng = np.random.default_rng(SEED)
    vocab = len(tokenizer)
    bundle = {"selected_logits": rng.choice(vocab, 64, replace=False).tolist(), "broad_logits": rng.choice(vocab, 64, replace=False).tolist(), "broad_sign": rng.choice([-1, 1], 64).tolist(), "late_hidden_indices": rng.choice(model.config.hidden_size, 64, replace=False).tolist(), "workspace_layers": cfg["target_bundle"]["workspace_layers"], "workspace_per_layer": cfg["target_bundle"]["workspace_per_layer"], "J_analogue": "J_NOT_COMPARABLE"}
    roles = {role: [] for role in ROLES}
    calibration_ranks = []
    calibration_clean = []
    for role in ROLES:
        for n, prior in enumerate(old[role], 1):
            sid = prior["base_trial_id"]
            prompt = lookup[sid]
            if hashlib.sha256(prompt.encode()).hexdigest() != prior["prompt_sha256"]:
                raise RuntimeError(f"V33 source prompt drift: {sid}")
            incoming, _, length, ids = prefix(model, tokenizer, prompt)
            if length > cfg["model2"]["context_limit_for_experiment"]:
                raise RuntimeError(f"V33 context limit exceeded {sid}")
            ranked = ranked_tokens(model, tokenizer, incoming, ids, k)
            if anchor not in ranked:
                raise RuntimeError(f"V33 anchor not top-{k}: {sid}")
            if role == "calibration":
                calibration_ranks.append(ranked)
            entry = {"base_trial_id": sid, "family": prior["family"], "source_role": prior["source_role"], "role": role, "prompt_sha256": prior["prompt_sha256"], "prefix_token_hash": hd(ids), "fork_total_length": length, "incoming_state_hashes": field_hashes(incoming), "ranked_token_hash": hd(ranked)}
            entry["incoming_state_hash"] = hd(entry["incoming_state_hashes"])
            entry["future_probe_tokens"] = [int(t) for t in ranked if usable(tokenizer.decode([int(t)]))][:cfg["future_probe_rule"]["count"]]
            if len(entry["future_probe_tokens"]) != 6:
                raise RuntimeError("V33 probe shortage")
            entry["future_probe_hash"] = hd(entry["future_probe_tokens"])
            roles[role].append(entry)
            if role == "calibration":
                clean = step(model, incoming, anchor, length)["cache"]
                for probe in entry["future_probe_tokens"]:
                    calibration_clean.append(step(model, clean, probe, length + 1, bundle)["targets"])
            if n % 10 == 0 or n == len(old[role]):
                print(f"V33 response-blind {role} {n}/{len(old[role])}", flush=True)
    library = common_library(calibration_ranks, tokenizer, cfg, anchor)
    allowed = {r["token_id"] for r in library}
    for role in ROLES:
        for entry in roles[role]:
            sid = entry["base_trial_id"]
            prompt = lookup[sid]
            incoming, _, _, ids = prefix(model, tokenizer, prompt)
            ranked = ranked_tokens(model, tokenizer, incoming, ids, k)
            eligible = [r for r in library if r["token_id"] in ranked]
            if not eligible:
                raise RuntimeError(f"V33 no token candidates: {sid}")
            eligible.sort(key=lambda r: hd([SEED, sid, r["token_id"]]))
            selected = eligible[0]
            entry.update({"primary_token_id": selected["token_id"], "primary_token_category": selected["category"], "primary_pair_hash": hd([anchor, selected["token_id"]]), "eligible_token_ids": sorted(x["token_id"] for x in eligible), "primary_prewrite_rank": ranked.index(selected["token_id"]) + 1, "anchor_prewrite_rank": ranked.index(anchor) + 1})
    scales = {}
    for block in ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace"):
        stacked = np.stack([x[block] for x in calibration_clean]).astype(np.float64)
        scales[block] = max(float(np.sqrt(np.mean((stacked - stacked.mean(axis=0))**2))), 1e-6)
    role_hashes = {role: hd([r["base_trial_id"] for r in roles[role]]) for role in ROLES}
    result = {**roles, "anchor_token_id": anchor, "token_library": library, "target_bundle": bundle, "calibration_clean_scales": scales, "role_hashes": role_hashes, "source_V32_state_ids_only": True, "model2_token_selection_independent": True, "causal_write_observed_before_freeze": False, "future_causal_response_observed_before_freeze": False, "independent_final_opened": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "design_v33.json"
    write_json_atomic(path, result)
    plan = {"roles": {role: [r["base_trial_id"] for r in roles[role]] for role in ROLES}, "factorial_conditions": cfg["factorial"], "primary_KV": cfg["primary_KV"], "estimand": {"D": "Ydonor-Y00", "C": "Y01-Y00", "R": "Y10-Y00", "RC": "Y11-Y00", "E_conv": "D-C", "R_given_C": "Y11-Y01", "I_RC": "Y11-Y10-Y01+Y00"}, "gates": {k: cfg[k] for k in ("replication_gate", "alignment_gate", "conv_dominance_gate", "structural_gate")}, "context_subset": {role: {family: [r["base_trial_id"] for r in roles[role] if r["family"] == family][:2] for family in cfg["families"]} for role in ("development", "validation")}, "adjudication_tree": ["CHANNEL_SEMANTICS_DIFFER", "HANDOFF_REPLICATES_BUT_REC_CORRECTION_DOES_NOT", "REC_CORRECTION_EXISTS_WITH_DIFFERENT_PRIMARY_CARRIER", "NO_ANALOGOUS_ORGANIZATION"], "phase_b_rule": cfg["phase_b_rule"], "final_rule": cfg["final_rule"], "thresholds_retuned": False}
    ppath = root / OUT / "execution_plan_v33.json"
    write_json_atomic(ppath, plan)
    stage = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)), str(ppath.relative_to(root)), "artifacts/cross_model_rec_conv_v33_architecture.freeze.json"], {"design_hash": hd(result), "execution_plan_hash": hd(plan), "roles": role_hashes, "causal_write_observed_before_freeze": False, "future_causal_response_observed_before_freeze": False})
    return {"freeze_digest": stage["freeze_digest"], "roles": {role: len(roles[role]) for role in ROLES}, "library": len(library), "scales": scales}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
