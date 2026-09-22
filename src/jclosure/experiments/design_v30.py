"""Freeze V30 panels, common token-pair library, and pre-write eligibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.protocol_v30 import verify, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

SOURCE = "src/jclosure/experiments/design_v30.py"
OUT = Path("results/v30/processed")
ROLES = ("calibration", "development", "validation", "independent_final")


def hd(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _special(surface):
    s = surface.strip()
    return not s or (s.startswith("<") and s.endswith(">"))


def library(root, cfg, tokenizer):
    p = json.loads((root / OUT / "prewrite_calibration_v30.json").read_text())
    rows = p["states"]
    if len(rows) != 25 or not p["no_current_token_write_observed"] or not p["no_future_response_observed"]:
        raise RuntimeError("V30 calibration provenance invalid")
    rankmax = int(cfg["token_library"]["calibration_common_rank_max"])
    common = set(rows[0]["ranked_token_ids_top4096"][:rankmax])
    for row in rows[1:]:
        common.intersection_update(row["ranked_token_ids_top4096"][:rankmax])
    anchor = int(cfg["token_library"]["anchor_token_id"])
    if anchor not in common or _special(tokenizer.decode([anchor])):
        raise RuntimeError("frozen anchor not calibration-plausible")
    ntrain = int(cfg["token_library"]["train_pairs"])
    nval = int(cfg["token_library"]["validation_pairs"])
    nfinal = int(cfg["token_library"]["final_pairs"])
    count = ntrain + nval + nfinal
    choices = [x for x in common if x != anchor and not _special(tokenizer.decode([x]))]
    choices.sort(key=lambda tid: (sum(row["ranked_token_ids_top4096"].index(tid) + 1 for row in rows) / len(rows), tid))
    if len(choices) < count:
        raise RuntimeError(f"insufficient common, non-special candidate tokens: {len(choices)}<{count}")
    chosen = choices[:count]
    chosen.sort(key=lambda tid: hd([cfg["seed"], "TOKEN_SPLIT", tid]))
    groups = {"TOKEN_TRAIN": chosen[:ntrain], "TOKEN_VALIDATION": chosen[ntrain:ntrain + nval], "TOKEN_FINAL": chosen[ntrain + nval:]}
    pairs = []
    for role, tokens in groups.items():
        for tid in tokens:
            pairs.append({"pair_id": hd([anchor, tid])[:16], "anchor_token_id": anchor, "candidate_token_id": int(tid), "candidate_surface": tokenizer.decode([int(tid)]), "role": role, "calibration_mean_rank": float(sum(row["ranked_token_ids_top4096"].index(tid) + 1 for row in rows) / len(rows)), "token_pair_hash": hd([anchor, tid])})
    return pairs, groups, rows


@torch.no_grad()
def prepare(root: Path):
    cfg = verify(root)["config"]
    old = [json.loads((root / f"results/v{v}/processed/design_v{v}.json").read_text()) for v in (28, 29)]
    excluded = {x["base_trial_id"] for d in old for role in ROLES for x in d[role]}
    bundle, *_ = v19._setup(root)
    model, tokenizer = bundle.hf_model, bundle.tokenizer
    device = next(model.parameters()).device
    pairs, groups, cal = library(root, cfg, tokenizer)
    if len({x["candidate_token_id"] for x in pairs}) != len(pairs):
        raise RuntimeError("token pair role overlap")
    cal_ids = {x["base_trial_id"] for x in cal}
    roles = {r: [] for r in ROLES}
    fams = cfg["families"]
    # Calibration is exactly the response-blind panel that defined the token library.
    for row in cal:
        roles["calibration"].append({"base_trial_id": row["base_trial_id"], "family": row["family"], "source_role": "train", "role": "calibration"})
    for fam in fams:
        for source, assignments in (("train", (("development", 24), ("independent_final", 10))), ("validation", (("validation", 12),))):
            f = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet")
            pool = f[~f.base_trial_id.isin(excluded | cal_ids)].to_dict("records")
            pool.sort(key=lambda row: hd([cfg["seed"], fam, source, row["base_trial_id"]]))
            offset = 0
            for role, count in assignments:
                if len(pool) < offset + count:
                    raise RuntimeError(f"insufficient V30 pool {fam}/{source}")
                roles[role].extend({"base_trial_id": x["base_trial_id"], "family": fam, "source_role": source, "role": role} for x in pool[offset:offset + count])
                offset += count
    all_ids = [x["base_trial_id"] for role in ROLES for x in roles[role]]
    if len(all_ids) != 255 or len(set(all_ids)) != 255:
        raise RuntimeError("V30 disjoint role count mismatch")
    rankmax = int(cfg["token_library"]["eligibility_rank_max"])
    library_tokens = [int(cfg["token_library"]["anchor_token_id"])] + [x["candidate_token_id"] for x in pairs]
    for role in ROLES:
        for n, item in enumerate(roles[role], 1):
            source = item["source_role"]
            fam = item["family"]
            f = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet", filters=[[("base_trial_id", "==", item["base_trial_id"])]])
            m = f.iloc[0].to_dict()
            ids = encode_direct_prompt(bundle, str(m["prompt"]))
            out = model(input_ids=ids[:, :-1].to(device), use_cache=True)
            rec, att = native_layers(out.past_key_values)
            top = torch.topk(out.logits[0, -1].float(), k=rankmax).indices.tolist()
            ranks = {tid: int(top.index(tid) + 1) if tid in top else None for tid in library_tokens}
            eligible = [p["pair_id"] for p in pairs if ranks[p["anchor_token_id"]] is not None and ranks[p["candidate_token_id"]] is not None]
            probes = [int(tid) for tid in top if not _special(tokenizer.decode([int(tid)]))][:int(cfg["future_probe_count"])]
            if len(probes) != cfg["future_probe_count"]:
                raise RuntimeError("insufficient prewrite future probes")
            item.update({"prompt_sha256": hashlib.sha256(str(m["prompt"]).encode()).hexdigest(), "prefix_token_hash": hd(ids[:, :-1].tolist()), "incoming_state_hashes": state_hashes(out.past_key_values, rec, att), "incoming_state_hash": total_hash(state_hashes(out.past_key_values, rec, att)), "fork_total_length": int(ids.shape[1]), "token_ranks": {str(k): v for k, v in ranks.items()}, "eligible_pair_ids": eligible, "eligible_TRAIN": sum(p["pair_id"] in eligible for p in pairs if p["role"] == "TOKEN_TRAIN"), "eligible_VALIDATION": sum(p["pair_id"] in eligible for p in pairs if p["role"] == "TOKEN_VALIDATION"), "eligible_FINAL": sum(p["pair_id"] in eligible for p in pairs if p["role"] == "TOKEN_FINAL"), "future_probe_tokens": probes, "future_probe_hash": hd(probes)})
            if n % 10 == 0 or n == len(roles[role]):
                print(f"V30 prewrite design {role} {n}/{len(roles[role])}", flush=True)
    coverage = {role: {"train_min": min(x["eligible_TRAIN"] for x in roles[role]), "train_median": float(pd.Series([x["eligible_TRAIN"] for x in roles[role]]).median()), "validation_min": min(x["eligible_VALIDATION"] for x in roles[role]), "final_min": min(x["eligible_FINAL"] for x in roles[role])} for role in ROLES}
    payload = {**roles, "families": fams, "token_pair_library": pairs, "token_groups": groups, "library_hash": hd(pairs), "role_hashes": {r: hd([x["base_trial_id"] for x in roles[r]]) for r in ROLES}, "eligibility_coverage": coverage, "target_bundle": old[-1]["target_bundle"], "target_hash": old[-1]["target_hash"], "gates": cfg["causal_gate"], "k_grid": cfg["k_grid"], "finalist_priority": cfg["finalist_priority"], "future_response_observed_before_freeze": 0, "natural_write_observed_before_freeze": False, "calibration_only_library": True, "V28_V29_ids_excluded": True, "token_library_size_above_suggestion_reason": "k=64 local PCA needs at least 64 independent TRAIN contrasts plus disjoint token-OOD contrasts"}
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "design_v30.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)), "results/v30/processed/prewrite_calibration_v30.json", "results/v29/processed/design_v29.json", "results/v28/processed/design_v28.json"], {"library_hash": payload["library_hash"], "role_hashes": payload["role_hashes"], "coverage": coverage, "future_response_observed_before_freeze": 0, "natural_write_observed_before_freeze": False})
    return {"freeze_digest": fr["freeze_digest"], "counts": {r: len(roles[r]) for r in ROLES}, "library_pairs": len(pairs), "coverage": coverage}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
