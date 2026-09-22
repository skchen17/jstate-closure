"""Freeze V31 response-blind state, token, composition and probe design."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.protocol_v31 import stage_freeze, verify
from jclosure.provenance import write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

OUT = Path("results/v31/processed")
SOURCE = "src/jclosure/experiments/design_v31.py"
ROLES = ("calibration", "development", "validation", "independent_final")
FUNCTION_WORDS = frozenset("a an and are as at be by can for from has have if in is it of on or the to was were with yes no not true false then else than because therefore so but into over under".split())


def hd(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def usable(surface):
    return bool(surface and surface.strip() and not (surface.startswith("<") and surface.endswith(">")) and "\ufffd" not in surface)


def category(surface):
    s = surface.strip()
    if not usable(surface):
        return "UNCLASSIFIED"
    if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", s):
        return "NUMERIC"
    if s.lower() in FUNCTION_WORDS:
        return "FUNCTION_WORD"
    if all(not c.isalnum() for c in s):
        return "PUNCTUATION_OR_STRUCTURE"
    if re.fullmatch(r"[A-Za-z]{3,}", s):
        return "LEXICAL_SURFACE"
    if re.fullmatch(r"[A-Za-z]{1,2}", s):
        return "SHORT_ALPHA_AMBIGUOUS"
    if any("\u4e00" <= c <= "\u9fff" for c in s):
        return "CJK_SURFACE_UNRESOLVED"
    return "UNCLASSIFIED"


def library(calibration, cfg):
    anchor = cfg["token_library"]["anchor_token_id"]
    common = {x["id"]: x for x in calibration["common_tokens"] if usable(x["surface"])}
    if anchor not in common:
        raise RuntimeError("anchor absent from response-blind common-token pool")
    all_ab = {x["AB"] for x in calibration["surface_composition_triples"]}
    triples = [x for x in calibration["surface_composition_triples"] if anchor not in (x["A"], x["B"], x["AB"]) and x["A"] not in all_ab and x["B"] not in all_ab]
    chosen_triples = []
    seen_ab = set()
    for x in triples:
        if x["AB"] not in seen_ab:
            chosen_triples.append(x)
            seen_ab.add(x["AB"])
    c = cfg["token_library"]
    n_train, n_val, n_final = (c["surface_composition_train"], c["surface_composition_validation"], c["surface_composition_final"])
    if len(chosen_triples) < n_train + n_val + n_final:
        raise RuntimeError(f"insufficient independent surface compositions: {len(chosen_triples)}")
    chosen_triples = chosen_triples[:n_train + n_val + n_final]
    splits = {"COMPOSITION_TRAIN": chosen_triples[:n_train], "COMPOSITION_VALIDATION": chosen_triples[n_train:n_train + n_val], "COMPOSITION_FINAL": chosen_triples[n_train + n_val:]}
    heldout_ab = {x["AB"] for x in splits["COMPOSITION_VALIDATION"] + splits["COMPOSITION_FINAL"]}
    required_train = {x["A"] for x in chosen_triples} | {x["B"] for x in chosen_triples} | {x["AB"] for x in splits["COMPOSITION_TRAIN"]}
    if anchor in required_train or required_train & heldout_ab:
        raise RuntimeError("composition component leakage or anchor conflict")
    train_ids = sorted(required_train, key=lambda tid: (common[tid]["mean_rank"], tid))
    for x in calibration["common_tokens"]:
        tid = x["id"]
        if tid != anchor and usable(x["surface"]) and tid not in train_ids and tid not in heldout_ab:
            train_ids.append(tid)
        if len(train_ids) >= c["train_pairs"]:
            break
    if len(train_ids) != c["train_pairs"]:
        raise RuntimeError("insufficient TRAIN tokens")
    role_ids = {"TOKEN_TRAIN": train_ids, "TOKEN_VALIDATION": [x["AB"] for x in splits["COMPOSITION_VALIDATION"]], "TOKEN_FINAL": [x["AB"] for x in splits["COMPOSITION_FINAL"]]}
    if len(set(sum(role_ids.values(), []))) != sum(map(len, role_ids.values())):
        raise RuntimeError("token-role overlap")
    pairs = [{"pair_id": hd([anchor, tid])[:16], "anchor_token_id": anchor, "candidate_token_id": tid, "candidate_surface": common[tid]["surface"], "category": category(common[tid]["surface"]), "role": role, "calibration_mean_rank": common[tid]["mean_rank"], "token_pair_hash": hd([anchor, tid])} for role, ids in role_ids.items() for tid in ids]
    for role, group in splits.items():
        for x in group:
            x["role"] = role
            x["composition_id"] = hd([x["A"], x["B"], x["AB"]])[:16]
            x["component_pair_ids"] = [hd([anchor, x["A"]])[:16], hd([anchor, x["B"]])[:16]]
            x["combined_pair_id"] = hd([anchor, x["AB"]])[:16]
    return pairs, role_ids, splits


@torch.no_grad()
def prepare(root: Path):
    cfg = verify(root)["config"]
    cal = json.loads((root / OUT / "prewrite_calibration_v31.json").read_text())
    if not cal["no_current_token_write_observed"] or not cal["no_future_response_observed"] or len(cal["states"]) != 25:
        raise RuntimeError("V31 calibration provenance mismatch")
    prior = [json.loads((root / f"results/v{v}/processed/design_v{v}.json").read_text()) for v in (28, 29, 30)]
    excluded = {x["base_trial_id"] for d in prior for role in ROLES for x in d[role]}
    cal_ids = {x["base_trial_id"] for x in cal["states"]}
    if cal_ids & excluded:
        raise RuntimeError("calibration state re-use")
    pairs, role_ids, comps = library(cal, cfg)
    roles = {r: [] for r in ROLES}
    roles["calibration"] = [{"base_trial_id": x["base_trial_id"], "family": x["family"], "source_role": "train", "role": "calibration"} for x in cal["states"]]
    for fam in cfg["families"]:
        for source, assignment in (("train", (("development", 24), ("independent_final", 10))), ("validation", (("validation", 12),))):
            frame = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet")
            pool = frame[~frame.base_trial_id.isin(excluded | cal_ids)].to_dict("records")
            pool.sort(key=lambda x: hd([cfg["seed"], fam, source, x["base_trial_id"]]))
            offset = 0
            for role, count in assignment:
                if len(pool) < offset + count:
                    raise RuntimeError(f"insufficient fresh V31 {fam}/{source} pool")
                roles[role].extend({"base_trial_id": x["base_trial_id"], "family": fam, "source_role": source, "role": role} for x in pool[offset:offset + count])
                offset += count
    ids = [x["base_trial_id"] for role in ROLES for x in roles[role]]
    if len(ids) != 255 or len(set(ids)) != 255 or set(ids) & excluded:
        raise RuntimeError("V31 panel size/disjointness mismatch")
    bundle, *_ = v19._setup(root)
    model, tokenizer = bundle.hf_model, bundle.tokenizer
    device = next(model.parameters()).device
    token_ids = [cfg["token_library"]["anchor_token_id"]] + [p["candidate_token_id"] for p in pairs]
    rankmax = cfg["token_library"]["eligibility_rank_max"]
    for role in ROLES:
        for n, item in enumerate(roles[role], 1):
            frame = pd.read_parquet(root / f"results/v18/processed/crossed_state_{item['source_role']}_{item['family']}_v18.parquet", filters=[[("base_trial_id", "==", item["base_trial_id"])]])
            prompt = str(frame.iloc[0]["prompt"])
            tokenized = encode_direct_prompt(bundle, prompt)
            out = model(input_ids=tokenized[:, :-1].to(device), use_cache=True)
            rec, att = native_layers(out.past_key_values)
            top = torch.topk(out.logits[0, -1].float(), k=rankmax).indices.tolist()
            ranks = {tid: i + 1 for i, tid in enumerate(top)}
            eligible = [p["pair_id"] for p in pairs if p["anchor_token_id"] in ranks and p["candidate_token_id"] in ranks]
            eligible_set = set(eligible)
            comp_eligible = {split: [x["composition_id"] for x in group if x["combined_pair_id"] in eligible_set and all(pid in eligible_set for pid in x["component_pair_ids"])] for split, group in comps.items()}
            probes = [int(tid) for tid in top if usable(tokenizer.decode([int(tid)]))][:cfg["future_probe_count"]]
            if len(probes) != cfg["future_probe_count"]:
                raise RuntimeError("future-probe shortfall")
            state = state_hashes(out.past_key_values, rec, att)
            item.update({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prefix_token_hash": hd(tokenized[:, :-1].tolist()), "incoming_state_hashes": state, "incoming_state_hash": total_hash(state), "fork_total_length": int(tokenized.shape[1]), "eligible_pair_ids": eligible, "eligible_compositions": comp_eligible, "eligible_by_token_role": {k: sum(hd([cfg["token_library"]["anchor_token_id"], tid])[:16] in eligible_set for tid in tids) for k, tids in role_ids.items()}, "future_probe_tokens": probes, "future_probe_hash": hd(probes)})
            if n % 10 == 0 or n == len(roles[role]):
                print(f"V31 response-blind design {role} {n}/{len(roles[role])}", flush=True)
    coverage = {role: {"minimum_pairs": min(len(x["eligible_pair_ids"]) for x in roles[role]), "minimum_composition_validation": min(len(x["eligible_compositions"]["COMPOSITION_VALIDATION"]) for x in roles[role]), "minimum_composition_final": min(len(x["eligible_compositions"]["COMPOSITION_FINAL"]) for x in roles[role])} for role in ROLES}
    fit = [x for fam in cfg["families"] for x in [y for y in roles["development"] if y["family"] == fam][:cfg["fit_states_per_family"]]]
    holdout = [x for x in roles["development"] if x["base_trial_id"] not in {y["base_trial_id"] for y in fit}]
    fit_pairs = {}
    train_pairs = [p for p in pairs if p["role"] == "TOKEN_TRAIN"]
    for i, item in enumerate(fit):
        order = train_pairs[(i * 10) % len(train_pairs):] + train_pairs[:(i * 10) % len(train_pairs)]
        fit_pairs[item["base_trial_id"]] = [p["pair_id"] for p in order if p["pair_id"] in item["eligible_pair_ids"]][:cfg["fit_contrasts_per_state"]]
        if len(fit_pairs[item["base_trial_id"]]) != cfg["fit_contrasts_per_state"]:
            raise RuntimeError(f"fit-pair shortfall: {item['base_trial_id']}")
    payload = {**roles, "families": cfg["families"], "token_pair_library": pairs, "token_groups": role_ids, "surface_composition_splits": comps, "fit_state_ids": [x["base_trial_id"] for x in fit], "development_holdout_ids": [x["base_trial_id"] for x in holdout], "fit_pair_ids_per_state": fit_pairs, "target_bundle": prior[-1]["target_bundle"], "target_hash": prior[-1]["target_hash"], "role_hashes": {r: hd([x["base_trial_id"] for x in roles[r]]) for r in ROLES}, "token_library_hash": hd(pairs), "composition_hash": hd(comps), "coverage": coverage, "future_response_observed_before_freeze": 0, "current_token_write_observed_before_freeze": False, "surface_composition_not_semantic_proof": True, "V28_V29_V30_formal_states_excluded": True, "independent_final_opened": False}
    path = root / OUT / "design_v31.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)), str(OUT / "prewrite_calibration_v31.json"), "artifacts/compositional_natural_writes_v31.freeze.json"], {"design_hash": hd(payload), "role_hashes": payload["role_hashes"], "token_library_hash": payload["token_library_hash"], "composition_hash": payload["composition_hash"], "coverage": coverage, "future_response_observed_before_freeze": 0, "current_token_write_observed_before_freeze": False})
    return {"freeze_digest": fr["freeze_digest"], "roles": {r: len(roles[r]) for r in ROLES}, "pairs": len(pairs), "compositions": {k: len(v) for k, v in comps.items()}, "coverage": coverage}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
