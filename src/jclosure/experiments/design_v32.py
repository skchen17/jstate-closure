"""Response-blind prospective V32 panels, ordinary token forks and probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.design_v31 import category, usable
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.protocol_v32 import stage_freeze, verify
from jclosure.provenance import write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/design_v32.py"
ROLES = ("calibration", "development", "validation", "independent_final")


def hd(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def select_states(root, cfg):
    previous = [json.loads((root / f"results/v{v}/processed/design_v{v}.json").read_text()) for v in cfg["source_exclusions"]]
    excluded = {row["base_trial_id"] for d in previous for role in ROLES for row in d[role]}
    roles = {role: [] for role in ROLES}
    requested = cfg["roles"]
    for family in cfg["families"]:
        for source in ("train", "validation"):
            frame = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{family}_v18.parquet")
            pool = [row for row in frame.to_dict("records") if row["base_trial_id"] not in excluded]
            pool.sort(key=lambda row: hd([cfg["seed"], family, source, row["base_trial_id"]]))
            assignments = (("calibration", requested["calibration_per_family"]), ("development", requested["development_per_family"]), ("independent_final", requested["independent_final_per_family"])) if source == "train" else (("validation", requested["validation_per_family"]),)
            offset = 0
            for role, size in assignments:
                if len(pool) < offset + size:
                    raise RuntimeError(f"V32 {family}/{source} fresh-state shortfall")
                for row in pool[offset:offset + size]:
                    roles[role].append({"base_trial_id": row["base_trial_id"], "family": family, "source_role": source, "role": role, "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode()).hexdigest()})
                offset += size
    ids = [x["base_trial_id"] for role in ROLES for x in roles[role]]
    if len(ids) != 225 or len(set(ids)) != len(ids) or set(ids) & excluded:
        raise RuntimeError("V32 panel disjointness violation")
    return roles, excluded


@torch.no_grad()
def prepare(root: Path):
    cfg = verify(root)["config"]
    roles, excluded = select_states(root, cfg)
    calibration = json.loads((root / "results/v31/processed/prewrite_calibration_v31.json").read_text())
    library = []
    for label in cfg["token_categories"]:
        tokens = [x for x in calibration["common_tokens"] if x["id"] != cfg["anchor_token_id"] and category(x["surface"]) == label]
        if len(tokens) < cfg["tokens_per_category"]:
            raise RuntimeError(f"V32 token category shortfall: {label}")
        for row in tokens[:cfg["tokens_per_category"]]:
            library.append({"token_id": int(row["id"]), "surface": row["surface"], "category": label, "calibration_mean_rank": float(row["mean_rank"]), "pair_hash": hd([cfg["anchor_token_id"], int(row["id"])])})
    if len({x["token_id"] for x in library}) != len(library):
        raise RuntimeError("V32 duplicate library token")
    bundle, *_ = v19._setup(root)
    model, tokenizer = bundle.hf_model, bundle.tokenizer
    device = next(model.parameters()).device
    for role in ROLES:
        for n, item in enumerate(roles[role], 1):
            frame = pd.read_parquet(root / f"results/v18/processed/crossed_state_{item['source_role']}_{item['family']}_v18.parquet", filters=[[("base_trial_id", "==", item["base_trial_id"])]])
            prompt = str(frame.iloc[0]["prompt"])
            if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
                raise RuntimeError("V32 prompt drift")
            ids = encode_direct_prompt(bundle, prompt)
            out = model(input_ids=ids[:, :-1].to(device), use_cache=True)
            rec, att = native_layers(out.past_key_values)
            ranked = torch.topk(out.logits[0, -1].float(), k=cfg["rank_max"]).indices.tolist()
            ranks = {int(tid): j + 1 for j, tid in enumerate(ranked)}
            if cfg["anchor_token_id"] not in ranks:
                raise RuntimeError("V32 anchor token not eligible")
            eligible = [x for x in library if x["token_id"] in ranks]
            if len(eligible) < cfg["contrasts_per_state"]:
                raise RuntimeError(f"V32 token coverage shortfall: {item['base_trial_id']}")
            eligible.sort(key=lambda x: hd([cfg["seed"], item["base_trial_id"], x["category"], x["token_id"]]))
            first = eligible[0]
            second = next((x for x in eligible if x["category"] != first["category"]), eligible[1])
            chosen = [first, second]
            probes = [int(tid) for tid in ranked if usable(tokenizer.decode([int(tid)]))][:cfg["future_probe_count"]]
            if len(probes) != cfg["future_probe_count"]:
                raise RuntimeError("V32 probe shortfall")
            hashes = state_hashes(out.past_key_values, rec, att)
            item.update({"prefix_token_hash": hd(ids[:, :-1].tolist()), "fork_total_length": int(ids.shape[1]), "incoming_state_hashes": hashes, "incoming_state_hash": total_hash(hashes), "primary_token_id": first["token_id"], "secondary_token_id": second["token_id"], "primary_pair_hash": first["pair_hash"], "secondary_pair_hash": second["pair_hash"], "future_probe_tokens": probes, "future_probe_hash": hd(probes), "eligible_token_ids": [x["token_id"] for x in eligible]})
            if n % 10 == 0 or n == len(roles[role]):
                print(f"V32 response-blind {role} {n}/{len(roles[role])}", flush=True)
    payload = {**roles, "families": cfg["families"], "token_library": library, "anchor_token_id": cfg["anchor_token_id"], "target_bundle": json.loads((root / "results/v31/processed/design_v31.json").read_text())["target_bundle"], "role_hashes": {role: hd([x["base_trial_id"] for x in roles[role]]) for role in ROLES}, "token_library_hash": hd(library), "layer_groups": cfg["layer_groups"], "layer_groups_hash": hd(cfg["layer_groups"]), "future_response_observed_before_freeze": 0, "current_token_write_observed_before_freeze": False, "V28_V29_V30_V31_formal_states_excluded": True, "independent_final_opened": False}
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "design_v32.json"
    write_json_atomic(path, payload)
    stage = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)), "results/v31/processed/prewrite_calibration_v31.json"], {"design_hash": hd(payload), "role_hashes": payload["role_hashes"], "token_library_hash": payload["token_library_hash"], "layer_groups_hash": payload["layer_groups_hash"], "future_response_observed_before_freeze": 0})
    return {"freeze_digest": stage["freeze_digest"], "roles": {role: len(roles[role]) for role in ROLES}, "token_library_size": len(library), "token_categories": {label: sum(x["category"] == label for x in library) for label in cfg["token_categories"]}, "prior_states_excluded": len(excluded)}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
