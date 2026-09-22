"""Response-blind feasibility calibration for prospective V31 token composition."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.provenance import write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

OUT = Path("results/v31/processed")
SEED = 310031
TOP = 8192
ROLES = ("calibration", "development", "validation", "independent_final")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def usable(s):
    return bool(s and s.strip() and not (s.startswith("<") and s.endswith(">")) and "\ufffd" not in s)


@torch.no_grad()
def run(root: Path):
    prior = [json.loads((root / f"results/v{v}/processed/design_v{v}.json").read_text()) for v in (28, 29, 30)]
    excluded = {x["base_trial_id"] for d in prior for role in ROLES for x in d[role]}
    families = prior[-1]["families"]
    states = []
    for fam in families:
        frame = pd.read_parquet(root / f"results/v18/processed/crossed_state_train_{fam}_v18.parquet")
        pool = frame[~frame.base_trial_id.isin(excluded)].to_dict("records")
        pool.sort(key=lambda x: digest([SEED, fam, x["base_trial_id"]]))
        if len(pool) < 5:
            raise RuntimeError(f"insufficient fresh V31 calibration states: {fam}")
        states.extend({"base_trial_id": x["base_trial_id"], "family": fam, "prompt": str(x["prompt"])} for x in pool[:5])
    bundle, *_ = v19._setup(root)
    model, tokenizer = bundle.hf_model, bundle.tokenizer
    device = next(model.parameters()).device
    rows = []
    for n, state in enumerate(states, 1):
        ids = encode_direct_prompt(bundle, state["prompt"])
        out = model(input_ids=ids[:, :-1].to(device), use_cache=False)
        ranked = torch.topk(out.logits[0, -1].float(), k=TOP).indices.tolist()
        rows.append({"base_trial_id": state["base_trial_id"], "family": state["family"], "prefix_token_hash": digest(ids[:, :-1].tolist()), "ranked_token_ids_top8192": ranked})
        if n % 5 == 0:
            print(f"V31 response-blind calibration {n}/25", flush=True)
    common = set(rows[0]["ranked_token_ids_top8192"])
    for row in rows[1:]:
        common.intersection_update(row["ranked_token_ids_top8192"])
    rank_maps = [{tid: i + 1 for i, tid in enumerate(row["ranked_token_ids_top8192"])} for row in rows]
    tokens = [{"id": tid, "surface": tokenizer.decode([tid]), "mean_rank": sum(m[tid] for m in rank_maps) / 25} for tid in common]
    tokens.sort(key=lambda x: (x["mean_rank"], x["id"]))
    surfaces = defaultdict(list)
    for item in tokens:
        if usable(item["surface"]):
            surfaces[item["surface"]].append(item["id"])
    triples = []
    for ab in tokens:
        whole = ab["surface"]
        if not usable(whole) or len(whole) < 3:
            continue
        for left, a_ids in surfaces.items():
            if not whole.startswith(left) or len(left) == len(whole):
                continue
            right = whole[len(left):]
            if not usable(right):
                continue
            for a_id in a_ids:
                for b_id in surfaces.get(right, []):
                    if len({a_id, b_id, ab["id"]}) == 3:
                        triples.append({"A": a_id, "B": b_id, "AB": ab["id"], "surface_A": left, "surface_B": right, "surface_AB": whole})
    triples.sort(key=lambda x: digest([SEED, "composition", x["A"], x["B"], x["AB"]]))
    cutoffs = {}
    for cutoff in (512, 1024, 2048, 4096, 8192):
        c = set(rows[0]["ranked_token_ids_top8192"][:cutoff])
        for row in rows[1:]:
            c.intersection_update(row["ranked_token_ids_top8192"][:cutoff])
        cutoffs[str(cutoff)] = len(c)
    payload = {"seed": SEED, "source_versions_excluded": [28, 29, 30], "states": rows, "families": families, "top_k": TOP, "common_tokens": tokens, "surface_composition_triples": triples, "common_by_rank_cutoff": cutoffs, "no_current_token_write_observed": True, "no_future_response_observed": True, "surface_composition_is_only_a_prewrite_proxy": True}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / OUT / "prewrite_calibration_v31.json", payload)
    return {"states": len(rows), "common_tokens": len(tokens), "surface_composition_triples": len(triples), "common_by_rank_cutoff": cutoffs}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
