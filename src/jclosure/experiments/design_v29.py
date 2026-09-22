"""Prospective V29 panels and pre-write fork-token selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.experiments.transaction_v28 import channel_hash
from jclosure.protocol_v29 import verify, stage_freeze
from jclosure.provenance import write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

SOURCE = "src/jclosure/experiments/design_v29.py"
OUT = Path("results/v29/processed")
ROLES = ("calibration", "development", "validation", "independent_final")


def hd(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _distinct_surface(tokenizer, a, b):
    x = tokenizer.decode([int(a)], skip_special_tokens=True).strip().casefold()
    y = tokenizer.decode([int(b)], skip_special_tokens=True).strip().casefold()
    return bool(y and y != x)


def choose_tokens(tokenizer, logits, k):
    top = torch.topk(logits.float(), k=k).indices.tolist()
    a = int(top[0])
    b = next((int(t) for t in top[1:] if _distinct_surface(tokenizer, a, t)), int(top[1]))
    return a, b, [int(t) for t in top[:4]]


@torch.no_grad()
def prepare(root: Path):
    cfg = verify(root)["config"]
    old = json.loads((root / "results/v28/processed/design_v28.json").read_text())
    excluded = {x["base_trial_id"] for role in ROLES for x in old[role]}
    # The V28 design had already excluded the earlier frozen final panels.
    roles = {k: [] for k in ROLES}
    fams = cfg["families"]
    for fam in fams:
        for source, assignment in (
            ("train", (("calibration", 5), ("development", 15), ("independent_final", 10))),
            ("validation", (("validation", 10),)),
        ):
            f = pd.read_parquet(root / f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet")
            pool = f[~f.base_trial_id.isin(excluded)].to_dict("records")
            pool.sort(key=lambda x: hd([cfg["seed"], fam, source, x["base_trial_id"]]))
            offset = 0
            for role, count in assignment:
                if len(pool) < offset + count:
                    raise RuntimeError(f"insufficient unseen states: {fam}/{source}")
                for row in pool[offset:offset + count]:
                    roles[role].append({
                        "base_trial_id": row["base_trial_id"],
                        "family": fam,
                        "source_role": source,
                        "role": role,
                        "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode()).hexdigest(),
                        "continuation_token": int(row["teacher_tokens_h8_or_h1"][0]),
                    })
                offset += count
    ids = [x["base_trial_id"] for role in ROLES for x in roles[role]]
    if len(ids) != 200 or len(ids) != len(set(ids)):
        raise RuntimeError("V29 role overlap/count mismatch")

    bundle, _, _, _, rec, att, *_ = v19._setup(root)
    model = bundle.hf_model
    tok = bundle.tokenizer
    device = next(model.parameters()).device
    for role in ROLES:
        for n, item in enumerate(roles[role], 1):
            source = item["source_role"]
            fam = item["family"]
            f = pd.read_parquet(
                root / f"results/v18/processed/crossed_state_{source}_{fam}_v18.parquet",
                filters=[[("base_trial_id", "==", item["base_trial_id"])]],
            )
            prompt = str(f.iloc[0]["prompt"])
            tokens = encode_direct_prompt(bundle, prompt)
            if tokens.shape[1] < 2:
                raise RuntimeError("prompt too short")
            output = model(input_ids=tokens[:, :-1].to(device), use_cache=True)
            a, b, probes = choose_tokens(tok, output.logits[0, -1], int(cfg["candidate_topk"]))
            item.update({
                "token_A": a,
                "token_B": b,
                "token_A_text": tok.decode([a]),
                "token_B_text": tok.decode([b]),
                "fork_token_hash": hd([a, b]),
                "probe_tokens": probes,
                "probe_hash": hd(probes),
                "prefix_token_hash": hd(tokens[:, :-1].tolist()),
                "incoming_state_hashes": {ch: channel_hash(output.past_key_values, ch, rec, att) for ch in ("REC", "Conv", "KV")},
                "fork_total_length": int(tokens.shape[1]),
                "selection_category": "TOP_PLAUSIBLE_DISTINCT_SURFACE",
            })
            if n % 10 == 0 or n == len(roles[role]):
                print(f"V29 design {role} {n}/{len(roles[role])}", flush=True)
    payload = {
        **roles,
        "families": fams,
        "role_hashes": {r: hd([x["base_trial_id"] for x in roles[r]]) for r in ROLES},
        "fork_rule": cfg["fork_token_rule"],
        "next_token_rule": cfg["next_token_rule"],
        "target_bundle": old["target_bundle"],
        "target_hash": old["target_hash"],
        "channel_rules": cfg["channels"],
        "gates": {k: v for k, v in cfg.items() if k.endswith("_min") or k.endswith("_max") or k.endswith("_required") or k.endswith("_fraction_min")},
        "final_rule": cfg["final_rule"],
        "future_response_observed_before_freeze": 0,
        "natural_outgoing_write_observed_before_freeze": False,
        "prior_v28_ids_excluded": True,
        "semantic_pairing_claim": False,
        "same_class_control_available": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = root / OUT / "design_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "design", [SOURCE, str(path.relative_to(root)), "results/v28/processed/design_v28.json"], {
        "role_hashes": payload["role_hashes"],
        "fork_selection_hash": hd([[x["base_trial_id"], x["token_A"], x["token_B"], x["probe_tokens"]] for r in ROLES for x in roles[r]]),
        "gates_hash": hd(payload["gates"]),
        "final_rule_hash": hd(cfg["final_rule"]),
        "future_response_observed_before_freeze": 0,
    })
    return {"freeze_digest": fr["freeze_digest"], "counts": {r: len(roles[r]) for r in ROLES}}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
