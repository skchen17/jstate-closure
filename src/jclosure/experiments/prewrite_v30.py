"""Response-blind calibration of cross-state token-pair feasibility for V30."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments import counterfactual_bank_v19 as v19
from jclosure.provenance import write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt

OUT = Path("results/v30/processed")
SEED = 300030


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def run(root: Path):
    old = json.loads((root / "results/v29/processed/design_v29.json").read_text())
    v28 = json.loads((root / "results/v28/processed/design_v28.json").read_text())
    excluded = {x["base_trial_id"] for d in (old, v28) for role in ("calibration", "development", "validation", "independent_final") for x in d[role]}
    families = old["families"]
    items = []
    for fam in families:
        f = pd.read_parquet(root / f"results/v18/processed/crossed_state_train_{fam}_v18.parquet")
        pool = f[~f.base_trial_id.isin(excluded)].to_dict("records")
        pool.sort(key=lambda x: h([SEED, fam, x["base_trial_id"]]))
        if len(pool) < 5:
            raise RuntimeError("insufficient calibration pool")
        items += [{"base_trial_id": x["base_trial_id"], "family": fam, "prompt": str(x["prompt"])} for x in pool[:5]]
    bundle, *_ = v19._setup(root)
    model = bundle.hf_model
    tok = bundle.tokenizer
    dev = next(model.parameters()).device
    rows = []
    counts = Counter()
    for n, item in enumerate(items, 1):
        ids = encode_direct_prompt(bundle, item["prompt"])
        out = model(input_ids=ids[:, :-1].to(dev), use_cache=False)
        top = torch.topk(out.logits[0, -1].float(), k=4096).indices.tolist()
        counts.update(top)
        rows.append({"base_trial_id": item["base_trial_id"], "family": item["family"], "prefix_token_hash": h(ids[:, :-1].tolist()), "ranked_token_ids_top4096": top})
        if n % 5 == 0:
            print(f"V30 prewrite calibration {n}/25", flush=True)
    probes = {str(n): sum(v >= n for v in counts.values()) for n in (5, 10, 15, 20, 25)}
    for cutoff in (128, 256, 512, 1024, 2048, 4096):
        c = Counter(x for row in rows for x in row["ranked_token_ids_top4096"][:cutoff])
        probes[f"rank{cutoff}_seen20"] = sum(v >= 20 for v in c.values())
        probes[f"rank{cutoff}_seen25"] = sum(v == 25 for v in c.values())
    output = {"seed": SEED, "states": rows, "families": families, "frequency_probe": probes, "all_tokens": [{"id": tid, "count_top4096": count, "surface": tok.decode([tid])} for tid, count in counts.most_common(500)], "no_current_token_write_observed": True, "no_future_response_observed": True}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / OUT / "prewrite_calibration_v30.json", output)
    return {"state_count": len(rows), "frequency_probe": probes, "prewrite_only": True}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
