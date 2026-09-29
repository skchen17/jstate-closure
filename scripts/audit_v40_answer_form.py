"""Calibration-only check whether answer-leading whitespace changes native task recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jclosure.experiments.calibration_v40_a2 import answer_token_ids, score
from jclosure.experiments.design_v40 import encode
from jclosure.experiments.runtime_v34 import load
from jclosure.protocol_v40_a3 import verify
from jclosure.provenance import sha256_file, write_json_atomic


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    cfg = verify(root)["config"]
    model, tokenizer = load(root, key)
    blank_ids = tokenizer.encode(" ", add_special_tokens=False)
    if len(blank_ids) != 1:
        raise RuntimeError(f"V40 whitespace not one token: {key}:{blank_ids}")
    blank = int(blank_ids[0])
    items = json.loads((root / "data/v40/calibration_pool_v40.json").read_text())["items"]
    device = next(model.parameters()).device
    rows = []
    for n, item in enumerate(items, 1):
        alphabet = answer_token_ids(tokenizer, cfg["answer_alphabets"][item["family"]])
        for role in ("recipient", "donor"):
            ids = torch.tensor([encode(tokenizer, key, item[f"{role}_prompt"])],
                               device=device)
            base = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            spaced = model(input_ids=torch.cat([ids,
                torch.tensor([[blank]], device=device)], dim=1),
                use_cache=False).logits[0, -1].float()
            rows.append({"state_id": item["state_id"], "family": item["family"],
                         "role": role,
                         "direct": score(base, alphabet, item["recipient_answer"],
                                         item["donor_answer"]),
                         "after_one_space": score(spaced, alphabet,
                                                  item["recipient_answer"],
                                                  item["donor_answer"]),
                         "leading_space_token_id": blank})
        print(f"V40 answer-form audit {key} {n}/{len(items)}", flush=True)
    families = {}
    for family in cfg["answer_alphabets"]:
        subset = [r for r in rows if r["family"] == family]
        native = {role: {r["state_id"]: r for r in subset if r["role"] == role}
                  for role in ("recipient", "donor")}
        families[family] = {name: {
            "recipient_correct": sum(r[name]["recipient_correct"]
                                     for r in native["recipient"].values()),
            "donor_correct": sum(r[name]["donor_recovered"]
                                 for r in native["donor"].values()),
            "both_correct": sum(native["recipient"][sid][name]["recipient_correct"]
                                and native["donor"][sid][name]["donor_recovered"]
                                for sid in native["recipient"]),
        } for name in ("direct", "after_one_space")}
    result = {"model": key, "status": "CALIBRATION_ONLY_ANSWER_FORM_AUDIT",
              "source_sha256": sha256_file(root / "scripts/audit_v40_answer_form.py"),
              "leading_space_token_id": blank, "families": families,
              "rows": rows, "development_validation_final_unopened": True}
    out = root / f"results/v40/processed/answer_form_audit_{key}_v40.json"
    if out.exists():
        raise RuntimeError("V40 answer-form audit exists")
    write_json_atomic(out, result)
    return {k: v for k, v in result.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
