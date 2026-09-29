"""Calibration-only audit of full, batch-cached, and tokenwise-cached continuations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.calibration_v40_a2 import (
    continuation_logits, encoded_plan, score,
)
from jclosure.experiments.design_v40 import tokenizer_for
from jclosure.experiments.runtime_v34 import load, step
from jclosure.protocol_v40_a2 import verify
from jclosure.provenance import sha256_file, write_json_atomic


@torch.no_grad()
def audit(root: Path, key: str) -> dict:
    cfg = verify(root)["config"]
    tokenizer = tokenizer_for(root, key)
    plans, preflight = encoded_plan(root, key, cfg, tokenizer, 1)
    plan = plans[0]
    task = plan["task"]
    model, _ = load(root, key)
    device = next(model.parameters()).device
    prefix = plan["prefix_ids"]
    state = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    prefix_cache = clone_hybrid_cache(state.past_key_values)
    del state
    rows = []
    alphabet = preflight["answer_ids"][task["family"]]
    for role, token in (("recipient", plan["recipient_token_id"]),
                        ("donor", plan["donor_token_id"])):
        fork_cache = step(model, prefix_cache, token, len(prefix)+1)["cache"]
        suffix = plan["suffix_ids"]
        batched = continuation_logits(model, fork_cache, suffix)
        work = fork_cache
        tokenwise = None
        for offset, symbol in enumerate(suffix, 1):
            result = step(model, work, symbol, len(prefix)+1+offset)
            work, tokenwise = result["cache"], result["logits"]
        full_ids = prefix + [token] + suffix
        direct_false = model(input_ids=torch.tensor([full_ids], device=device),
                             use_cache=False).logits[0, -1].float()
        direct_true = model(input_ids=torch.tensor([full_ids], device=device),
                            use_cache=True).logits[0, -1].float()
        predictions = {
            name: score(logits, alphabet, task["recipient_answer"],
                        task["donor_answer"])
            for name, logits in (("batched", batched), ("tokenwise", tokenwise),
                                 ("direct_use_cache_false", direct_false),
                                 ("direct_use_cache_true", direct_true))}
        rows.append({"role": role, "suffix_tokens": len(suffix),
                     "predictions": predictions,
                     "max_abs": {
                         "batched_vs_tokenwise": float((batched-tokenwise).abs().max().item()),
                         "batched_vs_direct_true": float((batched-direct_true).abs().max().item()),
                         "tokenwise_vs_direct_true": float((tokenwise-direct_true).abs().max().item()),
                         "direct_true_vs_false": float((direct_true-direct_false).abs().max().item()),
                     }})
    result = {"model": key, "state_id": task["state_id"],
              "protocol_freeze_digest": verify(root)["freeze_digest"],
              "source_sha256": sha256_file(root / "scripts/audit_v40_continuation.py"),
              "calibration_only": True, "rows": rows}
    path = root / f"results/v40/processed/continuation_audit_{key}_a2.json"
    if path.exists():
        raise RuntimeError("V40 continuation audit already exists")
    write_json_atomic(path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(audit(Path.cwd(), args.model), indent=2))
