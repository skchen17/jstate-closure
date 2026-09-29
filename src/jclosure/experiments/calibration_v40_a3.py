"""V40 calibration with sequential native token-time cache continuation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.calibration_v40_a2 import (
    CONDITION_CHANNELS, encoded_plan, score,
)
from jclosure.experiments.design_v40 import tokenizer_for
from jclosure.experiments.runtime_v34 import field_hashes, load, native_swap, step
from jclosure.protocol_v40_a1 import verify_stage as verify_startup_stage
from jclosure.protocol_v40_a2 import verify as verify_a2
from jclosure.protocol_v40_a3 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/calibration_v40_a3.py"


@torch.no_grad()
def sequential_logits(model, cache, tokens: list[int]) -> torch.Tensor:
    if not tokens:
        raise RuntimeError("V40 empty continuation")
    work = cache
    length = int(cache.get_seq_length())
    logits = None
    for token in tokens:
        length += 1
        output = step(model, work, token, length)
        work, logits = output["cache"], output["logits"]
    return logits.detach()


@torch.no_grad()
def run(root: Path, key: str, limit: int | None = None) -> dict:
    cfg = verify(root)["config"]
    verify_startup_stage(root, "startup_audit")
    if key not in cfg["models"]:
        raise ValueError(key)
    calibration_cfg = verify_a2(root)["config"]
    tokenizer = tokenizer_for(root, key)
    plans, preflight = encoded_plan(root, key, calibration_cfg, tokenizer, limit)
    model, _ = load(root, key)
    mode = "pilot" if limit is not None else "full"
    delays = cfg["pilot_delays"] if mode == "pilot" else cfg["full_calibration_delays"]
    run_id = f"a3-{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    out_dir = root / "results/v40/raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_{key}_{run_id}.jsonl"
    temporary = out_dir / f"calibration_{key}_{run_id}.jsonl.tmp"
    if path.exists() or temporary.exists():
        raise RuntimeError("V40 a3 raw run ID collision")
    rows = []
    device = next(model.parameters()).device
    with temporary.open("w", encoding="utf-8") as output:
        for n, plan in enumerate(plans, 1):
            task = plan["task"]
            sid = task["state_id"]
            try:
                prefix = plan["prefix_ids"]
                initial = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
                prefix_cache = clone_hybrid_cache(initial.past_key_values)
                del initial
                recipient = step(model, prefix_cache, plan["recipient_token_id"], len(prefix)+1)["cache"]
                donor = step(model, prefix_cache, plan["donor_token_id"], len(prefix)+1)["cache"]
                caches = {"BASE": recipient, "DONOR_NATIVE": donor}
                proofs = {"BASE": {"native": True, "hashes": field_hashes(recipient)},
                          "DONOR_NATIVE": {"native": True, "hashes": field_hashes(donor)}}
                for condition, channels in CONDITION_CHANNELS.items():
                    if condition == "BASE":
                        continue
                    caches[condition], proofs[condition] = native_swap(recipient, donor, channels, key)
                alphabet = preflight["answer_ids"][task["family"]]
                for delay in delays:
                    tokens = [preflight["neutral_token_id"]] * delay + plan["suffix_ids"]
                    for condition in (*cfg["conditions"], "DONOR_NATIVE"):
                        logits = sequential_logits(model, caches[condition], tokens)
                        result = score(logits, alphabet, task["recipient_answer"],
                                       task["donor_answer"])
                        equivalence = None
                        direct_result = None
                        if delay == 0 and condition in ("BASE", "DONOR_NATIVE"):
                            full_ids = (prefix +
                                        [plan["recipient_token_id"] if condition == "BASE"
                                         else plan["donor_token_id"]] + plan["suffix_ids"])
                            direct = model(input_ids=torch.tensor([full_ids], device=device),
                                           use_cache=True).logits[0, -1].float()
                            equivalence = float(torch.max(torch.abs(logits - direct)).item())
                            direct_result = score(direct, alphabet, task["recipient_answer"],
                                                  task["donor_answer"])
                        row = {"run_id": run_id, "model": key, "role": "calibration",
                               "state_id": sid, "family": task["family"],
                               "condition": condition, "delay": delay,
                               "status": "VALID", "continuation": "native_tokenwise",
                               "recipient_answer": task["recipient_answer"],
                               "donor_answer": task["donor_answer"],
                               "neutral_token_id": preflight["neutral_token_id"],
                               "native_cache_proof": proofs[condition],
                               "split_full_max_abs_logit": equivalence,
                               "direct_full_score": direct_result, **result}
                        rows.append(row)
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"V40 a3 calibration {key} {n}/{len(plans)} {sid}", flush=True)
            except Exception as exc:
                row = {"run_id": run_id, "model": key, "role": "calibration",
                       "state_id": sid, "family": task["family"],
                       "status": "INVALID", "reason": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"V40 a3 INVALID {key} {sid}: {row['reason']}", flush=True)
    os.replace(temporary, path)
    valid = [r for r in rows if r["status"] == "VALID"]
    invalid = [r for r in rows if r["status"] == "INVALID"]
    summary = {"run_id": run_id, "model": key, "mode": mode,
               "states_selected": len(plans), "delays": delays,
               "valid_rows": len(valid), "invalid_states": len(invalid),
               "invalid_reasons": invalid,
               "raw_path": str(path.relative_to(root)),
               "raw_sha256": sha256_file(path), "preflight": preflight,
               "calibration_only": True, "validation_and_final_unopened": True}
    summary_path = root / f"results/v40/processed/calibration_{key}_{run_id}.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"calibration_{key}_{run_id}",
                        [SOURCE, str(path.relative_to(root)),
                         str(summary_path.relative_to(root))],
                        {"model": key, "run_id": run_id, "mode": mode,
                         "raw_sha256": summary["raw_sha256"],
                         "calibration_only": True})
    return {**summary, "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.limit), indent=2))
