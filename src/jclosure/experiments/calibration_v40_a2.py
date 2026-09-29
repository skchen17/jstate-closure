"""V40 calibration-only natural channel swaps and task-grounded readout."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.design_v40 import encode, single_token_fork, tokenizer_for
from jclosure.experiments.runtime_v34 import field_hashes, load, native_swap, step
from jclosure.protocol_v40_a1 import verify_stage as verify_startup_stage
from jclosure.protocol_v40_a2 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/calibration_v40_a2.py"
CONDITION_CHANNELS = {
    "BASE": (), "REC_only": ("REC",), "Conv_only": ("Conv",),
    "KV_only": ("KV",), "REC_Conv": ("REC", "Conv"),
    "REC_Conv_KV": ("REC", "Conv", "KV"),
}


def answer_token_ids(tokenizer, alphabet: str) -> dict[str, int]:
    output = {}
    for answer in alphabet:
        ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"V40 answer not one token: {answer}:{ids}")
        output[answer] = int(ids[0])
    if len(set(output.values())) != len(output):
        raise RuntimeError("V40 duplicate answer tokens")
    return output


def score(logits: torch.Tensor, alphabet: dict[str, int],
          recipient_answer: str, donor_answer: str) -> dict:
    labels = list(alphabet)
    values = [float(logits[alphabet[x]].item()) for x in labels]
    pred = labels[max(range(len(values)), key=values.__getitem__)]
    return {"predicted_answer": pred, "candidate_logits": dict(zip(labels, values)),
            "recipient_correct": pred == recipient_answer,
            "donor_recovered": pred == donor_answer,
            "donor_minus_recipient_logit":
                float(logits[alphabet[donor_answer]].item() -
                      logits[alphabet[recipient_answer]].item())}


def encoded_plan(root: Path, key: str, cfg: dict, tokenizer, limit: int | None) -> tuple[list[dict], dict]:
    verify_startup_stage(root, f"design_{key}")
    pool = json.loads((root / "data/v40/calibration_pool_v40.json").read_text())["items"]
    design = json.loads((root / f"results/v40/processed/design_{key}_v40.json").read_text())["roles"]["calibration"]
    if len(pool) != len(design):
        raise RuntimeError("V40 calibration pool/design count drift")
    neutral = tokenizer.encode(cfg["neutral_text"], add_special_tokens=False)
    if len(neutral) != 1:
        raise RuntimeError("V40 neutral text not one token")
    neutral_id = int(neutral[0])
    alphabets = {family: answer_token_ids(tokenizer, chars)
                 for family, chars in cfg["answer_alphabets"].items()}
    planned = []
    for row, audit in zip(pool, design, strict=True):
        if row["state_id"] != audit["state_id"]:
            raise RuntimeError("V40 calibration pairing drift")
        base_r = encode(tokenizer, key, row["recipient_prompt"])
        base_d = encode(tokenizer, key, row["donor_prompt"])
        check = single_token_fork(base_r, base_d)
        if not check["eligible"] or any(check[k] != audit[k] for k in
             ("fork_index", "recipient_token_id", "donor_token_id",
              "shared_prefix_hash", "shared_suffix_hash")):
            raise RuntimeError(f"V40 fork audit drift: {row['state_id']}")
        i = check["fork_index"]
        for delay in cfg["calibration_delays"]:
            rtext = row["prefix"] + row["recipient_fork"] + cfg["neutral_text"] * delay + row["suffix"]
            dtext = row["prefix"] + row["donor_fork"] + cfg["neutral_text"] * delay + row["suffix"]
            rid, did = encode(tokenizer, key, rtext), encode(tokenizer, key, dtext)
            delayed = single_token_fork(rid, did)
            if (not delayed["eligible"] or delayed["fork_index"] != i or
                rid[:i+1] != base_r[:i+1] or
                rid[i+1:i+1+delay] != [neutral_id] * delay or
                rid[i+1+delay:] != base_r[i+1:] or
                did[i+1+delay:] != base_d[i+1:]):
                raise RuntimeError(f"V40 delay tokenization drift: {key}:{row['state_id']}:{delay}")
        planned.append({"task": row, "fork_index": i, "prefix_ids": base_r[:i],
                        "recipient_token_id": base_r[i], "donor_token_id": base_d[i],
                        "suffix_ids": base_r[i+1:]})
    if limit is not None:
        planned = planned[:limit]
    return planned, {"neutral_token_id": neutral_id, "answer_ids": alphabets,
                     "states_preflighted": len(pool), "states_selected": len(planned),
                     "no_model_forward": True}


def continuation_logits(model, cache, tokens: list[int]) -> torch.Tensor:
    if not tokens:
        raise RuntimeError("V40 empty suffix")
    device = next(model.parameters()).device
    work = clone_hybrid_cache(cache)
    length = work.get_seq_length() + len(tokens)
    out = model(input_ids=torch.tensor([tokens], device=device),
                past_key_values=work,
                attention_mask=torch.ones((1, length), device=device, dtype=torch.long),
                use_cache=True)
    return out.logits[0, -1].float().detach()


@torch.no_grad()
def run(root: Path, key: str, limit: int | None = None) -> dict:
    cfg = verify(root)["config"]
    verify_startup_stage(root, "startup_audit")
    if key not in cfg["models"]:
        raise ValueError(key)
    tokenizer = tokenizer_for(root, key)
    plans, preflight = encoded_plan(root, key, cfg, tokenizer, limit)
    model, _ = load(root, key)
    mode = "pilot" if limit is not None else "full"
    run_id = f"{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    out_dir = root / "results/v40/raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_{key}_{run_id}.jsonl"
    temporary = out_dir / f"calibration_{key}_{run_id}.jsonl.tmp"
    if path.exists() or temporary.exists():
        raise RuntimeError("V40 raw run ID collision")
    rows = []
    device = next(model.parameters()).device
    with temporary.open("w", encoding="utf-8") as output:
        for n, plan in enumerate(plans, 1):
            task = plan["task"]
            sid = task["state_id"]
            try:
                prefix_ids = plan["prefix_ids"]
                initial = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
                prefix_cache = clone_hybrid_cache(initial.past_key_values)
                del initial
                recipient = step(model, prefix_cache, plan["recipient_token_id"], len(prefix_ids)+1)["cache"]
                donor = step(model, prefix_cache, plan["donor_token_id"], len(prefix_ids)+1)["cache"]
                if recipient.get_seq_length() != donor.get_seq_length():
                    raise RuntimeError("V40 native cache length mismatch")
                caches = {"BASE": recipient, "DONOR_NATIVE": donor}
                proofs = {"BASE": {"native": True, "hashes": field_hashes(recipient)},
                          "DONOR_NATIVE": {"native": True, "hashes": field_hashes(donor)}}
                for condition, channels in CONDITION_CHANNELS.items():
                    if condition == "BASE":
                        continue
                    cache, proof = native_swap(recipient, donor, channels, key)
                    caches[condition], proofs[condition] = cache, proof
                alphabet = preflight["answer_ids"][task["family"]]
                for delay in cfg["calibration_delays"]:
                    tokens = [preflight["neutral_token_id"]] * delay + plan["suffix_ids"]
                    for condition in (*cfg["conditions"], "DONOR_NATIVE"):
                        logits = continuation_logits(model, caches[condition], tokens)
                        result = score(logits, alphabet, task["recipient_answer"],
                                       task["donor_answer"])
                        equivalence = None
                        if delay == 0 and condition in ("BASE", "DONOR_NATIVE"):
                            prompt_ids = (prefix_ids +
                                          [plan["recipient_token_id"] if condition == "BASE"
                                           else plan["donor_token_id"]] + plan["suffix_ids"])
                            direct = model(input_ids=torch.tensor([prompt_ids], device=device),
                                           use_cache=False).logits[0, -1].float()
                            equivalence = float(torch.max(torch.abs(logits - direct)).item())
                        row = {"run_id": run_id, "model": key, "role": "calibration",
                               "state_id": sid, "family": task["family"],
                               "condition": condition, "delay": delay, "status": "VALID",
                               "recipient_answer": task["recipient_answer"],
                               "donor_answer": task["donor_answer"],
                               "neutral_token_id": preflight["neutral_token_id"],
                               "native_cache_proof": proofs[condition],
                               "split_full_max_abs_logit": equivalence,
                               **result}
                        rows.append(row)
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"V40 calibration {key} {n}/{len(plans)} {sid}", flush=True)
            except Exception as exc:
                row = {"run_id": run_id, "model": key, "role": "calibration",
                       "state_id": sid, "family": task["family"],
                       "status": "INVALID", "reason": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"V40 calibration INVALID {key} {sid}: {row['reason']}", flush=True)
    os.replace(temporary, path)
    valid = [r for r in rows if r["status"] == "VALID"]
    invalid = [r for r in rows if r["status"] == "INVALID"]
    summary = {"run_id": run_id, "model": key, "mode": mode,
               "states_selected": len(plans), "valid_rows": len(valid),
               "invalid_states": len(invalid), "invalid_reasons": invalid,
               "raw_path": str(path.relative_to(root)),
               "raw_sha256": sha256_file(path),
               "preflight": preflight,
               "calibration_only": True,
               "validation_and_final_unopened": True}
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
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.preflight_only:
        cfg = verify(root)["config"]
        tokenizer = tokenizer_for(root, args.model)
        _, result = encoded_plan(root, args.model, cfg, tokenizer, args.limit)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(run(root, args.model, args.limit), indent=2))
