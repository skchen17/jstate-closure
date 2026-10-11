"""Frozen V40 task-grounded native-channel functional and temporal runs."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.calibration_v40_a2 import CONDITION_CHANNELS, score
from jclosure.experiments.calibration_v40_a3 import sequential_logits
from jclosure.experiments.design_v40 import encode, single_token_fork, tokenizer_for
from jclosure.experiments.runtime_v34 import field_hashes, load, native_swap, step
from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/formal_v40_a4.py"
CONDITIONS = (*CONDITION_CHANNELS, "DONOR_NATIVE")
ROLES = ("development", "validation", "independent_final")


def primary_plan(tokenizer, key: str, task: dict, design: dict) -> dict:
    recipient = encode(tokenizer, key, task["recipient_prompt"])
    donor = encode(tokenizer, key, task["donor_prompt"])
    fork = single_token_fork(recipient, donor)
    if not fork["eligible"] or task["state_id"] != design["state_id"] or any(
            fork[field] != design[field] for field in
            ("fork_index", "recipient_token_id", "donor_token_id")):
        raise RuntimeError("V40 a4 sealed primary fork drift")
    i = fork["fork_index"]
    return {"prefix": recipient[:i], "recipient_token": recipient[i],
            "donor_token": donor[i], "suffix": recipient[i + 1:]}


@torch.no_grad()
def native_pair(model, prefix: list[int], recipient_token: int, donor_token: int):
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    base = clone_hybrid_cache(out.past_key_values)
    del out
    recipient = step(model, base, recipient_token, len(prefix) + 1)["cache"]
    donor = step(model, base, donor_token, len(prefix) + 1)["cache"]
    if recipient.get_seq_length() != donor.get_seq_length():
        raise RuntimeError("V40 a4 native pair cache length mismatch")
    return recipient, donor


def full_score(logits, alphabet: dict[str, int], task: dict) -> dict:
    result = score(logits, alphabet, task["recipient_answer"], task["donor_answer"])
    result["task_correct"] = result["recipient_correct"]
    return result


@torch.no_grad()
def evaluate(model, cache, continuation: list[int], alphabet: dict[str, int], task: dict) -> dict:
    return full_score(sequential_logits(model, cache, continuation), alphabet, task)


def native_condition_caches(recipient, donor, key: str) -> tuple[dict, dict]:
    caches = {"BASE": recipient, "DONOR_NATIVE": donor}
    proofs = {"BASE": {"native": True, "hashes": field_hashes(recipient)},
              "DONOR_NATIVE": {"native": True, "hashes": field_hashes(donor)}}
    for condition, channels in CONDITION_CHANNELS.items():
        if condition == "BASE":
            continue
        caches[condition], proofs[condition] = native_swap(recipient, donor, channels, key)
        if not proofs[condition]["requested_exact"] or not proofs[condition]["untouched_exact"]:
            raise RuntimeError(f"V40 a4 write proof failed: {condition}")
    return caches, proofs


@torch.no_grad()
def direct_equivalence(model, tokens: list[int], cached_logits, alphabet, task):
    device = next(model.parameters()).device
    direct = model(input_ids=torch.tensor([tokens], device=device),
                   use_cache=True).logits[0, -1].float()
    direct_result = full_score(direct, alphabet, task)
    cached_result = full_score(cached_logits, alphabet, task)
    return {"full_vocab_max_abs": float(torch.max(torch.abs(cached_logits - direct)).item()),
            "candidate_prediction_agrees":
                cached_result["predicted_answer"] == direct_result["predicted_answer"],
            "direct_predicted_answer": direct_result["predicted_answer"]}


@torch.no_grad()
def primary_rows(model, key: str, role: str, task: dict, plan: dict,
                 answer_ids: dict, space: int, neutral: int,
                 delays: list[int], phase: str, run_id: str) -> tuple[list[dict], dict, dict]:
    recipient, donor = native_pair(model, plan["prefix"],
                                   plan["recipient_token"], plan["donor_token"])
    caches, proofs = native_condition_caches(recipient, donor, key)
    alphabet = answer_ids[task["family"]]
    rows = []
    for delay in delays:
        continuation = [neutral] * delay + plan["suffix"] + [space]
        for condition in CONDITIONS:
            logits = sequential_logits(model, caches[condition], continuation)
            equivalence = None
            if phase == "functional" and delay == 0 and condition in ("BASE", "DONOR_NATIVE"):
                token = plan["recipient_token"] if condition == "BASE" else plan["donor_token"]
                equivalence = direct_equivalence(model,
                    plan["prefix"] + [token] + plan["suffix"] + [space],
                    logits, alphabet, task)
            rows.append({"run_id": run_id, "model": key, "role": role,
                         "phase": phase, "state_id": task["state_id"],
                         "family": task["family"], "kind": "primary",
                         "condition": condition, "delay": delay, "status": "VALID",
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["donor_answer"],
                         "native_cache_proof": proofs[condition],
                         "split_full_audit": equivalence,
                         "native_tokenwise": True, **full_score(logits, alphabet, task)})
    return rows, caches, proofs


@torch.no_grad()
def control_rows(model, tokenizer, key: str, role: str, task: dict,
                 design: dict, plan: dict, caches: dict,
                 answer_ids: dict, space: int, items_by_id: dict,
                 run_id: str) -> list[dict]:
    alphabet = answer_ids[task["family"]]
    rows = []
    for kind in ("surface_only", "irrelevant_history"):
        control = design["controls"][kind]
        if not control["available"]:
            rows.append({"run_id": run_id, "model": key, "role": role,
                         "phase": "functional", "state_id": task["state_id"],
                         "family": task["family"], "kind": kind,
                         "status": "UNAVAILABLE", "reason": control["reason"]})
            continue
        left = encode(tokenizer, key, control["left_prompt"])
        right = encode(tokenizer, key, control["right_prompt"])
        fork = single_token_fork(left, right)
        if not fork["eligible"] or fork["fork_index"] != control["fork_index"]:
            raise RuntimeError(f"V40 a4 {kind} design drift")
        i = fork["fork_index"]
        native_left, native_right = native_pair(model, left[:i], left[i], right[i])
        hybrid, proof = native_swap(native_left, native_right,
                                    ("REC", "Conv", "KV"), key)
        control_task = {**task, "donor_answer": task["recipient_answer"]}
        continuation = left[i + 1:] + [space]
        for name, cache, cache_proof in (
                ("BASE", native_left, {"native": True, "hashes": field_hashes(native_left)}),
                ("DONOR_NATIVE", native_right, {"native": True, "hashes": field_hashes(native_right)}),
                ("REC_Conv_KV", hybrid, proof)):
            outcome = evaluate(model, cache, continuation, alphabet, control_task)
            rows.append({"run_id": run_id, "model": key, "role": role,
                         "phase": "functional", "state_id": task["state_id"],
                         "family": task["family"], "kind": kind,
                         "condition": name, "delay": 0, "status": "VALID",
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["recipient_answer"],
                         "native_cache_proof": cache_proof,
                         "external_task_variable_unchanged": True,
                         "native_tokenwise": True, **outcome})
    mismatch = design["controls"]["same_prefix_mismatch"]
    if mismatch["available"]:
        # Reuse the same natural pre-fork prefix, never the recipient post-fork cache.
        device = next(model.parameters()).device
        pre = model(input_ids=torch.tensor([plan["prefix"]], device=device), use_cache=True)
        alt = step(model, clone_hybrid_cache(pre.past_key_values),
                   mismatch["token_id"], len(plan["prefix"]) + 1)["cache"]
        del pre
        hybrid, proof = native_swap(caches["BASE"], alt, ("REC", "Conv"), key)
        for name, cache, cache_proof in (
                ("ALT_NATIVE", alt, {"native": True, "hashes": field_hashes(alt)}),
                ("ALT_REC_Conv", hybrid, proof)):
            outcome = evaluate(model, cache, plan["suffix"] + [space], alphabet, task)
            rows.append({"run_id": run_id, "model": key, "role": role,
                         "phase": "functional", "state_id": task["state_id"],
                         "family": task["family"], "kind": "same_prefix_mismatch",
                         "condition": name, "delay": 0, "status": "VALID",
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["donor_answer"],
                         "alternative_answer": mismatch["answer"],
                         "native_cache_proof": cache_proof,
                         "native_tokenwise": True, **outcome})
    else:
        rows.append({"run_id": run_id, "model": key, "role": role,
                     "phase": "functional", "state_id": task["state_id"],
                     "family": task["family"], "kind": "same_prefix_mismatch",
                     "status": "UNAVAILABLE", "reason": mismatch["reason"]})
    shuffled = design["controls"]["shuffled_REC_Conv"]
    if shuffled["available"]:
        source_task, source_design = items_by_id[shuffled["source_state_id"]]
        source_plan = primary_plan(tokenizer, key, source_task, source_design)
        _, source_donor = native_pair(model, source_plan["prefix"],
                                      source_plan["recipient_token"],
                                      source_plan["donor_token"])
        hybrid, proof = native_swap(caches["BASE"], source_donor,
                                    ("REC", "Conv"), key)
        outcome = evaluate(model, hybrid, plan["suffix"] + [space], alphabet, task)
        rows.append({"run_id": run_id, "model": key, "role": role,
                     "phase": "functional", "state_id": task["state_id"],
                     "family": task["family"], "kind": "shuffled_REC_Conv",
                     "condition": "SHUFFLED", "delay": 0, "status": "VALID",
                     "recipient_answer": task["recipient_answer"],
                     "donor_answer": task["donor_answer"],
                     "source_state_id": source_task["state_id"],
                     "source_donor_answer": source_task["donor_answer"],
                     "native_cache_proof": proof,
                     "native_tokenwise": True, **outcome})
    else:
        rows.append({"run_id": run_id, "model": key, "role": role,
                     "phase": "functional", "state_id": task["state_id"],
                     "family": task["family"], "kind": "shuffled_REC_Conv",
                     "status": "UNAVAILABLE", "reason": shuffled["reason"]})
    return rows


@torch.no_grad()
def run(root: Path, key: str, role: str, phase: str) -> dict:
    cfg = verify(root)["config"]
    if key not in ("Q", "F") or role not in ROLES or phase not in ("functional", "temporal"):
        raise ValueError((key, role, phase))
    if role == "validation":
        verify_stage(root, f"{phase}_{key}_development")
    elif role == "independent_final":
        verify_stage(root, "final_opening")
        verify_stage(root, f"{phase}_{key}_validation")
    if phase == "temporal":
        verify_stage(root, f"functional_{key}_{role}")
    design = json.loads((root / "results/v40/processed/formal_design_a4_v40.json").read_text())["models"][key]
    items = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    plans = design["roles"][role]
    if len(items) != len(plans):
        raise RuntimeError("V40 a4 role count drift")
    by_id = {task["state_id"]: (task, plan) for task, plan in zip(items, plans, strict=True)}
    tokenizer = tokenizer_for(root, key)
    model, _ = load(root, key)
    run_id = f"a4-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir = root / "results/v40/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"formal_{phase}_{key}_{role}_{run_id}.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    if path.exists() or temp.exists():
        raise RuntimeError("V40 a4 raw ID collision")
    counts = {"VALID": 0, "INVALID": 0, "UNAVAILABLE": 0}
    with temp.open("w", encoding="utf-8") as handle:
        for n, (task, design_row) in enumerate(zip(items, plans, strict=True), 1):
            state_rows = []
            try:
                plan = primary_plan(tokenizer, key, task, design_row)
                delays = [0] if phase == "functional" else [d for d in cfg["temporal_delays"] if d]
                state_rows, caches, _ = primary_rows(model, key, role, task, plan,
                    design["answer_ids"], design["space_token_id"],
                    design["neutral_token_id"], delays, phase, run_id)
                if phase == "functional":
                    state_rows.extend(control_rows(model, tokenizer, key, role, task,
                        design_row, plan, caches, design["answer_ids"],
                        design["space_token_id"], by_id, run_id))
            except Exception as exc:
                state_rows.append({"run_id": run_id, "model": key, "role": role,
                                   "phase": phase, "state_id": task["state_id"],
                                   "family": task["family"], "status": "INVALID",
                                   "reason": f"{type(exc).__name__}: {exc}"})
            for row in state_rows:
                counts[row["status"]] += 1
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            print(f"V40 a4 {phase} {key} {role} {n}/{len(items)} "
                  f"{task['state_id']} {state_rows[-1]['status']}", flush=True)
    os.replace(temp, path)
    summary = {"run_id": run_id, "model": key, "role": role, "phase": phase,
               "state_count": len(items), "row_counts": counts,
               "raw_path": str(path.relative_to(root)), "raw_sha256": sha256_file(path),
               "design_sha256": sha256_file(root / "results/v40/processed/formal_design_a4_v40.json"),
               "native_tokenwise": True, "all_states_preserved": True}
    summary_path = root / f"results/v40/processed/formal_{phase}_{key}_{role}_v40.json"
    if summary_path.exists():
        raise RuntimeError("V40 a4 summary exists; raw run preserved, do not overwrite")
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"{phase}_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         str(summary_path.relative_to(root))],
                        {"model": key, "role": role, "phase": phase,
                         "raw_sha256": summary["raw_sha256"]})
    return {**summary, "freeze_digest": seal["freeze_digest"]}


@torch.no_grad()
def pilot(root: Path, key: str) -> dict:
    """Calibration-only end-to-end dry run; never eligible as formal evidence."""
    design = json.loads((root / "results/v40/processed/formal_design_a4_v40.json").read_text())["models"][key]
    items = json.loads((root / "data/v40/calibration_pool_v40.json").read_text())["items"]
    plans = design["roles"]["calibration"]
    if len(items) != len(plans) or len(items) != 16:
        raise RuntimeError("V40 a4 calibration plan drift")
    tokenizer = tokenizer_for(root, key)
    model, _ = load(root, key)
    task, design_row = items[0], plans[0]
    by_id = {item["state_id"]: (item, row) for item, row in zip(items, plans, strict=True)}
    run_id = f"a4-pilot-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    plan = primary_plan(tokenizer, key, task, design_row)
    rows, caches, _ = primary_rows(model, key, "calibration", task, plan,
        design["answer_ids"], design["space_token_id"],
        design["neutral_token_id"], [0], "functional", run_id)
    rows.extend(control_rows(model, tokenizer, key, "calibration", task,
        design_row, plan, caches, design["answer_ids"],
        design["space_token_id"], by_id, run_id))
    path = root / f"results/v40/raw/pilot_{key}_{run_id}.jsonl"
    if path.exists():
        raise RuntimeError("V40 a4 pilot collision")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "calibration_only": True},
                                    sort_keys=True, allow_nan=False) + "\n")
    return {"model": key, "calibration_only": True,
            "state_id": task["state_id"], "rows": len(rows),
            "path": str(path.relative_to(root)), "sha256": sha256_file(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("functional", "temporal", "pilot"))
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", nargs="?", choices=ROLES)
    args = parser.parse_args()
    if args.phase == "pilot":
        if args.role is not None:
            raise SystemExit("pilot has no role argument")
        result = pilot(Path.cwd(), args.model)
    else:
        if args.role is None:
            raise SystemExit("formal phase needs a role argument")
        result = run(Path.cwd(), args.model, args.role, args.phase)
    print(json.dumps(result, indent=2))
