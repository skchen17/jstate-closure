"""Task-grounded Qwen REC depth-trajectory factorial with natural continuation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.experiments.calibration_v40_a2 import score
from jclosure.experiments.formal_v40_a4 import native_pair, primary_plan
from jclosure.experiments.runtime_v34 import field_hashes, load, native_swap, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v40_a4 import verify_stage as verify_a4_stage
from jclosure.protocol_v40_a5 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

SOURCE = "src/jclosure/experiments/trajectory_v40_a5.py"
ORDER = ("R000", "R100", "R010", "R001", "R110", "R101", "R011", "R111")
ROLES = ("development", "validation", "independent_final")


def group_layers(model) -> dict[str, list[int]]:
    recurrent = [i for i, layer in enumerate(model.model.layers)
                 if layer.layer_type == "linear_attention"]
    if len(recurrent) != 24:
        raise RuntimeError(f"V40 a5 Q recurrent-layer drift: {recurrent}")
    return {name: recurrent[i:i + 8] for name, i in
            (("early", 0), ("middle", 8), ("late", 16))}


def selective_rec(base, donor, layers: list[int]) -> tuple[object, dict]:
    selected = set(layers)
    work = clone_hybrid_cache(base)
    proofs = []
    for i in selected:
        source = donor.layers[i].recurrent_states
        target = work.layers[i].recurrent_states
        if source.shape != target.shape or source.dtype != target.dtype:
            raise RuntimeError(f"V40 a5 REC topology mismatch: {i}")
        work.layers[i].recurrent_states = source.detach().clone()
        if not torch.equal(work.layers[i].recurrent_states, source):
            raise RuntimeError(f"V40 a5 REC exact write failed: {i}")
        proofs.append({"layer": i, "requested_hash": thash(source),
                       "realized_hash": thash(work.layers[i].recurrent_states)})
    for i, actual in enumerate(work.layers):
        for field in ("recurrent_states", "conv_states", "keys", "values"):
            value = getattr(actual, field, None)
            expected = (donor if i in selected and field == "recurrent_states" else base).layers[i]
            reference = getattr(expected, field, None)
            if isinstance(value, torch.Tensor) and not torch.equal(value, reference):
                raise RuntimeError(f"V40 a5 untouched field changed: {i}:{field}")
            if isinstance(value, torch.Tensor) != isinstance(reference, torch.Tensor):
                raise RuntimeError(f"V40 a5 field presence changed: {i}:{field}")
    hashes = field_hashes(work)
    base_hashes = field_hashes(base)
    if hashes["Conv"] != base_hashes["Conv"] or hashes["KV"] != base_hashes["KV"]:
        raise RuntimeError("V40 a5 non-REC channel drift")
    return work, {"selected_layers": sorted(selected),
                  "exact_selected": all(p["requested_hash"] == p["realized_hash"]
                                        for p in proofs),
                  "untouched_exact": True,
                  "recipient_conv_hash": base_hashes["Conv"],
                  "recipient_kv_hash": base_hashes["KV"],
                  "realized_conv_hash": hashes["Conv"],
                  "realized_kv_hash": hashes["KV"]}


def _state_norms(cache, layers: list[int]) -> dict:
    return {str(i): {
        field: float(torch.linalg.vector_norm(getattr(cache.layers[i], field).float()).item())
        for field in ("recurrent_states", "conv_states")}
        for i in layers}


def _state_hashes(cache, layers: list[int]) -> dict:
    return {str(i): {
        field: thash(getattr(cache.layers[i], field))
        for field in ("recurrent_states", "conv_states")}
        for i in layers}


@torch.no_grad()
def continuation(model, cache, tokens: list[int], delay: int,
                 capture_layers: list[int]) -> tuple[torch.Tensor, dict, dict]:
    if not tokens:
        raise RuntimeError("V40 a5 empty continuation")
    checkpoints = {0: "first_continuation_token", len(tokens) - 1: "final_answer_space"}
    if delay:
        checkpoints[delay - 1] = "post_neutral"
    work = cache
    length = int(cache.get_seq_length())
    hidden, state_norms, state_hashes = {}, {}, {}
    logits = None
    for index, token in enumerate(tokens):
        length += 1
        if index in checkpoints:
            with ActivationRecorder(model.model.layers, capture_layers) as recorder:
                out = step(model, work, token, length)
            tag = checkpoints[index]
            hidden[tag] = {str(li): recorder.activations[li][0, -1].float().cpu().numpy()
                           for li in capture_layers}
            state_norms[tag] = _state_norms(out["cache"], capture_layers)
            state_hashes[tag] = _state_hashes(out["cache"], capture_layers)
        else:
            out = step(model, work, token, length)
        work, logits = out["cache"], out["logits"]
    return logits.detach(), hidden, state_norms, state_hashes


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else None


@torch.no_grad()
def state_rows(model, tokenizer, role: str, task: dict, design_row: dict,
               answer_ids: dict, space: int, neutral: int,
               cfg: dict, run_id: str) -> list[dict]:
    plan = primary_plan(tokenizer, "Q", task, design_row)
    recipient, donor = native_pair(model, plan["prefix"],
                                   plan["recipient_token"], plan["donor_token"])
    common, common_proof = native_swap(recipient, donor, ("Conv",), "Q")
    if not common_proof["requested_exact"] or not common_proof["untouched_exact"]:
        raise RuntimeError("V40 a5 common Conv write failed")
    groups = group_layers(model)
    capture_layers = [values[-1] for values in groups.values()]
    caches = {"BASE": recipient, "DONOR_NATIVE": donor}
    proofs = {"BASE": {"native": True, "hashes": field_hashes(recipient)},
              "DONOR_NATIVE": {"native": True, "hashes": field_hashes(donor)}}
    for label in ORDER:
        layer_ids = [i for group in cfg["conditions"][label] for i in groups[group]]
        caches[label], rec_proof = selective_rec(common, donor, layer_ids)
        proofs[label] = {"common_conv_proof": common_proof,
                         "rec_proof": rec_proof,
                         "no_future_output_copy": True}
    alphabet = answer_ids[task["family"]]
    rows = []
    for delay in cfg["delays"]:
        tokens = [neutral] * delay + plan["suffix"] + [space]
        outcomes = {}
        for label in ("BASE", "DONOR_NATIVE", *ORDER):
            logits, hidden, norms, hashes = continuation(model, caches[label],
                                                          tokens, delay, capture_layers)
            outcomes[label] = {"answer": score(logits, alphabet,
                task["recipient_answer"], task["donor_answer"]),
                "hidden": hidden, "state_norms": norms, "state_hashes": hashes}
        for label in ("BASE", "DONOR_NATIVE", *ORDER):
            item = outcomes[label]
            comparisons = {}
            for checkpoint, activations in item["hidden"].items():
                comparisons[checkpoint] = {layer: {
                    "to_native_donor": cosine(value, outcomes["DONOR_NATIVE"]["hidden"][checkpoint][layer]),
                    "to_common_conv": cosine(value, outcomes["R000"]["hidden"][checkpoint][layer])}
                    for layer, value in activations.items()}
            rows.append({"run_id": run_id, "model": "Q", "role": role,
                         "phase": "trajectory", "state_id": task["state_id"],
                         "family": task["family"], "condition": label,
                         "delay": delay, "status": "VALID",
                         "recipient_answer": task["recipient_answer"],
                         "donor_answer": task["donor_answer"],
                         "capture_layers": capture_layers,
                         "native_cache_proof": proofs[label],
                         "checkpoint_state_norms": item["state_norms"],
                         "checkpoint_state_hashes": item["state_hashes"],
                         "checkpoint_hidden_cosines": comparisons,
                         "native_downstream": True,
                         "future_output_copied": False,
                         **item["answer"]})
    return rows


@torch.no_grad()
def pilot(root: Path) -> dict:
    if (root / "artifacts/channel_function_v40_a5.freeze.json").exists():
        raise RuntimeError("V40 a5 pilot must precede formal freeze")
    import yaml
    cfg = yaml.safe_load((root / "configs/channel_function_v40_trajectory_a5.yaml").read_text())
    design = json.loads((root / "results/v40/processed/formal_design_a4_v40.json").read_text())["models"]["Q"]
    task = json.loads((root / "data/v40/calibration_pool_v40.json").read_text())["items"][0]
    plan = design["roles"]["calibration"][0]
    model, tokenizer = load(root, "Q")
    run_id = f"a5-pilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    rows = state_rows(model, tokenizer, "calibration", task, plan,
                      design["answer_ids"], design["space_token_id"],
                      design["neutral_token_id"], cfg, run_id)
    path = root / f"results/v40/raw/pilot_trajectory_Q_{run_id}.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "calibration_only": True},
                                    sort_keys=True, allow_nan=False) + "\n")
    return {"calibration_only": True, "rows": len(rows),
            "path": str(path.relative_to(root)), "sha256": sha256_file(path)}


@torch.no_grad()
def run(root: Path, role: str) -> dict:
    cfg = verify(root)["config"]
    if role not in ROLES:
        raise ValueError(role)
    verify_a4_stage(root, f"functional_Q_{role}")
    if role == "validation":
        verify_stage(root, "trajectory_Q_development")
    elif role == "independent_final":
        verify_a4_stage(root, "final_opening")
        verify_stage(root, "trajectory_Q_validation")
    design = json.loads((root / "results/v40/processed/formal_design_a4_v40.json").read_text())["models"]["Q"]
    items = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    plans = design["roles"][role]
    if len(items) != len(plans):
        raise RuntimeError("V40 a5 role count drift")
    model, tokenizer = load(root, "Q")
    run_id = f"a5-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    path = root / f"results/v40/raw/trajectory_Q_{role}_{run_id}.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    if path.exists() or temp.exists():
        raise RuntimeError("V40 a5 raw ID collision")
    valid, invalid = 0, 0
    with temp.open("w", encoding="utf-8") as handle:
        for n, (task, plan) in enumerate(zip(items, plans, strict=True), 1):
            try:
                rows = state_rows(model, tokenizer, role, task, plan,
                                  design["answer_ids"], design["space_token_id"],
                                  design["neutral_token_id"], cfg, run_id)
            except Exception as exc:
                rows = [{"run_id": run_id, "model": "Q", "role": role,
                         "phase": "trajectory", "state_id": task["state_id"],
                         "family": task["family"], "status": "INVALID",
                         "reason": f"{type(exc).__name__}: {exc}"}]
            valid += sum(r["status"] == "VALID" for r in rows)
            invalid += sum(r["status"] == "INVALID" for r in rows)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            print(f"V40 a5 trajectory Q {role} {n}/{len(items)} "
                  f"{task['state_id']} {rows[-1]['status']}", flush=True)
    os.replace(temp, path)
    summary = {"model": "Q", "role": role, "run_id": run_id,
               "states": len(items), "valid_rows": valid, "invalid_rows": invalid,
               "raw_path": str(path.relative_to(root)), "raw_sha256": sha256_file(path),
               "future_output_copied": False}
    summary_path = root / f"results/v40/processed/trajectory_Q_{role}_v40.json"
    if summary_path.exists():
        raise RuntimeError("V40 a5 summary exists; preserve raw rerun")
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"trajectory_Q_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         str(summary_path.relative_to(root))],
                        {"role": role, "raw_sha256": summary["raw_sha256"],
                         "future_output_copied": False})
    return {**summary, "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=(*ROLES, "pilot"))
    args = parser.parse_args()
    print(json.dumps(pilot(Path.cwd()) if args.role == "pilot" else run(Path.cwd(), args.role), indent=2))
