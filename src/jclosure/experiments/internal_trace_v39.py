"""Eight-condition, one-probe native internal interaction trace with tensor banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.functional_hooks_v39 import FunctionalIntervention
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import load, step
from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/internal_trace_v39.py"
HOOK_SOURCE = "src/jclosure/experiments/functional_hooks_v39.py"
ORDER = ("R000", "R100", "R010", "R001", "R110", "R101", "R011", "R111")


def arr(value):
    if isinstance(value, (tuple, list)):
        return np.concatenate([arr(part) for part in value])
    if not isinstance(value, torch.Tensor):
        raise TypeError(type(value))
    return value.detach().to(torch.float32).cpu().numpy().reshape(-1).copy()


def ahash(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def register_blocks(model, key, capture):
    handles = []
    for li, block in enumerate(model.model.layers):
        def pre(_module, args, _kwargs, li=li):
            capture.setdefault(li, {})["BLOCK_INPUT"] = arr(args[0])
        def post(_module, _args, value, li=li):
            capture.setdefault(li, {})["BLOCK_OUTPUT"] = arr(value[0] if isinstance(value, tuple) else value)
        handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(block.register_forward_hook(post))
        if hasattr(block, "self_attn"):
            def attn(_module, _args, value, li=li):
                capture.setdefault(li, {})["ATTENTION_OUTPUT"] = arr(value[0] if isinstance(value, tuple) else value)
            handles.append(block.self_attn.register_forward_hook(attn))
        mlp = getattr(block, "feed_forward", None) or getattr(block, "mlp", None)
        if mlp is not None:
            def ff(_module, _args, value, li=li):
                capture.setdefault(li, {})["MLP_OUTPUT"] = arr(value[0] if isinstance(value, tuple) else value)
            handles.append(mlp.register_forward_hook(ff))
    return handles


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"factorial_gate_{role}")
    audit = verify_stage(root, f"trace_interface_audit_{key}")
    if not audit["all_bitwise"]:
        raise RuntimeError("V39 expanded hook equality audit failed")
    gate = json.loads((root / OUT / f"factorial_gate_{role}_v39.json").read_text())
    if key == "F" and not (gate["falcon_q234_pass"] and gate["falcon_higher_order_pass"]):
        raise RuntimeError("Falcon interaction replication gate failed; do not trace")
    cfg = verify(root)["config"]
    plan = json.loads((root / OUT / "execution_plan_v39.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v39.json").read_text())
    panel = json.loads((root / OUT / "panel_v39.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for r in
               ("calibration", "development", "validation", "independent_final") for row in panel[r]}
    by_id = {row["base_trial_id"]: row for row in design[role]}
    ids = [sid for family in cfg["families"] for sid in
           plan["internal_trace_subsets"][role][family]]
    groups = design["relative_depth_layers"]
    model, tokenizer = load(root, key)
    bank_dir = root / OUT / f"internal_tensor_bank_{key}_{role}_v39"
    if bank_dir.exists() and any(bank_dir.iterdir()):
        raise RuntimeError("V39 internal tensor bank exists; never rewrite")
    bank_dir.mkdir(parents=True, exist_ok=True)
    rows, bank_hashes = [], {}
    for n, sid in enumerate(ids, 1):
        item = by_id[sid]
        caches = prepare(model, tokenizer, key, item, prompts[sid])
        conditions = {}
        for label in ORDER:
            chosen = cfg["conditions"][label]
            selected = [layer for group in chosen for layer in groups[group]]
            cache, _ = (_selective_rec(caches["conv"], caches["joint"], selected)
                        if selected else (caches["conv"], []))
            captures = {}
            handles = register_blocks(model, key, captures)
            try:
                with FunctionalIntervention(model, key) as hook:
                    output = step(model, cache, item["future_probe_tokens"][0],
                                  caches["length"] + 1)
                    for li, factors in hook.capture.items():
                        dest = captures.setdefault(li, {})
                        for name, value in factors.items():
                            if isinstance(value, (torch.Tensor, tuple, list)):
                                dest[name] = arr(value)
            finally:
                for handle in handles:
                    handle.remove()
            conditions[label] = captures
            del output
        bank = {}
        for li in range(len(model.model.layers)):
            stage_names = set.intersection(*(set(conditions[c].get(li, {})) for c in ORDER))
            for stage in sorted(stage_names):
                values = [conditions[c][li][stage] for c in ORDER]
                if len({x.shape for x in values}) != 1:
                    raise RuntimeError(f"V39 internal shape drift: {key}:{sid}:{li}:{stage}")
                x0, x1, x2, x3, x12, x13, x23, x123 = values
                effect = x123 - x0
                interaction = x123 - x12 - x13 - x23 + x1 + x2 + x3 - x0
                en, inorm = float(np.linalg.norm(effect)), float(np.linalg.norm(interaction))
                fraction = inorm / max(en, float(cfg["internal_effect_norm_floor"]))
                projection = float(np.dot(interaction, effect) / max(en*en, 1e-12))
                cosine = float(np.dot(interaction, effect) / max(inorm*en, 1e-12))
                bank[f"I_l{li}_{stage}"] = interaction.astype(np.float32)
                bank[f"E_l{li}_{stage}"] = effect.astype(np.float32)
                rows.append({"state_id": sid, "model": key, "role": role,
                             "family": item["family"], "probe_index": 0,
                             "layer": li, "stage": stage, "width": len(effect),
                             "effect_norm": en, "interaction_norm": inorm,
                             "interaction_fraction": fraction,
                             "signed_projection": projection, "cosine": cosine,
                             "interaction_sha256": ahash(interaction),
                             "effect_sha256": ahash(effect),
                             "condition_sha256_json": json.dumps(
                                 {label: ahash(conditions[label][li][stage]) for label in ORDER},
                                 sort_keys=True)})
        bank_path = bank_dir / f"state_{n:03d}.npz"
        np.savez_compressed(bank_path, **bank)
        bank_hashes[str(bank_path.relative_to(root))] = sha256_file(bank_path)
        print(f"V39 internal trace {key} {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    table = root / OUT / f"internal_trace_{key}_{role}_v39.parquet"
    frame.to_parquet(table, index=False, compression="zstd")
    grouped = []
    threshold = cfg["internal_emergence_gate"]["fraction_min"]
    prevalence_min = cfg["internal_emergence_gate"]["state_prevalence_min"]
    for (layer, stage), g in frame.groupby(["layer", "stage"]):
        prevalence = {family: float((f.interaction_fraction >= threshold).mean())
                      for family, f in g.groupby("family")}
        grouped.append({"layer": int(layer), "stage": stage,
                        "median_fraction": float(g.interaction_fraction.median()),
                        "family_prevalence": prevalence,
                        "families_passing": sum(p >= prevalence_min for p in prevalence.values())})
    stable = []
    for row in grouped:
        nxt = next((other for other in grouped if other["layer"] == row["layer"]+1
                    and other["stage"] == row["stage"]), None)
        if row["families_passing"] >= 4 and nxt and nxt["families_passing"] >= 4:
            stable.append({"layer": row["layer"], "stage": row["stage"],
                           "families_passing": row["families_passing"],
                           "next_layer_families_passing": nxt["families_passing"]})
    summary = {"model": key, "role": role, "states": len(ids), "conditions": ORDER,
               "one_frozen_probe_per_state": True, "layers": len(model.model.layers),
               "rows": len(frame), "stage_layer_summary": grouped,
               "stable_next_layer_interactions": stable,
               "tensor_bank_sha256": bank_hashes,
               "table_sha256": sha256_file(table),
               "descriptive_trace_not_causal_mediation": True}
    path = root / OUT / f"internal_trace_{key}_{role}_v39.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"internal_trace_{key}_{role}",
                        [SOURCE, HOOK_SOURCE, str(path.relative_to(root)),
                         str(table.relative_to(root)), *bank_hashes,
                         f"artifacts/interaction_genesis_v39_factorial_gate_{role}.freeze.json"],
                        {"model": key, "role": role, "summary_sha256": sha256_file(path),
                         "descriptive_trace_not_causal_mediation": True})
    return {"freeze_digest": seal["freeze_digest"], "model": key,
            "states": len(ids), "rows": len(frame), "stable": stable[:20]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("F", "Q"))
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
