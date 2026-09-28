"""Fresh V39 replay, exact native REC writeback, and algebra equality audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import field_hashes, load, signature, step
from jclosure.protocol_v39 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/calibration_v39.py"
BLOCKS = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, "design")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v39.json").read_text())
    panel = json.loads((root / OUT / "panel_v39.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for role in
               ("calibration", "development", "validation", "independent_final") for row in panel[role]}
    model, tokenizer = load(root, key)
    rows = []
    groups = design["relative_depth_layers"]
    layers = groups["Q2"] + groups["Q3"] + groups["Q4"]
    for n, item in enumerate(design["calibration"], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, prompts[sid])
        probe = item["future_probe_tokens"][0]
        native = step(model, caches["conv"], probe, caches["length"] + 1, design["target_bundle"])
        with FunctionalIntervention(model, key) as hook:
            instrumented = step(model, caches["conv"], probe, caches["length"] + 1, design["target_bundle"])
        if not torch.equal(native["logits"], instrumented["logits"]):
            raise RuntimeError(f"V39 instrumental logits not bitwise {key}:{sid}")
        if field_hashes(native["cache"]) != field_hashes(instrumented["cache"]):
            raise RuntimeError(f"V39 instrumental cache not bitwise {key}:{sid}")
        if any(not np.array_equal(native["targets"][block], instrumented["targets"][block])
               for block in BLOCKS) or len(hook.capture) != 24:
            raise RuntimeError(f"V39 endpoint or layer capture drift {key}:{sid}")
        hybrid, proof = _selective_rec(caches["conv"], caches["joint"], layers)
        if field_hashes(hybrid)["KV"] != caches["recipient_KV_hash"]:
            raise RuntimeError(f"V39 recipient KV changed {key}:{sid}")
        if field_hashes(hybrid)["Conv"] != field_hashes(caches["conv"])["Conv"]:
            raise RuntimeError(f"V39 donor Conv changed {key}:{sid}")
        if not all(p["requested_hash"] == p["realized_hash"] for p in proof):
            raise RuntimeError(f"V39 REC writeback failed {key}:{sid}")
        # Genuine eight-condition endpoint equality on fresh calibration states.
        sig = {}
        proofs = {}
        for label, selected in cfg["conditions"].items():
            chosen = [layer for group in selected for layer in groups[group]]
            cache, current_proof = (_selective_rec(caches["conv"], caches["joint"], chosen)
                                    if chosen else (caches["conv"], []))
            proofs[label] = current_proof
            if field_hashes(cache)["KV"] != caches["recipient_KV_hash"]:
                raise RuntimeError(f"V39 factorial KV drift: {key}:{sid}:{label}")
            sig[label] = signature([step(model, cache, token, caches["length"] + 1,
                                         design["target_bundle"])
                                    for token in item["future_probe_tokens"]],
                                   design["calibration_clean_scales"])
        if not all(np.isfinite(value).all() for value in sig.values()):
            raise RuntimeError(f"V39 nonfinite calibration signature {key}:{sid}")
        if not np.array_equal(sig["R111"], signature(
                [step(model, hybrid, token, caches["length"] + 1, design["target_bundle"])
                 for token in item["future_probe_tokens"]], design["calibration_clean_scales"])):
            raise RuntimeError(f"V39 R111 native replay drift: {key}:{sid}")
        e = {name: value - sig["R000"] for name, value in sig.items()}
        e100, e010, e001 = (e[name] for name in ("R100", "R010", "R001"))
        e110, e101, e011, e111 = (e[name] for name in ("R110", "R101", "R011", "R111"))
        additive = e100 + e010 + e001
        second = e110 + e101 + e011 - e100 - e010 - e001
        third = e111 - e110 - e101 - e011 + e100 + e010 + e001
        identity_residual = float(np.max(np.abs(second + third - e111)))
        if not np.allclose(second + third, e111,
                           rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"V39 inclusion-exclusion equality failed {key}:{sid}")
        norm_e, norm_i = float(np.linalg.norm(e111)), float(np.linalg.norm(third))
        if norm_e > 1e-12 and norm_i > 1e-12:
            p = float(np.dot(third, e111) / norm_e**2)
            fraction = norm_i / norm_e
            cosine = float(np.dot(third, e111) / (norm_i * norm_e))
            metric_residual = abs(p - fraction * cosine)
            if metric_residual > 1e-10 or abs(p) > fraction + 1e-10:
                raise RuntimeError(f"V39 corrected metric identity failed {key}:{sid}")
        else:
            metric_residual = 0.0
        rows.append({"state_id": sid, "model": key, "family": item["family"],
                     "instrumental_bitwise": True, "captured_layers": len(hook.capture),
                     "rec_layers_written": len(proof), "rec_writeback_exact": True,
                     "recipient_KV_unchanged": True, "donor_Conv_unchanged": True,
                     "factorial_conditions": len(sig),
                     "all_condition_rec_writeback_exact": all(
                         all(p["requested_hash"] == p["realized_hash"] for p in proof)
                         for proof in proofs.values()),
                     "formula_algebra_equal": True,
                     "identity_max_abs_residual": identity_residual,
                     "corrected_metric_identity_residual": metric_residual,
                     "q234_correction_norm": norm_e,
                     "additive_norm": float(np.linalg.norm(additive))})
        if n % 5 == 0 or n == len(design["calibration"]):
            print(f"V39 calibration {key} {n}/{len(design['calibration'])}", flush=True)
    frame = pd.DataFrame(rows)
    table = root / OUT / f"calibration_{key}_v39.parquet"
    frame.to_parquet(table, index=False, compression="zstd")
    summary = {"model": key, "states": len(frame), "all_bitwise": True,
               "all_writeback_exact": True, "all_formula_algebra_equal": True,
               "actual_eight_condition_equality": True,
               "max_identity_residual": float(frame.identity_max_abs_residual.max()),
               "max_metric_identity_residual": float(frame.corrected_metric_identity_residual.max()),
               "formal_outcomes_seen": False, "table_sha256": sha256_file(table)}
    path = root / OUT / f"calibration_{key}_v39.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"calibration_{key}", [SOURCE, str(table.relative_to(root)),
                         str(path.relative_to(root)),
                         "artifacts/interaction_genesis_v39_design.freeze.json"],
                        {"model": key, "all_bitwise": True, "all_writeback_exact": True,
                         "all_formula_algebra_equal": True, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
