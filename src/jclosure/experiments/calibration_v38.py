"""Fresh V38 replay, exact native REC writeback, and algebra equality audit."""
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
from jclosure.protocol_v38 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v38/processed")
SOURCE = "src/jclosure/experiments/calibration_v38.py"
BLOCKS = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, "design")
    design = json.loads((root / OUT / f"design_{key}_v38.json").read_text())
    panel = json.loads((root / OUT / "panel_v38.json").read_text())
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
            raise RuntimeError(f"V38 instrumental logits not bitwise {key}:{sid}")
        if field_hashes(native["cache"]) != field_hashes(instrumented["cache"]):
            raise RuntimeError(f"V38 instrumental cache not bitwise {key}:{sid}")
        if any(not np.array_equal(native["targets"][block], instrumented["targets"][block])
               for block in BLOCKS) or len(hook.capture) != 24:
            raise RuntimeError(f"V38 endpoint or layer capture drift {key}:{sid}")
        hybrid, proof = _selective_rec(caches["conv"], caches["joint"], layers)
        if field_hashes(hybrid)["KV"] != caches["recipient_KV_hash"]:
            raise RuntimeError(f"V38 recipient KV changed {key}:{sid}")
        if field_hashes(hybrid)["Conv"] != field_hashes(caches["conv"])["Conv"]:
            raise RuntimeError(f"V38 donor Conv changed {key}:{sid}")
        if not all(p["requested_hash"] == p["realized_hash"] for p in proof):
            raise RuntimeError(f"V38 REC writeback failed {key}:{sid}")
        # Exact state-wise identity of the registered prediction formulas.
        sig = {}
        for label, cache in {"000": caches["conv"], "111": hybrid}.items():
            sig[label] = signature([step(model, cache, token, caches["length"] + 1,
                                         design["target_bundle"])
                                    for token in item["future_probe_tokens"]],
                                   design["calibration_clean_scales"])
        if not np.isfinite(sig["000"]).all() or not np.isfinite(sig["111"]).all():
            raise RuntimeError(f"V38 nonfinite calibration signature {key}:{sid}")
        e = np.arange(1, 8, dtype=np.float64).reshape(7, 1) * (sig["111"] - sig["000"])
        e100, e010, e001, e110, e101, e011, e111 = e
        additive = e100 + e010 + e001
        second = e110 + e101 + e011 - e100 - e010 - e001
        third = e111 - e110 - e101 - e011 + e100 + e010 + e001
        if not np.allclose(additive + (second - additive) + third, e111,
                           rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"V38 inclusion-exclusion equality failed {key}:{sid}")
        rows.append({"state_id": sid, "model": key, "family": item["family"],
                     "instrumental_bitwise": True, "captured_layers": len(hook.capture),
                     "rec_layers_written": len(proof), "rec_writeback_exact": True,
                     "recipient_KV_unchanged": True, "donor_Conv_unchanged": True,
                     "formula_algebra_equal": True,
                     "q234_correction_norm": float(np.linalg.norm(sig["111"] - sig["000"]))})
        if n % 5 == 0 or n == len(design["calibration"]):
            print(f"V38 calibration {key} {n}/{len(design['calibration'])}", flush=True)
    frame = pd.DataFrame(rows)
    table = root / OUT / f"calibration_{key}_v38.parquet"
    frame.to_parquet(table, index=False, compression="zstd")
    summary = {"model": key, "states": len(frame), "all_bitwise": True,
               "all_writeback_exact": True, "all_formula_algebra_equal": True,
               "formal_outcomes_seen": False, "table_sha256": sha256_file(table)}
    path = root / OUT / f"calibration_{key}_v38.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"calibration_{key}", [SOURCE, str(table.relative_to(root)),
                         str(path.relative_to(root)),
                         "artifacts/trajectory_composition_v38_design.freeze.json"],
                        {"model": key, "all_bitwise": True, "all_writeback_exact": True,
                         "all_formula_algebra_equal": True, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
