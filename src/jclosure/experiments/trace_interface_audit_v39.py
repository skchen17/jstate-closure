"""Bitwise equality of the expanded V39 internal tracing hook and native decode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.functional_hooks_v39 import FunctionalIntervention
from jclosure.experiments.runtime_v34 import field_hashes, load, step
from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/trace_interface_audit_v39.py"
HOOK_SOURCE = "src/jclosure/experiments/functional_hooks_v39.py"
BLOCKS = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")


@torch.no_grad()
def run(root: Path, key: str) -> dict:
    verify_stage(root, f"calibration_{key}")
    design = json.loads((root / OUT / f"design_{key}_v39.json").read_text())
    model, tokenizer = load(root, key)
    rows = []
    for item in design["calibration"]:
        caches = prepare(model, tokenizer, key, item, item["prompt"])
        probe = item["future_probe_tokens"][0]
        native = step(model, caches["conv"], probe, caches["length"] + 1,
                      design["target_bundle"])
        with FunctionalIntervention(model, key) as hook:
            traced = step(model, caches["conv"], probe, caches["length"] + 1,
                          design["target_bundle"])
        logits_equal = torch.equal(native["logits"], traced["logits"])
        cache_equal = field_hashes(native["cache"]) == field_hashes(traced["cache"])
        target_equal = all(np.array_equal(native["targets"][name], traced["targets"][name])
                           for name in BLOCKS)
        count_equal = len(hook.capture) == 24
        if not all((logits_equal, cache_equal, target_equal, count_equal)):
            raise RuntimeError(f"V39 expanded trace hook changed native decode: {key}:{item['base_trial_id']}")
        rows.append({"state_id": item["base_trial_id"], "model": key,
                     "family": item["family"], "logits_bitwise": logits_equal,
                     "cache_bitwise": cache_equal, "targets_bitwise": target_equal,
                     "captured_recurrent_layers": len(hook.capture)})
    table = root / OUT / f"trace_interface_audit_{key}_v39.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False, compression="zstd")
    summary = {"model": key, "states": len(rows), "all_bitwise": True,
               "all_24_recurrent_layers_captured": True,
               "expanded_hook_sha256": sha256_file(root / HOOK_SOURCE),
               "table_sha256": sha256_file(table)}
    path = root / OUT / f"trace_interface_audit_{key}_v39.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"trace_interface_audit_{key}",
                        [SOURCE, HOOK_SOURCE, str(path.relative_to(root)),
                         str(table.relative_to(root)),
                         f"artifacts/interaction_genesis_v39_calibration_{key}.freeze.json"],
                        {"model": key, "all_bitwise": True,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("F", "Q"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
