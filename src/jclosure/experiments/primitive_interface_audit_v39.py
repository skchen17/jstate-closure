"""Native equality audit of Falcon primitive hook at every recurrent layer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.primitive_hook_v39 import FalconPrimitiveHook, FACTOR_KEYS
from jclosure.experiments.runtime_v34 import field_hashes, load, step
from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/primitive_interface_audit_v39.py"
HOOK_SOURCE = "src/jclosure/experiments/primitive_hook_v39.py"
BLOCKS = ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace")


@torch.no_grad()
def run(root: Path) -> dict:
    verify_stage(root, "calibration_F")
    design = json.loads((root / OUT / "design_F_v39.json").read_text())
    model, tokenizer = load(root, "F")
    rows = []
    for n, item in enumerate(design["calibration"], 1):
        caches = prepare(model, tokenizer, "F", item, item["prompt"])
        probe = item["future_probe_tokens"][0]
        native = step(model, caches["conv"], probe, caches["length"]+1,
                      design["target_bundle"])
        for layer in design["recurrent_layers"]:
            with FalconPrimitiveHook(model, layer) as hook:
                replay = step(model, caches["conv"], probe, caches["length"]+1,
                              design["target_bundle"])
            valid = (torch.equal(native["logits"], replay["logits"]) and
                     field_hashes(native["cache"]) == field_hashes(replay["cache"]) and
                     all(np.array_equal(native["targets"][block], replay["targets"][block])
                         for block in BLOCKS) and
                     all(name in hook.capture for name in FACTOR_KEYS))
            if not valid:
                raise RuntimeError(f"V39 primitive hook altered native decode: {item['base_trial_id']}:{layer}")
            rows.append({"state_id": item["base_trial_id"], "family": item["family"],
                         "layer": int(layer), "native_bitwise": True,
                         "all_primitive_factors_captured": True})
        if n % 5 == 0:
            print(f"V39 primitive equality {n}/{len(design['calibration'])}", flush=True)
    table = root / OUT / "primitive_interface_audit_F_v39.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False, compression="zstd")
    summary = {"model": "F", "states": len(design["calibration"]), "rows": len(rows),
               "recurrent_layers": len(design["recurrent_layers"]), "all_bitwise": True,
               "all_primitive_factors_captured": True,
               "hook_sha256": sha256_file(root / HOOK_SOURCE),
               "table_sha256": sha256_file(table)}
    path = root / OUT / "primitive_interface_audit_F_v39.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, "primitive_interface_audit_F",
                        [SOURCE, HOOK_SOURCE, str(path.relative_to(root)),
                         str(table.relative_to(root)),
                         "artifacts/interaction_genesis_v39_calibration_F.freeze.json"],
                        {"all_bitwise": True, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
