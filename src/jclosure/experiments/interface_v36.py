"""Calibration-only bitwise replay of the installed single-token recurrence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/interface_v36.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "design")
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    records = []
    for item in design["calibration"]:
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V36 calibration incoming drift {key}:{sid}")
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        probe = int(item["future_probe_tokens"][0])
        baseline = step(model, recipient, probe, length + 1, design["target_bundle"])
        with FunctionalIntervention(model, key) as hook:
            replay = step(model, recipient, probe, length + 1, design["target_bundle"])
        if not torch.equal(baseline["logits"], replay["logits"]):
            raise RuntimeError(f"V36 non-bitwise native logits {key}:{sid}")
        if field_hashes(baseline["cache"]) != field_hashes(replay["cache"]):
            raise RuntimeError(f"V36 non-bitwise native cache {key}:{sid}")
        for block in ("logits", "semantic", "broad_vocabulary", "late_hidden", "workspace"):
            if not np.array_equal(baseline["targets"][block], replay["targets"][block]):
                raise RuntimeError(f"V36 non-bitwise endpoint {key}:{sid}:{block}")
        for layer in design["local_layers"]:
            record = hook.capture[layer]
            for stage in ("POSTCONV_INPUT", "TRANSFORMED_CONTROL", "TRUE_UPDATE", "RECURRENT_READ"):
                if stage not in record:
                    raise RuntimeError(f"V36 missing stage {key}:{layer}:{stage}")
        records.append({"state_id": sid, "probe": probe, "prefix_hash": item["prefix_token_hash"],
                        "token_triple_hash": item["token_triple_hash"],
                        "local_layer_hash": design["local_layer_hash"],
                        "baseline_logit_hash": thash(baseline["logits"]),
                        "replay_logit_hash": thash(replay["logits"]),
                        "bitwise": True})
    result = {"model": key, "n_calibration_states": len(records), "bitwise_native_replay_all": True,
              "local_layers": design["local_layers"], "records": records,
              "development_or_validation_observed": False}
    path = root / OUT / f"interface_{key}_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"interface_{key}", [SOURCE, str(path.relative_to(root)),
                        "src/jclosure/experiments/functional_hooks_v34.py",
                        "artifacts/computational_origin_v36_design.freeze.json"],
                        {"model": key, "n": len(records), "bitwise": True, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "model": key, "n": len(records)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model), indent=2))
