"""Calibration-only comparison of source-derived equations with installed mixers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.operator_v36 import factors, native_local, parts
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, step
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/equation_audit_v36.py"
# Code-frozen before opening calibration responses. BF16/fp32 mixed arithmetic:
# require elementwise max error <= 0.02 + 0.03 * RMS(native raw).
ABS_TOL = 0.02
RMS_REL_TOL = 0.03


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "design")
    verify_stage(root, f"interface_{key}")
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    rows = []
    for item in design["calibration"]:
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V36 equation incoming drift {key}:{sid}")
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        donor = step(model, incoming, item["donor_token_id"], length)["cache"]
        conv, _ = native_swap(recipient, donor, ["Conv"], key)
        rec_only, _ = native_swap(recipient, donor, ["REC"], key)
        h = {}
        handles = []
        for layer in design["local_layers"]:
            module = model.model.layers[layer].linear_attn if key == "Q" else model.model.layers[layer].mamba
            def capture(_m, args, kwargs, layer=layer):
                h[layer] = kwargs.get("hidden_states", args[0] if args else None).detach().clone()
            handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        try:
            step(model, conv, int(item["future_probe_tokens"][0]), length + 1)
        finally:
            for handle in handles: handle.remove()
        for layer in design["local_layers"]:
            module = model.model.layers[layer].linear_attn if key == "Q" else model.model.layers[layer].mamba
            for name, cache in (("recipient", recipient), ("donor", donor)):
                # Hold local Conv at recipient while varying incoming REC state.
                controlled = rec_only if name == "donor" else recipient
                f = factors(key, module, h[layer], recipient)
                prediction = parts(key, f, cache.layers[layer].recurrent_states)["raw"].float()
                observed = native_local(module, h[layer], controlled)["raw"].float()
                error = float((prediction - observed).abs().max().item())
                rms = float(observed.square().mean().sqrt().item())
                limit = ABS_TOL + RMS_REL_TOL * rms
                if prediction.shape != observed.shape or error > limit:
                    raise RuntimeError(f"V36 equation mismatch {key}:{sid}:{layer}:{name} {error}>{limit}")
                rows.append({"model": key, "state_id": sid, "family": item["family"], "layer": layer,
                             "condition": name, "max_abs_error": error, "native_raw_rms": rms,
                             "tolerance": limit, "pass": True})
    frame = pd.DataFrame(rows)
    path = root / OUT / f"equation_audit_{key}_v36.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {"model": key, "states": 20, "rows": len(frame), "all_pass": bool(frame["pass"].all()),
               "max_error": float(frame.max_abs_error.max()), "max_tolerance": float(frame.tolerance.max()),
               "abs_tolerance": ABS_TOL, "rms_relative_tolerance": RMS_REL_TOL,
               "rows_sha256": sha256_file(path)}
    summary_path = root / OUT / f"equation_audit_{key}_v36.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"equation_{key}", [SOURCE,
                        "src/jclosure/experiments/operator_v36.py", str(path.relative_to(root)),
                        str(summary_path.relative_to(root)),
                        f"artifacts/computational_origin_v36_interface_{key}.freeze.json"],
                        {"model": key, "all_pass": True, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model), indent=2))
