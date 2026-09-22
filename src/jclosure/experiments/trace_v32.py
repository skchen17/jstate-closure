"""V32 next-token sublayer trace and development-only recurrent-site nomination."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.design_v32 import hd
from jclosure.experiments.factorial_v32 import cosine
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/trace_v32.py"


@contextlib.contextmanager
def hook_trace(blocks, rec):
    values = {}
    handles = []
    def capture(layer, name):
        def fn(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            values[(layer, name)] = tensor.detach().float().reshape(-1).cpu().numpy().copy()
        return fn
    try:
        for layer in rec:
            module = blocks[layer].linear_attn
            for name, submodule in (("qkv_projection", module.in_proj_qkv), ("gate_a", module.in_proj_a), ("gate_b", module.in_proj_b), ("normalized_recurrent_read", module.norm), ("recurrent_output", module)):
                handles.append(submodule.register_forward_hook(capture(layer, name)))
        yield values
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, f"factorial_{role}")
    if role == "validation":
        verify_stage(root, "trace_development")
        verify_stage(root, "site_selection")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    ids = {sid for family in cfg["families"] for sid in plan["roles"][role][family]["trace_state_ids"]}
    bundle, dense, *_ = context(root)
    blocks = bundle.layers
    rows = []
    for n, item in enumerate([x for x in design[role] if x["base_trial_id"] in ids], 1):
        sid = item["base_trial_id"]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError("V32 trace incoming state drift")
        a = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, design, cfg["readout_layer"])
        b = step(bundle, dense, incoming, token(item["primary_token_id"]), length, design, cfg["readout_layer"])
        conditions = {"Y00": a["cache"], "Y10": partial(a["cache"], b["cache"], rec, (), rec), "Y01": partial(a["cache"], b["cache"], rec, rec, ()), "Y11": partial(a["cache"], b["cache"], rec, rec, rec), "Ydonor": b["cache"]}
        probe = token(item["future_probe_tokens"][0])
        outputs, sublayers = {}, {}
        for name, cache in conditions.items():
            with hook_trace(blocks, rec) as captured:
                outputs[name] = step(bundle, dense, cache, probe, length + 1, design, cfg["readout_layer"], layers=range(len(blocks)))
            sublayers[name] = captured
        final_residual_target = outputs["Ydonor"]["activations"][len(blocks) - 1] - outputs["Y01"]["activations"][len(blocks) - 1]
        for layer in range(len(blocks)):
            r11 = outputs["Y11"]["activations"][layer] - outputs["Y01"]["activations"][layer]
            e = outputs["Ydonor"]["activations"][layer] - outputs["Y01"]["activations"][layer]
            rows.append({"role": role, "state_id": sid, "family": item["family"], "layer": layer, "boundary": "post_block_residual", "conditional_norm": float(np.linalg.norm(r11)), "boundary_residual_alignment": cosine(r11, e), "final_residual_alignment": cosine(r11, final_residual_target), "probe_hash": item["future_probe_hash"], "probe_id": item["future_probe_tokens"][0], "trace_only_one_of_six_probes": True})
            if layer not in rec:
                continue
            for boundary in ("qkv_projection", "gate_a", "gate_b", "normalized_recurrent_read", "recurrent_output"):
                r11 = sublayers["Y11"][(layer, boundary)] - sublayers["Y01"][(layer, boundary)]
                e = sublayers["Ydonor"][(layer, boundary)] - sublayers["Y01"][(layer, boundary)]
                rows.append({"role": role, "state_id": sid, "family": item["family"], "layer": layer, "boundary": boundary, "conditional_norm": float(np.linalg.norm(r11)), "boundary_residual_alignment": cosine(r11, e), "final_residual_alignment": cosine(r11, final_residual_target) if r11.shape == final_residual_target.shape else None, "probe_hash": item["future_probe_hash"], "probe_id": item["future_probe_tokens"][0], "trace_only_one_of_six_probes": True})
        print(f"V32 trace {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"next_token_trace_{role}_v32.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    first = frame[(frame.boundary == "recurrent_output") & (frame.conditional_norm > 1e-6)].groupby("state_id").layer.min().to_dict()
    summary = {"role": role, "states": len(ids), "rows": len(frame), "trace_probe_count": 1, "functional_probe_count": cfg["future_probe_count"], "first_measurable_recurrent_output_by_state": first, "first_measurable_is_not_causal_site": True, "response_sha256": sha256_file(path)}
    jpath = root / OUT / f"next_token_trace_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"trace_{role}", [SOURCE, str(path.relative_to(root)), str(jpath.relative_to(root)), f"artifacts/rec_conv_mechanism_v32_factorial_{role}.freeze.json"], {"summary_sha256": sha256_file(jpath), "trace_hash": sha256_file(path)})
    return {"freeze_digest": stage["freeze_digest"], "states": len(ids), "rows": len(frame), "first_measurable_median_layer": float(np.median(list(first.values()))) if first else None}


def select_site(root: Path):
    verify_stage(root, "trace_development")
    cfg = verify(root)["config"]
    frame = pd.read_parquet(root / OUT / "next_token_trace_development_v32.parquet")
    candidates = frame[(frame.boundary == "recurrent_output") & frame.final_residual_alignment.notna()]
    medians = candidates.groupby("layer").final_residual_alignment.median().to_dict()
    if not medians:
        raise RuntimeError("V32 no valid recurrent-output trace candidates")
    selected = sorted(medians, key=lambda layer: (-medians[layer], layer))[0]
    payload = {"candidate_boundary": "recurrent_output", "candidate_layer": int(selected), "development_median_final_residual_alignment": float(medians[selected]), "all_development_candidate_medians": {str(k): float(v) for k, v in medians.items()}, "selection_rule": cfg["trace_candidate_rule"], "validation_not_inspected": True, "independent_final_not_inspected": True}
    path = root / OUT / "site_selection_v32.json"
    write_json_atomic(path, payload)
    stage = stage_freeze(root, "site_selection", [SOURCE, str(path.relative_to(root)), "results/v32/processed/next_token_trace_development_v32.parquet", "artifacts/rec_conv_mechanism_v32_trace_development.freeze.json"], {"site_hash": hd(["recurrent_output", int(selected)]), "selection_sha256": sha256_file(path)})
    return {"freeze_digest": stage["freeze_digest"], **payload}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("development", "validation", "select"))
    args = parser.parse_args()
    print(json.dumps(select_site(Path.cwd()) if args.command == "select" else run(Path.cwd(), args.command), indent=2))
