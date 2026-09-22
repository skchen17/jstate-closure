"""Diagnostic mediation probes at the V32 development-nominated recurrent layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import signature
from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.factorial_v32 import cosine
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.intervene_v32 import capture_or_replace
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/subsite_v32.py"
COMPONENTS = ("gate_a", "gate_b", "qkv_projection", "normalized_recurrent_read", "post_residual_mlp_normalized_input")


@torch.no_grad()
def run(root, role):
    verify_stage(root, "site_selection")
    verify_stage(root, f"factorial_{role}")
    cfg = verify(root)["config"]
    d = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    site = json.loads((root / OUT / "site_selection_v32.json").read_text())
    ids = {sid for family in cfg["families"] for sid in plan["roles"][role][family]["trace_state_ids"]}
    f = pd.read_parquet(root / OUT / f"factorial_{role}_v32.parquet")
    vec = np.load(root / OUT / f"factorial_vectors_{role}_v32.npz")["vectors"]
    bundle, dense, *_ = context(root)
    scale = scales(root)
    layer = site["candidate_layer"]
    block = bundle.layers[layer]
    modules = {"gate_a": block.linear_attn.in_proj_a, "gate_b": block.linear_attn.in_proj_b, "qkv_projection": block.linear_attn.in_proj_qkv, "normalized_recurrent_read": block.linear_attn.norm, "post_residual_mlp_normalized_input": block.post_attention_layernorm}
    rows, audit = [], []
    for n, item in enumerate([x for x in d[role] if x["base_trial_id"] in ids], 1):
        sid = item["base_trial_id"]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError("V32 subsite incoming drift")
        a = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, d, cfg["readout_layer"])
        b = step(bundle, dense, incoming, token(item["primary_token_id"]), length, d, cfg["readout_layer"])
        c01 = partial(a["cache"], b["cache"], rec, rec, ())
        c11 = partial(a["cache"], b["cache"], rec, rec, rec)
        z = f[(f.state_id == sid) & f.primary].iloc[0]
        y00, _y10, y01, y11, yd = vec[int(z.vector_index)].astype(np.float64)
        error01, error11 = float(np.linalg.norm(yd-y01)), float(np.linalg.norm(yd-y11))
        benefit = error01 - error11
        for component in COMPONENTS:
            module = modules[component]
            removed, inserted = [], []
            for probe_id in item["future_probe_tokens"]:
                probe = token(probe_id)
                with capture_or_replace(module) as q01:
                    step(bundle, dense, c01, probe, length + 1, d, cfg["readout_layer"])
                with capture_or_replace(module) as q11:
                    step(bundle, dense, c11, probe, length + 1, d, cfg["readout_layer"])
                with capture_or_replace(module, q01["captured"]) as remove:
                    removed.append(step(bundle, dense, c11, probe, length + 1, d, cfg["readout_layer"]))
                with capture_or_replace(module, q11["captured"]) as insert:
                    inserted.append(step(bundle, dense, c01, probe, length + 1, d, cfg["readout_layer"]))
                exact = all(q["calls"] == 1 for q in (q01,q11,remove,insert)) and remove["output_hash"] == q01["input_hash"] and insert["output_hash"] == q11["input_hash"]
                if not exact:
                    raise RuntimeError("V32 diagnostic subsite patch failed")
                audit.append({"role": role, "state_id": sid, "family": item["family"], "component": component, "layer": layer, "probe_id": probe_id, "remove_requested_hash": q01["input_hash"], "remove_realized_hash": remove["output_hash"], "insert_requested_hash": q11["input_hash"], "insert_realized_hash": insert["output_hash"], "exact": True})
            yi, yr = signature(removed, scale), signature(inserted, scale)
            rows.append({"role": role, "state_id": sid, "family": item["family"], "component": component, "layer": layer, "benefit": benefit, "removed_benefit_fraction": (float(np.linalg.norm(yd-yi))-error11)/max(benefit,1e-12), "restored_benefit_fraction": (error01-float(np.linalg.norm(yd-yr)))/max(benefit,1e-12), "remove_direction_cosine": cosine(y11-yi,y11-y01), "restore_direction_cosine": cosine(yr-y01,y11-y01), "probe_hash": item["future_probe_hash"], "exact": True, "diagnostic_only": True})
        print(f"V32 subsite {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    paths = {"subsite": root / OUT / f"subsite_{role}_v32.parquet", "audit": root / OUT / f"subsite_audit_{role}_v32.parquet"}
    frame.to_parquet(paths["subsite"], index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(paths["audit"], index=False, compression="zstd")
    summary = {"role": role, "states": len(ids), "components": list(COMPONENTS), "medians": {name: {k: float(group[k].median()) for k in ("removed_benefit_fraction","restored_benefit_fraction","remove_direction_cosine","restore_direction_cosine")} for name, group in frame.groupby("component")}, "diagnostic_only": True, "response_sha256": {k: sha256_file(v) for k,v in paths.items()}}
    jpath = root / OUT / f"subsite_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"subsite_{role}", [SOURCE, *(str(p.relative_to(root)) for p in paths.values()), str(jpath.relative_to(root)), "artifacts/rec_conv_mechanism_v32_site_selection.freeze.json"], {"summary_sha256": sha256_file(jpath), "diagnostic_only": True})
    return {"freeze_digest": stage["freeze_digest"], "states": len(ids), "medians": summary["medians"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development","validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
