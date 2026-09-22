"""Frozen 4×4 REC-group × Conv-group six-probe causal interaction map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.causal_global_v30 import signature
from jclosure.experiments.conv_depth_v30 import partial
from jclosure.experiments.design_v32 import hd
from jclosure.experiments.factorial_v32 import cosine
from jclosure.experiments.forks_formal_v29 import native_layers, state_hashes, total_hash
from jclosure.experiments.global_basis_v30 import token
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/layer_map_v32.py"


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, f"factorial_{role}")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    ids = {sid for family in cfg["families"] for sid in plan["roles"][role][family]["trace_state_ids"]}
    groups = cfg["layer_groups"]
    factorial = pd.read_parquet(root / OUT / f"factorial_{role}_v32.parquet")
    vectors = np.load(root / OUT / f"factorial_vectors_{role}_v32.npz")["vectors"]
    bundle, dense, *_ = context(root)
    scale = scales(root)
    rows = []
    for n, item in enumerate([x for x in design[role] if x["base_trial_id"] in ids], 1):
        sid = item["base_trial_id"]
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError("V32 layer-map incoming drift")
        a = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, design, cfg["readout_layer"])
        b = step(bundle, dense, incoming, token(item["primary_token_id"]), length, design, cfg["readout_layer"])
        main = factorial[(factorial.state_id == sid) & factorial.primary].iloc[0]
        y00, _y10, _y01, _y11, yd = vectors[int(main.vector_index)].astype(np.float64)
        probes = [token(x) for x in item["future_probe_tokens"]]
        future = lambda cache: signature([step(bundle, dense, cache, probe, length + 1, design, cfg["readout_layer"]) for probe in probes], scale)
        for conv_name, conv_layers in groups.items():
            ccache = partial(a["cache"], b["cache"], rec, conv_layers, ())
            c = future(ccache)
            ce = float(np.linalg.norm(yd-c))
            for rec_name, rec_layers in groups.items():
                joint_cache = partial(a["cache"], b["cache"], rec, conv_layers, rec_layers)
                j = future(joint_cache)
                if state_hashes(joint_cache, rec, att)["KV"] != state_hashes(a["cache"], rec, att)["KV"]:
                    raise RuntimeError("V32 map KV drift")
                rows.append({"role": role, "state_id": sid, "family": item["family"], "REC_group": rec_name, "Conv_group": conv_name, "REC_layers_hash": hd(rec_layers), "Conv_layers_hash": hd(conv_layers), "same_group": rec_name == conv_name, "REC_before_Conv": list(groups).index(rec_name) < list(groups).index(conv_name), "REC_after_Conv": list(groups).index(rec_name) > list(groups).index(conv_name), "conv_relative_l2": ce / max(float(np.linalg.norm(yd-y00)), 1e-12), "joint_relative_l2": float(np.linalg.norm(yd-j)) / max(float(np.linalg.norm(yd-y00)), 1e-12), "paired_l2_improvement": (ce - float(np.linalg.norm(yd-j))) / max(float(np.linalg.norm(yd-y00)), 1e-12), "residual_alignment_cosine": cosine(j-c, yd-c), "probe_hash": item["future_probe_hash"], "exact_native_fields": True})
        print(f"V32 layer map {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"rec_conv_layer_map_{role}_v32.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    pair_medians = frame.groupby(["REC_group", "Conv_group"]).paired_l2_improvement.median().to_dict()
    summary = {"role": role, "states": len(ids), "rows": len(frame), "group_count": len(groups), "pair_median_improvement": {f"{a}/{b}": float(v) for (a,b),v in pair_medians.items()}, "same_group_median": float(frame[frame.same_group].paired_l2_improvement.median()), "cross_group_median": float(frame[~frame.same_group].paired_l2_improvement.median()), "response_sha256": sha256_file(path), "group_hash": hd(groups), "independent_final_opened": False}
    jpath = root / OUT / f"rec_conv_layer_map_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"layer_map_{role}", [SOURCE, str(path.relative_to(root)), str(jpath.relative_to(root)), f"artifacts/rec_conv_mechanism_v32_factorial_{role}.freeze.json"], {"summary_sha256": sha256_file(jpath), "group_hash": hd(groups)})
    return {"freeze_digest": stage["freeze_digest"], "states": len(ids), "rows": len(frame), "same_group_median": summary["same_group_median"], "cross_group_median": summary["cross_group_median"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
