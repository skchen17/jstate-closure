"""Descriptive recipient/Conv/joint read and residual traces at quartile boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import capture, prepare
from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.runtime_v34 import load
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v35 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/flow_v35.py"


@torch.no_grad()
def run(root: Path, key: str):
    verify_stage(root, "control_plan")
    verify_stage(root, "full_gate_validation")
    plan = json.loads((root / OUT / "control_plan_v35.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v35.json").read_text())
    panel = json.loads((root / OUT / "panel_v35.json").read_text())
    lookup = prompt_lookup(root, panel)
    selected = set(plan["flow_states"])
    boundaries = {q: design["relative_depth_layers"][q][-1] for q in plan["quartiles"]}
    model, tokenizer = load(root, key)
    rows = []
    for n, item in enumerate((x for x in design["validation"] if x["base_trial_id"] in selected), 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, lookup[sid])
        probe = item["future_probe_tokens"][plan["flow_probe_index"]]
        snapshots = {}
        for branch, cache in (("RECIPIENT", caches["recipient"]),
                              ("CONV", caches["conv"]), ("JOINT", caches["joint"])):
            with ActivationRecorder(model.model.layers, at=boundaries.values(), clone=True, detach=True) as recorder:
                _, reads = capture(model, key, cache, probe, caches["length"], design["target_bundle"])
            snapshots[branch] = {q: {"residual": recorder.activations[layer][0, -1].float().cpu().numpy(),
                                     "read": reads[layer]["RECURRENT_READ"][0, -1].float().cpu().numpy(),
                                     "residual_hash": thash(recorder.activations[layer]),
                                     "read_hash": thash(reads[layer]["RECURRENT_READ"])}
                                 for q, layer in boundaries.items()}
        for q, layer in boundaries.items():
            for kind in ("residual", "read"):
                rec, conv, joint = [snapshots[branch][q][kind].ravel().astype(np.float64)
                                    for branch in ("RECIPIENT", "CONV", "JOINT")]
                rows.append({"model_key": key, "state_id": sid, "family": item["family"],
                             "probe_token_id": int(probe), "quartile_boundary": q,
                             "native_layer_index": layer, "tensor_kind": kind,
                             "conv_vs_recipient_l2": float(np.linalg.norm(conv - rec)),
                             "joint_vs_recipient_l2": float(np.linalg.norm(joint - rec)),
                             "joint_vs_conv_l2": float(np.linalg.norm(joint - conv)),
                             "recipient_hash": snapshots["RECIPIENT"][q][f"{kind}_hash"],
                             "conv_hash": snapshots["CONV"][q][f"{kind}_hash"],
                             "joint_hash": snapshots["JOINT"][q][f"{kind}_hash"],
                             "descriptive_not_causal_gate": True})
        print(f"V35 flow {key} {n}/{len(selected)}", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 5 * 4 * 2:
        raise RuntimeError("V35 flow record incomplete")
    path = root / OUT / f"flow_{key}_validation_v35.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {"model_key": key, "states": 5, "probe_per_state": 1,
               "boundaries": boundaries, "descriptive_not_primary_causal_evidence": True,
               "median_joint_vs_conv_l2_within_model": {
                   f"{q}:{kind}": float(part.joint_vs_conv_l2.median())
                   for (q, kind), part in frame.groupby(["quartile_boundary", "tensor_kind"])},
               "parquet_sha256": sha256_file(path)}
    summary_path = root / OUT / f"flow_{key}_validation_v35.json"
    write_json_atomic(summary_path, summary)
    seal = stage_freeze(root, f"flow_{key}", [SOURCE, str(path.relative_to(root)),
                                               str(summary_path.relative_to(root)),
                                               "artifacts/hierarchical_read_v35_control_plan.freeze.json",
                                               "artifacts/hierarchical_read_v35_full_gate_validation.freeze.json"],
                        {"model_key": key, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model), indent=2))
