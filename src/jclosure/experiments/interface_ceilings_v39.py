"""Non-mechanistic mixer/residual I234 subtraction ceilings on dev trace states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import load, signature, step
from jclosure.protocol_v39 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v39/processed")
SOURCE = "src/jclosure/experiments/interface_ceilings_v39.py"
ORDER = ("R000", "R100", "R010", "R001", "R110", "R101", "R011", "R111")


def _endpoint_first_probe(root, key):
    arrays = {}
    state_ids = None
    for stage in ("singles", "pairs", "full"):
        path = root / OUT / f"trajectory_{stage}_{key}_development_v39.npz"
        with np.load(path) as z:
            if state_ids is None:
                state_ids = z["state_ids"].tolist()
                width = z["vectors"].shape[-1] // 6
            elif state_ids != z["state_ids"].tolist():
                raise RuntimeError("V39 ceiling state alignment drift")
            for idx, label in enumerate(z["conditions"].tolist()):
                arrays[label] = z["vectors"][:, idx, :width].astype(np.float64)
    return {sid: {label: arrays[label][i] for label in ORDER}
            for i, sid in enumerate(state_ids)}


@torch.no_grad()
def run(root: Path) -> dict:
    verify_stage(root, "trace_synthesis")
    verify_stage(root, "candidate_site_plan")
    design = json.loads((root / OUT / "design_F_v39.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v39.json").read_text())
    groups = design["relative_depth_layers"]
    ids = [sid for family in ("boolean_logic", "modular_arithmetic", "short_graph_traversal",
                              "simple_state_transition", "variable_binding")
           for sid in plan["internal_trace_subsets"]["development"][family]]
    by_id = {row["base_trial_id"]: row for row in design["development"]}
    endpoints = _endpoint_first_probe(root, "F")
    model, tokenizer = load(root, "F")
    rows, vectors = [], []
    for n, sid in enumerate(ids, 1):
        item = by_id[sid]
        caches = prepare(model, tokenizer, "F", item, item["prompt"])
        hybrid, _ = _selective_rec(caches["conv"], caches["joint"],
                                   groups["Q2"]+groups["Q3"]+groups["Q4"])
        probe = item["future_probe_tokens"][0]
        y = endpoints[sid]
        x0, x1, x2, x3, x12, x13, x23, x123 = (y[label] for label in ORDER)
        original = x123-x12-x13-x23+x1+x2+x3-x0
        with np.load(root / OUT / f"internal_tensor_bank_F_development_v39/state_{n:03d}.npz") as bank:
            for layer in groups["Q4"]:
                for stage in ("MIXER_OUTPUT", "BLOCK_OUTPUT"):
                    module = (model.model.layers[layer].mamba if stage == "MIXER_OUTPUT"
                              else model.model.layers[layer])
                    interaction = torch.from_numpy(bank[f"I_l{layer}_{stage}"].copy())
                    audit = {}
                    def replace(_module, _args, output):
                        native = output[0] if isinstance(output, tuple) else output
                        i = interaction.to(device=native.device, dtype=torch.float32).reshape(native.shape)
                        requested = (native.to(torch.float32)-i).to(native.dtype)
                        realized = requested.detach().clone()
                        audit["requested"] = hashlib.sha256(requested.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
                        audit["realized"] = hashlib.sha256(realized.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
                        return (realized, *output[1:]) if isinstance(output, tuple) else realized
                    handle = module.register_forward_hook(replace)
                    try:
                        modified = step(model, hybrid, probe, caches["length"]+1,
                                        design["target_bundle"])
                    finally:
                        handle.remove()
                    if audit.get("requested") != audit.get("realized"):
                        raise RuntimeError(f"V39 ceiling writeback failed: {sid}:{layer}:{stage}")
                    changed = signature([modified], design["calibration_clean_scales"])
                    after = original + changed - x123
                    denominator = max(float(np.dot(original, original)), 1e-12)
                    rows.append({"state_id": sid, "family": item["family"], "model": "F",
                                 "layer": int(layer), "stage": stage, "probe_index": 0,
                                 "removed_direction": float(np.dot(original-after, original)/denominator),
                                 "remaining_norm_ratio": float(np.linalg.norm(after)/
                                                               max(np.linalg.norm(original),1e-12)),
                                 "requested_realized_exact": True,
                                 "downstream_recomputed": True,
                                 "interface_ceiling_not_primitive_mechanism": True,
                                 "vector_index": len(vectors)})
                    vectors.append(np.stack([original, after]).astype(np.float32))
        print(f"V39 interface ceilings {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    table = root / OUT / "interface_ceilings_F_development_v39.parquet"
    archive = root / OUT / "interface_ceilings_vectors_F_development_v39.npz"
    frame.to_parquet(table, index=False, compression="zstd")
    np.savez_compressed(archive, vectors=np.stack(vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        stages=frame.stage.to_numpy(str), layers=frame.layer.to_numpy(int))
    result = {"model": "F", "role": "development_explanatory", "states": len(ids),
              "layers": groups["Q4"], "one_probe_per_state": True,
              "stage_medians": {stage: {"removed_direction": float(g.removed_direction.median()),
                                       "remaining_norm_ratio": float(g.remaining_norm_ratio.median())}
                                for stage, g in frame.groupby("stage")},
              "not_mechanism_evidence": True,
              "files_sha256": {"rows": sha256_file(table), "vectors": sha256_file(archive)}}
    path = root / OUT / "interface_ceilings_F_development_v39.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, "interface_ceilings_F_development",
                        [SOURCE, str(path.relative_to(root)), str(table.relative_to(root)),
                         str(archive.relative_to(root)),
                         "artifacts/interaction_genesis_v39_trace_synthesis.freeze.json"],
                        {"model": "F", "not_mechanism_evidence": True,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
