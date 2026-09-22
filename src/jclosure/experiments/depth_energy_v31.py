"""TRAIN-only Conv depth-energy patterns; descriptive, not a carrier test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.primitive_matrix_v31 import OUT, SCRATCH
from jclosure.protocol_v31 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/depth_energy_v31.py"


def run(root: Path):
    verify_stage(root, "primitive_collect")
    manifest = json.loads((root / OUT / "primitive_training_rows_v31.json").read_text())
    matrix = np.memmap(SCRATCH / "train_write_matrix_v31.f32", dtype=np.float32, mode="r", shape=(manifest["nrows"], manifest["dimension"]))
    specs = []
    offset = 0
    for layer, name, shape, count in manifest["layout"]:
        if name == "conv_states":
            specs.append((layer, offset, offset + count))
        offset += count
    if len(specs) != 24 or offset != manifest["dimension"]:
        raise RuntimeError("V31 exact Conv layout mismatch")
    rows = []
    for i, meta in enumerate(manifest["rows"]):
        x = matrix[i]
        energy = [(layer, float(np.dot(x[begin:end], x[begin:end]))) for layer, begin, end in specs]
        total = max(sum(v for _, v in energy), 1e-12)
        for layer, e in energy:
            rows.append({"train_row": i, "state_id": meta["state_id"], "family": meta["family"], "token_id": meta["candidate_token_id"], "token_category": meta["category"], "conv_layer": layer, "conv_energy": e, "conv_energy_fraction": e / total})
        if (i + 1) % 100 == 0:
            print(f"V31 TRAIN Conv energy {i+1}/{manifest['nrows']}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / "train_conv_depth_energy_v31.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    category = frame.groupby(["token_category", "conv_layer"]).conv_energy_fraction.median().reset_index()
    bycat = {cat: [float(x) for x in group.sort_values("conv_layer").conv_energy_fraction] for cat, group in category.groupby("token_category")}
    summary = {"TRAIN_write_rows": manifest["nrows"], "Conv_layers": 24, "token_category_median_energy_profiles": bycat, "descriptive_only_not_causal_route": True, "response_sha256": sha256_file(path)}
    jp = root / OUT / "train_conv_depth_energy_v31.json"
    write_json_atomic(jp, summary)
    freeze = stage_freeze(root, "train_depth_energy", [SOURCE, str(path.relative_to(root)), str(jp.relative_to(root)), "artifacts/compositional_natural_writes_v31_primitive_collect.freeze.json"], {"summary_sha256": sha256_file(jp), "rows": len(frame)})
    return {"freeze_digest": freeze["freeze_digest"], "rows": len(frame), "categories": list(bycat)}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
