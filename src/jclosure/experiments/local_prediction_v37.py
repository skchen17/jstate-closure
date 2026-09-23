"""Freeze exact 3×3 local predictions before native counterfactual continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.local_v36 import _capture, _module
from jclosure.experiments.native_operator_v37 import predict_with_factors
from jclosure.experiments.operator_v36 import factors
from jclosure.experiments.runtime_v34 import load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v37 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/local_prediction_v37.py"
LETTERS = ("A", "B", "C")


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    if role not in ("calibration", "development", "validation"):
        raise ValueError(role)
    verify_stage(root, "design")
    if role != "calibration":
        gate = verify_stage(root, f"read_gate_{role}")
        if not gate["both_pass"]:
            raise RuntimeError("Full read gate failed; no local mechanism experiment")
    if role == "validation":
        verify_stage(root, "local_analysis_development")
    design = json.loads((root / OUT / f"design_{key}_v37.json").read_text())
    panel = json.loads((root / OUT / "panel_v37.json").read_text())
    lookup = {row["base_trial_id"]: row["prompt"] for role_name in
              ("calibration", "development", "validation", "independent_final")
              for row in panel[role_name]}
    model, tokenizer = load(root, key)
    layers = list(design["local_layers"].values())
    rows, raw_vectors, mixer_vectors = [], [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, lookup[sid])
        caches = {letter: step(model, incoming, token, length)["cache"]
                  for letter, token in zip(LETTERS,
                                           (item["recipient_token_id"], item["donor_token_id"],
                                            item["third_token_id"]), strict=True)}
        conv_base, _ = native_swap(caches["A"], caches["B"], ["Conv"], key)
        for pi, probe in enumerate(item["future_probe_tokens"]):
            _, hidden, _, _ = _capture(model, key, conv_base, probe, length,
                                       design["target_bundle"], layers)
            for position, layer in design["local_layers"].items():
                module = _module(model, key, layer)
                h = hidden[layer]
                raw_cells, mix_cells = [], []
                factor_hashes = {}
                for operator in LETTERS:
                    f = factors(key, module, h, caches[operator])
                    factor_hashes[operator] = {name: thash(value) for name, value in f.items()
                                               if isinstance(value, torch.Tensor)}
                    raw_col, mix_col = [], []
                    for state in LETTERS:
                        p = predict_with_factors(key, module, h, f,
                                                 caches[state].layers[layer].recurrent_states)
                        raw_col.append(p["raw"].float().cpu().numpy().ravel())
                        mix_col.append(p["mixer"].float().cpu().numpy().ravel())
                    raw_cells.append(raw_col)
                    mix_cells.append(mix_col)
                # First axis is state, second is operator; computation above iterates operator first.
                raw = np.asarray(raw_cells, np.float32).transpose(1, 0, 2).copy()
                mix = np.asarray(mix_cells, np.float32).transpose(1, 0, 2).copy()
                raw_vectors.append(raw)
                mixer_vectors.append(mix)
                row = {"state_id": sid, "model": key, "role": role, "family": item["family"],
                       "position": position, "layer": layer, "probe_index": pi,
                       "vector_index": len(rows), "hidden_input_hash": thash(h),
                       "factor_hashes_json": json.dumps(factor_hashes, sort_keys=True),
                       "raw_prediction_hash": thash(torch.from_numpy(raw)),
                       "mixer_prediction_hash": thash(torch.from_numpy(mix)),
                       "state_hashes_json": json.dumps({letter: thash(caches[letter].layers[layer].recurrent_states)
                                                        for letter in LETTERS}, sort_keys=True),
                       "prediction_before_counterfactual_continuation": True}
                for si, state in enumerate(LETTERS):
                    for oi, operator in enumerate(LETTERS):
                        label = f"{state}{operator}"
                        row[f"raw_norm_{label}"] = float(np.linalg.norm(raw[si, oi]))
                        row[f"mixer_norm_{label}"] = float(np.linalg.norm(mix[si, oi]))
                rows.append(row)
        if n % 5 == 0 or n == len(design[role]):
            print(f"V37 predictions {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"local_prediction_{key}_{role}_v37.parquet",
             "vectors": root / OUT / f"local_prediction_vectors_{key}_{role}_v37.npz"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], raw=np.stack(raw_vectors), mixer=np.stack(mixer_vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        positions=frame.position.to_numpy(str),
                        probe_indices=frame.probe_index.to_numpy(np.int8),
                        state_axis=np.asarray(LETTERS), operator_axis=np.asarray(LETTERS))
    summary = {"model": key, "role": role, "states": len(design[role]), "prediction_rows": len(frame),
               "local_positions": design["local_layers"],
               "files_sha256": {name: sha256_file(path) for name, path in files.items()},
               "counterfactual_continuation_observed": False,
               "prior_v36_validation_not_reused": True}
    path = root / OUT / f"local_prediction_{key}_{role}_v37.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"local_prediction_{key}_{role}",
                        [SOURCE, "src/jclosure/experiments/native_operator_v37.py",
                         str(path.relative_to(root)), *[str(file.relative_to(root)) for file in files.values()]],
                        {"model": key, "role": role, "summary_sha256": sha256_file(path),
                         "counterfactual_continuation_observed": False})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("calibration", "development", "validation"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
