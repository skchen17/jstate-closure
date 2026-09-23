"""Persist per-state/token/layer/native-factor/operator/prediction hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.local_v36 import _capture, _module
from jclosure.experiments.operator_v36 import factors, parts
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/factor_hashes_v36.py"


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"local_{key}_{role}")
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    rows = []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError("V36 factor hash incoming drift")
        caches = {"A": step(model, incoming, item["recipient_token_id"], length)["cache"],
                  "B": step(model, incoming, item["donor_token_id"], length)["cache"],
                  "C": step(model, incoming, item["third_token_id"], length)["cache"]}
        conv, _ = native_swap(caches["A"], caches["B"], ["Conv"], key)
        for pi, probe in enumerate(item["future_probe_tokens"]):
            _, h, _, _ = _capture(model, key, conv, probe, length,
                                  design["target_bundle"], design["local_layers"])
            for layer in design["local_layers"]:
                sa = caches["A"].layers[layer].recurrent_states
                sb = caches["B"].layers[layer].recurrent_states
                for operator in ("A", "B", "C"):
                    native = caches[operator]
                    f = factors(key, _module(model, key, layer), h[layer], native)
                    factor_hashes = {name: thash(value) for name, value in f.items()
                                     if isinstance(value, torch.Tensor)}
                    effect = parts(key, f, sb)["raw"].float() - parts(key, f, sa)["raw"].float()
                    rows.append({"model": key, "role": role, "state_id": sid,
                                 "family": item["family"], "probe_index": pi,
                                 "probe_token_hash": hd(probe), "token_triple_hash": item["token_triple_hash"],
                                 "layer": layer, "layer_hash": hd(layer), "operator_token": operator,
                                 "local_input_hash": thash(h[layer]),
                                 "REC_A_hash": thash(sa), "REC_B_hash": thash(sb),
                                 "Conv_operator_hash": thash(native.layers[layer].conv_states),
                                 "native_factor_hashes": json.dumps(factor_hashes, sort_keys=True),
                                 "operator_hash": hd(factor_hashes),
                                 "predicted_effect_hash": thash(effect),
                                 "response_used_to_choose_operator": False})
        if n % 20 == 0 or n == len(design[role]):
            print(f"V36 factor hashes {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    path = root / OUT / f"factor_hashes_{key}_{role}_v36.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    result = {"model": key, "role": role, "states": len(design[role]), "rows": len(frame),
              "distinct_state_ids": int(frame.state_id.nunique()),
              "all_factor_hashes_present": bool(frame.native_factor_hashes.notna().all()),
              "rows_sha256": sha256_file(path), "future_response_used_to_choose_operator": False}
    summary_path = root / OUT / f"factor_hashes_{key}_{role}_v36.json"
    write_json_atomic(summary_path, result)
    seal = stage_freeze(root, f"factor_hashes_{key}_{role}", [SOURCE,
                        str(path.relative_to(root)), str(summary_path.relative_to(root)),
                        f"artifacts/computational_origin_v36_local_{key}_{role}.freeze.json"],
                        {"model": key, "role": role, "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
