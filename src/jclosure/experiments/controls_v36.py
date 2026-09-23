"""Development/validation controls for the failed local read-operator candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.local_v36 import _capture, _cos, _module, _norm, _patch_step
from jclosure.experiments.operator_v36 import factors, parts
from jclosure.experiments.runtime_v34 import field_hashes, hd, load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/controls_v36.py"


def _old_update_raw(key, f, pa, pb):
    old = pb["decayed"] - pa["decayed"]
    update = pb["update"] - pa["update"]
    if key == "Q":
        q = f["q"]
        old_raw = (old * q.unsqueeze(-1)).sum(dim=-2).reshape(-1)
        update_raw = (update * q.unsqueeze(-1)).sum(dim=-2).reshape(-1)
    else:
        c = f["C"]
        old_raw = torch.einsum("bhdn,bhn->bhd", old.to(c.dtype), c).reshape(-1)
        update_raw = torch.einsum("bhdn,bhn->bhd", update.to(c.dtype), c).reshape(-1)
    return old_raw.float(), update_raw.float()


@torch.no_grad()
def run(root: Path, key: str, role: str):
    if role not in ("development", "validation"):
        raise ValueError(role)
    verify_stage(root, f"local_{key}_{role}")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v36.json").read_text())
    panel = json.loads((root / OUT / "panel_v36.json").read_text())
    lookup = prompt_lookup(root, panel)
    model, tokenizer = load(root, key)
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v36.parquet")
    baseline = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v36.npz")["vectors"].astype(np.float64)
    base_by_id = {r.state_id: baseline[int(r.vector_index)] for r in factorial.itertuples()}
    rows, state_rows, proofs = [], [], []
    floor = float(cfg["operator_gate"]["denominator_benefit_floor_absolute"])
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        incoming, length, ids, _ = prefix(model, tokenizer, key, lookup[sid])
        if hd(ids) != item["prefix_token_hash"] or field_hashes(incoming) != item["incoming_state_hashes"]:
            raise RuntimeError(f"V36 control incoming drift {key}:{sid}")
        a = step(model, incoming, item["recipient_token_id"], length)["cache"]
        b = step(model, incoming, item["donor_token_id"], length)["cache"]
        c = step(model, incoming, item["third_token_id"], length)["cache"]
        conv, _ = native_swap(a, b, ["Conv"], key)
        upper = {layer: [] for layer in design["local_layers"]}
        for pi, probe in enumerate(item["future_probe_tokens"]):
            _, h, _, _ = _capture(model, key, conv, probe, length, design["target_bundle"], design["local_layers"])
            _, _, donor_outputs, _ = _capture(model, key, b, probe, length, design["target_bundle"], design["local_layers"])
            for layer in design["local_layers"]:
                module = _module(model, key, layer)
                fb = factors(key, module, h[layer], b)
                fc = factors(key, module, h[layer], c)
                sa, sb = a.layers[layer].recurrent_states, b.layers[layer].recurrent_states
                pa, pb = parts(key, fb, sa), parts(key, fb, sb)
                delta = pb["raw"].float() - pa["raw"].float()
                old_raw, update_raw = _old_update_raw(key, fb, pa, pb)
                shuffled = dict(fb)
                if key == "Q":
                    shuffled["k"] = fc["k"]
                else:
                    shuffled["C"] = fc["C"]
                shuffled_delta = (parts(key, shuffled, sb)["raw"].float() -
                                  parts(key, shuffled, sa)["raw"].float())
                sign_flip = (2 * sa.float() - sb.float()).to(sa.dtype)
                sign_delta = (parts(key, fb, sign_flip)["raw"].float() - pa["raw"].float())
                patched, proof = _patch_step(model, key, conv, probe, length,
                                             design["target_bundle"], layer, donor_outputs[layer])
                upper[layer].append(patched)
                proofs.append({"state_id": sid, "layer": layer, "probe_index": pi,
                               "donor_mixer_requested_hash": thash(donor_outputs[layer]),
                               "upper_patch_proof": json.dumps(proof, sort_keys=True)})
                rows.append({"model": key, "role": role, "state_id": sid, "family": item["family"],
                             "layer": layer, "probe_index": pi,
                             "old_raw_effect_norm": _norm(old_raw),
                             "state_dependent_update_raw_effect_norm": _norm(update_raw),
                             "old_raw_projection_to_full": float(torch.dot(old_raw, delta.ravel()) / max(float(delta.square().sum()), 1e-12)),
                             "dependent_update_projection_to_full": float(torch.dot(update_raw, delta.ravel()) / max(float(delta.square().sum()), 1e-12)),
                             "algebraic_sum_cosine": _cos(old_raw, delta.ravel() - update_raw),
                             "shuffled_operator_cosine_to_native": _cos(shuffled_delta, delta),
                             "shuffled_operator_norm_ratio": _norm(shuffled_delta) / max(_norm(delta), 1e-6),
                             "sign_flip_opposite_cosine": _cos(sign_delta, -delta),
                             "off_manifold_controls_not_causal_native_token_branches": True})
        reference = base_by_id[sid]
        donor_y, conv_y, joint_y = reference[4], reference[2], reference[3]
        baseline_error = _norm(donor_y - conv_y)
        benefit = baseline_error - _norm(donor_y - joint_y)
        for layer in design["local_layers"]:
            upper_y = signature(upper[layer], design["calibration_clean_scales"])
            upper_benefit = baseline_error - _norm(donor_y - upper_y)
            state_rows.append({"model": key, "role": role, "state_id": sid,
                               "family": item["family"], "layer": layer,
                               "natural_rec_benefit_absolute": benefit,
                               "donor_mixer_upper_benefit_absolute": upper_benefit,
                               "donor_mixer_upper_fraction": upper_benefit / benefit if benefit >= floor else None,
                               "denominator_eligible": benefit >= floor,
                               "six_probes_one_state": True})
        if n % 10 == 0 or n == len(design[role]):
            print(f"V36 controls {key} {role} {n}/{len(design[role])}", flush=True)
    frame, state_frame = pd.DataFrame(rows), pd.DataFrame(state_rows)
    files = {"rows": root / OUT / f"controls_{key}_{role}_v36.parquet",
             "state": root / OUT / f"controls_state_{key}_{role}_v36.parquet",
             "proof": root / OUT / f"controls_proof_{key}_{role}_v36.parquet"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    state_frame.to_parquet(files["state"], index=False, compression="zstd")
    pd.DataFrame(proofs).to_parquet(files["proof"], index=False, compression="zstd")
    result = {"model": key, "role": role, "states": len(design[role]),
              "median_upper_fraction": float(state_frame.donor_mixer_upper_fraction.median()),
              "median_old_projection": float(frame.old_raw_projection_to_full.median()),
              "median_dependent_update_projection": float(frame.dependent_update_projection_to_full.median()),
              "median_shuffled_cosine": float(frame.shuffled_operator_cosine_to_native.median()),
              "median_sign_flip_opposite_cosine": float(frame.sign_flip_opposite_cosine.median()),
              "files_sha256": {name: sha256_file(path) for name, path in files.items()},
              "final_responses_seen": False}
    path = root / OUT / f"controls_{key}_{role}_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"controls_{key}_{role}", [SOURCE, str(path.relative_to(root)),
                        *(str(file.relative_to(root)) for file in files.values()),
                        f"artifacts/computational_origin_v36_local_{key}_{role}.freeze.json"],
                        {"model": key, "role": role, "summary_sha256": sha256_file(path),
                         "final_responses_seen": False})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
