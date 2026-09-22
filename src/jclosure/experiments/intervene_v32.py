"""Bidirectional six-probe recurrent-output interception at frozen V32 site."""
from __future__ import annotations

import argparse
import contextlib
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
from jclosure.experiments.pre_readout_v27 import metadata, prefix, scales, step
from jclosure.experiments.transaction_v28 import context, thash
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/intervene_v32.py"


@contextlib.contextmanager
def capture_or_replace(module, replacement=None):
    state = {"calls": 0, "captured": None, "exact": None, "input_hash": None, "output_hash": None}
    def hook(_module, _inputs, output):
        state["calls"] += 1
        state["captured"] = output.detach().clone()
        state["input_hash"] = thash(output)
        if replacement is None:
            return None
        if output.shape != replacement.shape or output.dtype != replacement.dtype or output.device != replacement.device:
            raise RuntimeError("V32 intervention field shape/dtype/device mismatch")
        patched = replacement.detach().clone()
        state["exact"] = bool(torch.equal(patched, replacement))
        state["output_hash"] = thash(patched)
        return patched
    handle = module.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@torch.no_grad()
def run(root: Path, role: str):
    verify_stage(root, "site_selection")
    verify_stage(root, f"factorial_{role}")
    if role == "validation":
        verify_stage(root, "intervention_development")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v32.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v32.json").read_text())
    site = json.loads((root / OUT / "site_selection_v32.json").read_text())
    layer = site["candidate_layer"]
    ids = {sid for family in cfg["families"] for sid in plan["roles"][role][family]["trace_state_ids"]}
    factorial = pd.read_parquet(root / OUT / f"factorial_{role}_v32.parquet")
    saved = np.load(root / OUT / f"factorial_vectors_{role}_v32.npz")
    bundle, dense, *_ = context(root)
    module = bundle.layers[layer].linear_attn
    scale = scales(root)
    rows, audits, vectors = [], [], []
    for n, item in enumerate([x for x in design[role] if x["base_trial_id"] in ids], 1):
        sid = item["base_trial_id"]
        row = factorial[(factorial.state_id == sid) & factorial.primary].iloc[0]
        y00, y10, y01, y11, yd = saved["vectors"][int(row.vector_index)].astype(np.float64)
        m = metadata(root, item)
        incoming, _, length = prefix(bundle, str(m["prompt"]))
        rec, att = native_layers(incoming)
        if total_hash(state_hashes(incoming, rec, att)) != item["incoming_state_hash"]:
            raise RuntimeError("V32 intervention incoming drift")
        a = step(bundle, dense, incoming, token(cfg["anchor_token_id"]), length, design, cfg["readout_layer"])
        b = step(bundle, dense, incoming, token(item["primary_token_id"]), length, design, cfg["readout_layer"])
        c01 = partial(a["cache"], b["cache"], rec, rec, ())
        c11 = partial(a["cache"], b["cache"], rec, rec, rec)
        before01, before11 = state_hashes(c01, rec, att), state_hashes(c11, rec, att)
        intercepted, restored, replay01, replay11 = [], [], [], []
        for probe_id in item["future_probe_tokens"]:
            probe = token(probe_id)
            with capture_or_replace(module) as q01:
                base01 = step(bundle, dense, c01, probe, length + 1, design, cfg["readout_layer"])
            with capture_or_replace(module) as q11:
                base11 = step(bundle, dense, c11, probe, length + 1, design, cfg["readout_layer"])
            replay01.append(base01)
            replay11.append(base11)
            if q01["calls"] != 1 or q11["calls"] != 1:
                raise RuntimeError("V32 target module was not called exactly once")
            with capture_or_replace(module, q01["captured"]) as remove:
                intercepted.append(step(bundle, dense, c11, probe, length + 1, design, cfg["readout_layer"]))
            with capture_or_replace(module, q11["captured"]) as insert:
                restored.append(step(bundle, dense, c01, probe, length + 1, design, cfg["readout_layer"]))
            if remove["calls"] != 1 or insert["calls"] != 1 or not remove["exact"] or not insert["exact"]:
                raise RuntimeError("V32 recurrent output patch not exact")
            audits.append({"role": role, "state_id": sid, "family": item["family"], "probe_id": probe_id, "site_layer": layer, "boundary": "recurrent_output", "remove_requested_hash": q01["input_hash"], "remove_realized_hash": remove["output_hash"], "insert_requested_hash": q11["input_hash"], "insert_realized_hash": insert["output_hash"], "remove_exact": remove["output_hash"] == q01["input_hash"], "insert_exact": insert["output_hash"] == q11["input_hash"], "recipient_native_KV": True, "cache_length": int(c01.get_seq_length()), "input_cache_unchanged": state_hashes(c01, rec, att) == before01 and state_hashes(c11, rec, att) == before11})
        yi = signature(intercepted, scale)
        yr = signature(restored, scale)
        if not np.allclose(signature(replay01, scale), y01, rtol=1e-5, atol=1e-5) or not np.allclose(signature(replay11, scale), y11, rtol=1e-5, atol=1e-5):
            raise RuntimeError("V32 intervention baseline replay drift")
        if yi.shape != y01.shape or yr.shape != y11.shape:
            raise RuntimeError("V32 intervention signature shape mismatch")
        err01 = float(np.linalg.norm(yd - y01))
        err11 = float(np.linalg.norm(yd - y11))
        benefit = err01 - err11
        removed = (float(np.linalg.norm(yd - yi)) - err11) / max(benefit, 1e-12)
        restored_fraction = (err01 - float(np.linalg.norm(yd - yr))) / max(benefit, 1e-12)
        remove_cos = cosine(y11 - yi, y11 - y01)
        insert_cos = cosine(yr - y01, y11 - y01)
        rows.append({"role": role, "state_id": sid, "family": item["family"], "site_layer": layer, "boundary": "recurrent_output", "probe_count": len(item["future_probe_tokens"]), "probe_hash": item["future_probe_hash"], "factorial_vector_index": int(row.vector_index), "benefit": benefit, "benefit_positive": benefit > 0, "removed_benefit_fraction": removed, "restored_benefit_fraction": restored_fraction, "remove_direction_cosine": remove_cos, "restore_direction_cosine": insert_cos, "intercept_relative_l2": float(np.linalg.norm(yd - yi)) / max(float(np.linalg.norm(yd-y00)), 1e-12), "restore_relative_l2": float(np.linalg.norm(yd - yr)) / max(float(np.linalg.norm(yd-y00)), 1e-12), "writeback_exact": True, "vector_index": len(vectors)})
        vectors.append(np.stack([yi, yr]).astype(np.float32))
        print(f"V32 intervention {role} {n}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    paths = {"intervention": root / OUT / f"site_intervention_{role}_v32.parquet", "audit": root / OUT / f"site_intervention_audit_{role}_v32.parquet", "vectors": root / OUT / f"site_intervention_vectors_{role}_v32.npz"}
    frame.to_parquet(paths["intervention"], index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(paths["audit"], index=False, compression="zstd")
    np.savez_compressed(paths["vectors"], vectors=np.stack(vectors), state_ids=frame.state_id.to_numpy(str))
    gate = cfg["localization_gate"]
    good = frame.benefit_positive & (frame.removed_benefit_fraction >= gate["benefit_removed_min"]) & (frame.restored_benefit_fraction >= gate["benefit_restored_min"]) & (frame.remove_direction_cosine >= gate["donor_direction_cosine_min"]) & (frame.restore_direction_cosine >= gate["donor_direction_cosine_min"])
    family = {fam: float(good.loc[g.index].mean()) for fam, g in frame.groupby("family")}
    summary = {"role": role, "candidate_layer": layer, "states": len(frame), "success_rows": int(good.sum()), "family_success_fraction": family, "families_passing": sum(x >= gate["family_success_fraction_min"] for x in family.values()), "all_audit_exact": bool(pd.DataFrame(audits)[["remove_exact", "insert_exact", "input_cache_unchanged"]].all().all()), "response_sha256": {name: sha256_file(path) for name, path in paths.items()}, "independent_final_opened": False}
    jpath = root / OUT / f"site_intervention_{role}_v32.json"
    write_json_atomic(jpath, summary)
    stage = stage_freeze(root, f"intervention_{role}", [SOURCE, *(str(p.relative_to(root)) for p in paths.values()), str(jpath.relative_to(root)), "artifacts/rec_conv_mechanism_v32_site_selection.freeze.json"], {"summary_sha256": sha256_file(jpath), "site_layer": layer, "all_audit_exact": summary["all_audit_exact"]})
    return {"freeze_digest": stage["freeze_digest"], **{k: summary[k] for k in ("states", "success_rows", "families_passing", "all_audit_exact")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
