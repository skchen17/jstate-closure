"""Frozen-subset context, Conv-dependence, REC-only and KV controls for read mediation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.design_v34 import prompt_lookup
from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.mediate_v34 import cosine
from jclosure.experiments.runtime_v34 import hd, load, native_swap, prefix, signature, step
from jclosure.experiments.transaction_v28 import thash
from jclosure.protocol_v34 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/followup_controls_v34.py"


def _capture(model, key, cache, probe, length, bundle):
    with FunctionalIntervention(model, key) as hook:
        result = step(model, cache, probe, length + 1, bundle)
    return result, hook.capture


def _patch(model, key, cache, probe, length, bundle, reference):
    with FunctionalIntervention(model, key, "RECURRENT_READ", reference) as hook:
        result = step(model, cache, probe, length + 1, bundle)
    if len(hook.capture) != 24 or not all(x.get("exact_writeback") for x in hook.capture.values()):
        raise RuntimeError("V34 read control writeback failed")
    return result, hd([x["requested_hash"] for _, x in sorted(hook.capture.items())])


def _stage_hash(capture):
    return hd([thash(x["RECURRENT_READ"]) for _, x in sorted(capture.items())])


@torch.no_grad()
def run(root: Path, key: str, role: str = "validation") -> dict:
    verify_stage(root, "stage_analysis_development")
    verify_stage(root, "stage_analysis_validation")
    analysis = json.loads((root / OUT / "stage_analysis_validation_v34.json").read_text())
    if "RECURRENT_READ" not in analysis["shared_stage_passes_development_and_validation"]:
        raise RuntimeError("V34 read-stage controls not authorized")
    plan = json.loads((root / OUT / "mediation_plan_v34.json").read_text())
    panel = json.loads((root / OUT / "panel_v34.json").read_text())
    design = json.loads((root / OUT / f"design_{key}_v34.json").read_text())
    lookup = prompt_lookup(root, panel)
    selected = {sid for values in plan["context_subset"][role].values() for sid in values}
    kv_selected = {sid for values in plan["KV_subset"][role].values() for sid in values}
    if len(selected) != 10 or len(kv_selected) != 5 or not kv_selected <= selected:
        raise RuntimeError("V34 frozen control subset drift")
    factorial = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v34.parquet")
    vectors = np.load(root / OUT / f"factorial_vectors_{key}_{role}_v34.npz")["vectors"].astype(np.float64)
    state_rows = {x.state_id: x for x in factorial.itertuples()}
    model, tokenizer = load(root, key)
    rows, kv_rows = [], []
    for index, item in enumerate((x for x in design[role] if x["base_trial_id"] in selected), 1):
        sid = item["base_trial_id"]
        incoming, length, _, _ = prefix(model, tokenizer, key, lookup[sid])
        recipient = step(model, incoming, item["recipient_token_id"], length)["cache"]
        donor = step(model, incoming, item["donor_token_id"], length)["cache"]
        conv, _ = native_swap(recipient, donor, ["Conv"], key)
        rec, _ = native_swap(recipient, donor, ["REC"], key)
        joint, _ = native_swap(recipient, donor, ["REC", "Conv"], key)
        bundle = design["target_bundle"]
        probes = item["future_probe_tokens"]
        joint_refs = []; rec_refs = []
        for probe in probes:
            _, ref = _capture(model, key, joint, probe, length, bundle)
            joint_refs.append(ref)
            _, ref = _capture(model, key, rec, probe, length, bundle)
            rec_refs.append(ref)
        matched = []; wrong = []; rec_only = []; matched_hashes = []; wrong_hashes = []
        for i, probe in enumerate(probes):
            out, h = _patch(model, key, conv, probe, length, bundle, joint_refs[i])
            matched.append(out); matched_hashes.append(h)
            out, h = _patch(model, key, conv, probe, length, bundle, joint_refs[(i + 1) % len(probes)])
            wrong.append(out); wrong_hashes.append(h)
            out, _ = _patch(model, key, conv, probe, length, bundle, rec_refs[i])
            rec_only.append(out)
        y00, y10, yconv, yjoint, yd = vectors[int(state_rows[sid].vector_index)]
        scales = design["calibration_clean_scales"]
        matched_y, wrong_y, rec_y = [signature(x, scales) for x in (matched, wrong, rec_only)]
        ec, ej = float(np.linalg.norm(yd - yconv)), float(np.linalg.norm(yd - yjoint))
        benefit = ec - ej
        if benefit <= 0: raise RuntimeError("V34 frozen control state has nonpositive benefit")
        rows.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                     "matched_restore": (ec - float(np.linalg.norm(yd - matched_y))) / benefit,
                     "wrong_token_restore": (ec - float(np.linalg.norm(yd - wrong_y))) / benefit,
                     "rec_only_stage_restore": (ec - float(np.linalg.norm(yd - rec_y))) / benefit,
                     "matched_cosine": cosine(matched_y - yconv, yjoint - yconv),
                     "wrong_token_cosine": cosine(wrong_y - yconv, yjoint - yconv),
                     "rec_only_stage_cosine": cosine(rec_y - yconv, yjoint - yconv),
                     "matched_stage_hash": hd([_stage_hash(x) for x in joint_refs]),
                     "wrong_token_stage_hash": hd([_stage_hash(joint_refs[(i + 1) % len(probes)]) for i in range(len(probes))]),
                     "rec_only_stage_hash": hd([_stage_hash(x) for x in rec_refs]),
                     "matched_requested_realized_hash": hd(matched_hashes),
                     "wrong_requested_realized_hash": hd(wrong_hashes),
                     "joint_vs_rec_only_stage_different_probes": sum(_stage_hash(a) != _stage_hash(b) for a, b in zip(joint_refs, rec_refs)),
                     "recipient_native_KV": True, "writeback_exact": True,
                     "wrong_token_off_manifold_caveat": True})
        if sid in kv_selected:
            conv_kv, _ = native_swap(recipient, donor, ["Conv", "KV"], key)
            joint_kv, _ = native_swap(recipient, donor, ["REC", "Conv", "KV"], key)
            natural_conv = []; natural_joint = []; inserted = []
            for probe in probes:
                out, _ = _capture(model, key, conv_kv, probe, length, bundle)
                natural_conv.append(out)
                out, ref = _capture(model, key, joint_kv, probe, length, bundle)
                natural_joint.append(out)
                out, _ = _patch(model, key, conv_kv, probe, length, bundle, ref)
                inserted.append(out)
            yc, yj, yi = [signature(x, scales) for x in (natural_conv, natural_joint, inserted)]
            ec_kv, ej_kv, ei_kv = [float(np.linalg.norm(yd - x)) for x in (yc, yj, yi)]
            den = ec_kv - ej_kv
            kv_rows.append({"model_key": key, "role": role, "state_id": sid, "family": item["family"],
                            "conv_error_donor_KV": ec_kv, "joint_error_donor_KV": ej_kv,
                            "read_insert_error_donor_KV": ei_kv, "positive_KV_benefit": den > 0,
                            "read_restored_fraction_donor_KV": (ec_kv - ei_kv) / den if den > 1e-12 else None,
                            "writeback_exact": True})
        print(f"V34 controls {key} {role} {index}/{len(selected)}", flush=True)
    paths = {"context_conv_rec": root / OUT / f"followup_controls_{key}_{role}_v34.parquet",
             "KV": root / OUT / f"followup_KV_{key}_{role}_v34.parquet"}
    pd.DataFrame(rows).to_parquet(paths["context_conv_rec"], index=False, compression="zstd")
    pd.DataFrame(kv_rows).to_parquet(paths["KV"], index=False, compression="zstd")
    frame = pd.DataFrame(rows); kv_frame = pd.DataFrame(kv_rows)
    summary = {"model_key": key, "role": role, "states": len(rows), "KV_states": len(kv_rows),
               "matched_restore_median": float(frame.matched_restore.median()),
               "wrong_token_restore_median": float(frame.wrong_token_restore.median()),
               "rec_only_stage_restore_median": float(frame.rec_only_stage_restore.median()),
               "joint_vs_rec_only_stage_different_fraction": float((frame.joint_vs_rec_only_stage_different_probes == 6).mean()),
               "KV_read_restore_median_valid": float(kv_frame.read_restored_fraction_donor_KV.median()),
               "wrong_token_off_manifold_caveat": True,
               "files_sha256": {name: sha256_file(path) for name, path in paths.items()}}
    path = root / OUT / f"followup_controls_{key}_{role}_v34.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"followup_controls_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)), *(str(p.relative_to(root)) for p in paths.values()),
                         "artifacts/functional_mediation_v34_stage_analysis_validation.freeze.json",
                         "artifacts/functional_mediation_v34_mediation_plan.freeze.json"],
                        {"model_key": key, "role": role, "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation"), default="validation", nargs="?")
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
