"""Prospectively staged Q2/Q3/Q4 REC-state factorial trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.natural_reconstruction_v37 import _selective_rec
from jclosure.experiments.runtime_v34 import field_hashes, load, signature, step
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v38/processed")
SOURCE = "src/jclosure/experiments/trajectory_v38.py"
CONDITIONS = {"singles": ("R000", "R100", "R010", "R001"),
              "pairs": ("R110", "R101", "R011"), "full": ("R111",)}
ROLES = ("development", "validation", "independent_final")


def _path(root: Path, key: str, role: str, stage: str, kind: str) -> Path:
    suffix = {"rows": "parquet", "vectors": "npz", "summary": "json"}[kind]
    return root / OUT / f"trajectory_{stage}_{key}_{role}_v38.{suffix}"


def _saved(root: Path, key: str, role: str, stage: str):
    verify_stage(root, f"trajectory_{stage}_{key}_{role}")
    frame = pd.read_parquet(_path(root, key, role, stage, "rows"))
    archive = np.load(_path(root, key, role, stage, "vectors"))
    return frame, archive["vectors"].astype(np.float64), tuple(archive["conditions"].tolist())


@torch.no_grad()
def observe(root: Path, key: str, role: str, stage: str) -> dict:
    if role not in ROLES or stage not in CONDITIONS:
        raise ValueError((role, stage))
    verify_stage(root, "design")
    high = verify_stage(root, "high_level_development")
    if not high["both_pass"]:
        raise RuntimeError("V38 high-level development gate failed; stop formal trajectory")
    if role == "validation":
        verify_stage(root, "high_level_validation")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening["opened"]:
            raise RuntimeError("V38 independent final remains sealed")
    if stage == "pairs":
        verify_stage(root, f"trajectory_singles_{key}_{role}")
    if stage == "full":
        for model_key in ("Q", "F"):
            verify_stage(root, f"trajectory_predict_{model_key}_{role}")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / f"design_{key}_v38.json").read_text())
    panel = json.loads((root / OUT / "panel_v38.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for r in
               ("calibration", *ROLES) for row in panel[r]}
    groups = design["relative_depth_layers"]
    model, tokenizer = load(root, key)
    rows, vectors = [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, prompts[sid])
        ys, proofs = [], []
        for condition in CONDITIONS[stage]:
            chosen = cfg["conditions"][condition]
            if chosen:
                layers = [layer for name in chosen for layer in groups[name]]
                cache, proof = _selective_rec(caches["conv"], caches["joint"], layers)
                if field_hashes(cache)["KV"] != caches["recipient_KV_hash"] or \
                   field_hashes(cache)["Conv"] != field_hashes(caches["conv"])["Conv"]:
                    raise RuntimeError(f"V38 non-REC state changed {key}:{sid}:{condition}")
                if not all(p["requested_hash"] == p["realized_hash"] for p in proof):
                    raise RuntimeError(f"V38 REC writeback not exact {key}:{sid}:{condition}")
            else:
                cache, proof = caches["conv"], []
            outputs = [step(model, cache, token, caches["length"] + 1,
                            design["target_bundle"])
                       for token in item["future_probe_tokens"]]
            ys.append(signature(outputs, design["calibration_clean_scales"]))
            proofs.append({"condition": condition, "groups": chosen,
                           "layers": len(proof), "writeback_exact": True,
                           "recipient_KV_unchanged": True, "donor_Conv_unchanged": True,
                           "future_output_copy": False})
        rows.append({"state_id": sid, "model": key, "role": role, "family": item["family"],
                     "vector_index": len(vectors), "proof_json": json.dumps(proofs, sort_keys=True),
                     "six_frozen_probes": True, "native_downstream": True})
        vectors.append(np.stack(ys).astype(np.float32))
        if n % 10 == 0 or n == len(design[role]):
            print(f"V38 trajectory {stage} {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {kind: _path(root, key, role, stage, kind) for kind in ("rows", "vectors")}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        conditions=np.asarray(CONDITIONS[stage]))
    summary = {"model": key, "role": role, "stage": stage, "states": len(frame),
               "conditions": CONDITIONS[stage], "future_output_copy": False,
               "files_sha256": {kind: sha256_file(path) for kind, path in files.items()}}
    path = _path(root, key, role, stage, "summary")
    write_json_atomic(path, summary)
    dependencies = ["artifacts/trajectory_composition_v38_design.freeze.json"]
    if stage == "pairs":
        dependencies.append(f"artifacts/trajectory_composition_v38_trajectory_singles_{key}_{role}.freeze.json")
    if stage == "full":
        dependencies += [f"artifacts/trajectory_composition_v38_trajectory_predict_{k}_{role}.freeze.json"
                         for k in ("Q", "F")]
    seal = stage_freeze(root, f"trajectory_{stage}_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         *[str(p.relative_to(root)) for p in files.values()], *dependencies],
                        {"model": key, "role": role, "stage": stage,
                         "summary_sha256": sha256_file(path), "future_output_copy": False})
    return {"freeze_digest": seal["freeze_digest"], **summary}


def predict(root: Path, key: str, role: str) -> dict:
    if role not in ROLES:
        raise ValueError(role)
    sf, sv, sc = _saved(root, key, role, "singles")
    pf, pv, pc = _saved(root, key, role, "pairs")
    if sc != CONDITIONS["singles"] or pc != CONDITIONS["pairs"] or \
       sf.state_id.tolist() != pf.state_id.tolist():
        raise RuntimeError("V38 staged condition or state mismatch")
    e100, e010, e001 = (sv[:, i] - sv[:, 0] for i in (1, 2, 3))
    e110, e101, e011 = (pv[:, i] - sv[:, 0] for i in range(3))
    additive = e100 + e010 + e001
    second_order = e110 + e101 + e011 - e100 - e010 - e001
    interactions = np.stack([e110-e100-e010, e101-e100-e001,
                             e011-e010-e001], axis=1)
    predictions = np.stack([additive, second_order], axis=1).astype(np.float32)
    frame = sf[["state_id", "model", "role", "family", "vector_index"]].copy()
    frame["additive_norm"] = np.linalg.norm(additive, axis=1)
    frame["second_order_norm"] = np.linalg.norm(second_order, axis=1)
    frame["pair_interaction_norm_sum"] = np.linalg.norm(interactions, axis=2).sum(axis=1)
    files = {kind: _path(root, key, role, "predict", kind) for kind in ("rows", "vectors")}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=predictions,
                        pair_interactions=interactions.astype(np.float32),
                        state_ids=frame.state_id.to_numpy(str),
                        conditions=np.asarray(["additive", "second_order"]))
    summary = {"model": key, "role": role, "states": len(frame),
               "R111_observed_before_prediction_freeze": False,
               "prediction_formulas": verify(root)["config"]["primary_interaction_formulas"],
               "files_sha256": {kind: sha256_file(path) for kind, path in files.items()}}
    path = _path(root, key, role, "predict", "summary")
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"trajectory_predict_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         *[str(p.relative_to(root)) for p in files.values()],
                         f"artifacts/trajectory_composition_v38_trajectory_singles_{key}_{role}.freeze.json",
                         f"artifacts/trajectory_composition_v38_trajectory_pairs_{key}_{role}.freeze.json"],
                        {"model": key, "role": role,
                         "R111_observed_before_prediction_freeze": False,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("singles", "pairs", "predict", "full"))
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=ROLES)
    args = parser.parse_args()
    print(json.dumps(predict(Path.cwd(), args.model, args.role) if args.stage == "predict"
                     else observe(Path.cwd(), args.model, args.role, args.stage), indent=2))
