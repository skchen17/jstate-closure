"""Fresh high-level REC+Conv donor-error gate on V38 pools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.depth_common_v35 import prepare
from jclosure.experiments.runtime_v34 import load, signature, step
from jclosure.protocol_v38 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v38/processed")
SOURCE = "src/jclosure/experiments/high_level_v38.py"
ROLES = ("development", "validation", "independent_final")


@torch.no_grad()
def run(root: Path, key: str, role: str) -> dict:
    verify_stage(root, "design")
    for k in ("Q", "F"):
        seal = verify_stage(root, f"calibration_{k}")
        if not seal["all_bitwise"] or not seal["all_writeback_exact"]:
            raise RuntimeError("V38 calibration gate failed")
    if role not in ROLES:
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "high_level_development")
    if role == "independent_final":
        opened = verify_stage(root, "final_opening")
        if not opened["opened"]:
            raise RuntimeError("V38 independent final sealed")
    design = json.loads((root / OUT / f"design_{key}_v38.json").read_text())
    panel = json.loads((root / OUT / "panel_v38.json").read_text())
    prompts = {row["base_trial_id"]: row["prompt"] for r in
               ("calibration", *ROLES) for row in panel[r]}
    model, tokenizer = load(root, key)
    rows, vectors = [], []
    for n, item in enumerate(design[role], 1):
        sid = item["base_trial_id"]
        caches = prepare(model, tokenizer, key, item, prompts[sid])
        ys = {}
        for name in ("conv", "joint", "donor"):
            outputs = [step(model, caches[name], token, caches["length"] + 1,
                            design["target_bundle"])
                       for token in item["future_probe_tokens"]]
            ys[name] = signature(outputs, design["calibration_clean_scales"])
        econv = float(np.linalg.norm(ys["donor"] - ys["conv"]))
        ejoint = float(np.linalg.norm(ys["donor"] - ys["joint"]))
        rows.append({"state_id": sid, "model": key, "role": role, "family": item["family"],
                     "vector_index": len(vectors), "donor_error_conv": econv,
                     "donor_error_joint": ejoint,
                     "relative_reduction": (econv - ejoint) / max(econv, 1e-12),
                     "benefit_absolute": econv - ejoint,
                     "positive": ejoint < econv,
                     "recipient_native_KV": True, "donor_Conv": True,
                     "later_output_copy": False})
        vectors.append(np.stack([ys["conv"], ys["joint"], ys["donor"]]).astype(np.float32))
        if n % 10 == 0 or n == len(design[role]):
            print(f"V38 high level {key} {role} {n}/{len(design[role])}", flush=True)
    frame = pd.DataFrame(rows)
    files = {"rows": root / OUT / f"high_level_{key}_{role}_v38.parquet",
             "vectors": root / OUT / f"high_level_vectors_{key}_{role}_v38.npz"}
    frame.to_parquet(files["rows"], index=False, compression="zstd")
    np.savez_compressed(files["vectors"], vectors=np.stack(vectors),
                        state_ids=frame.state_id.to_numpy(str),
                        conditions=np.asarray(["conv", "joint", "donor"]))
    summary = {"model": key, "role": role, "states": len(frame),
               "positive_fraction": float(frame.positive.mean()),
               "median_reduction": float(frame.relative_reduction.median()),
               "families_passing": int(sum(bool(g.positive.mean() >= 0.8 and
                    g.relative_reduction.median() >= 0.2)
                    for _, g in frame.groupby("family"))),
               "files_sha256": {k: sha256_file(v) for k, v in files.items()}}
    path = root / OUT / f"high_level_{key}_{role}_v38.json"
    write_json_atomic(path, summary)
    seal = stage_freeze(root, f"high_level_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         *[str(p.relative_to(root)) for p in files.values()],
                         "artifacts/trajectory_composition_v38_design.freeze.json"],
                        {"model": key, "role": role,
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **summary}


def adjudicate(root: Path, role: str) -> dict:
    if role not in ("development", "validation"):
        raise ValueError(role)
    cfg = verify(root)["config"]
    gate = cfg["high_level_gate"]
    summaries = {}
    for key in ("Q", "F"):
        verify_stage(root, f"high_level_{key}_{role}")
        summaries[key] = json.loads((root / OUT / f"high_level_{key}_{role}_v38.json").read_text())
    passes = {key: (x["positive_fraction"] >= gate["positive_fraction_min"] and
                    x["median_reduction"] >= gate["median_donor_error_reduction_min"] and
                    x["families_passing"] >= gate["families_required"])
              for key, x in summaries.items()}
    result = {"role": role, "model_summaries": summaries, "model_passes": passes,
              "both_pass": all(passes.values()), "gate": gate}
    path = root / OUT / f"high_level_{role}_v38.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"high_level_{role}", [SOURCE, str(path.relative_to(root)),
                        *[f"artifacts/trajectory_composition_v38_high_level_{key}_{role}.freeze.json"
                          for key in ("Q", "F")]],
                        {"role": role, "both_pass": result["both_pass"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F", "adjudicate"))
    parser.add_argument("role", choices=ROLES)
    args = parser.parse_args()
    print(json.dumps(adjudicate(Path.cwd(), args.role) if args.model == "adjudicate"
                     else run(Path.cwd(), args.model, args.role), indent=2))
