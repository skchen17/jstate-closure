"""Reaggregate V36 controls at the independent state unit before reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/control_analysis_v36.py"


def _ci(values, seed):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x): return [None, None]
    rng = np.random.default_rng(seed)
    samples = np.median(x[rng.integers(0, len(x), size=(2000, len(x)))], axis=1)
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def run(root: Path, key: str, role: str):
    verify_stage(root, f"controls_{key}_{role}")
    cfg = verify(root)["config"]
    raw = pd.read_parquet(root / OUT / f"controls_{key}_{role}_v36.parquet")
    future = pd.read_parquet(root / OUT / f"controls_state_{key}_{role}_v36.parquet")
    state = raw.groupby(["state_id", "family"], as_index=False).agg(
        old_projection=("old_raw_projection_to_full", "median"),
        dependent_update_projection=("dependent_update_projection_to_full", "median"),
        shuffled_cosine=("shuffled_operator_cosine_to_native", "median"),
        sign_flip_cosine=("sign_flip_opposite_cosine", "median"))
    future = future.groupby("state_id", as_index=False).agg(upper_fraction=("donor_mixer_upper_fraction", "median"))
    state = state.merge(future, on="state_id", validate="one_to_one")
    path = root / OUT / f"control_state_analysis_{key}_{role}_v36.parquet"
    state.to_parquet(path, index=False, compression="zstd")
    seed = cfg["seed"] + (0 if key == "Q" else 1) + (0 if role == "development" else 100)
    result = {"model": key, "role": role, "n_states": len(state),
              "median_upper_fraction": float(state.upper_fraction.median()),
              "upper_fraction_bootstrap_ci": _ci(state.upper_fraction, seed),
              "median_old_projection": float(state.old_projection.median()),
              "median_dependent_update_projection": float(state.dependent_update_projection.median()),
              "median_shuffled_cosine": float(state.shuffled_cosine.median()),
              "median_sign_flip_opposite_cosine": float(state.sign_flip_cosine.median()),
              "families": {fam: {"n": len(part), "median_upper_fraction": float(part.upper_fraction.median())}
                           for fam, part in state.groupby("family")},
              "state_table_sha256": sha256_file(path), "bootstrap_unit": "state"}
    summary_path = root / OUT / f"control_analysis_{key}_{role}_v36.json"
    write_json_atomic(summary_path, result)
    seal = stage_freeze(root, f"control_analysis_{key}_{role}", [SOURCE,
                        str(path.relative_to(root)), str(summary_path.relative_to(root)),
                        f"artifacts/computational_origin_v36_controls_{key}_{role}.freeze.json"],
                        {"model": key, "role": role, "state_unit": True,
                         "summary_sha256": sha256_file(summary_path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", choices=("Q", "F"))
    p.add_argument("role", choices=("development", "validation")); a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.model, a.role), indent=2))
