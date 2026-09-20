#!/usr/bin/env python3
"""Preserve and audit every V18 pilot state metadata record superseded by amendments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.crossed_bank_v18 import OUT, SCRATCH, _state_paths
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    root = Path.cwd()
    archive = SCRATCH / "state_layer_amendment_2_prior" / "train"
    paths = sorted(archive.glob("state_*.parquet"))
    if not paths:
        raise RuntimeError("No V18 pilot state archive")
    rows, audit = [], []
    for path in paths:
        key = path.stem.removeprefix("state_")
        response, current = _state_paths("train", key)
        if not response.exists() or not current.exists():
            raise RuntimeError(f"V18 archive pair incomplete: {key}")
        prior = pd.read_parquet(path)
        corrected = pd.read_parquet(current)
        if len(prior) != 1 or len(corrected) != 1:
            raise RuntimeError(f"V18 archive state row count: {key}")
        if "state_j_layer" in prior or int(corrected.iloc[0].state_j_layer) != 23:
            raise RuntimeError(f"V18 archive correction status invalid: {key}")
        if not np.array_equal(np.asarray(corrected.iloc[0].current_j),
                              np.asarray(corrected.iloc[0].history_j_last4)[-1]):
            raise RuntimeError(f"V18 corrected J/history mismatch: {key}")
        rows.append(prior)
        audit.append({"base_trial_id": key, "prior_state_sha256": sha256_file(path),
                      "corrected_state_sha256": sha256_file(current),
                      "retained_response_sha256": sha256_file(response),
                      "live_current_j_v13_rel_l2": float(corrected.iloc[0].live_current_j_v13_rel_l2)})
    output = root / OUT / "state_layer_amendment_2_prior_v18.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(output, index=False, compression="zstd")
    manifest = {"pilot_state_count": len(rows), "prior_state_metadata_archive": str(output.relative_to(root)),
                "prior_state_metadata_sha256": sha256_file(output),
                "original_individual_records_retained_outside_repository": str(archive),
                "response_records_retained_unchanged": True,
                "maximum_live_current_j_v13_rel_l2": max(row["live_current_j_v13_rel_l2"] for row in audit),
                "median_live_current_j_v13_rel_l2": float(np.median([row["live_current_j_v13_rel_l2"] for row in audit])),
                "records": audit}
    write_json_atomic(root / OUT / "state_layer_amendment_2_complete_v18.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("pilot_state_count", "maximum_live_current_j_v13_rel_l2",
                                                    "median_live_current_j_v13_rel_l2")}, indent=2))


if __name__ == "__main__":
    main()
