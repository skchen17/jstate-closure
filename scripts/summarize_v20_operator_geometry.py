#!/usr/bin/env python3
"""Secondary descriptive aggregation of frozen V20 finite-operator geometry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments import operator_bank_v20 as bank
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    root = Path.cwd()
    states_path = root / bank.OUT / "operator_singular_spectra_v20.parquet"
    pairs_path = root / bank.OUT / "operator_rotation_pairs_v20.parquet"
    states = pd.read_parquet(states_path)
    pairs = pd.read_parquet(pairs_path)
    family = {}
    for name, group in states.groupby("family", sort=True):
        family[name] = {
            "natural_operator_state_count": int((group.q_name == "P0").sum()),
            "counterfactual_operator_state_count": int((group.q_name != "P0").sum()),
            "natural_r95_median": float(group[group.q_name == "P0"].r95.median()),
            "counterfactual_r95_median": float(group[group.q_name != "P0"].r95.median()),
        }
    result = {
        "source_state_spectra_sha256": sha256_file(states_path),
        "source_pair_geometry_sha256": sha256_file(pairs_path),
        "family_finite_operator_rank": family,
        "pair_singular_spectrum_Jensen_Shannon_median": float(pairs.spectrum_Jensen_Shannon.median()),
        "pair_principal_angle_degrees_median": float(pairs.mean_principal_angle_degrees.median()),
        "pair_gain_ratio_median": float(pairs.gain_ratio_Pq_over_P0.median()),
        "pair_rank_change_median": float(pairs.rank_change.median()),
        "rank_definition": "r95 on each measured 18-positive-action by 288-target finite-response matrix",
        "empirical_dimension_limit": "Not an exact JVP rank, global operator manifold dimension, or physical cache dimension.",
    }
    write_json_atomic(root / bank.OUT / "operator_geometry_secondary_v20.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
