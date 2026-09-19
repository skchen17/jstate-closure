"""Freeze the disjoint V14 diagnostic/development panels; final remains unselected."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v14 import freeze_stage, verify_base


def _hash_ids(items: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def main() -> None:
    root = Path.cwd()
    verify_base(root)
    diagnostic = json.loads(
        (root / "results/v14/processed/jvp_finite_writeback_audit_v14.json").read_text()
    )["panel_ids"]
    development = (
        pd.read_parquet(
            root / "results/v13/processed/moving_tangent_oracle_development_v13.parquet"
        )
        .base_trial_id.astype(str)
        .unique()
        .tolist()
    )
    if set(diagnostic) & set(development):
        raise RuntimeError("diagnostic train and development validation panels overlap")
    if len(diagnostic) != 5 or len(development) != 10:
        raise RuntimeError("V14 frozen panel sizes differ from predeclared protocol")
    output = freeze_stage(
        root,
        "splits",
        [
            "scripts/freeze_v14_splits.py",
            "results/v14/processed/jvp_finite_writeback_audit_v14.json",
            "results/v13/processed/moving_tangent_oracle_development_v13.parquet",
        ],
        {
            "diagnostic_train": {
                "count": len(diagnostic),
                "base_trial_ids": sorted(diagnostic),
                "id_sha256": _hash_ids(diagnostic),
            },
            "development_validation": {
                "count": len(development),
                "base_trial_ids": sorted(development),
                "id_sha256": _hash_ids(development),
            },
            "independent_final": {
                "status": "NOT_YET_CREATED_OR_SELECTED",
                "reason": "numerical-JVP agreement gate failed before finalist freeze",
            },
            "diagnostic_development_disjoint": True,
        },
    )
    print(output["freeze_digest"])


if __name__ == "__main__":
    main()
