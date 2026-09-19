"""Freeze V15 diagnostic/development IDs; never repurpose V13/V14 final IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v15 import stage_freeze, verify


def _hash_ids(items: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def main() -> None:
    root = Path.cwd()
    verify(root)
    previous = json.loads((root / "artifacts/finite_causal_control_v14_splits.freeze.json").read_text())
    measured = json.loads((root / "results/v15/processed/actuator_transfer_corrected_v15.json").read_text())
    train = list(previous["diagnostic_train"]["base_trial_ids"])
    validation = list(previous["development_validation"]["base_trial_ids"])
    measured_ids = set(pd.read_parquet(root / measured["raw_records"])["base_trial_id"].astype(str))
    if measured_ids != set(train):
        raise RuntimeError("V15 measured diagnostic IDs differ from frozen V14 train panel")
    if set(train) & set(validation) or len(train) != 5 or len(validation) != 10:
        raise RuntimeError("V15 train/validation panels invalid")
    result = stage_freeze(
        root, "splits", [
            "scripts/freeze_v15_splits.py",
            "artifacts/finite_causal_control_v14_splits.freeze.json",
            "results/v15/processed/actuator_transfer_corrected_v15.json",
        ],
        {
            "diagnostic_train": {"base_trial_ids": train, "count": len(train), "id_sha256": _hash_ids(train)},
            "development_validation": {"base_trial_ids": validation, "count": len(validation), "id_sha256": _hash_ids(validation)},
            "independent_final": {"status": "NOT_CREATED_OR_OPENED_PENDING_FINALIST_GATE"},
            "diagnostic_development_disjoint": True,
        },
    )
    print(result["freeze_digest"])


if __name__ == "__main__":
    main()
