"""Nullify the numerically invalid V12 hybrid-device scaling attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.config import load_config
from jclosure.experiments.runtime_v12_amendment_3 import (
    AMENDMENT_PATH as AMENDMENT_3_PATH,
)
from jclosure.protocol_v12 import verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL = "local_causal_geometry_v12_hybrid_nullification_1"
SCHEMA_VERSION = 19
FREEZE_PATH = Path("artifacts/causal_geometry_v12_hybrid_nullification_1.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/protocol_v12_hybrid_nullification.py")
INVALID_PARQUET = Path(
    "results/v12/processed/data_rank_scaling_causal_v12_INVALID_hybrid_am3.parquet"
)
INVALID_JSON = Path(
    "results/v12/processed/data_rank_scaling_causal_v12_INVALID_hybrid_am3.json"
)


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = Path.cwd()
    base = verify_base_freeze(
        root, load_config(root / "configs/causal_geometry_v12.yaml")
    )
    frame = pd.read_parquet(root / INVALID_PARQUET)
    required = (
        "teacher_j_effect_norm",
        "decoded_j_effect_norm",
        "direction_cosine",
        "output_direction_cosine",
    )
    finite_counts = {name: int(frame[name].notna().sum()) for name in required}
    if any(finite_counts.values()):
        raise RuntimeError(
            "hybrid record is not uniformly invalid; manual adjudication required"
        )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL,
        "parent_base_freeze_digest": base["freeze_digest"],
        "parent_runtime_amendment": str(AMENDMENT_3_PATH),
        "status": "NULLIFIED_NUMERICAL_INVALIDITY",
        "reason": (
            "accelerate auto-map across CPU/GPU produced non-finite teacher and "
            "decoded trajectories in every scaling record"
        ),
        "record_count": int(len(frame)),
        "finite_required_metric_counts": finite_counts,
        "usable_for_scientific_inference": False,
        "usable_for_model_or_method_selection": False,
        "replacement_execution": "runtime_amendment_2_cpu",
        "hashes": {
            str(path): sha256_file(root / path)
            for path in (CODE_PATH, AMENDMENT_3_PATH, INVALID_PARQUET, INVALID_JSON)
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / FREEZE_PATH, value)
    print(value["freeze_digest"])


if __name__ == "__main__":
    main()
