"""V13 batched exact-JVP amendment 6: conservative batch-8 diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments import runtime_v13_jvp_batched as v1
from jclosure.experiments import runtime_v13_jvp_batched_v4 as v4
from jclosure.experiments import runtime_v13_jvp_batched_v5 as v5
from jclosure.experiments.runtime_v13 import (
    AMENDMENT_PATH as RUNTIME_AMENDMENT_PATH,
)
from jclosure.experiments.runtime_v13 import _verify as _verify_runtime
from jclosure.geometry import DenseJMap
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path("artifacts/causal_geometry_v13_jvp_batched_v6.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v6.py")
V1_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched.py")
V4_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v4.py")
V5_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v5.py")
PROTOCOL = "causal_path_geometry_v13_jvp_batched_amendment_6"
MAXIMUM_BATCH_SIZE = 8
DIAGNOSTIC_ROOT = Path("results/v13/processed/jvp_matrices_batched_diagnostic_b8")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze(root: Path) -> dict[str, Any]:
    runtime = _verify_runtime(root)
    parent = json.loads((root / v5.AMENDMENT_PATH).read_text())
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "purpose": "test whether conservative batch 8 preserves every frozen probe-prefix rank",
        "parent_runtime_freeze_digest": runtime["freeze_digest"],
        "parent_batched_jvp_freeze_digest": parent["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "probe_operator_changed": False,
        "target_bundle_changed": False,
        "derivative_backend": "torch.autograd.functional.jvp",
        "maximum_batch_size": MAXIMUM_BATCH_SIZE,
        "diagnostic_matrix_root": str(DIAGNOSTIC_ROOT),
        "spectrum_acceptance_gate": parent["spectrum_acceptance_gate"],
        "fallback": "use the frozen scalar exact-JVP runtime if any gate fails",
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(V1_CODE_PATH): sha256_file(root / V1_CODE_PATH),
            str(V4_CODE_PATH): sha256_file(root / V4_CODE_PATH),
            str(V5_CODE_PATH): sha256_file(root / V5_CODE_PATH),
            str(RUNTIME_AMENDMENT_PATH): sha256_file(root / RUNTIME_AMENDMENT_PATH),
            str(geometry.GEOMETRY_FREEZE): sha256_file(root / geometry.GEOMETRY_FREEZE),
            "configs/causal_geometry_v13.yaml": sha256_file(
                root / "configs/causal_geometry_v13.yaml"
            ),
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    return value


def main() -> None:
    root = Path.cwd()
    if "--freeze-amendment" in sys.argv:
        print(_freeze(root)["freeze_digest"])
        return
    diagnostic = "--diagnostic-matrix" in sys.argv
    if diagnostic:
        sys.argv.remove("--diagnostic-matrix")
        geometry.JVP_MATRIX_ROOT = DIAGNOSTIC_ROOT
    v1.AMENDMENT_PATH = AMENDMENT_PATH
    v1.PROTOCOL = PROTOCOL
    v1.DEFAULT_BATCH_SIZE = MAXIMUM_BATCH_SIZE
    DenseJMap.dense_state = v4._dense_state_scalar_order
    v1.main()


if __name__ == "__main__":
    main()
