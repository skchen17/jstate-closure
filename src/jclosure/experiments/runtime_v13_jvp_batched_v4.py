"""V13 batched exact-JVP amendment 4: scalar-order dense-J readout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments import runtime_v13_jvp_batched as v1
from jclosure.experiments import runtime_v13_jvp_batched_v2 as v2
from jclosure.experiments import runtime_v13_jvp_batched_v3 as v3
from jclosure.experiments.runtime_v13 import (
    AMENDMENT_PATH as RUNTIME_AMENDMENT_PATH,
)
from jclosure.experiments.runtime_v13 import _verify as _verify_runtime
from jclosure.geometry import DenseJMap
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path("artifacts/causal_geometry_v13_jvp_batched_v4.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v4.py")
V1_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched.py")
V2_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v2.py")
V3_CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched_v3.py")
PROTOCOL = "causal_path_geometry_v13_jvp_batched_amendment_4"


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze(root: Path) -> dict[str, Any]:
    runtime = _verify_runtime(root)
    parent = json.loads((root / v3.AMENDMENT_PATH).read_text())
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "purpose": "preserve scalar matvec accumulation order in each batched dense-J readout",
        "parent_runtime_freeze_digest": runtime["freeze_digest"],
        "parent_batched_jvp_freeze_digest": parent["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "probe_operator_changed": False,
        "target_bundle_changed": False,
        "derivative_backend": "torch.autograd.functional.jvp",
        "maximum_batch_size": v3.MAXIMUM_BATCH_SIZE,
        "dense_j_batch_rule": "stack independently evaluated scalar-order DenseJMap.dense_state calls",
        "equivalence_gate": parent["equivalence_gate"],
        "resume_rule": parent["resume_rule"],
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            str(V1_CODE_PATH): sha256_file(root / V1_CODE_PATH),
            str(V2_CODE_PATH): sha256_file(root / V2_CODE_PATH),
            str(V3_CODE_PATH): sha256_file(root / V3_CODE_PATH),
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


def _dense_state_scalar_order(
    self: DenseJMap, h: torch.Tensor, layer: int
) -> torch.Tensor:
    if h.ndim == 1:
        return v2._dense_state_batched(self, h, layer)
    if h.ndim != 2:
        raise ValueError("dense_state expects [d_model] or [batch,d_model]")
    return torch.stack(
        [v2._dense_state_batched(self, row, layer) for row in h], dim=0
    )


def main() -> None:
    root = Path.cwd()
    if "--freeze-amendment" in sys.argv:
        print(_freeze(root)["freeze_digest"])
        return
    v1.AMENDMENT_PATH = AMENDMENT_PATH
    v1.PROTOCOL = PROTOCOL
    v1.DEFAULT_BATCH_SIZE = v3.MAXIMUM_BATCH_SIZE
    DenseJMap.dense_state = _dense_state_scalar_order
    v1.main()


if __name__ == "__main__":
    main()
