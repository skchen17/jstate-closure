"""Frozen runtime-only amendment for memory-efficient V12 J-state loading.

The original loader casts the full GPU unembedding matrix to float32 before selecting
the frozen concept vocabulary.  On a shared 24 GiB GPU this transiently needs another
2.37 GiB.  This amendment selects the same rows first and then performs the same
float32 CPU conversion.  It changes neither numerical inputs nor any estimand.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder
from jclosure.protocol_v12 import (
    BASE_FREEZE_PATH,
    JVP_FREEZE_PATH,
    PREPARED_FREEZE_PATH,
    verify_base_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

PROTOCOL_AMENDMENT = "local_causal_geometry_v12_runtime_amendment_1"
SCHEMA_AMENDMENT = 16
AMENDMENT_PATH = Path("artifacts/causal_geometry_v12_runtime_amendment_1.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v12_amendment.py")


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_encoder_memory_efficient(
    context: Any, bundle: Any
) -> tuple[Any, Any, DenseJMap]:
    size = int(context.config["single_arm_v4"]["dictionary_size"])
    vocabulary = ConceptVocabulary.from_json(
        context.root / "results/processed" / f"concept_vocabulary_v2_{size}.json"
    )
    weight = bundle.unembedding_weight.detach()
    ids = torch.tensor(vocabulary.token_ids, dtype=torch.long, device=weight.device)
    selected_unembedding = weight.index_select(0, ids).cpu().float()
    jacobians = {
        int(layer): jacobian.detach().cpu().float()
        for layer, jacobian in bundle.lens.jacobians.items()
    }
    chunk = int(context.config["jstate"].get("direction_chunk_size", 512))

    def build(layer: int) -> torch.Tensor:
        jacobian = jacobians[int(layer)]
        pieces = [
            selected_unembedding[start : start + chunk] @ jacobian
            for start in range(0, selected_unembedding.shape[0], chunk)
        ]
        return torch.cat(pieces, dim=0)

    encoder = JStateEncoder(
        None,
        vocabulary,
        k=int(context.config["jstate"]["k"]),
        raw_builder=build,
        available_layers=tuple(jacobians),
        protocol_version=PROTOCOL_AMENDMENT,
        direction_chunk_size=chunk,
    )
    return vocabulary, encoder, DenseJMap.from_encoder(encoder)


def _freeze() -> None:
    from jclosure.config import load_config

    root = Path.cwd()
    config = load_config(root / "configs/causal_geometry_v12.yaml")
    base = verify_base_freeze(root, config)
    parents = [BASE_FREEZE_PATH, PREPARED_FREEZE_PATH, JVP_FREEZE_PATH]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_AMENDMENT,
        "protocol_version": PROTOCOL_AMENDMENT,
        "purpose": "memory-order-only correction after pre-measurement CUDA OOM",
        "parent_base_freeze_digest": base["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "algorithm_changed": False,
        "numerical_equivalence": (
            "select frozen vocabulary rows before float32 CPU copy instead of casting "
            "the full unembedding matrix before selecting the same rows"
        ),
        "failure_stage": "model/encoder loading before the first V12 measurement record",
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
            **{str(path): sha256_file(root / path) for path in parents},
        },
    }
    value["freeze_digest"] = _digest(value)
    write_json_atomic(root / AMENDMENT_PATH, value)
    print(value["freeze_digest"])


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("v12 runtime amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"v12 runtime amendment input changed: {name}")
    return value


def main() -> None:
    try:
        position = sys.argv.index("--target")
        target = sys.argv[position + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            "usage: runtime_v12_amendment --target freeze|causal|jvp ..."
        ) from exc
    del sys.argv[position : position + 2]
    if target == "freeze":
        _freeze()
        return
    _verify(Path.cwd())
    if target == "causal":
        from jclosure.experiments import causal_v12 as module
    elif target == "jvp":
        from jclosure.experiments import jvp_v12 as module
    else:
        raise SystemExit(f"unknown target: {target}")
    module._load_encoder = _load_encoder_memory_efficient
    module.PROTOCOL_V12 = PROTOCOL_AMENDMENT
    module.SCHEMA_VERSION_V12 = SCHEMA_AMENDMENT
    module.main()


if __name__ == "__main__":
    main()
