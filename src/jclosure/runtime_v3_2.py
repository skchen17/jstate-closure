"""Runtime helpers with protocol-v3.2 config semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.datasets import TaskExample, task_examples_from_json
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.runtime_v3_1 import (
    answer_token_id,
    construct_initial_sequence,
    encode_direct_prompt,
    match_donor,
    replacement_transform,
    restore_sequence,
    teacher_preanswer_prefix,
    tensor_digest,
)

PROTOCOL_V32 = "corrective_causal_protocol_v3_2"
MEMORY_PROTOCOL_V32 = "compact_memory_exploratory_v3_2"


def v32_thresholds(config: dict[str, Any]) -> V3ClampThresholds:
    values = config["v3_2"]
    return V3ClampThresholds(
        dense_cosine=float(values["dense_cosine_threshold"]),
        dense_top10_overlap=float(values["top10_overlap_threshold"]),
        rms_drift=float(values["rms_drift_threshold"]),
        formal_displacement=float(values["formal_displacement_fraction"]),
        sensitivity_displacement=0.05,
    )


def restoration_is_optimized(method: str) -> bool:
    if method == "dense_local":
        return False
    if method == "dense_optimized":
        return True
    raise ValueError(f"unknown restoration method: {method}")


def load_v32_domain(root: Path, domain: str) -> list[TaskExample]:
    manifest = json.loads(
        (root / "artifacts/v3_2_data_manifest.json").read_text(encoding="utf-8")
    )
    return task_examples_from_json(root / manifest["causal_domains"][domain]["path"])


def fit_naturality_models_v32(
    root: Path,
    records: list[dict[str, Any]],
    layers: list[int],
    *,
    scope: str,
    config: dict[str, Any],
) -> dict[int, NaturalityModel]:
    models: dict[int, NaturalityModel] = {}
    for layer in layers:
        states: list[np.ndarray] = []
        for record in records:
            payload = torch.load(root / record["activation_path"], map_location="cpu")
            sequence = payload["activations"][layer].float().numpy()
            if scope == "final":
                states.append(sequence[-1])
            elif scope == "all_non_padding":
                states.extend(sequence)
            else:
                raise ValueError(f"unsupported naturality scope: {scope}")
        models[layer] = NaturalityModel(
            int(config["v3_2"]["pca_dimension"]),
            int(config["v3_2"]["nearest_neighbors"]),
            float(config["v3_2"]["naturality_quantile"]),
        ).fit(np.stack(states))
    return models


__all__ = [
    "MEMORY_PROTOCOL_V32",
    "PROTOCOL_V32",
    "answer_token_id",
    "construct_initial_sequence",
    "encode_direct_prompt",
    "fit_naturality_models_v32",
    "load_v32_domain",
    "match_donor",
    "replacement_transform",
    "restoration_is_optimized",
    "restore_sequence",
    "teacher_preanswer_prefix",
    "tensor_digest",
    "v32_thresholds",
]
