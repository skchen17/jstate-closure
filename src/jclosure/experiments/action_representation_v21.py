"""Exact V13 requested-score-coordinate kernels and V20 descriptor baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jclosure.experiments import bank_v21 as bank
from jclosure.protocol_v21 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/action_representation_v21.py"
OUTPUT = bank.OUT / "action_representation_v21.json"


def _coverage(train: np.ndarray, validation: np.ndarray) -> list[dict]:
    norm_train = np.linalg.norm(train, axis=1)
    basis, singular, vh = np.linalg.svd(train, full_matrices=False)
    rank = int((singular > max(singular[0] * 1e-9, 1e-10)).sum())
    right = vh[:rank]
    results = []
    for row in validation:
        norm = max(float(np.linalg.norm(row)), 1e-12)
        cosine = train @ row / np.maximum(norm_train * norm, 1e-12)
        residual = float(np.linalg.norm(row - row @ right.T @ right) / norm)
        results.append({"nearest_train_abs_cosine": float(np.max(np.abs(cosine))),
                        "train_span_relative_residual": residual,
                        "train_span_rank": rank,
                        "validation_norm_over_train_median": norm / max(float(np.median(norm_train)), 1e-12)})
    return results


def _kernel_features(train: np.ndarray, validation: np.ndarray) -> dict:
    median = max(float(np.median(np.linalg.norm(train, axis=1))), 1e-12)
    both = np.vstack([train, validation]) / median
    return {"train_normalization_norm": median,
            "exact_requested_Gram": (both @ both.T).tolist(),
            "coverage": _coverage(train, validation)}


def prepare(root: Path) -> dict:
    roles = verify_stage(root, "roles")
    frozen = json.loads((root / "results/v20/processed/action_descriptors_v20.json").read_text())
    directions = torch.load(root / "artifacts/causal/v13/probe_directions_v13.pt", map_location="cpu", weights_only=False)
    actions = roles["train_actions"] + roles["validation_actions"]
    coords = [int(x["coordinate_index"]) for x in actions]
    z0 = np.stack([np.asarray(frozen["positive_base_scale_z"][str(x)], dtype=np.float64) for x in coords])
    score = np.asarray(directions["score_directions"], dtype=np.float64)
    if score.shape != (512, 14397):
        raise RuntimeError("V21 original continuous score coordinate shape changed")
    full = np.stack([score[int(x["direction_index"])] * float(x["base_alpha"]) for x in actions])
    slices = {"REC": [0, 4799], "Conv": [4799, 9598], "KV": [9598, 14397]}
    z1 = _kernel_features(full[:12], full[12:])
    z2 = {channel: _kernel_features(full[:12, span[0]:span[1]], full[12:, span[0]:span[1]])
          for channel, span in slices.items()}
    result = {"action_coordinate_order": coords,
              "V20_train_action_count": 12, "V20_validation_action_count": 6,
              "Z0_exact_V20_descriptor": z0.tolist(),
              "Z0_coverage": _coverage(z0[:12], z0[12:]),
              "Z1_full_score_coordinate_dimension": 14397,
              "Z1_exact_linear_kernel": z1,
              "Z2_architecture_score_slices": slices,
              "Z2_channel_exact_linear_kernels": z2,
              "Z1_interpretation": "Exact dot-product Gram of the complete frozen V13 14397-D requested score coordinates at frozen action alpha, not a one-hot action identity or anchor-cosine approximation.",
              "Z2_interpretation": "Three exact channel-specific Grams retaining REC/Conv/KV structure.",
              "sign_rule": "requested score coordinates and Gram cross-products change sign with action sign",
              "scale_rule": "requested score coordinates scale linearly with action amplitude",
              "final_action_directions_used": False,
              "validation_response_labels_used": False,
              "V13_direction_file_sha256": sha256_file(root / "artifacts/causal/v13/probe_directions_v13.pt")}
    target = root / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, result)
    freeze = stage_freeze(root, "action_representations",
                          [SOURCE, "artifacts/action_coordinate_geometry_v21_roles.freeze.json",
                           "results/v20/processed/action_descriptors_v20.json", str(OUTPUT)],
                          {"action_representation_sha256": sha256_file(target),
                           "Z0_Z1_Z2_frozen_before_V21_model_fit": True,
                           "Z3_realized_action_pending_separate_freeze": True,
                           "Z4_Z5_oracle_pending_separate_freeze": True,
                           "V20_final_six_action_responses_opened": False})
    return {"freeze_digest": freeze["freeze_digest"], "output_sha256": sha256_file(target),
            "Z1_validation_coverage": z1["coverage"]}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
