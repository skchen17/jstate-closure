"""Finite-scale directional curvature diagnostic for V14.

This is not asserted to be a differential Hessian when BF16 writeback fails
the frozen infinitesimal JVP/finite-difference gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.config import load_config
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.numerics_v14 import _cosine, _evaluate, _panels, _slices
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v14/processed")
RANK = 9
FIT_RADIUS = 1.0
HELDOUT = [0.5, 2.0]
MIXED_PAIRS = [(0, 1), (1, 2)]


def _combined_rows(
    directions: dict[str, Any], basis: np.ndarray
) -> list[dict[str, torch.Tensor]]:
    weights = torch.from_numpy(basis.astype(np.float32))
    blocks = {}
    for name in ("recurrent", "conv", "kv"):
        source = directions[name]
        flattened = source.reshape(source.shape[0], -1)
        combined = weights @ flattened.float()
        blocks[name] = combined.reshape((len(weights), *source.shape[1:]))
    return [{name: blocks[name][i] for name in blocks} for i in range(len(weights))]


def _effect_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | None]:
    norm = float(np.linalg.norm(actual))
    return {
        "direction_cosine": _cosine(actual, predicted),
        "relative_l2": float(np.linalg.norm(actual - predicted) / max(norm, 1e-20)),
        "magnitude_ratio": float(np.linalg.norm(predicted) / max(norm, 1e-20)),
        "absolute_l2": float(np.linalg.norm(actual - predicted)),
        "actual_norm": norm,
    }


def analyze(root: Path) -> dict[str, Any]:
    verify_base(root)
    freeze_path = root / "artifacts/finite_causal_control_v14_curvature.freeze.json"
    if not freeze_path.exists():
        frozen = freeze_stage(
            root,
            "curvature",
            [
                "src/jclosure/experiments/curvature_v14.py",
                "results/v14/processed/numerical_snr_summary_v14.json",
                "results/v13/processed/causal_probe_scaling_v13.parquet",
            ],
            {
                "local_rank": RANK,
                "rank_source": "V13 median r95 at 512 probes",
                "curvature_fit_radius": FIT_RADIUS,
                "heldout_radii": HELDOUT,
                "mixed_pairs": MIXED_PAIRS,
                "source_split": "V13 train anchors",
                "interpretation": "finite-scale diagnostic only until JVP equivalence passes",
            },
        )
    else:
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    directions = torch.load(
        root / geometry.DIRECTIONS, map_location="cpu", weights_only=False
    )
    metadata = geometry._pair_metadata(root)
    tasks = {
        task.example_id: task for _, task in load_tasks(root / geometry.SELECTION_PATH)
    }
    bundle = _load_model(config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    context = type("EncoderContext", (), {"root": root, "config": config})()
    _, _, dense_map = _load_encoder_memory_efficient(context, bundle)
    v8 = config["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp = config["causal_geometry_v13"]["jvp"]
    targets = ["j", "logits", "semantic_continuous", "workspace"]
    rows = []
    mixed_rows = []
    for anchor_number, anchor in enumerate(_panels(root, 1)):
        base_id = str(anchor["base_trial_id"])
        task = tasks[str(metadata[base_id]["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        token = _teacher_tokens(
            bundle, clean, count=1, measured_layers=measured, dense_map=dense_map
        )[0]
        with np.load(root / anchor["matrix_path"], allow_pickle=False) as payload:
            matrix = payload["matrix"].astype(np.float64)
            _, _, vh = np.linalg.svd(matrix, full_matrices=False)
            basis = vh[:RANK]
            selection = _slices(payload)
            kwargs = dict(
                bundle=bundle,
                dense_map=dense_map,
                token=token,
                prompt_length=int(clean["prompt_length"]),
                selected_j=payload["selected_j"].astype(int),
                selected_logits=payload["selected_logits"].astype(int),
                workspace_layers=[int(x) for x in jvp["workspace_layers"]],
                workspace_count=int(jvp["selected_workspace_count"]),
                main_layer=max(measured),
            )
        basis_rows = _combined_rows(directions, basis)
        base_cache = clean["cache"]
        reference = _evaluate(cache=base_cache, **kwargs)
        device = next(bundle.hf_model.parameters()).device

        def evaluate(
            row: dict[str, torch.Tensor],
            alpha: float,
            base_cache: Any = base_cache,
            device: Any = device,
            kwargs: dict[str, Any] = kwargs,
        ) -> dict[str, np.ndarray]:
            edited = _apply_direction(
                base_cache,
                torch.tensor(alpha, dtype=torch.float32, device=device),
                row,
                recurrent,
                attention,
            )
            return _evaluate(cache=edited, **kwargs)

        for direction_index, row in enumerate(basis_rows):
            plus = evaluate(row, FIT_RADIUS)
            minus = evaluate(row, -FIT_RADIUS)
            heldout = {radius: evaluate(row, radius) for radius in HELDOUT}
            for target in targets:
                exact = matrix[selection[target]] @ basis[direction_index]
                second = (plus[target] + minus[target] - 2 * reference[target]) / (
                    FIT_RADIUS**2
                )
                for radius in HELDOUT:
                    actual = heldout[radius][target] - reference[target]
                    linear = radius * exact
                    quadratic = linear + 0.5 * radius**2 * second
                    for method, prediction in (
                        ("linear", linear),
                        ("quadratic", quadratic),
                    ):
                        rows.append(
                            {
                                "source_freeze_digest": frozen["freeze_digest"],
                                "base_trial_id": base_id,
                                "family": anchor["family"],
                                "direction_index": direction_index,
                                "target": target,
                                "radius": radius,
                                "surrogate": method,
                                "second_derivative_norm": float(np.linalg.norm(second)),
                                "first_derivative_norm": float(np.linalg.norm(exact)),
                                **_effect_metrics(actual, prediction),
                            }
                        )
        for i, j in MIXED_PAIRS:
            outputs = {}
            for left, right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                joint = {
                    name: left * basis_rows[i][name] + right * basis_rows[j][name]
                    for name in ("recurrent", "conv", "kv")
                }
                outputs[(left, right)] = evaluate(joint, FIT_RADIUS)
            for target in targets:
                mixed = (
                    outputs[(1, 1)][target]
                    - outputs[(1, -1)][target]
                    - outputs[(-1, 1)][target]
                    + outputs[(-1, -1)][target]
                ) / (4 * FIT_RADIUS**2)
                mixed_rows.append(
                    {
                        "source_freeze_digest": frozen["freeze_digest"],
                        "base_trial_id": base_id,
                        "family": anchor["family"],
                        "direction_i": i,
                        "direction_j": j,
                        "target": target,
                        "mixed_second_difference_norm": float(np.linalg.norm(mixed)),
                    }
                )
        write_json_atomic(
            root / OUT / "curvature_progress_v14.json",
            {
                "completed_anchors": anchor_number + 1,
                "total_anchors": 5,
            },
        )
    OUT_PATH = root / OUT / "local_causal_curvature_v14.parquet"
    MIXED_PATH = root / OUT / "local_causal_mixed_curvature_v14.parquet"
    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False, compression="zstd")
    pd.DataFrame(mixed_rows).to_parquet(MIXED_PATH, index=False, compression="zstd")
    frame = pd.DataFrame(rows)
    pooled_frame = (
        frame.groupby(["radius", "target", "surrogate"])
        .agg(
            direction=("direction_cosine", "median"),
            relative_l2=("relative_l2", "median"),
            magnitude=("magnitude_ratio", "median"),
        )
        .reset_index()
    )

    def valid_radius(method: str) -> float | None:
        eligible = pooled_frame[
            (pooled_frame.surrogate == method)
            & (pooled_frame.direction >= 0.8)
            & (pooled_frame.relative_l2 <= 0.2)
            & (pooled_frame.magnitude.between(0.8, 1.2))
        ]
        valid = [
            float(radius)
            for radius in HELDOUT
            if len(eligible[eligible.radius == radius]) == len(targets)
        ]
        return max(valid) if valid else None

    pooled = pooled_frame.to_dict("records")
    summary = {
        "protocol_version": "finite_causal_control_v14_curvature",
        "source_freeze_digest": frozen["freeze_digest"],
        "finite_scale_diagnostic_not_differential_hessian": True,
        "first_order_valid_radius": valid_radius("linear"),
        "second_order_valid_radius": valid_radius("quadratic"),
        "pooled": pooled,
        "records": {
            "directional": {
                "path": str(OUT_PATH.relative_to(root)),
                "sha256": sha256_file(OUT_PATH),
            },
            "mixed": {
                "path": str(MIXED_PATH.relative_to(root)),
                "sha256": sha256_file(MIXED_PATH),
            },
        },
    }
    write_json_atomic(root / OUT / "local_causal_curvature_v14.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(Path.cwd()), sort_keys=True)[:2000])
