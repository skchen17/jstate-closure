"""Ablate REC, convolution, and KV realization of shared local coordinates."""

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
from jclosure.experiments.curvature_v14 import _combined_rows
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.numerics_v14 import _cosine, _evaluate, _panels
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v14 import freeze_stage, verify_base
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v14/processed")
RANK = 3
SCALE = 1.0
CHANNELS = ("recurrent", "conv", "kv")


def run(root: Path) -> dict[str, Any]:
    verify_base(root)
    stage = root / "artifacts/finite_causal_control_v14_channel_ablation.freeze.json"
    if not stage.exists():
        frozen = freeze_stage(
            root,
            "channel_ablation",
            [
                "src/jclosure/experiments/channels_v14.py",
                "results/v13/processed/causal_probe_scaling_v13.parquet",
            ],
            {
                "source_split": "V13 train anchors",
                "shared_coordinate_rank": RANK,
                "finite_scale": SCALE,
                "material_break_rule": "J cosine < 0.8 or J effect norm ratio < 0.8",
                "model_weights_changed": False,
            },
        )
    else:
        frozen = json.loads(stage.read_text(encoding="utf-8"))
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
    min_effect = float(
        json.loads((root / OUT / "numerical_snr_summary_v14.json").read_text())[
            "min_causal_effect_norm"
        ]
    )
    device = next(bundle.hf_model.parameters()).device
    rows = []
    for anchor in _panels(root, 1):
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
        reference = _evaluate(cache=clean["cache"], **kwargs)
        for coordinate_index, row in enumerate(basis_rows):
            channel_norms = {
                name: float(torch.linalg.vector_norm(row[name].float()).item())
                for name in CHANNELS
            }
            effects = {}
            for ablation in ("none", *CHANNELS):
                amended = {
                    name: (
                        torch.zeros_like(row[name]) if name == ablation else row[name]
                    )
                    for name in CHANNELS
                }
                cache = _apply_direction(
                    clean["cache"],
                    torch.tensor(SCALE, device=device),
                    amended,
                    recurrent,
                    attention,
                )
                output = _evaluate(cache=cache, **kwargs)
                effects[ablation] = {
                    name: output[name] - reference[name] for name in reference
                }
            for ablation in CHANNELS:
                values = {
                    "source_freeze_digest": frozen["freeze_digest"],
                    "base_trial_id": base_id,
                    "family": anchor["family"],
                    "coordinate_index": coordinate_index,
                    "removed_channel": ablation,
                    **{
                        f"{name}_direction_norm": norm
                        for name, norm in channel_norms.items()
                    },
                }
                for target in reference:
                    full = effects["none"][target]
                    cut = effects[ablation][target]
                    values[f"{target}_cosine"] = _cosine(full, cut)
                    values[f"{target}_norm_ratio"] = float(
                        np.linalg.norm(cut) / max(np.linalg.norm(full), 1e-20)
                    )
                    values[f"{target}_absolute_loss"] = float(
                        np.linalg.norm(full - cut)
                    )
                values["j_full_effect_norm"] = float(
                    np.linalg.norm(effects["none"]["j"])
                )
                values["direction_snr_label"] = (
                    "SNR_QUALIFIED"
                    if values["j_full_effect_norm"] >= min_effect
                    else "BELOW_DIRECTION_SNR_THRESHOLD"
                )
                values["joint_channel_required"] = (
                    bool(
                        values["j_cosine"] is None
                        or values["j_cosine"] < 0.8
                        or values["j_norm_ratio"] < 0.8
                    )
                    if values["direction_snr_label"] == "SNR_QUALIFIED"
                    else None
                )
                rows.append(values)
    path = root / OUT / "joint_channel_ablation_v14.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    frame = pd.DataFrame(rows)
    summary = {
        "protocol_version": "finite_causal_control_v14_channel_ablation",
        "source_freeze_digest": frozen["freeze_digest"],
        "joint_required_fraction": float(frame.joint_channel_required.dropna().mean())
        if frame.joint_channel_required.notna().any()
        else None,
        "qualified_rows": int(frame.joint_channel_required.notna().sum()),
        "by_channel": frame[frame.joint_channel_required.notna()]
        .groupby("removed_channel")
        .agg(
            required_fraction=("joint_channel_required", "mean"),
            j_cosine=("j_cosine", "median"),
            j_norm_ratio=("j_norm_ratio", "median"),
        )
        .reset_index()
        .to_dict("records"),
        "records": str(path.relative_to(root)),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(root / OUT / "joint_channel_ablation_v14.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), sort_keys=True)[:2000])
