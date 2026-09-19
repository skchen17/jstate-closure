"""V15 raw transfer audit of the real persistent-state writeback interface."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.config import load_config
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.numerics_v14 import _bf16_delta_then_writeback, _panels
from jclosure.experiments.persistent_channels_v7 import _prefill
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v15 import freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")


def components(
    cache: Any, row: dict[str, torch.Tensor], recurrent: list[int], attention: list[int]
) -> list[tuple[str, int, str, torch.Tensor, torch.Tensor]]:
    output = []
    for index, layer in enumerate(recurrent):
        for name, channel in (("recurrent_states", "recurrent"), ("conv_states", "conv")):
            old = getattr(cache.layers[layer], name)
            direction = row[channel][index].to(old.device).float()[None]
            output.append((channel, layer, name, old, direction))
    for index, layer in enumerate(attention):
        for part, name in enumerate(("keys", "values")):
            old = getattr(cache.layers[layer], name)
            direction = row["kv"][index, part].to(old.device).float()[None]
            length = min(old.shape[-2], direction.shape[-2])
            output.append((name, layer, name, old[..., :length, :], direction[..., :length, :]))
    return output


def local_bf16_ulp(value: torch.Tensor) -> torch.Tensor:
    # BF16 normal spacing, with subnormal spacing for true zero/subnormal values.
    x = value.float().abs()
    return torch.where(
        x < 2.0**-126,
        torch.full_like(x, 2.0**-133),
        torch.exp2(torch.floor(torch.log2(x.clamp_min(2.0**-126))) - 7),
    )


def _mode_cache(cache: Any, mode: str) -> Any:
    output = clone_hybrid_cache(cache)
    names = {
        "fp32_shadow_rec_conv": ("recurrent_states", "conv_states"),
        "fp32_rec_only": ("recurrent_states",),
        "fp32_conv_only": ("conv_states",),
        "fp32_kv_only": ("keys", "values"),
    }.get(mode, ())
    for layer in output.layers:
        for name in names:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, name, value.float())
    return output


def apply(cache: Any, scale: float, row: dict[str, torch.Tensor], recurrent: list[int], attention: list[int], mode: str) -> Any:
    if mode == "native_bf16_add":
        return _bf16_delta_then_writeback(cache, scale, row, recurrent, attention)
    amplitude = torch.tensor(scale, dtype=torch.float32, device=cache.layers[recurrent[0]].recurrent_states.device)
    return _apply_direction(cache, amplitude, row, recurrent, attention)


def transfer_metrics(old: torch.Tensor, direction: torch.Tensor, realized: torch.Tensor, scale: float) -> dict[str, float | None]:
    requested = scale * direction.float()
    actual = realized.float() - old.float()
    a = requested.reshape(-1).double()
    b = actual.reshape(-1).double()
    norm_a = float(torch.linalg.vector_norm(a).item())
    norm_b = float(torch.linalg.vector_norm(b).item())
    cosine = float(torch.dot(a, b).item() / (norm_a * norm_b)) if norm_a * norm_b > 1e-25 else None
    ulp = local_bf16_ulp(old).reshape(-1).float()
    abs_a = requested.reshape(-1).abs()
    nonzero = abs_a > 0
    nonzero_count = int(nonzero.sum().item())
    if nonzero_count:
        zero_fraction = float(((b == 0) & nonzero).sum().item() / nonzero_count)
        sign_flip = float((((a * b) < 0) & nonzero).sum().item() / nonzero_count)
        survive = float(((b != 0) & nonzero).sum().item() / nonzero_count)
        below_ulp = float((abs_a[nonzero] < ulp[nonzero]).float().mean().item())
        below_half_ulp = float((abs_a[nonzero] < 0.5 * ulp[nonzero]).float().mean().item())
        median_ulp = float(torch.median(ulp[nonzero]).item())
    else:
        zero_fraction = sign_flip = survive = below_ulp = below_half_ulp = 0.0
        median_ulp = None
    return {
        "requested_norm": norm_a,
        "realized_norm": norm_b,
        "cosine": cosine,
        "zero_fraction": zero_fraction,
        "sign_flip_fraction": sign_flip,
        "relative_norm_loss": (norm_a - norm_b) / norm_a if norm_a > 0 else None,
        "gain": norm_b / norm_a if norm_a > 0 else None,
        "max_absolute_error": float((actual - requested).abs().max().item()),
        "median_effective_bf16_ulp": median_ulp,
        "below_one_ulp_fraction": below_ulp,
        "below_half_ulp_fraction": below_half_ulp,
        "surviving_fraction": survive,
    }


def ulp_relative_scale(parts: list[tuple[str, int, str, torch.Tensor, torch.Tensor]], channel: str, multiplier: float) -> float:
    ratios = []
    for current, _, _, old, direction in parts:
        if current != channel:
            continue
        absolute = direction.abs()
        nonzero = absolute > 0
        if bool(nonzero.any()):
            ratios.append((local_bf16_ulp(old)[nonzero] / absolute[nonzero]).flatten())
    if not ratios:
        return math.nan
    return float(torch.median(torch.cat(ratios)).item() * multiplier)


def load_context(root: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(root / "configs/causal_geometry_v13.yaml")
    bundle = _load_model(config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    encoder_context = type("EncoderContext", (), {"root": root, "config": config})()
    _, _, dense_map = _load_encoder_memory_efficient(encoder_context, bundle)
    metadata = geometry._pair_metadata(root)
    tasks = {task.example_id: task for _, task in load_tasks(root / geometry.SELECTION_PATH)}
    directions = torch.load(root / geometry.DIRECTIONS, map_location="cpu", weights_only=False)
    return bundle, dense_map, config, metadata, {"tasks": tasks, "directions": directions}


def raw_audit(root: Path) -> dict[str, Any]:
    frozen = verify(root)
    section = frozen["config"]["diagnostic"]
    bundle, dense_map, config, metadata, values = load_context(root)
    v8 = config["persistent_state_v8"]
    recurrent = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    panel = _panels(root, int(section["anchors_per_family"]))
    records: list[dict[str, Any]] = []
    for anchor_number, anchor in enumerate(panel):
        base_id = str(anchor["base_trial_id"])
        task = values["tasks"][str(metadata[base_id]["prompt_id"])]
        clean = _prefill(
            bundle, task.prompt, measured_layers=measured, dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]), candidate=None,
        )
        cache = clean["cache"]
        for direction_index in section["direction_indices"]:
            row = {name: values["directions"][name][int(direction_index)] for name in ("recurrent", "conv", "kv")}
            base_parts = components(cache, row, recurrent, attention)
            scales = [("absolute", float(x), None) for x in section["scales"]]
            for channel in ("recurrent", "conv", "keys", "values"):
                for multiplier in section["ulp_multipliers"]:
                    scale = ulp_relative_scale(base_parts, channel, float(multiplier))
                    if math.isfinite(scale):
                        scales.append(("ulp_relative", scale, f"{channel}:{multiplier}"))
            for mode in section["modes"]:
                source = _mode_cache(cache, str(mode))
                parts = components(source, row, recurrent, attention)
                for scale_kind, scale, ulp_label in scales:
                    edited = apply(source, scale, row, recurrent, attention, str(mode))
                    for channel, layer, name, old, direction in parts:
                        new = getattr(edited.layers[layer], name)
                        if channel in ("keys", "values"):
                            new = new[..., : old.shape[-2], :]
                        records.append({
                            "protocol_version": "quantization_aware_causal_actuation_v15",
                            "freeze_digest": frozen["freeze_digest"],
                            "role": "diagnostic_train",
                            "base_trial_id": base_id,
                            "family": str(anchor["family"]),
                            "direction_index": int(direction_index),
                            "direction_family": values["directions"]["labels"][int(direction_index)],
                            "mode": str(mode),
                            "scale_kind": scale_kind,
                            "ulp_scale_label": ulp_label,
                            "scale": scale,
                            "channel": channel,
                            "layer": layer,
                            "dtype_before": str(old.dtype),
                            "dtype_after": str(new.dtype),
                            "element_count": int(old.numel()),
                            **transfer_metrics(old, direction, new, scale),
                        })
                    del edited
                del source
        write_json_atomic(root / OUT / "transfer_progress_v15.json", {"completed_anchors": anchor_number + 1, "total_anchors": len(panel)})
    frame = pd.DataFrame(records)
    path = root / OUT / "actuator_transfer_v15.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    absolute = frame[(frame.scale_kind == "absolute") & (frame.mode == "native_fp32_add_bf16_writeback")]
    pooled = absolute.groupby(["channel", "scale"]).agg(
        median_gain=("gain", "median"), median_cosine=("cosine", "median"),
        median_zero_fraction=("zero_fraction", "median"), median_surviving_fraction=("surviving_fraction", "median"),
        median_below_one_ulp=("below_one_ulp_fraction", "median"), median_below_half_ulp=("below_half_ulp_fraction", "median"),
        count=("gain", "size"),
    ).reset_index()
    deadzones = {}
    for channel, group in pooled.groupby("channel"):
        group = group.sort_values("scale")
        change = group[group.median_surviving_fraction > 0]
        reliable = group[(group.median_gain >= section["reliable_state_gain_min"]) & (group.median_gain <= section["reliable_state_gain_max"]) & (group.median_cosine >= section["reliable_state_cosine_min"])]
        dead = group[group.median_zero_fraction >= section["deadzone_zero_fraction_threshold"]]
        deadzones[channel] = {
            "ACTUATOR_DEADZONE_MAX_TESTED_SCALE": float(dead.scale.max()) if len(dead) else None,
            "MIN_STATE_CHANGE_SCALE": float(change.scale.min()) if len(change) else None,
            "MIN_RELIABLE_STATE_SCALE": float(reliable.scale.min()) if len(reliable) else None,
            "REALIZED_REQUESTED_GAIN_AT_SCALE_1": float(group.loc[group.scale == 1.0, "median_gain"].iloc[0]),
        }
    modes = frame[frame.scale_kind == "absolute"].groupby(["mode", "channel", "scale"]).agg(median_gain=("gain", "median"), median_cosine=("cosine", "median"), median_surviving_fraction=("surviving_fraction", "median")).reset_index()
    summary = {
        "protocol_version": frozen["protocol_version"], "freeze_digest": frozen["freeze_digest"],
        "role": "diagnostic_train", "records": str(path.relative_to(root)), "records_sha256": sha256_file(path),
        "panel_ids": [str(x["base_trial_id"]) for x in panel],
        "deadzones": deadzones, "curves": pooled.to_dict("records"),
        "precision_modes": modes.to_dict("records"),
        "fp32_kv_consumption": "UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT",
        "fp32_shadow_is_diagnostic_not_canonical": True,
    }
    write_json_atomic(root / OUT / "actuator_transfer_v15.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "raw", "verify"), required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "freeze":
        print(freeze(root)["freeze_digest"])
    elif args.stage == "verify":
        print(verify(root)["freeze_digest"])
    else:
        print(json.dumps(raw_audit(root)["deadzones"], sort_keys=True))


if __name__ == "__main__":
    main()
