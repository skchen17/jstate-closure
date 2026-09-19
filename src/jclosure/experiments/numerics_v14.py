"""Finite-difference and writeback-SNR audit of frozen V13 exact-JVP columns."""

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
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.protocol_v14 import freeze_base, verify_base
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder

OUT = Path("results/v14/processed")


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator < 1e-20 else float(np.dot(a, b) / denominator)


def _fp32_cache(cache: Any) -> Any:
    output = clone_hybrid_cache(cache)
    for layer in output.layers:
        for name in ("recurrent_states", "conv_states", "keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, name, value.float())
    return output


def _bf16_delta_then_writeback(
    cache: Any,
    epsilon: float,
    row: dict[str, torch.Tensor],
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> Any:
    """Quantize the increment before addition, unlike V13's FP32 accumulate."""
    output = clone_hybrid_cache(cache)
    for index, layer in enumerate(recurrent_layers):
        target = output.layers[layer]
        for name, source in (
            ("recurrent_states", row["recurrent"][index]),
            ("conv_states", row["conv"][index]),
        ):
            old = getattr(target, name)
            increment = (float(epsilon) * source.to(old.device).float()).to(old.dtype)
            setattr(target, name, old + increment[None])
    for index, layer in enumerate(attention_layers):
        target = output.layers[layer]
        for part_index, name in enumerate(("keys", "values")):
            old = getattr(target, name)
            source = row["kv"][index, part_index].to(old.device).float()
            length = min(old.shape[-2], source.shape[-2])
            increment = (float(epsilon) * source[:, :length]).to(old.dtype)
            setattr(
                target,
                name,
                torch.cat(
                    (old[..., :length, :] + increment[None], old[..., length:, :]),
                    dim=-2,
                ),
            )
    return output


def _cache_difference(left: Any, right: Any) -> tuple[float, float]:
    squared, reference = 0.0, 0.0
    for first, second in zip(left.layers, right.layers, strict=True):
        for name in ("recurrent_states", "conv_states", "keys", "values"):
            a, b = getattr(first, name, None), getattr(second, name, None)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                delta = b.float() - a.float()
                squared += float(torch.sum(delta.square()).item())
                reference += float(torch.sum(a.float().square()).item())
    return math.sqrt(squared), math.sqrt(reference)


def _evaluate(
    bundle: Any,
    dense_map: Any,
    cache: Any,
    token: int,
    prompt_length: int,
    selected_j: np.ndarray,
    selected_logits: np.ndarray,
    workspace_layers: list[int],
    workspace_count: int,
    main_layer: int,
) -> dict[str, np.ndarray]:
    device = next(bundle.hf_model.parameters()).device
    with torch.no_grad():
        with ActivationRecorder(
            bundle.layers, at=workspace_layers, clone=False, detach=True
        ) as recorder:
            output = bundle.hf_model(
                input_ids=torch.tensor([[token]], device=device),
                attention_mask=torch.ones(
                    (1, prompt_length + 1), device=device, dtype=torch.long
                ),
                past_key_values=clone_hybrid_cache(cache),
                use_cache=True,
            )
        logits = output.logits[0, -1].float()
        selection = torch.as_tensor(selected_logits, device=logits.device)
        hidden = recorder.activations[main_layer][0, -1].float()
        j = dense_map.dense_state(hidden, main_layer)
        j_selection = torch.as_tensor(selected_j, device=j.device)
        workspace = torch.cat(
            [
                recorder.activations[layer][0, -1].float()[:workspace_count]
                for layer in workspace_layers
            ]
        )
        return {
            "j": j[j_selection].cpu().numpy().astype(np.float64),
            "logits": logits[selection].cpu().numpy().astype(np.float64),
            "semantic_continuous": torch.log_softmax(logits, dim=-1)[selection]
            .cpu()
            .numpy()
            .astype(np.float64),
            "workspace": workspace.cpu().numpy().astype(np.float64),
        }


def _slices(payload: Any) -> dict[str, np.ndarray]:
    raw = json.loads(str(payload["slices_json"].item()))
    layers = [23, 26, 30]
    return {
        "j": np.asarray(raw["j_h1"], dtype=int),
        "logits": np.asarray(raw["logits_h1"], dtype=int),
        "semantic_continuous": np.asarray(raw["semantic_h1"], dtype=int),
        "workspace": np.asarray(
            [idx for layer in layers for idx in raw[f"workspace_l{layer}_h1"]],
            dtype=int,
        ),
    }


def _panels(root: Path, count: int) -> list[dict[str, Any]]:
    frame = pd.read_parquet(
        root / "results/v13/processed/causal_probe_scaling_v13.parquet"
    )
    frame = frame[
        (frame.probe_family == "mixed")
        & (frame.probe_direction_count == 512)
        & (frame.token_position == 0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby("family", sort=True):
        rows.extend(group.sort_values("base_trial_id").head(count).to_dict("records"))
    if len(rows) != 5 * count:
        raise RuntimeError("frozen V13 anchor panel size mismatch")
    return rows


def audit(root: Path) -> dict[str, Any]:
    freeze = verify_base(root)
    section = freeze["config"]["numerical_audit"]
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    v13 = load_config(root / "configs/causal_geometry_v13.yaml")
    direction_payload = torch.load(
        root / geometry.DIRECTIONS, map_location="cpu", weights_only=False
    )
    directions = [int(i) for i in section["direction_indices"]]
    metadata = geometry._pair_metadata(root)
    tasks = {
        task.example_id: task for _, task in load_tasks(root / geometry.SELECTION_PATH)
    }
    bundle = _load_model(v13)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    context = type("EncoderContext", (), {"root": root, "config": v13})()
    _, _, dense_map = _load_encoder_memory_efficient(context, bundle)
    v8 = v13["persistent_state_v8"]
    measured = [int(x) for x in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [int(x) for x in v8["persistent_components"]["recurrent_layers"]]
    attention_layers = [int(x) for x in v8["persistent_components"]["attention_layers"]]
    jvp = v13["causal_geometry_v13"]["jvp"]
    workspace_layers = [int(x) for x in jvp["workspace_layers"]]
    workspace_count = int(jvp["selected_workspace_count"])
    modes = [str(x) for x in section["precision_modes"]]
    records: list[dict[str, Any]] = []
    clean_records: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(
        _panels(root, int(section["anchors_per_family"]))
    ):
        base_id = str(anchor["base_trial_id"])
        pair = metadata[base_id]
        task = tasks[str(pair["prompt_id"])]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=1,
            measured_layers=measured,
            dense_map=dense_map,
        )
        with np.load(
            root / anchor["matrix_path"], allow_pickle=False
        ) as matrix_payload:
            if sha256_file(root / anchor["matrix_path"]) != anchor["matrix_sha256"]:
                raise RuntimeError("frozen exact-JVP matrix hash mismatch")
            matrix = matrix_payload["matrix"].astype(np.float64)
            selected_j = matrix_payload["selected_j"].astype(int)
            selected_logits = matrix_payload["selected_logits"].astype(int)
            index_slices = _slices(matrix_payload)
        base_cache = clean["cache"]
        kwargs = dict(
            bundle=bundle,
            dense_map=dense_map,
            token=tokens[0],
            prompt_length=int(clean["prompt_length"]),
            selected_j=selected_j,
            selected_logits=selected_logits,
            workspace_layers=workspace_layers,
            workspace_count=workspace_count,
            main_layer=main_layer,
        )
        reference = _evaluate(cache=base_cache, **kwargs)
        clean_outputs = [reference]
        for _ in range(int(section["clean_repeats"]) - 1):
            clean_outputs.append(_evaluate(cache=base_cache, **kwargs))
        for target in section["targets"]:
            noise = [
                float(np.linalg.norm(item[target] - reference[target]))
                for item in clean_outputs[1:]
            ]
            clean_records.append(
                {
                    "base_trial_id": base_id,
                    "family": anchor["family"],
                    "target": target,
                    "clean_repeat_noise_max": max(noise),
                    "clean_repeat_noise_mean": float(np.mean(noise)),
                }
            )
        for direction_index in directions:
            row = {
                name: direction_payload[name][direction_index]
                for name in ("recurrent", "conv", "kv")
            }
            exact = {
                target: matrix[index_slices[target], direction_index]
                for target in section["targets"]
            }
            for mode in modes:
                source_cache = (
                    _fp32_cache(base_cache)
                    if mode == "fp32_cache_if_supported"
                    else base_cache
                )
                for eps in [float(x) for x in section["epsilons"]]:
                    outputs: dict[int, dict[str, np.ndarray]] = {}
                    realized: dict[int, float] = {}
                    error: str | None = None
                    for sign in (-1, 1):
                        amplitude = torch.tensor(
                            sign * eps,
                            device=next(bundle.hf_model.parameters()).device,
                            dtype=torch.float32,
                        )
                        try:
                            if mode == "bf16_delta_then_writeback":
                                edited = _bf16_delta_then_writeback(
                                    source_cache,
                                    sign * eps,
                                    row,
                                    recurrent_layers,
                                    attention_layers,
                                )
                            else:
                                edited = _apply_direction(
                                    source_cache,
                                    amplitude,
                                    row,
                                    recurrent_layers,
                                    attention_layers,
                                )
                            realized[sign], _ = _cache_difference(source_cache, edited)
                            outputs[sign] = _evaluate(cache=edited, **kwargs)
                        except (RuntimeError, ValueError, TypeError) as exc:
                            error = f"{type(exc).__name__}: {str(exc)[:300]}"
                            break
                    for target in section["targets"]:
                        common = {
                            "protocol_version": "finite_causal_control_v14",
                            "source_freeze_digest": freeze["freeze_digest"],
                            "base_trial_id": base_id,
                            "family": anchor["family"],
                            "direction_index": direction_index,
                            "direction_family": direction_payload["labels"][
                                direction_index
                            ],
                            "precision_mode": mode,
                            "epsilon": eps,
                            "target": target,
                            "exact_norm": float(np.linalg.norm(exact[target])),
                            "clean_repeat_noise_max": next(
                                item["clean_repeat_noise_max"]
                                for item in clean_records
                                if item["base_trial_id"] == base_id
                                and item["target"] == target
                            ),
                            "writeback_plus_norm": realized.get(1),
                            "writeback_minus_norm": realized.get(-1),
                            "error": error,
                        }
                        if error is None:
                            finite = (outputs[1][target] - outputs[-1][target]) / (
                                2 * eps
                            )
                            delta = outputs[1][target] - reference[target]
                            common.update(
                                finite_norm=float(np.linalg.norm(finite)),
                                cosine=_cosine(exact[target], finite),
                                relative_l2=float(
                                    np.linalg.norm(finite - exact[target])
                                    / max(float(np.linalg.norm(exact[target])), 1e-20)
                                ),
                                absolute_l2=float(
                                    np.linalg.norm(finite - exact[target])
                                ),
                                target_effect_norm=float(np.linalg.norm(delta)),
                                signed_projection=float(
                                    np.dot(delta, exact[target])
                                    / max(float(np.linalg.norm(exact[target])), 1e-20)
                                ),
                                sign_agreement=bool(np.dot(finite, exact[target]) > 0),
                                signal_to_noise=float(
                                    np.linalg.norm(delta)
                                    / max(common["clean_repeat_noise_max"], 1e-20)
                                ),
                            )
                        records.append(common)
        write_json_atomic(
            root / OUT / "numerical_audit_progress_v14.json",
            {
                "completed_anchors": anchor_index + 1,
                "total_anchors": 5 * int(section["anchors_per_family"]),
            },
        )
    frame = pd.DataFrame(records)
    path = root / OUT / "jvp_finite_writeback_audit_v14.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    clean_path = root / OUT / "clean_repeat_noise_v14.parquet"
    pd.DataFrame(clean_records).to_parquet(clean_path, index=False)
    valid = frame[frame.error.isna()].copy()
    pooled = (
        valid.groupby(["precision_mode", "epsilon", "target"], dropna=False)
        .agg(
            median_cosine=("cosine", "median"),
            median_relative_l2=("relative_l2", "median"),
            median_effect_norm=("target_effect_norm", "median"),
            median_snr=("signal_to_noise", "median"),
            zero_writeback_fraction=(
                "writeback_plus_norm",
                lambda x: float((x == 0).mean()),
            ),
            count=("cosine", "size"),
        )
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "protocol_version": "finite_causal_control_v14",
        "source_freeze_digest": freeze["freeze_digest"],
        "records": str(path.relative_to(root)),
        "records_sha256": sha256_file(path),
        "clean_repeat_records": str(clean_path.relative_to(root)),
        "clean_repeat_records_sha256": sha256_file(clean_path),
        "panel_ids": [
            str(x["base_trial_id"])
            for x in _panels(root, int(section["anchors_per_family"]))
        ],
        "pooled": pooled,
        "unsupported_modes": sorted(
            frame.loc[frame.error.notna(), "precision_mode"].unique().tolist()
        ),
    }
    write_json_atomic(root / OUT / "jvp_finite_writeback_audit_v14.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "audit", "check"), required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "freeze":
        print(freeze_base(root)["freeze_digest"])
    elif args.stage == "check":
        print(len(_panels(root, 1)), "anchors")
    else:
        print(json.dumps(audit(root), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
