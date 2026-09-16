"""Exact autograd JVP probes of the local persistent-state causal operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.compress_persistent_v8 import (
    _capture_manifests,
    _load_raw_deltas,
)
from jclosure.experiments.decoded_causal_v10 import _pair_metadata, _raw_order
from jclosure.experiments.persistent_channels_v7 import _prefill, _teacher_tokens
from jclosure.experiments.persistent_state_v8 import FORMAL_DATA, _semantic_ids
from jclosure.experiments.sufficiency_v9 import _load_data
from jclosure.model import load_model_bundle
from jclosure.protocol_v12 import (
    JVP_FREEZE_PATH,
    PROTOCOL_V12,
    SCHEMA_VERSION_V12,
    build_derived_freeze,
    verify_base_freeze,
    verify_derived_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.recorder import ActivationRecorder
from jclosure.state_models_v12 import V12StateBank

JVP_DIRECTIONS = Path("artifacts/causal/v12/jvp_probe_directions.pt")
JVP_DIRECTION_SUMMARY = Path("results/v12/processed/jvp_probe_directions_v12.json")
JVP_SPECTRA = Path("results/v12/processed/local_causal_jacobian_v12.parquet")
JVP_SUMMARY = Path("results/v12/processed/local_causal_jacobian_v12.json")
JVP_MATRIX_ROOT = Path("results/v12/processed/jvp_matrices")


def _direction_block(
    block: torch.Tensor,
    train: np.ndarray,
    source_scores: np.ndarray,
    score_directions: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    fit_scores = source_scores[train].astype(np.float64)
    fit_scores -= fit_scores.mean(axis=0, keepdims=True)
    alpha = score_directions.astype(np.float64) @ np.linalg.pinv(
        fit_scores, rcond=1e-10
    )
    flat = block.reshape(block.shape[0], -1)
    output = torch.empty((len(alpha), flat.shape[1]), dtype=torch.bfloat16)
    alpha_device = torch.from_numpy(alpha.astype(np.float32)).to(device)
    for start in range(0, flat.shape[1], chunk_size):
        stop = min(start + chunk_size, flat.shape[1])
        fit = flat[train, start:stop].float()
        mean = fit.double().mean(dim=0).float()
        values = alpha_device @ (fit - mean).to(device)
        output[:, start:stop] = values.cpu().to(torch.bfloat16)
    return output.reshape((len(alpha), *block.shape[1:]))


def _prepare_directions(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    data = _load_data(context.root)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    train = np.asarray(
        [
            lookup[str(value)]
            for value in base["train_scaling"]["nested_base_trial_ids"]["600"]
        ],
        dtype=int,
    )
    bank = V12StateBank(data, train)
    section = context.config["causal_geometry_v12"]["jvp"]
    count = int(section["probe_directions"])
    leading = int(section["leading_directions"])
    rank = bank.arch_pca.rank
    tail = np.unique(
        np.geomspace(leading, rank - 1, count - leading).round().astype(int)
    ).tolist()
    candidates = list(range(leading)) + tail
    candidates.extend(
        index for index in range(leading, rank) if index not in candidates
    )
    components = np.asarray(candidates[:count], dtype=int)
    coefficient_sd = bank.arch_pca.singular / max(len(train) - 1, 1) ** 0.5
    standardized = bank.arch_pca.basis[components] * coefficient_sd[components, None]
    score_directions = standardized * bank.arch_pca.scale
    raw, raw_ids, _, _, _ = _load_raw_deltas(context, _capture_manifests(context.root))
    raw = _raw_order(raw, raw_ids, data["ids"])
    rank_per_channel = int(data["features"].shape[1])
    requested = int(context.config["model"].get("device", 0))
    device = torch.device(f"cuda:{requested}" if torch.cuda.is_available() else "cpu")
    chunk = int(
        context.config["causal_geometry_v12"]["decoder"]["reconstruction_chunk"]
    )
    raw_names = {"recurrent": "recurrent", "conv": "conv", "kv": "kv_full"}
    directions = {}
    channel_energy = {}
    for channel_index, name in enumerate(("recurrent", "conv", "kv")):
        start = channel_index * rank_per_channel
        stop = (channel_index + 1) * rank_per_channel
        current = score_directions[:, start:stop]
        directions[name] = _direction_block(
            raw[raw_names[name]],
            train,
            bank.blocks[name],
            current,
            device=device,
            chunk_size=chunk,
        )
        channel_energy[name] = np.sum(current.astype(np.float64) ** 2, axis=1).tolist()
    path = context.root / JVP_DIRECTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "v12_frozen_empirical_joint_pc_raw_directions_bf16",
            "components": components.tolist(),
            "recurrent": directions["recurrent"],
            "conv": directions["conv"],
            "kv": directions["kv"],
        },
        path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "source_freeze_digest": base["freeze_digest"],
        "operator_domain": "64 frozen empirical global joint-PC raw-state directions, each scaled by one training score SD",
        "full_raw_jacobian_claimed": False,
        "components": components.tolist(),
        "component_singular_values": bank.arch_pca.singular[components].tolist(),
        "channel_score_energy": channel_energy,
        "artifact": str(JVP_DIRECTIONS),
        "artifact_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / JVP_DIRECTION_SUMMARY, summary)
    return summary


def _freeze_jvp(context: Any) -> dict[str, Any]:
    summary = json.loads((context.root / JVP_DIRECTION_SUMMARY).read_text())
    return build_derived_freeze(
        context.root,
        context.config,
        path=JVP_FREEZE_PATH,
        purpose="freeze v12 exact-autograd JVP empirical input operator",
        inputs=[JVP_DIRECTIONS, JVP_DIRECTION_SUMMARY],
        payload={
            "operator_domain": summary["operator_domain"],
            "component_count": len(summary["components"]),
            "full_raw_jacobian_claimed": False,
        },
    )


def _advance_cache(bundle: Any, cache: Any, token: int, length: int) -> Any:
    device = next(bundle.hf_model.parameters()).device
    with torch.no_grad():
        output = bundle.hf_model(
            input_ids=torch.tensor([[token]], device=device),
            attention_mask=torch.ones((1, length + 1), dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
    return output.past_key_values


def _apply_direction(
    cache: Any,
    epsilon: torch.Tensor,
    row: dict[str, torch.Tensor],
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> Any:
    output = clone_hybrid_cache(cache)
    for index, layer in enumerate(recurrent_layers):
        target = output.layers[layer]
        target.recurrent_states = (
            target.recurrent_states.float()
            + epsilon
            * row["recurrent"][index].to(target.recurrent_states.device).float()[None]
        ).to(target.recurrent_states.dtype)
        target.conv_states = (
            target.conv_states.float()
            + epsilon * row["conv"][index].to(target.conv_states.device).float()[None]
        ).to(target.conv_states.dtype)
    for index, layer in enumerate(attention_layers):
        target = output.layers[layer]
        for part_index, name in enumerate(("keys", "values")):
            base = getattr(target, name).float()
            delta = row["kv"][index, part_index].to(base.device).float()
            length = min(base.shape[-2], delta.shape[-2])
            value = torch.cat(
                (
                    base[..., :length, :] + epsilon * delta[None, :, :length, :],
                    base[..., length:, :],
                ),
                dim=-2,
            )
            setattr(target, name, value.to(getattr(target, name).dtype))
    return output


def _spectrum(
    matrix: np.ndarray, thresholds: list[float]
) -> tuple[dict[str, Any], np.ndarray]:
    _, singular, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-20)
    probability = energy / max(float(energy.sum()), 1e-20)
    entropy = -float(
        np.sum(probability[probability > 0] * np.log(probability[probability > 0]))
    )
    row = {
        "largest_singular_value": float(singular[0]),
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-20)),
        "effective_rank": float(np.exp(entropy)),
        "singular_values": singular.astype(np.float32).tolist(),
        "cumulative_sensitivity": cumulative.astype(np.float32).tolist(),
    }
    for threshold in thresholds:
        rank = int(np.searchsorted(cumulative, threshold) + 1)
        row[f"rank_{int(round(threshold * 100))}"] = (
            rank if rank <= len(singular) else None
        )
    return row, vh.astype(np.float32)


def _run_jvp(
    context: Any, base: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    directions = torch.load(
        context.root / JVP_DIRECTIONS, map_location="cpu", weights_only=False
    )
    metadata = _pair_metadata(context.root)
    tasks = {
        task.example_id: task for _, task in load_tasks(context.root / FORMAL_DATA)
    }
    bundle = load_model_bundle(context.config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    _, _, dense_map = _load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    section = context.config["causal_geometry_v12"]["jvp"]
    positions = [int(value) for value in section["token_positions"]]
    horizons = [int(value) for value in section["target_horizons"]]
    count = int(section["probe_directions"])
    target_data = _load_data(context.root)["targets"]
    j_variance = target_data["next_delta"].var(axis=0)
    selected_j = np.argsort(-j_variance)[:256].astype(int)
    records = []
    matrix_root = context.root / JVP_MATRIX_ROOT
    matrix_root.mkdir(parents=True, exist_ok=True)
    device = next(bundle.hf_model.parameters()).device
    selected_j_tensor = torch.as_tensor(selected_j, device=device, dtype=torch.long)
    for anchor_index, base_id in enumerate(base["jvp_anchors"]["base_trial_ids"]):
        pair = metadata[str(base_id)]
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
            count=max(positions) + max(horizons),
            measured_layers=measured,
            dense_map=dense_map,
        )
        for position in positions:
            base_cache = clone_hybrid_cache(clean["cache"])
            current_length = int(clean["prompt_length"])
            for offset in range(position):
                base_cache = _advance_cache(
                    bundle, base_cache, tokens[offset], current_length
                )
                current_length += 1
            local_tokens = tokens[position : position + max(horizons)]
            semantic_ids = _semantic_ids(
                bundle.tokenizer,
                task.semantic_actions[min(position, len(task.semantic_actions) - 1)],
            )
            selected_logits = list(dict.fromkeys(semantic_ids))[
                : int(section["selected_logit_count"])
            ]
            if len(selected_logits) < int(section["selected_logit_count"]):
                with torch.no_grad():
                    probe = bundle.hf_model(
                        input_ids=torch.tensor([[local_tokens[0]]], device=device),
                        attention_mask=torch.ones(
                            (1, current_length + 1), dtype=torch.long, device=device
                        ),
                        past_key_values=clone_hybrid_cache(base_cache),
                        use_cache=True,
                    )
                top = torch.topk(
                    probe.logits[0, -1], int(section["selected_logit_count"])
                ).indices.tolist()
                selected_logits = list(dict.fromkeys([*selected_logits, *top]))[
                    : int(section["selected_logit_count"])
                ]
            columns = []
            for direction_index in range(count):
                row = {
                    name: directions[name][direction_index]
                    for name in ("recurrent", "conv", "kv")
                }

                def target(
                    epsilon: torch.Tensor,
                    base_cache: Any = base_cache,
                    row: dict[str, torch.Tensor] = row,
                    local_tokens: list[int] = local_tokens,
                    current_length: int = current_length,
                    selected_logits: list[int] = selected_logits,
                ) -> torch.Tensor:
                    cache = _apply_direction(
                        base_cache, epsilon, row, recurrent_layers, attention_layers
                    )
                    output_parts = []
                    for step, token in enumerate(local_tokens, start=1):
                        with ActivationRecorder(
                            bundle.layers, at=[main_layer], clone=False, detach=False
                        ) as recorder:
                            model_output = bundle.hf_model(
                                input_ids=torch.tensor([[token]], device=device),
                                attention_mask=torch.ones(
                                    (1, current_length + step),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                past_key_values=cache,
                                use_cache=True,
                            )
                        cache = model_output.past_key_values
                        if step in horizons:
                            hidden = recorder.activations[main_layer][0, -1].float()
                            j_state = dense_map.dense_state(hidden, main_layer)
                            output_parts.append(j_state[selected_j_tensor])
                            output_parts.append(
                                model_output.logits[0, -1, selected_logits].float()
                            )
                    return torch.cat(output_parts)

                epsilon = torch.zeros((), device=device, dtype=torch.float32)
                _, derivative = torch.autograd.functional.jvp(
                    target,
                    epsilon,
                    torch.ones_like(epsilon),
                    create_graph=False,
                    strict=True,
                )
                columns.append(derivative.detach().cpu().numpy().astype(np.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            matrix = np.stack(columns, axis=1)
            spectrum, vh = _spectrum(
                matrix, [float(value) for value in section["energy_thresholds"]]
            )
            matrix_path = (
                matrix_root / f"anchor_{anchor_index:02d}_position_{position}.npz"
            )
            np.savez_compressed(
                matrix_path,
                matrix=matrix,
                right_singular_vectors=vh,
                selected_j=selected_j,
                selected_logits=np.asarray(selected_logits),
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION_V12,
                    "protocol_version": PROTOCOL_V12,
                    "base_trial_id": str(base_id),
                    "prompt_id": str(pair["prompt_id"]),
                    "family": str(pair["family"]),
                    "token_position": position,
                    "probe_direction_count": count,
                    "target_dimension": int(matrix.shape[0]),
                    "operator_scope": "empirical_64_direction_restriction",
                    "matrix_path": str(matrix_path.relative_to(context.root)),
                    "matrix_sha256": sha256_file(matrix_path),
                    **spectrum,
                }
            )
            write_json_atomic(
                context.raw_dir / context.run_id / "jvp_progress.json",
                {
                    "status": "RUNNING",
                    "completed_local_states": len(records),
                    "total_local_states": len(base["jvp_anchors"]["base_trial_ids"])
                    * len(positions),
                },
            )
    frame = pd.DataFrame(records)
    path = context.root / JVP_SPECTRA
    frame.to_parquet(path, index=False, compression="zstd")
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "exact_autograd_jvp": True,
        "flash_sdp_disabled_for_higher_order_derivative": True,
        "operator_scope": "64 frozen empirical joint-PC input directions; selected 256 J coordinates plus 16 logits at h1/h2/h4",
        "full_raw_jacobian_claimed": False,
        "state_count": len(records),
        "mean_stable_rank": float(frame["stable_rank"].mean()),
        "mean_effective_rank": float(frame["effective_rank"].mean()),
        "median_rank_90": float(frame["rank_90"].median()),
        "median_rank_95": float(frame["rank_95"].median()),
        "median_rank_99": float(frame["rank_99"].median()),
        "records": str(JVP_SPECTRA),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / JVP_SUMMARY, summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "exact local causal JVP geometry v12", "configs/causal_geometry_v12.yaml"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare-directions", "freeze-jvp", "run-jvp"),
    )
    args = parser.parse_args()
    context = initialize_context("jvp-v12", args)
    try:
        base = verify_base_freeze(context.root, context.config)
        if args.stage == "prepare-directions":
            result = _prepare_directions(context, base)
        elif args.stage == "freeze-jvp":
            result = _freeze_jvp(context)
        else:
            freeze = verify_derived_freeze(
                context.root, context.config, JVP_FREEZE_PATH
            )
            result = _run_jvp(context, base, freeze)
        context.finish("COMPLETED_V12_JVP_STAGE", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
