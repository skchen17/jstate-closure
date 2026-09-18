"""Batched exact-JVP execution amendment for the frozen V13 operator.

Each batch member has an independent scalar epsilon and an independent frozen
probe direction.  The target is block diagonal across the batch, so a JVP with
an all-ones tangent returns exactly one directional derivative per member.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import ALL_STATE_ATTRIBUTES, clone_hybrid_cache
from jclosure.experiments import geometry_v13 as geometry
from jclosure.experiments.runtime_v13 import (
    AMENDMENT_PATH as RUNTIME_AMENDMENT_PATH,
)
from jclosure.experiments.runtime_v13 import _load_encoder_memory_efficient, _load_model
from jclosure.experiments.runtime_v13 import _verify as _verify_runtime
from jclosure.provenance import sha256_file, write_json_atomic

AMENDMENT_PATH = Path("artifacts/causal_geometry_v13_jvp_batched.freeze.json")
CODE_PATH = Path("src/jclosure/experiments/runtime_v13_jvp_batched.py")
PROTOCOL = "causal_path_geometry_v13_jvp_batched_amendment_1"
DEFAULT_BATCH_SIZE = 8


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "freeze_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _freeze(root: Path) -> dict[str, Any]:
    runtime = _verify_runtime(root)
    jvp_freeze = json.loads((root / geometry.GEOMETRY_FREEZE).read_text())
    value: dict[str, Any] = {
        "schema_version": 22,
        "protocol_version": PROTOCOL,
        "purpose": "batch independent frozen directions without changing the exact autograd JVP estimand",
        "parent_runtime_freeze_digest": runtime["freeze_digest"],
        "parent_jvp_freeze_digest": jvp_freeze["freeze_digest"],
        "estimand_changed": False,
        "split_changed": False,
        "probe_operator_changed": False,
        "target_bundle_changed": False,
        "derivative_backend": "torch.autograd.functional.jvp",
        "batch_independence_rule": "one scalar epsilon and one frozen direction per batch member; no cross-member state",
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "equivalence_gate": {
            "reference": "an already completed scalar-JVP local-state matrix",
            "maximum_absolute_difference": 5e-4,
            "minimum_column_cosine": 0.99999,
        },
        "resume_rule": "existing matrices are hash-read and re-analysed; incomplete local states are recomputed atomically",
        "hashes": {
            str(CODE_PATH): sha256_file(root / CODE_PATH),
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


def _verify(root: Path) -> dict[str, Any]:
    value = json.loads((root / AMENDMENT_PATH).read_text())
    if value.get("freeze_digest") != _digest(value):
        raise RuntimeError("V13 batched-JVP amendment digest mismatch")
    for name, expected in value["hashes"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"V13 batched-JVP amendment input changed: {name}")
    _verify_runtime(root)
    return value


def _repeat_cache(cache: Any, batch_size: int) -> Any:
    output = clone_hybrid_cache(cache)
    for layer in output.layers:
        for name in ALL_STATE_ATTRIBUTES:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                if value.shape[0] != 1:
                    raise RuntimeError(f"expected singleton cache batch for {name}")
                repeats = (batch_size,) + (1,) * (value.ndim - 1)
                setattr(layer, name, value.repeat(repeats))
    return output


def _apply_direction_batch(
    cache: Any,
    epsilon: torch.Tensor,
    rows: dict[str, torch.Tensor],
    recurrent_layers: list[int],
    attention_layers: list[int],
) -> Any:
    batch_size = int(epsilon.numel())
    output = _repeat_cache(cache, batch_size)
    for index, layer in enumerate(recurrent_layers):
        target = output.layers[layer]
        recurrent = target.recurrent_states.float()
        recurrent_delta = rows["recurrent"][:, index].to(recurrent.device).float()
        scale = epsilon.to(recurrent.device).reshape(
            (batch_size,) + (1,) * (recurrent.ndim - 1)
        )
        target.recurrent_states = (recurrent + scale * recurrent_delta).to(
            target.recurrent_states.dtype
        )
        conv = target.conv_states.float()
        conv_delta = rows["conv"][:, index].to(conv.device).float()
        conv_scale = epsilon.to(conv.device).reshape(
            (batch_size,) + (1,) * (conv.ndim - 1)
        )
        target.conv_states = (conv + conv_scale * conv_delta).to(
            target.conv_states.dtype
        )
    for index, layer in enumerate(attention_layers):
        target = output.layers[layer]
        for part_index, name in enumerate(("keys", "values")):
            original = getattr(target, name)
            base = original.float()
            delta = rows["kv"][:, index, part_index].to(base.device).float()
            length = min(base.shape[-2], delta.shape[-2])
            scale = epsilon.to(base.device).reshape(
                (batch_size,) + (1,) * (base.ndim - 1)
            )
            value = torch.cat(
                (
                    base[..., :length, :] + scale * delta[..., :length, :],
                    base[..., length:, :],
                ),
                dim=-2,
            )
            setattr(target, name, value.to(original.dtype))
    return output


def _append_records(
    *,
    matrix: np.ndarray,
    matrix_path: Path,
    root: Path,
    labels: list[str],
    probe_sizes: list[int],
    base_id: str,
    pair: dict[str, Any],
    position: int,
    records: list[dict[str, Any]],
    sensitivity: dict[str, list[float]],
) -> None:
    column_norms = np.linalg.norm(matrix.astype(np.float64), axis=0)
    for probe_family in sorted(set(labels)):
        mask = np.asarray([label == probe_family for label in labels], dtype=bool)
        sensitivity[probe_family].extend(column_norms[mask].tolist())
    relative = str(matrix_path.relative_to(root))
    digest = sha256_file(matrix_path)
    for probe_size in probe_sizes:
        spectrum, _ = geometry._spectrum(matrix[:, :probe_size], [0.90, 0.95, 0.99])
        records.append(
            {
                "schema_version": geometry.SCHEMA_VERSION_V13,
                "protocol_version": PROTOCOL,
                "base_trial_id": base_id,
                "prompt_id": pair["prompt_id"],
                "family": pair["family"],
                "token_position": position,
                "probe_family": "mixed",
                "probe_direction_count": probe_size,
                "matrix_path": relative,
                "matrix_sha256": digest,
                **spectrum,
            }
        )
    for probe_family in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == probe_family]
        spectrum, _ = geometry._spectrum(matrix[:, indices], [0.90, 0.95, 0.99])
        records.append(
            {
                "schema_version": geometry.SCHEMA_VERSION_V13,
                "protocol_version": PROTOCOL,
                "base_trial_id": base_id,
                "prompt_id": pair["prompt_id"],
                "family": pair["family"],
                "token_position": position,
                "probe_family": probe_family,
                "probe_direction_count": len(indices),
                "matrix_path": relative,
                "matrix_sha256": digest,
                **spectrum,
            }
        )


def _run_jvp_batched(
    context: Any,
    *,
    batch_size: int,
    validate_only: bool,
) -> dict[str, Any]:
    amendment = _verify(context.root)
    freeze = geometry.verify_stage_freeze(
        context.root, context.config, geometry.GEOMETRY_FREEZE
    )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    directions = torch.load(
        context.root / geometry.DIRECTIONS, map_location="cpu", weights_only=False
    )
    data = geometry._load_features(context.root)
    metadata = geometry._pair_metadata(context.root)
    tasks = {
        task.example_id: task
        for _, task in geometry.load_tasks(context.root / geometry.SELECTION_PATH)
    }
    bundle = geometry.load_model_bundle(context.config)
    for parameter in bundle.hf_model.parameters():
        parameter.requires_grad_(False)
    _, _, dense_map = geometry._load_encoder(context, bundle)
    v8 = context.config["persistent_state_v8"]
    measured = [int(value) for value in v8["intervention"]["measured_layers"]]
    main_layer = max(measured)
    recurrent_layers = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention_layers = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    section = context.config["causal_geometry_v13"]["jvp"]
    anchors = geometry._anchor_ids(data, int(section["anchors_per_family"]))
    positions = [int(value) for value in section["token_positions"]]
    horizons = [int(value) for value in section["target_horizons"]]
    max_count = int(section["maximum_probe_directions"])
    probe_sizes = [int(value) for value in section["probe_sizes"]]
    j_variance = (
        data["endpoint__perturbed_j_h1"].astype(np.float32)
        - data["endpoint__clean_j_h1"].astype(np.float32)
    ).var(0)
    selected_j = np.argsort(-j_variance)[: int(section["selected_j_count"])]
    device = next(bundle.hf_model.parameters()).device
    selected_j_tensor = torch.as_tensor(selected_j, device=device)
    matrix_root = context.root / geometry.JVP_MATRIX_ROOT
    matrix_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    labels = list(directions["labels"])
    sensitivity: dict[str, list[float]] = defaultdict(list)
    equivalence: dict[str, Any] | None = None
    computed_states = 0

    for anchor_index, base_id in enumerate(anchors):
        pair = metadata[base_id]
        matrix_paths = [
            matrix_root / f"anchor_{anchor_index:02d}_position_{position}.npz"
            for position in positions
        ]
        if all(path.exists() for path in matrix_paths) and not validate_only:
            for position, matrix_path in zip(positions, matrix_paths, strict=True):
                with np.load(matrix_path, allow_pickle=False) as saved:
                    matrix = saved["matrix"].astype(np.float32)
                _append_records(
                    matrix=matrix,
                    matrix_path=matrix_path,
                    root=context.root,
                    labels=labels,
                    probe_sizes=probe_sizes,
                    base_id=base_id,
                    pair=pair,
                    position=position,
                    records=records,
                    sensitivity=sensitivity,
                )
            continue

        task = tasks[str(pair["prompt_id"])]
        clean = geometry._prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(v8["intervention"]["layer"]),
            candidate=None,
        )
        tokens = geometry._teacher_tokens(
            bundle,
            clean,
            count=max(positions) + max(horizons),
            measured_layers=measured,
            dense_map=dense_map,
        )
        for position in positions:
            matrix_path = (
                matrix_root / f"anchor_{anchor_index:02d}_position_{position}.npz"
            )
            reference = None
            if matrix_path.exists():
                with np.load(matrix_path, allow_pickle=False) as saved:
                    reference = saved["matrix"].astype(np.float32)
                if not validate_only:
                    _append_records(
                        matrix=reference,
                        matrix_path=matrix_path,
                        root=context.root,
                        labels=labels,
                        probe_sizes=probe_sizes,
                        base_id=base_id,
                        pair=pair,
                        position=position,
                        records=records,
                        sensitivity=sensitivity,
                    )
                    continue
            elif validate_only:
                continue

            base_cache = clone_hybrid_cache(clean["cache"])
            current_length = int(clean["prompt_length"])
            for offset in range(position):
                base_cache = geometry._advance_cache(
                    bundle, base_cache, tokens[offset], current_length
                )
                current_length += 1
            local_tokens = tokens[position : position + max(horizons)]
            with torch.no_grad():
                probe = bundle.hf_model(
                    input_ids=torch.tensor([[local_tokens[0]]], device=device),
                    attention_mask=torch.ones(
                        (1, current_length + 1), dtype=torch.long, device=device
                    ),
                    past_key_values=clone_hybrid_cache(base_cache),
                    use_cache=True,
                )
            top_logits = torch.topk(
                probe.logits[0, -1], int(section["selected_logit_count"])
            ).indices.tolist()
            semantic_ids = geometry._semantic_ids(
                bundle.tokenizer,
                task.semantic_actions[min(position, len(task.semantic_actions) - 1)],
            )
            selected_logits = list(dict.fromkeys([*semantic_ids, *top_logits]))[
                : int(section["selected_logit_count"])
            ]
            columns: list[np.ndarray] = []
            slices: dict[str, list[int]] = defaultdict(list)
            direction_limit = max_count if reference is None else reference.shape[1]
            for start_index in range(0, direction_limit, batch_size):
                stop_index = min(start_index + batch_size, direction_limit)
                rows = {
                    name: directions[name][start_index:stop_index]
                    for name in ("recurrent", "conv", "kv")
                }
                current_batch = stop_index - start_index

                def target(
                    epsilon: torch.Tensor,
                    base_cache: Any = base_cache,
                    rows: dict[str, torch.Tensor] = rows,
                    local_tokens: list[int] = local_tokens,
                    current_length: int = current_length,
                    selected_logits: list[int] = selected_logits,
                    slices_ref: dict[str, list[int]] = slices,
                    record_slices: bool = start_index == 0,
                    current_batch: int = current_batch,
                ) -> torch.Tensor:
                    cache = _apply_direction_batch(
                        base_cache,
                        epsilon,
                        rows,
                        recurrent_layers,
                        attention_layers,
                    )
                    parts: list[torch.Tensor] = []
                    for step, token in enumerate(local_tokens, start=1):
                        with geometry.ActivationRecorder(
                            bundle.layers,
                            at=[int(value) for value in section["workspace_layers"]],
                            clone=False,
                            detach=False,
                        ) as recorder:
                            output = bundle.hf_model(
                                input_ids=torch.full(
                                    (current_batch, 1),
                                    token,
                                    dtype=torch.long,
                                    device=device,
                                ),
                                attention_mask=torch.ones(
                                    (current_batch, current_length + step),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                past_key_values=cache,
                                use_cache=True,
                            )
                        cache = output.past_key_values
                        if step in horizons:
                            offset = sum(part.shape[-1] for part in parts)
                            hidden = recorder.activations[main_layer][:, -1].float()
                            j_state = dense_map.dense_state(hidden, main_layer)
                            parts.append(j_state[:, selected_j_tensor])
                            if record_slices:
                                slices_ref[f"j_h{step}"].extend(
                                    range(offset, offset + len(selected_j))
                                )
                            offset += len(selected_j)
                            logits = output.logits[:, -1, selected_logits].float()
                            parts.append(logits)
                            if record_slices:
                                slices_ref[f"logits_h{step}"].extend(
                                    range(offset, offset + len(selected_logits))
                                )
                            offset += len(selected_logits)
                            semantic = torch.log_softmax(
                                output.logits[:, -1].float(), dim=-1
                            )[:, selected_logits]
                            parts.append(semantic)
                            if record_slices:
                                slices_ref[f"semantic_h{step}"].extend(
                                    range(offset, offset + len(selected_logits))
                                )
                            for workspace_layer in section["workspace_layers"]:
                                workspace = recorder.activations[int(workspace_layer)][
                                    :, -1
                                ].float()[:, : int(section["selected_workspace_count"])]
                                offset = sum(part.shape[-1] for part in parts)
                                parts.append(workspace)
                                if record_slices:
                                    slices_ref[
                                        f"workspace_l{workspace_layer}_h{step}"
                                    ].extend(
                                        range(offset, offset + workspace.shape[-1])
                                    )
                    return torch.cat(parts, dim=-1)

                epsilon = torch.zeros(
                    (current_batch,), device=device, dtype=torch.float32
                )
                _, derivative = torch.autograd.functional.jvp(
                    target,
                    epsilon,
                    torch.ones_like(epsilon),
                    create_graph=False,
                    strict=True,
                )
                columns.append(derivative.detach().cpu().numpy().astype(np.float32).T)
                if start_index == 0:
                    slices = {
                        name: sorted(set(indices)) for name, indices in slices.items()
                    }
                write_json_atomic(
                    context.raw_dir / context.run_id / "jvp_progress.json",
                    {
                        "status": "RUNNING",
                        "completed_local_states": anchor_index * len(positions)
                        + positions.index(position),
                        "total_local_states": len(anchors) * len(positions),
                        "completed_directions_current_state": stop_index,
                        "batch_size": batch_size,
                    },
                )
            matrix = np.concatenate(columns, axis=1)
            if reference is not None:
                difference = np.abs(matrix.astype(np.float64) - reference)
                norms = np.linalg.norm(matrix.astype(np.float64), axis=0)
                reference_norms = np.linalg.norm(reference.astype(np.float64), axis=0)
                cosines = np.sum(
                    matrix.astype(np.float64) * reference.astype(np.float64), axis=0
                ) / np.maximum(norms * reference_norms, 1e-20)
                equivalence = {
                    "reference_matrix": str(matrix_path.relative_to(context.root)),
                    "batch_size": batch_size,
                    "maximum_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                    "minimum_column_cosine": float(cosines.min()),
                    "mean_column_cosine": float(cosines.mean()),
                    "passed": bool(
                        difference.max()
                        <= amendment["equivalence_gate"]["maximum_absolute_difference"]
                        and cosines.min()
                        >= amendment["equivalence_gate"]["minimum_column_cosine"]
                    ),
                }
                write_json_atomic(
                    context.root
                    / "results/v13/processed/jvp_batch_equivalence_v13.json",
                    equivalence,
                )
                if not equivalence["passed"]:
                    raise RuntimeError(f"batched JVP equivalence failed: {equivalence}")
                return equivalence
            temporary = matrix_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                matrix=matrix,
                selected_j=selected_j,
                selected_logits=np.asarray(selected_logits),
                slices_json=np.asarray(json.dumps(slices)),
            )
            temporary.replace(matrix_path)
            _append_records(
                matrix=matrix,
                matrix_path=matrix_path,
                root=context.root,
                labels=labels,
                probe_sizes=probe_sizes,
                base_id=base_id,
                pair=pair,
                position=position,
                records=records,
                sensitivity=sensitivity,
            )
            computed_states += 1
            write_json_atomic(
                context.raw_dir / context.run_id / "jvp_progress.json",
                {
                    "status": "RUNNING",
                    "completed_local_states": anchor_index * len(positions)
                    + positions.index(position)
                    + 1,
                    "total_local_states": len(anchors) * len(positions),
                    "completed_directions_current_state": max_count,
                    "batch_size": batch_size,
                },
            )

    if validate_only:
        raise RuntimeError("no scalar-JVP reference matrix was available")
    frame = pd.DataFrame(records)
    path = context.root / geometry.JVP_RECORDS
    frame.to_parquet(path, index=False, compression="zstd")
    mixed = frame[frame.probe_family == "mixed"]
    curves = {
        str(size): {
            "median_r90": float(group.rank_90.median()),
            "median_r95": float(group.rank_95.median()),
            "median_r99": float(group.rank_99.median()),
            "mean_stable_rank": float(group.stable_rank.mean()),
            "mean_effective_rank": float(group.effective_rank.mean()),
        }
        for size, group in mixed.groupby("probe_direction_count")
    }
    robustness = {
        str(name): {
            "median_r95": float(group.rank_95.median()),
            "mean_r95": float(group.rank_95.mean()),
        }
        for name, group in frame[frame.probe_family != "mixed"].groupby("probe_family")
    }
    summary = {
        "schema_version": geometry.SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "runtime_amendment_digest": amendment["freeze_digest"],
        "exact_autograd_jvp": True,
        "batched_independent_directions": True,
        "batch_size": batch_size,
        "full_raw_jacobian_claimed": False,
        "local_state_count": len(anchors) * len(positions),
        "newly_computed_local_states": computed_states,
        "probe_scaling": curves,
        "probe_family_robustness": robustness,
        "probe_family_mean_column_sensitivity": {
            name: float(np.mean(values)) for name, values in sensitivity.items()
        },
        "records": str(geometry.JVP_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / geometry.JVP_SUMMARY, summary)
    return summary


def main() -> None:
    root = Path.cwd()
    if "--freeze-amendment" in sys.argv:
        print(_freeze(root)["freeze_digest"])
        return
    batch_size = DEFAULT_BATCH_SIZE
    if "--jvp-batch-size" in sys.argv:
        index = sys.argv.index("--jvp-batch-size")
        batch_size = int(sys.argv[index + 1])
        del sys.argv[index : index + 2]
    validate_only = "--validate-only" in sys.argv
    if validate_only:
        sys.argv.remove("--validate-only")
    if batch_size < 1 or batch_size > DEFAULT_BATCH_SIZE:
        raise SystemExit(f"batch size must be in [1, {DEFAULT_BATCH_SIZE}]")
    _verify(root)
    geometry._load_encoder = _load_encoder_memory_efficient
    geometry.load_model_bundle = _load_model
    geometry.PROTOCOL_V13 = PROTOCOL

    original = geometry._run_jvp

    def run(context: Any) -> dict[str, Any]:
        return _run_jvp_batched(
            context, batch_size=batch_size, validate_only=validate_only
        )

    geometry._run_jvp = run
    try:
        geometry.main()
    finally:
        geometry._run_jvp = original


if __name__ == "__main__":
    main()
