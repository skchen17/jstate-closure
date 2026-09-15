"""Mechanism-aligned REC/conv and KV layer/head/token localization."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.cache_v7 import clone_hybrid_cache, make_chimeric_cache
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.persistent_channels_v7 import (
    _candidate_states,
    _prefill,
    _source_items,
    _tasks,
    _teacher_forced_trajectory,
    _teacher_tokens,
    _trajectory_metrics,
)
from jclosure.protocol_v7_stage2 import build_stage_freeze, verify_stage_freeze
from jclosure.provenance import append_jsonl, write_json_atomic
from jclosure.records_v7 import PROTOCOL_V7
from jclosure.statistics import clustered_bootstrap_ci


def _specs(section: dict[str, Any]) -> list[dict[str, Any]]:
    rec_layers = [int(value) for value in section["downstream_recurrent_layers"]]
    kv_layers = [int(value) for value in section["downstream_attention_layers"]]
    output: list[dict[str, Any]] = [
        {"name": "clean"},
        {"name": "full", "kv": True, "matrix": True, "conv": True},
        {"name": "recurrent_matrix_all", "matrix": True},
        {"name": "conv_all", "conv": True},
        {"name": "recurrent_both_all", "matrix": True, "conv": True},
    ]
    for layer in rec_layers:
        output.extend(
            (
                {
                    "name": f"rec_layer_{layer}_both",
                    "matrix": True,
                    "conv": True,
                    "rec_layers": [layer],
                },
                {
                    "name": f"rec_layer_{layer}_matrix",
                    "matrix": True,
                    "rec_layers": [layer],
                },
                {
                    "name": f"rec_layer_{layer}_conv",
                    "conv": True,
                    "rec_layers": [layer],
                },
            )
        )
    for layer in kv_layers:
        output.append({"name": f"kv_layer_{layer}", "kv": True, "kv_layers": [layer]})
        for head in range(4):
            output.append(
                {
                    "name": f"kv_layer_{layer}_head_{head}",
                    "kv": True,
                    "kv_layers": [layer],
                    "kv_heads": [head],
                }
            )
    for head in range(4):
        output.append({"name": f"kv_head_{head}", "kv": True, "kv_heads": [head]})
    for window in section["localization"]["attention_token_windows"]:
        output.append(
            {
                "name": f"kv_last_{int(window)}_tokens",
                "kv": True,
                "kv_window": int(window),
            }
        )
    return output


def _cache_from_spec(clean: Any, perturbed: Any, spec: dict[str, Any]) -> Any:
    if spec["name"] == "clean":
        return clone_hybrid_cache(clean)
    positions = None
    if "kv_window" in spec:
        length = clean.layers[27].keys.shape[-2]
        start = max(0, length - int(spec["kv_window"]))
        positions = range(start, length)
    return make_chimeric_cache(
        clean,
        perturbed,
        kv_from_perturbed=bool(spec.get("kv", False)),
        recurrent_from_perturbed=bool(spec.get("matrix", False)),
        conv_from_perturbed=bool(spec.get("conv", False)),
        attention_layers=spec.get("kv_layers"),
        recurrent_layers=spec.get("rec_layers"),
        kv_heads=spec.get("kv_heads"),
        kv_positions=positions,
    )


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 1e-20 else float(np.dot(left, right) / denominator)


def _run(context: Any, bundle: Any, freeze: dict[str, Any], limit: int | None) -> None:
    items = _source_items(
        context,
        json.loads(
            (context.root / "artifacts/persistent_channels_v7.freeze.json").read_text()
        ),
    )
    if limit is not None:
        items = items[: int(limit)]
    tasks = _tasks(context)
    _, candidates = _candidate_states(context)
    section = context.config["persistent_channels_v7"]
    measured_layers = [int(value) for value in section["measured_layers"]]
    main_layer = max(measured_layers)
    _, _, dense_map = _load_encoder(context, bundle)
    device = next(bundle.hf_model.parameters()).device
    specs = _specs(section)
    records = []
    for item in items:
        task = tasks[item["prompt_id"]]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured_layers,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        perturbed = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured_layers,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=torch.from_numpy(candidates[int(item["artifact_index"])]).to(
                device
            ),
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=1,
            measured_layers=measured_layers,
            dense_map=dense_map,
        )
        trajectories = {
            spec["name"]: _teacher_forced_trajectory(
                bundle,
                _cache_from_spec(clean["cache"], perturbed["cache"], spec),
                tokens,
                prompt_length=clean["prompt_length"],
                measured_layers=measured_layers,
                dense_map=dense_map,
            )
            for spec in specs
        }
        clean_trajectory = trajectories["clean"]
        full_delta = (
            trajectories["full"]["j"][main_layer][0]
            - clean_trajectory["j"][main_layer][0]
        )
        full_output = trajectories["full"]["logits"][0] - clean_trajectory["logits"][0]
        full_norm = float(np.linalg.norm(full_delta))
        for spec in specs:
            name = spec["name"]
            trajectory = trajectories[name]
            metrics = _trajectory_metrics(
                clean_trajectory, trajectory, main_layer=main_layer
            )
            delta = metrics.pop("next_j_delta")
            output = metrics.pop("output_delta")
            metrics.update(
                {
                    "direction_cosine_to_full": _safe_cosine(delta, full_delta),
                    "magnitude_ratio_to_full": float(
                        np.linalg.norm(delta) / max(full_norm, 1e-20)
                    ),
                    "output_delta_cosine_to_full": _safe_cosine(output, full_output),
                }
            )
            records.append(
                {
                    "schema_version": 9,
                    "protocol_version": PROTOCOL_V7,
                    "record_type": "channel_localization_trial",
                    "run_id": context.run_id,
                    "base_trial_id": item["base_trial_id"],
                    "prompt_id": item["prompt_id"],
                    "family": item["family"],
                    "split": item["role"],
                    "condition": name,
                    "valid": full_norm > 0,
                    "spec": spec,
                    "metrics": metrics,
                }
            )
    path = context.raw_dir / context.run_id / "channel_localization_v7.jsonl"
    append_jsonl(path, records)
    context.finish(
        "COMPLETED_LOCALIZATION",
        source_freeze_digest=freeze["freeze_digest"],
        records=str(path.relative_to(context.root)),
        record_count=len(records),
    )


def _ci(frame: pd.DataFrame, column: str, seed: int, resamples: int) -> dict[str, Any]:
    if frame.empty:
        return {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
    return asdict(
        clustered_bootstrap_ci(
            frame,
            cluster_col="prompt_id",
            value_col=column,
            seed=seed,
            n_resamples=resamples,
        )
    )


def _analyze(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    manifests = []
    for path in context.raw_dir.glob("channel-localization-v7-*/manifest.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "COMPLETED_LOCALIZATION":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed localization run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"]).read_text().splitlines()
    ]
    frame = pd.DataFrame(rows)
    for column in (
        "next_j_l2",
        "output_js_divergence",
        "direction_cosine_to_full",
        "magnitude_ratio_to_full",
        "output_delta_cosine_to_full",
    ):
        frame[column] = frame["metrics"].map(lambda value, key=column: value.get(key))
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    summaries = {}
    for split in ("localization_fit", "attribution_test"):
        subset = frame[(frame["split"] == split) & frame["valid"]]
        summaries[split] = {
            condition: {
                metric: _ci(values.dropna(subset=[metric]), metric, seed, resamples)
                for metric in (
                    "next_j_l2",
                    "output_js_divergence",
                    "direction_cosine_to_full",
                    "magnitude_ratio_to_full",
                    "output_delta_cosine_to_full",
                )
            }
            for condition, values in subset.groupby("condition", sort=True)
        }
    fit = summaries["localization_fit"]
    rec_layers = [
        int(value)
        for value in context.config["persistent_channels_v7"][
            "downstream_recurrent_layers"
        ]
    ]
    kv_layers = [
        int(value)
        for value in context.config["persistent_channels_v7"][
            "downstream_attention_layers"
        ]
    ]
    selected_rec_layers = sorted(
        rec_layers,
        key=lambda layer: fit[f"rec_layer_{layer}_both"]["direction_cosine_to_full"][
            "estimate"
        ],
        reverse=True,
    )
    selected_kv_layer = max(
        kv_layers,
        key=lambda layer: fit[f"kv_layer_{layer}"]["direction_cosine_to_full"][
            "estimate"
        ],
    )
    selected_kv_heads = sorted(
        range(4),
        key=lambda head: fit[f"kv_head_{head}"]["direction_cosine_to_full"]["estimate"],
        reverse=True,
    )
    output_frame = frame.copy()
    output_frame["metrics"] = output_frame["metrics"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    output_frame["spec"] = output_frame["spec"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    output_path = context.processed_dir / "channel_localization_v7.parquet"
    output_frame.to_parquet(output_path, index=False, compression="zstd")
    summary = {
        "schema_version": 9,
        "protocol_version": PROTOCOL_V7,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "source_freeze_digest": source["source_freeze_digest"],
        "summaries": summaries,
        "selected_recurrent_layers_ranked": selected_rec_layers,
        "selected_kv_layer": selected_kv_layer,
        "selected_kv_heads_ranked": selected_kv_heads,
        "records": str(output_path.relative_to(context.root)),
    }
    write_json_atomic(context.processed_dir / "channel_localization_v7.json", summary)
    context.finish("COMPLETED_LOCALIZATION_ANALYSIS", summary=summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "v7 persistent-channel localization", "configs/persistent_channels_v7.yaml"
    )
    parser.add_argument("--stage", choices=("freeze", "run", "analyze"), required=True)
    args = parser.parse_args()
    context = initialize_context("channel-localization-v7", args)
    try:
        if args.stage == "freeze":
            freeze = build_stage_freeze(context.root, context.config, "localization")
            context.finish("COMPLETED_FREEZE", freeze=freeze)
            return
        freeze = verify_stage_freeze(context.root, context.config, "localization")
        if args.stage == "analyze":
            _analyze(context, freeze)
            return
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        from jclosure.model import load_model_bundle

        _run(context, load_model_bundle(context.config), freeze, args.limit)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
