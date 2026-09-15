"""Correct v7 endpoint serialization, then run frozen aligned compression.

The original v7 attribution metrics remain immutable and valid.  Only the
auxiliary endpoint archive was affected: the save loop used the final ``full``
trajectory for all four labels.  This runner uses the already frozen v6
clean/intervened endpoint arrays and never rewrites the defective archive.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jclosure.arch_compression_v7 import (
    apply_delta_blocks,
    component_order,
    fit_dual_pca,
    flatten_blocks,
    pca_scores,
    reconstruct_blocks,
)
from jclosure.experiments.arch_compression_v7 import (
    _compression_alpha,
    _cosine_rows,
    _paired_ci,
    _ridge,
)
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _source_items,
    _tasks,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.metrics import jensen_shannon_from_logits, token_log_odds
from jclosure.protocol_v7 import FREEZE_PATH as ATTRIBUTION_FREEZE_PATH
from jclosure.protocol_v7_corrective import (
    PROTOCOL_V7_CORRECTIVE,
    SCHEMA_VERSION_V7_CORRECTIVE,
    build_freeze,
    verify_freeze,
)
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.records_v7 import ChannelCompressionRecord
from jclosure.statistics import clustered_bootstrap_ci

CORRECTED_CEILING = Path(
    "results/v7/processed/raw_arch_channel_ceiling_v7_corrected.json"
)
CORRECTED_COMPRESSION = Path("results/v7/processed/arch_compression_v7_corrected.json")


def _load_capture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(
        (root / "results/v7/processed/raw_arch_channel_ceiling_v7.json").read_text()
    )
    artifact = root / str(summary["artifact"])
    if sha256_file(artifact) != summary["artifact_sha256"]:
        raise RuntimeError("architecture-channel capture hash mismatch")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    return summary, payload


def _ordered_v6_arrays(context: Any, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    relative = context.config["persistent_channels_v7_corrective"]["source_v6_artifact"]
    with np.load(context.root / relative, allow_pickle=False) as source:
        source_index = {
            str(value): index for index, value in enumerate(source["base_trial_id"])
        }
        order = [source_index[str(value)] for value in payload["base_trial_id"]]
        return {
            "current_j": source["j_intervened"][order, 0].astype(np.float32),
            "clean_next_j": source["j_clean"][order, 1].astype(np.float32),
            "full_next_j": source["j_intervened"][order, 1].astype(np.float32),
        }


def _predictive_comparison(
    current_j: np.ndarray,
    channel_scores: np.ndarray,
    target_delta: np.ndarray,
    train: list[int],
    test: list[int],
    *,
    alpha: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    baseline = _ridge(
        current_j[train], current_j[test], target_delta[train], alpha=alpha
    )
    augmented_input = np.concatenate((current_j, channel_scores), axis=1)
    augmented = _ridge(
        augmented_input[train],
        augmented_input[test],
        target_delta[train],
        alpha=alpha,
    )
    target = target_delta[test]
    baseline_cosine = _cosine_rows(baseline, target)
    augmented_cosine = _cosine_rows(augmented, target)
    baseline_rmse = np.sqrt(np.mean((baseline - target) ** 2, axis=1))
    augmented_rmse = np.sqrt(np.mean((augmented - target) ** 2, axis=1))
    return {
        "j_only_delta_cosine": _paired_ci(baseline_cosine, seed, resamples),
        "full_channel_delta_cosine": _paired_ci(augmented_cosine, seed, resamples),
        "delta_cosine_gain": _paired_ci(
            augmented_cosine - baseline_cosine, seed, resamples
        ),
        "j_only_delta_rmse": _paired_ci(baseline_rmse, seed, resamples),
        "full_channel_delta_rmse": _paired_ci(augmented_rmse, seed, resamples),
        "delta_rmse_improvement": _paired_ci(
            baseline_rmse - augmented_rmse, seed, resamples
        ),
        "target_delta_l2": _paired_ci(np.linalg.norm(target, axis=1), seed, resamples),
    }


def _ceiling(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    capture, payload = _load_capture(context.root)
    blocks = flatten_blocks(payload)
    split = np.asarray(payload["split"])
    train = np.flatnonzero(split == "localization_fit").tolist()
    test = np.flatnonzero(split == "attribution_test").tolist()
    model = fit_dual_pca(blocks, train)
    scores = pca_scores(model, blocks, list(range(len(split)))).numpy()
    endpoint = _ordered_v6_arrays(context, payload)
    target_delta = endpoint["full_next_j"] - endpoint["clean_next_j"]
    section = context.config["persistent_channels_v7_corrective"]
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    comparison = _predictive_comparison(
        endpoint["current_j"],
        scores,
        target_delta,
        train,
        test,
        alpha=float(section["predictor"]["alpha"]),
        seed=seed,
        resamples=resamples,
    )
    attribution = json.loads(
        (
            context.root / "results/v7/processed/persistent_channel_attribution_v7.json"
        ).read_text()
    )
    gain_cosine = comparison["delta_cosine_gain"]
    gain_rmse = comparison["delta_rmse_improvement"]
    authorized = bool(
        attribution["classification"] != "transient_or_unresolved"
        and (gain_cosine["lower"] > 0 or gain_rmse["lower"] > 0)
    )
    result = {
        "schema_version": SCHEMA_VERSION_V7_CORRECTIVE,
        "protocol_version": PROTOCOL_V7_CORRECTIVE,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "source_capture_run_id": capture["run_id"],
        "source_capture_artifact": capture["artifact"],
        "source_capture_artifact_sha256": capture["artifact_sha256"],
        "effective_raw_rank": model.rank,
        "train_count": len(train),
        "test_count": len(test),
        "ridge_alpha": float(section["predictor"]["alpha"]),
        **comparison,
        "direct_causal_swap_ceiling": attribution["effects"]["pooled"],
        "raw_channel_ceiling_authorized": authorized,
        "correction_note": (
            "The immutable v7 endpoint archive aliased all conditions to full. "
            "This analysis uses the frozen v6 clean/intervened step-1 J arrays."
        ),
        "scope_note": (
            "Predictive ceiling is intervention-delta prediction on 33 held-out "
            "pairs; direct cache swapping is the causal ceiling."
        ),
    }
    write_json_atomic(context.root / CORRECTED_CEILING, result)
    context.finish("COMPLETED_CORRECTED_CEILING", summary=result)
    return result


def _run(context: Any, bundle: Any, freeze: dict[str, Any]) -> None:
    ceiling = json.loads((context.root / CORRECTED_CEILING).read_text())
    if ceiling["source_freeze_digest"] != freeze["freeze_digest"]:
        raise RuntimeError("corrected ceiling freeze mismatch")
    if not ceiling["raw_channel_ceiling_authorized"]:
        context.finish(
            "GATED_NO_STRONG_RAW_CEILING",
            source_freeze_digest=freeze["freeze_digest"],
        )
        return
    _, payload = _load_capture(context.root)
    blocks = flatten_blocks(payload)
    split = np.asarray(payload["split"])
    train = np.flatnonzero(split == "localization_fit").tolist()
    test = np.flatnonzero(split == "attribution_test").tolist()
    model = fit_dual_pca(blocks, train)
    train_scores = pca_scores(model, blocks, train)
    test_scores = pca_scores(model, blocks, test)
    endpoint = _ordered_v6_arrays(context, payload)
    target_delta = torch.from_numpy(endpoint["full_next_j"] - endpoint["clean_next_j"])
    target_absolute = torch.from_numpy(endpoint["full_next_j"])
    predictive_order = component_order(train_scores, target_absolute[train])
    causal_order = component_order(train_scores, target_delta[train])
    section = context.config["persistent_channels_v7"]
    rec_layers = [int(value) for value in payload["recurrent_layers"]]
    kv_layers = [int(value) for value in payload["attention_layers"]]
    methods = [str(value) for value in section["compression"]["methods"]]
    dimensions = [int(value) for value in section["compression"]["dimensions"]]
    coefficients: dict[tuple[str, int], torch.Tensor] = {}
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for method in methods:
        for dimension in dimensions:
            key = (method, dimension)
            coefficients[key], metadata[key] = _compression_alpha(
                method,
                dimension,
                model,
                train_scores,
                test_scores,
                predictive_order,
                causal_order,
                int(context.seed),
            )
    parent = json.loads((context.root / ATTRIBUTION_FREEZE_PATH).read_text())
    items = {item["base_trial_id"]: item for item in _source_items(context, parent)}
    tasks = _tasks(context)
    measured = [int(value) for value in section["measured_layers"]]
    main_layer = max(measured)
    _, _, dense_map = _load_encoder(context, bundle)
    records: list[dict[str, Any]] = []
    for test_row, global_index in enumerate(test):
        base_id = str(payload["base_trial_id"][global_index])
        item = items[base_id]
        task = tasks[item["prompt_id"]]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=int(section["intervention_layer"]),
            candidate=None,
        )
        tokens = _teacher_tokens(
            bundle, clean, count=1, measured_layers=measured, dense_map=dense_map
        )
        clean_trajectory = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        observed = tuple(block[global_index] for block in blocks)
        full_trajectory = _teacher_forced_trajectory(
            bundle,
            apply_delta_blocks(
                clean["cache"],
                observed,
                recurrent_layers=rec_layers,
                attention_layers=kv_layers,
            ),
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_j = clean_trajectory["j"][main_layer][0]
        full_j = full_trajectory["j"][main_layer][0]
        full_delta = full_j - clean_j
        full_norm = float(np.linalg.norm(full_delta))
        clean_logits = clean_trajectory["logits"][0]
        full_logits = full_trajectory["logits"][0]
        target_token = int(np.argmax(clean_logits))
        full_odds_delta = token_log_odds(
            torch.from_numpy(full_logits), target_token
        ) - token_log_odds(torch.from_numpy(clean_logits), target_token)
        for method in methods:
            for dimension in dimensions:
                key = (method, dimension)
                reconstructed = reconstruct_blocks(model, coefficients[key], test_row)
                trajectory = _teacher_forced_trajectory(
                    bundle,
                    apply_delta_blocks(
                        clean["cache"],
                        reconstructed,
                        recurrent_layers=rec_layers,
                        attention_layers=kv_layers,
                    ),
                    tokens,
                    prompt_length=clean["prompt_length"],
                    measured_layers=measured,
                    dense_map=dense_map,
                )
                candidate_j = trajectory["j"][main_layer][0]
                delta = candidate_j - clean_j
                norm = float(np.linalg.norm(delta))
                cosine = float(np.dot(delta, full_delta) / max(norm * full_norm, 1e-20))
                magnitude = norm / max(full_norm, 1e-20)
                residual_gain = float(
                    np.linalg.norm(candidate_j - full_j) / max(full_norm, 1e-20)
                )
                gap_closed = 1 - residual_gain
                causal_gap = cosine * float(np.exp(-abs(np.log(max(magnitude, 1e-20)))))
                candidate_logits = trajectory["logits"][0]
                odds_delta = token_log_odds(
                    torch.from_numpy(candidate_logits), target_token
                ) - token_log_odds(torch.from_numpy(clean_logits), target_token)
                record = ChannelCompressionRecord(
                    run_id=context.run_id,
                    channel="mixed_kv_recurrent_conv",
                    method=method,
                    dimension=dimension,
                    split="attribution_test",
                    predictive_gap_closed=gap_closed,
                    causal_gap_closed=causal_gap,
                    conditional_residual_gain=residual_gain,
                    causal_direction_cosine=cosine,
                    magnitude_ratio=magnitude,
                    semantic_delta_agreement=float(
                        (np.argmax(candidate_logits) != np.argmax(clean_logits))
                        == (np.argmax(full_logits) != np.argmax(clean_logits))
                    ),
                    output_sign_agreement=float(
                        np.sign(odds_delta) == np.sign(full_odds_delta)
                    ),
                    metadata={
                        **metadata[key],
                        "base_trial_id": base_id,
                        "prompt_id": item["prompt_id"],
                        "family": item["family"],
                        "full_delta_l2": full_norm,
                        "output_js_to_full": jensen_shannon_from_logits(
                            torch.from_numpy(candidate_logits),
                            torch.from_numpy(full_logits),
                        ),
                    },
                ).to_dict()
                record["schema_version"] = SCHEMA_VERSION_V7_CORRECTIVE
                record["protocol_version"] = PROTOCOL_V7_CORRECTIVE
                records.append({**record, "record_type": "channel_compression_trial"})
    path = context.raw_dir / context.run_id / "arch_compression_v7_corrected.jsonl"
    append_jsonl(path, records)
    context.finish(
        "COMPLETED_CORRECTED_COMPRESSION",
        source_freeze_digest=freeze["freeze_digest"],
        records=str(path.relative_to(context.root)),
        record_count=len(records),
        effective_sample_rank=model.rank,
    )


def _ci(frame: pd.DataFrame, column: str, seed: int, resamples: int) -> dict[str, Any]:
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
    for path in context.raw_dir.glob("arch-compression-v7-corrective-*/manifest.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "COMPLETED_CORRECTED_COMPRESSION":
            manifests.append(value)
    if not manifests:
        raise RuntimeError("no completed corrected architecture compression run")
    source = sorted(manifests, key=lambda value: value["run_id"])[-1]
    rows = [
        json.loads(line)
        for line in (context.root / source["records"]).read_text().splitlines()
    ]
    frame = pd.DataFrame(rows)
    frame["prompt_id"] = frame["metadata"].map(lambda value: value["prompt_id"])
    frame["family"] = frame["metadata"].map(lambda value: value["family"])
    seed = int(context.config["reproducibility"]["bootstrap_seed"])
    resamples = int(context.config["persistent_channels_v7"]["bootstrap_resamples"])
    metrics = (
        "predictive_gap_closed",
        "causal_gap_closed",
        "conditional_residual_gain",
        "causal_direction_cosine",
        "magnitude_ratio",
        "semantic_delta_agreement",
        "output_sign_agreement",
    )
    summaries = []
    for (method, dimension), values in frame.groupby(
        ["method", "dimension"], sort=True
    ):
        summaries.append(
            {
                "method": method,
                "dimension": int(dimension),
                "effective_dimension": int(
                    values.iloc[0]["metadata"]["effective_dimension"]
                ),
                "rank_limited": bool(values.iloc[0]["metadata"].get("rank_limited")),
                **{metric: _ci(values, metric, seed, resamples) for metric in metrics},
            }
        )
    rules = context.config["persistent_channels_v7"]["compression"]
    passing = [
        value
        for value in summaries
        if value["predictive_gap_closed"]["lower"]
        >= float(rules["predictive_gap_closed"])
        and value["causal_gap_closed"]["lower"] >= float(rules["causal_gap_closed"])
        and value["conditional_residual_gain"]["upper"]
        <= float(rules["maximum_conditional_gain"])
        and value["causal_direction_cosine"]["lower"]
        >= float(rules["causal_direction_minimum"])
    ]
    # The empirical fit rank is 32.  Nominal 64--512 dimensions therefore do
    # not establish a state of those dimensions on independent variation.
    independently_identified = [value for value in passing if not value["rank_limited"]]
    selected = (
        min(independently_identified, key=lambda value: value["dimension"])
        if independently_identified
        else None
    )
    serial = frame.copy()
    serial["metadata"] = serial["metadata"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    record_path = (
        context.root / "results/v7/processed/arch_compression_v7_corrected.parquet"
    )
    serial.to_parquet(record_path, index=False, compression="zstd")
    ceiling = json.loads((context.root / CORRECTED_CEILING).read_text())
    result = {
        "schema_version": SCHEMA_VERSION_V7_CORRECTIVE,
        "protocol_version": PROTOCOL_V7_CORRECTIVE,
        "run_id": context.run_id,
        "source_run_id": source["run_id"],
        "source_freeze_digest": freeze["freeze_digest"],
        "effective_sample_rank": source["effective_sample_rank"],
        "raw_ceiling": ceiling,
        "summaries": summaries,
        "nominal_passing_count": len(passing),
        "independently_identified_passing_count": len(independently_identified),
        "selected": selected,
        "candidate_sufficient_persistent_state": selected is not None,
        "autonomous_controller_authorized": selected is not None,
        "records": str(record_path.relative_to(context.root)),
        "limitation": (
            "All cache-delta encoders are fit on 33 pairs; empirical rank is 32. "
            "Nominal 64-512D settings are identical rank-limited reconstructions."
        ),
    }
    write_json_atomic(context.root / CORRECTED_COMPRESSION, result)
    context.finish("COMPLETED_CORRECTED_COMPRESSION_ANALYSIS", summary=result)
    return result


def main() -> None:
    parser = standard_parser(
        "v7 corrective architecture compression",
        "configs/persistent_channels_v7_corrective.yaml",
    )
    parser.add_argument(
        "--stage", choices=("freeze", "ceiling", "run", "analyze"), required=True
    )
    args = parser.parse_args()
    context = initialize_context("arch-compression-v7-corrective", args)
    try:
        if args.stage == "freeze":
            freeze = build_freeze(context.root, context.config)
            context.finish("COMPLETED_CORRECTIVE_FREEZE", freeze=freeze)
            return
        freeze = verify_freeze(context.root, context.config)
        if args.stage == "ceiling":
            _ceiling(context, freeze)
            return
        if args.stage == "analyze":
            _analyze(context, freeze)
            return
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        from jclosure.model import load_model_bundle

        _run(context, load_model_bundle(context.config), freeze)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
