"""Corrected sub-512 sufficiency and residual-localization audit for v10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.sufficiency_v9 import (
    _apply_ridge_weights,
    _bootstrap_ci,
    _cosine_rows,
    _load_data,
    _ridge_weights,
    _top10_agreement,
    build_ordered_latents,
    safe_gap_ratio,
)
from jclosure.protocol_v10 import (
    PROTOCOL_V10,
    SCHEMA_VERSION_V10,
    build_candidate_freeze,
    build_freeze,
    verify_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic

SWEEP = Path("results/v10/processed/corrected_sub512_sufficiency_v10.parquet")
SWEEP_SUMMARY = Path("results/v10/processed/corrected_sub512_sufficiency_v10.json")
RESIDUAL = Path("results/v10/processed/residual_localization_audit_v10.parquet")
RESIDUAL_SUMMARY = Path("results/v10/processed/residual_localization_audit_v10.json")
SEMANTIC = Path("results/v10/processed/semantic_sufficiency_audit_v10.parquet")
SEMANTIC_SUMMARY = Path("results/v10/processed/semantic_sufficiency_audit_v10.json")


def semantic_retention(
    compact: float, full: float, epsilon: float
) -> dict[str, Any]:
    identified = abs(full) >= epsilon
    return {
        "semantic_retention_relative_to_full": (
            float(compact / full) if identified else None
        ),
        "semantic_retention_identified": identified,
        "semantic_compact_minus_full": float(compact - full),
    }


def select_confirmatory_candidates(
    eligible: list[int], dimensions: list[int], reference: int, fallback: list[int]
) -> tuple[list[int], str]:
    eligible = sorted(set(eligible))
    if not eligible:
        return sorted(set(fallback)), "DIAGNOSTIC_ONLY_NO_ALL_FAMILY_PASS"
    smallest = eligible[0]
    output = [smallest]
    interior = [value for value in eligible if smallest < value < reference]
    if interior:
        target = (smallest + reference) / 2
        output.append(min(interior, key=lambda value: (abs(value - target), value)))
    if reference in dimensions:
        output.append(reference)
    return sorted(set(output))[:3], "OBSERVATIONAL_GATE_PASS"


def _predict(
    x: np.ndarray,
    targets: dict[str, np.ndarray],
    train: np.ndarray,
    evaluate: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    weights = _ridge_weights(x[train], x[evaluate], alpha)
    return {
        name: _apply_ridge_weights(weights, target[train])
        for name, target in targets.items()
    }


def _metric_record(
    *,
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    ceiling: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    local: np.ndarray,
    epsilon: float,
    seeds: list[int],
    resamples: int,
) -> dict[str, Any]:
    base = _cosine_rows(baseline["next_delta"][local], truth["next_delta"][local])
    compact = _cosine_rows(
        candidate["next_delta"][local], truth["next_delta"][local]
    )
    full = _cosine_rows(ceiling["next_delta"][local], truth["next_delta"][local])
    base_semantic = _top10_agreement(
        baseline["next_absolute"][local], truth["next_absolute"][local]
    )
    compact_semantic = _top10_agreement(
        candidate["next_absolute"][local], truth["next_absolute"][local]
    )
    full_semantic = _top10_agreement(
        ceiling["next_absolute"][local], truth["next_absolute"][local]
    )
    conditional = full - compact
    semantic_residual = full_semantic - compact_semantic
    semantic_values = semantic_retention(
        float(compact_semantic.mean()), float(full_semantic.mean()), epsilon
    )
    return {
        **safe_gap_ratio(
            float(base.mean()), float(compact.mean()), float(full.mean()), epsilon
        ),
        "conditional_residual_gain": _bootstrap_ci(
            conditional, seeds, resamples
        ),
        "semantic_baseline": float(base_semantic.mean()),
        "semantic_compact": float(compact_semantic.mean()),
        "semantic_full_ceiling": float(full_semantic.mean()),
        **semantic_values,
        "semantic_conditional_residual_gain": _bootstrap_ci(
            semantic_residual, seeds, resamples
        ),
        "semantic_compact_ci": _bootstrap_ci(
            compact_semantic, seeds, resamples
        ),
        "semantic_full_ci": _bootstrap_ci(full_semantic, seeds, resamples),
    }


def _gate(metrics: dict[str, Any], section: dict[str, Any]) -> dict[str, bool]:
    epsilon = float(section["gap_denominator_epsilon"])
    predictive = (
        metrics["gap_closed"] >= float(section["predictive_gap_closed_minimum"])
        if metrics["identified"]
        else abs(metrics["candidate_score"] - metrics["ceiling_score"]) <= epsilon
    )
    conditional = bool(
        metrics["conditional_residual_gain"]["upper"]
        <= float(section["conditional_residual_gain_maximum"])
    )
    retention = metrics["semantic_retention_relative_to_full"]
    semantic_relative = bool(
        (retention is None or retention >= float(section["semantic_relative_retention_minimum"]))
        and metrics["semantic_conditional_residual_gain"]["upper"]
        <= float(section["semantic_residual_gain_maximum"])
    )
    return {
        "predictive_pass": bool(predictive),
        "conditional_pass": conditional,
        "semantic_relative_pass": semantic_relative,
        "observational_pass": bool(predictive and conditional and semantic_relative),
    }


def _local_indices(evaluation: np.ndarray, group: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(evaluation.tolist())}
    return np.asarray([lookup[value] for value in group], dtype=int)


def _folds(ids: np.ndarray, train: np.ndarray, count: int) -> list[np.ndarray]:
    assigned = np.asarray(
        [int(hashlib.sha256(ids[index].encode()).hexdigest(), 16) % count for index in train]
    )
    return [train[assigned == fold] for fold in range(count)]


def cross_fitted_residuals(
    compact: np.ndarray,
    block: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    folds: list[np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_lookup = {value: index for index, value in enumerate(train.tolist())}
    residual_train = np.zeros((len(train), block.shape[1]), dtype=np.float32)
    for held in folds:
        held_set = set(held.tolist())
        fit = np.asarray([value for value in train if value not in held_set])
        weights = _ridge_weights(compact[fit], compact[held], alpha)
        predicted = _apply_ridge_weights(weights, block[fit])
        positions = [train_lookup[value] for value in held]
        residual_train[positions] = block[held] - predicted
    weights = _ridge_weights(compact[train], compact[evaluation], alpha)
    residual_evaluation = block[evaluation] - _apply_ridge_weights(
        weights, block[train]
    )
    denominator_train = float(np.mean(block[train].astype(np.float64) ** 2))
    denominator_eval = float(np.mean(block[evaluation].astype(np.float64) ** 2))
    return residual_train, residual_evaluation, {
        "train_oof_residual_fraction": float(
            np.mean(residual_train.astype(np.float64) ** 2)
            / max(denominator_train, 1e-20)
        ),
        "evaluation_residual_fraction": float(
            np.mean(residual_evaluation.astype(np.float64) ** 2)
            / max(denominator_eval, 1e-20)
        ),
    }


def _predict_split(
    train_x: np.ndarray,
    evaluation_x: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return _apply_ridge_weights(
        _ridge_weights(train_x, evaluation_x, alpha), target[train]
    )


def _residual_audit(
    *,
    context: Any,
    data: dict[str, Any],
    latent_bank: dict[str, np.ndarray],
    train: np.ndarray,
    evaluation: np.ndarray,
    validation: np.ndarray,
    final: np.ndarray,
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dimensions = [int(value) for value in section["residual_audit_dimensions"]]
    method = str(section["primary_method"])
    alpha = float(section["ridge_alpha"])
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    fold_rows = _folds(data["ids"], train, int(section["residualization_folds"]))
    target = data["targets"]["next_delta"]
    current = data["targets"]["current_j"]
    rank = data["features"].shape[1]
    layerwise = data["layerwise"]
    blocks = {
        "combined_reference": data["full_features"],
        "recurrent": layerwise[:, :rank],
        "conv": layerwise[:, rank : 2 * rank],
        "kv": layerwise[:, 2 * rank :],
    }
    groups = [("validation", "pooled", validation), ("final_test", "pooled", final)]
    for split_name, indices in (("validation", validation), ("final_test", final)):
        groups.extend(
            (split_name, str(family), indices[data["families"][indices] == family])
            for family in sorted(set(data["families"]))
        )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for dimension in dimensions:
        compact = latent_bank[method][:, :dimension]
        candidate_train_x = np.concatenate((current[train], compact[train]), axis=1)
        candidate_eval_x = np.concatenate(
            (current[evaluation], compact[evaluation]), axis=1
        )
        candidate_prediction = _predict_split(
            candidate_train_x, candidate_eval_x, target, train, alpha
        )
        residuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        diagnostics: dict[str, Any] = {}
        source_predictions: dict[str, np.ndarray] = {}
        for name, block in blocks.items():
            residual_train, residual_eval, detail = cross_fitted_residuals(
                compact, block, train, evaluation, fold_rows, alpha
            )
            residuals[name] = (residual_train, residual_eval)
            diagnostics[name] = detail
            source_predictions[name] = _predict_split(
                np.concatenate((candidate_train_x, residual_train), axis=1),
                np.concatenate((candidate_eval_x, residual_eval), axis=1),
                target,
                train,
                alpha,
            )
        architecture_train = np.concatenate(
            [residuals[name][0] for name in ("recurrent", "conv", "kv")], axis=1
        )
        architecture_eval = np.concatenate(
            [residuals[name][1] for name in ("recurrent", "conv", "kv")], axis=1
        )
        architecture_prediction = _predict_split(
            np.concatenate((candidate_train_x, architecture_train), axis=1),
            np.concatenate((candidate_eval_x, architecture_eval), axis=1),
            target,
            train,
            alpha,
        )
        local_truth = target[evaluation]
        candidate_score = _cosine_rows(candidate_prediction, local_truth)
        gains = {
            name: _cosine_rows(prediction, local_truth) - candidate_score
            for name, prediction in source_predictions.items()
        }
        gains["architecture_joint"] = (
            _cosine_rows(architecture_prediction, local_truth) - candidate_score
        )
        gains["interaction"] = gains["architecture_joint"] - sum(
            gains[name] for name in ("recurrent", "conv", "kv")
        )
        dimension_summary: dict[str, Any] = {"residualization": diagnostics}
        for split_name, family, indices in groups:
            local = _local_indices(evaluation, indices)
            for source, values in gains.items():
                ci = _bootstrap_ci(values[local], seeds, resamples)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION_V10,
                        "protocol_version": PROTOCOL_V10,
                        "record_type": "cross_fitted_residual_localization",
                        "run_id": context.run_id,
                        "method": method,
                        "dimension": dimension,
                        "split": split_name,
                        "family": family,
                        "source": source,
                        "conditional_gain": ci["estimate"],
                        "lower": ci["lower"],
                        "upper": ci["upper"],
                        "metadata": json.dumps(ci, sort_keys=True),
                    }
                )
                if split_name == "final_test" and family == "pooled":
                    dimension_summary[source] = ci
        summary[str(dimension)] = dimension_summary
    return rows, summary


def _analysis(context: Any, freeze: dict[str, Any]) -> None:
    section = context.config["causal_sufficiency_v10"]
    data = _load_data(context.root)
    targets = data["targets"]
    train = np.flatnonzero(data["splits"] == "train")
    validation = np.flatnonzero(data["splits"] == "validation")
    final = np.flatnonzero(data["splits"] == "final_test")
    evaluation = np.concatenate((validation, final))
    latent_bank, latent_meta = build_ordered_latents(
        data["features"],
        train,
        targets,
        context.config["sufficiency_v9"]["semantic_objective_weights"],
    )
    if latent_meta["rank"] != int(freeze["effective_feature_rank"]):
        raise RuntimeError("v10 latent rank differs from frozen rank")
    alpha = float(section["ridge_alpha"])
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    epsilon = float(section["gap_denominator_epsilon"])
    common_targets = {
        "next_delta": targets["next_delta"],
        "next_absolute": targets["next_absolute"],
    }
    baseline = _predict(targets["current_j"], common_targets, train, evaluation, alpha)
    ceiling = _predict(
        np.concatenate((targets["current_j"], data["full_features"]), axis=1),
        common_targets,
        train,
        evaluation,
        alpha,
    )
    truth = {name: target[evaluation] for name, target in common_targets.items()}
    groups: list[tuple[str, str, np.ndarray]] = []
    for split_name, indices in (("validation", validation), ("final_test", final)):
        groups.append((split_name, "pooled", indices))
        groups.extend(
            (split_name, str(family), indices[data["families"][indices] == family])
            for family in sorted(set(data["families"]))
        )
    rows: list[dict[str, Any]] = []
    gate_index: dict[tuple[str, int, str], list[bool]] = {}
    progress = context.raw_dir / context.run_id / "audit_progress.json"
    jobs = [
        (str(method), int(dimension))
        for method in section["methods"]
        for dimension in section["dimensions"]
    ]
    for job_index, (method, dimension) in enumerate(jobs):
        compact = latent_bank[method][:, :dimension]
        candidate = _predict(
            np.concatenate((targets["current_j"], compact), axis=1),
            common_targets,
            train,
            evaluation,
            alpha,
        )
        for split_name, family, indices in groups:
            local = _local_indices(evaluation, indices)
            metrics = _metric_record(
                baseline=baseline,
                candidate=candidate,
                ceiling=ceiling,
                truth=truth,
                local=local,
                epsilon=epsilon,
                seeds=seeds,
                resamples=resamples,
            )
            gates = _gate(metrics, section)
            gate_index.setdefault((method, dimension, split_name), []).append(
                gates["observational_pass"]
            )
            conditional = metrics["conditional_residual_gain"]
            semantic_ci = metrics["semantic_conditional_residual_gain"]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION_V10,
                    "protocol_version": PROTOCOL_V10,
                    "record_type": "corrected_sub512_sufficiency",
                    "run_id": context.run_id,
                    "method": method,
                    "dimension": dimension,
                    "split": split_name,
                    "family": family,
                    "count": int(len(indices)),
                    "baseline_score": metrics["baseline_score"],
                    "candidate_score": metrics["candidate_score"],
                    "ceiling_score": metrics["ceiling_score"],
                    "candidate_baseline_delta": metrics["candidate_baseline_delta"],
                    "ceiling_baseline_delta": metrics["ceiling_baseline_delta"],
                    "predictive_gap_closed": metrics["gap_closed"],
                    "gap_identified": metrics["identified"],
                    "conditional_residual_gain": conditional["estimate"],
                    "conditional_lower": conditional["lower"],
                    "conditional_upper": conditional["upper"],
                    "semantic_baseline": metrics["semantic_baseline"],
                    "semantic_compact": metrics["semantic_compact"],
                    "semantic_full_ceiling": metrics["semantic_full_ceiling"],
                    "semantic_retention_relative_to_full": metrics[
                        "semantic_retention_relative_to_full"
                    ],
                    "semantic_residual_gain": semantic_ci["estimate"],
                    "semantic_residual_lower": semantic_ci["lower"],
                    "semantic_residual_upper": semantic_ci["upper"],
                    **gates,
                    "metadata": json.dumps(metrics, sort_keys=True),
                }
            )
        write_json_atomic(
            progress,
            {
                "status": "RUNNING",
                "completed": job_index + 1,
                "total": len(jobs),
                "last": f"{method}/{dimension}",
            },
        )
    primary = str(section["primary_method"])
    eligible = [
        int(dimension)
        for dimension in section["dimensions"]
        if all(gate_index.get((primary, int(dimension), "validation"), []))
    ]
    selection = section["candidate_selection"]
    candidates, selection_status = select_confirmatory_candidates(
        eligible,
        [int(value) for value in section["dimensions"]],
        int(selection["include_reference_dimension"]),
        [int(value) for value in selection["fallback_diagnostic_dimensions"]],
    )
    residual_rows, residual_summary = _residual_audit(
        context=context,
        data=data,
        latent_bank=latent_bank,
        train=train,
        evaluation=evaluation,
        validation=validation,
        final=final,
        section=section,
    )
    context.root.joinpath(SWEEP).parent.mkdir(parents=True, exist_ok=True)
    sweep_frame = pd.DataFrame(rows)
    sweep_frame.to_parquet(context.root / SWEEP, index=False, compression="zstd")
    pd.DataFrame(residual_rows).to_parquet(
        context.root / RESIDUAL, index=False, compression="zstd"
    )
    semantic_columns = [
        "schema_version",
        "protocol_version",
        "record_type",
        "run_id",
        "method",
        "dimension",
        "split",
        "family",
        "count",
        "semantic_baseline",
        "semantic_compact",
        "semantic_full_ceiling",
        "semantic_retention_relative_to_full",
        "semantic_residual_gain",
        "semantic_residual_lower",
        "semantic_residual_upper",
        "semantic_relative_pass",
    ]
    sweep_frame[semantic_columns].to_parquet(
        context.root / SEMANTIC, index=False, compression="zstd"
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "status": "COMPLETED_CORRECTED_OBSERVATIONAL_AUDIT",
        "effective_rank": latent_meta["rank"],
        "validation_all_family_eligible_dimensions": eligible,
        "smallest_observationally_sufficient_dimension": (
            min(eligible) if eligible else None
        ),
        "selected_confirmatory_candidates": candidates,
        "candidate_selection_status": selection_status,
        "selection_split": "validation",
        "final_test_not_used_for_candidate_selection": True,
        "records": str(SWEEP),
        "records_sha256": sha256_file(context.root / SWEEP),
    }
    write_json_atomic(context.root / SWEEP_SUMMARY, summary)
    write_json_atomic(
        context.root / RESIDUAL_SUMMARY,
        {
            "schema_version": SCHEMA_VERSION_V10,
            "protocol_version": PROTOCOL_V10,
            "run_id": context.run_id,
            "source_freeze_digest": freeze["freeze_digest"],
            "v9_comparator_audit": {
                "mathematically_consistent_with_corrected_joint": False,
                "problem": "v9 appended raw block scores to compact coordinates without train-only residualization",
                "correction": "five-fold cross-fitted block residualization on train and full-train residualization on held-out splits",
                "reference_space_note": "architecture_joint uses 3x599 block-specific scores and is richer than the 599D combined ceiling",
            },
            "dimensions": residual_summary,
            "records": str(RESIDUAL),
            "records_sha256": sha256_file(context.root / RESIDUAL),
        },
    )
    write_json_atomic(
        context.root / SEMANTIC_SUMMARY,
        {
            "schema_version": SCHEMA_VERSION_V10,
            "protocol_version": PROTOCOL_V10,
            "run_id": context.run_id,
            "source_freeze_digest": freeze["freeze_digest"],
            "absolute_quality_definition": "top-10 agreement with held-out full next-J target",
            "relative_sufficiency_definition": "compact/full semantic agreement plus full-minus-compact residual CI",
            "records": str(SEMANTIC),
            "records_sha256": sha256_file(context.root / SEMANTIC),
        },
    )
    write_json_atomic(
        progress, {"status": "COMPLETED", "completed": len(jobs), "total": len(jobs)}
    )
    context.finish("COMPLETED_V10_AUDIT", summary=summary)


def main() -> None:
    parser = standard_parser(
        "corrected observational and residual audit v10",
        "configs/causal_sufficiency_v10.yaml",
    )
    parser.add_argument(
        "--stage", required=True, choices=("freeze", "audit", "freeze-candidates")
    )
    args = parser.parse_args()
    context = initialize_context("sufficiency-v10", args)
    try:
        if args.stage == "freeze":
            context.finish("COMPLETED_V10_FREEZE", freeze=build_freeze(context.root, context.config))
            return
        freeze = verify_freeze(context.root, context.config)
        if args.stage == "freeze-candidates":
            context.finish(
                "COMPLETED_V10_CANDIDATE_FREEZE",
                candidate_freeze=build_candidate_freeze(context.root, context.config),
            )
            return
        if args.dry_run:
            context.finish("DRY_RUN", source_freeze_digest=freeze["freeze_digest"])
            return
        _analysis(context, freeze)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
