"""Coordinate-stable amendment to the protocol-v10 residual audit."""

from __future__ import annotations

import json
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
    build_ordered_latents,
)
from jclosure.experiments.sufficiency_v10 import (
    RESIDUAL,
    RESIDUAL_SUMMARY,
    SWEEP,
    _folds,
    _local_indices,
    _predict_split,
    cross_fitted_residuals,
)
from jclosure.protocol_v10 import PROTOCOL_V10, SCHEMA_VERSION_V10
from jclosure.protocol_v10_residual_amendment import (
    ARCHIVED_RESIDUAL_JSON,
    ARCHIVED_RESIDUAL_PARQUET,
    PROTOCOL_V10_RESIDUAL_AMENDMENT,
    build_amendment_freeze,
    verify_amendment_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic


def _oof_prediction(
    x: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    folds: list[np.ndarray],
    alpha: float,
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(train.tolist())}
    output = np.zeros((len(train), target.shape[1]), dtype=np.float32)
    for held in folds:
        held_set = set(held.tolist())
        fit = np.asarray([value for value in train if value not in held_set])
        prediction = _apply_ridge_weights(
            _ridge_weights(x[fit], x[held], alpha), target[fit]
        )
        output[[lookup[value] for value in held]] = prediction
    return output


def frozen_base_residual_prediction(
    residual_train_x: np.ndarray,
    residual_eval_x: np.ndarray,
    target_residual_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Predict only the incremental correction; never refit the frozen base."""
    weights = _ridge_weights(residual_train_x, residual_eval_x, alpha)
    prediction = _apply_ridge_weights(weights, target_residual_train)
    return prediction - target_residual_train.mean(axis=0, keepdims=True)


def _audit(context: Any, amendment: dict[str, Any]) -> None:
    section = context.config["causal_sufficiency_v10"]
    data = _load_data(context.root)
    train = np.flatnonzero(data["splits"] == "train")
    validation = np.flatnonzero(data["splits"] == "validation")
    final = np.flatnonzero(data["splits"] == "final_test")
    evaluation = np.concatenate((validation, final))
    latent_bank, _ = build_ordered_latents(
        data["features"],
        train,
        data["targets"],
        context.config["sufficiency_v9"]["semantic_objective_weights"],
    )
    method = str(section["primary_method"])
    dimensions = [int(value) for value in section["residual_audit_dimensions"]]
    alpha = float(section["ridge_alpha"])
    seeds = [int(value) for value in section["confirmation_seeds"]]
    resamples = int(section["bootstrap_resamples"])
    folds = _folds(data["ids"], train, int(section["residualization_folds"]))
    target = data["targets"]["next_delta"]
    current = data["targets"]["current_j"]
    rank = data["features"].shape[1]
    layerwise = data["layerwise"]
    blocks = {
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
    sweep = pd.read_parquet(context.root / SWEEP)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    maximum_reference_difference = 0.0
    for dimension in dimensions:
        compact = latent_bank[method][:, :dimension]
        candidate_x = np.concatenate((current, compact), axis=1)
        candidate_prediction = _predict_split(
            candidate_x[train], candidate_x[evaluation], target, train, alpha
        )
        candidate_oof = _oof_prediction(candidate_x, target, train, folds, alpha)
        target_residual_train = target[train] - candidate_oof
        full_x = np.concatenate((current, data["full_features"]), axis=1)
        full_prediction = _predict_split(
            full_x[train], full_x[evaluation], target, train, alpha
        )
        local_truth = target[evaluation]
        candidate_score = _cosine_rows(candidate_prediction, local_truth)
        gains: dict[str, np.ndarray] = {
            "combined_reference": _cosine_rows(full_prediction, local_truth)
            - candidate_score
        }
        diagnostics: dict[str, Any] = {}
        residuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, block in blocks.items():
            residual_train, residual_eval, detail = cross_fitted_residuals(
                compact, block, train, evaluation, folds, alpha
            )
            residuals[name] = (residual_train, residual_eval)
            diagnostics[name] = detail
            correction = frozen_base_residual_prediction(
                residual_train, residual_eval, target_residual_train, alpha
            )
            gains[name] = (
                _cosine_rows(candidate_prediction + correction, local_truth)
                - candidate_score
            )
        architecture_train = np.concatenate(
            [residuals[name][0] for name in ("recurrent", "conv", "kv")], axis=1
        )
        architecture_eval = np.concatenate(
            [residuals[name][1] for name in ("recurrent", "conv", "kv")], axis=1
        )
        architecture_correction = frozen_base_residual_prediction(
            architecture_train, architecture_eval, target_residual_train, alpha
        )
        gains["architecture_joint"] = (
            _cosine_rows(candidate_prediction + architecture_correction, local_truth)
            - candidate_score
        )
        gains["interaction"] = gains["architecture_joint"] - sum(
            gains[name] for name in ("recurrent", "conv", "kv")
        )
        dimension_summary: dict[str, Any] = {
            "residualization": diagnostics,
            "base_prediction_frozen": True,
        }
        for split_name, family, indices in groups:
            local = _local_indices(evaluation, indices)
            for source, values in gains.items():
                ci = _bootstrap_ci(values[local], seeds, resamples)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION_V10,
                        "protocol_version": PROTOCOL_V10_RESIDUAL_AMENDMENT,
                        "record_type": "frozen_base_cross_fitted_residual_localization",
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
                if source == "combined_reference":
                    expected = sweep[
                        (sweep["method"] == method)
                        & (sweep["dimension"] == dimension)
                        & (sweep["split"] == split_name)
                        & (sweep["family"] == family)
                    ]
                    if len(expected) != 1:
                        raise RuntimeError("missing unique unified-comparator row")
                    difference = abs(
                        ci["estimate"]
                        - float(expected.iloc[0]["conditional_residual_gain"])
                    )
                    maximum_reference_difference = max(
                        maximum_reference_difference, difference
                    )
                if split_name == "final_test" and family == "pooled":
                    dimension_summary[source] = ci
        summary[str(dimension)] = dimension_summary
    if maximum_reference_difference > 1e-7:
        raise RuntimeError(
            "combined_reference does not reproduce unified comparator: "
            f"{maximum_reference_difference}"
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(context.root / RESIDUAL, index=False, compression="zstd")
    payload = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": PROTOCOL_V10,
        "methodological_amendment": PROTOCOL_V10_RESIDUAL_AMENDMENT,
        "run_id": context.run_id,
        "source_amendment_freeze_digest": amendment["freeze_digest"],
        "supersedes": {
            "json": str(ARCHIVED_RESIDUAL_JSON),
            "parquet": str(ARCHIVED_RESIDUAL_PARQUET),
            "reason": amendment["reason"],
        },
        "estimand": {
            "combined_reference": (
                "exact unified full-information prediction minus the frozen compact "
                "prediction; no coordinate-dependent refit"
            ),
            "architecture_sources": (
                "five-fold OOF B-E[B|C] residuals predict OOF target residuals and "
                "are added to a frozen compact prediction"
            ),
            "base_prediction_frozen": True,
            "heldout_residualization_uses_train_only": True,
        },
        "unified_comparator_max_abs_difference": maximum_reference_difference,
        "dimensions": summary,
        "records": str(RESIDUAL),
        "records_sha256": sha256_file(context.root / RESIDUAL),
    }
    write_json_atomic(context.root / RESIDUAL_SUMMARY, payload)
    context.finish("COMPLETED_V10_RESIDUAL_AMENDMENT", summary=payload)


def main() -> None:
    parser = standard_parser(
        "coordinate-stable protocol-v10 residual amendment",
        "configs/causal_sufficiency_v10.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "audit"))
    args = parser.parse_args()
    context = initialize_context("residual-amendment-v10", args)
    try:
        if args.stage == "freeze":
            context.finish(
                "COMPLETED_V10_RESIDUAL_AMENDMENT_FREEZE",
                freeze=build_amendment_freeze(context.root, context.config),
            )
            return
        amendment = verify_amendment_freeze(context.root, context.config)
        if args.dry_run:
            context.finish("DRY_RUN", amendment_freeze_digest=amendment["freeze_digest"])
            return
        _audit(context, amendment)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
