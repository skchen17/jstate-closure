"""Causal tangent, variance alignment, trajectory, and v12 adjudication."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.protocol_v12 import PROTOCOL_V12, SCHEMA_VERSION_V12, verify_base_freeze
from jclosure.provenance import sha256_file, write_json_atomic

from .causal_v12 import (
    CONFIRM_RECORDS,
    CONFIRM_SUMMARY,
    MEASUREMENT_SUMMARY,
    ORACLE_STAGE2_RECORDS,
    REPLACEMENT_SUMMARY,
    SCALING_CAUSAL_RECORDS,
)
from .jvp_v12 import JVP_DIRECTION_SUMMARY, JVP_SPECTRA, JVP_SUMMARY
from .prepare_v12 import SCALING_RECORDS

TANGENT_RECORDS = Path("results/v12/processed/causal_tangent_geometry_v12.parquet")
TANGENT_SUMMARY = Path("results/v12/processed/causal_tangent_geometry_v12.json")
VARIANCE_RECORDS = Path("results/v12/processed/variance_vs_causal_v12.parquet")
VARIANCE_SUMMARY = Path("results/v12/processed/variance_vs_causal_v12.json")
TRAJECTORY_RECORDS = Path(
    "results/v12/processed/causal_trajectory_dynamics_v12.parquet"
)
TRAJECTORY_SUMMARY = Path("results/v12/processed/causal_trajectory_dynamics_v12.json")
SCALING_SUMMARY = Path("results/v12/processed/data_rank_scaling_analysis_v12.json")
ADJUDICATION = Path("results/v12/processed/adjudication_v12.json")


def _angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(left @ right.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1, 1)))


def _tangent(context: Any) -> dict[str, Any]:
    frame = pd.read_parquet(context.root / JVP_SPECTRA)
    loaded = {}
    for index, row in frame.iterrows():
        loaded[index] = np.load(context.root / row["matrix_path"])[
            "right_singular_vectors"
        ]
    rows = []
    for left_index, right_index in itertools.combinations(frame.index.tolist(), 2):
        left_row, right_row = frame.loc[left_index], frame.loc[right_index]
        if left_row["base_trial_id"] == right_row["base_trial_id"]:
            relation = "same_prompt_successive_position"
        elif left_row["family"] == right_row["family"]:
            relation = "within_family_different_prompt"
        else:
            relation = "across_family"
        token_distance = abs(
            int(left_row["token_position"]) - int(right_row["token_position"])
        )
        for rank in (8, 16, 32):
            angle = _angles(loaded[left_index][:rank], loaded[right_index][:rank])
            rows.append(
                {
                    "left_base_trial_id": left_row["base_trial_id"],
                    "right_base_trial_id": right_row["base_trial_id"],
                    "relation": relation,
                    "token_distance": token_distance,
                    "rank": rank,
                    "mean_principal_angle_degrees": float(angle.mean()),
                    "maximum_principal_angle_degrees": float(angle.max()),
                    "topk_overlap": float(
                        np.sum(np.cos(np.radians(angle)) ** 2) / rank
                    ),
                    "grassmann_distance": float(
                        np.linalg.norm(np.sin(np.radians(angle)))
                    ),
                }
            )
    output = pd.DataFrame(rows)
    path = context.root / TANGENT_RECORDS
    output.to_parquet(path, index=False, compression="zstd")
    grouped = output.groupby(["relation", "rank"]).mean(numeric_only=True).reset_index()
    rank16 = grouped[grouped["rank"] == 16]
    local = json.loads((context.root / JVP_SUMMARY).read_text())
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "relation_rank_means": grouped.to_dict(orient="records"),
        "large_tangent_rotation": bool(
            float(rank16["mean_principal_angle_degrees"].mean()) >= 30.0
        ),
        "local_rank_probe_limited": bool(float(local["median_rank_95"]) >= 60),
        "records": str(TANGENT_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / TANGENT_SUMMARY, summary)
    return summary


def _variance(context: Any) -> dict[str, Any]:
    frame = pd.read_parquet(context.root / JVP_SPECTRA)
    direction = json.loads((context.root / JVP_DIRECTION_SUMMARY).read_text())
    variances = (
        np.asarray(direction["component_singular_values"], dtype=np.float64) ** 2
    )
    channel_energy = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in direction["channel_score_energy"].items()
    }
    rows = []
    for _, record in frame.iterrows():
        payload = np.load(context.root / record["matrix_path"])
        matrix = payload["matrix"].astype(np.float64)
        vh = payload["right_singular_vectors"].astype(np.float64)
        column_energy = np.sum(matrix**2, axis=0)
        for rank in (8, 16, 32):
            causal = vh[:rank]
            overlap = float(np.sum(causal[:, :rank] ** 2) / rank)
            angles = _angles(causal, np.eye(len(variances), dtype=np.float64)[:rank])
            causal_variance = float(np.trace(causal @ np.diag(variances) @ causal.T))
            leading_variance = float(variances[:rank].sum())
            causal_sensitivity_pca = float(
                column_energy[:rank].sum() / max(column_energy.sum(), 1e-20)
            )
            sensitivity_per_variance = column_energy / np.maximum(variances, 1e-20)
            tail = np.arange(len(variances)) >= len(variances) // 2
            high_causal = sensitivity_per_variance >= np.quantile(
                sensitivity_per_variance, 0.75
            )
            channel_contributions = {}
            causal_weights = np.sum(causal**2, axis=0)
            for name, values in channel_energy.items():
                channel_contributions[name] = float(
                    np.sum(causal_weights * values)
                    / max(np.sum(causal_weights * sum(channel_energy.values())), 1e-20)
                )
            rows.append(
                {
                    "base_trial_id": record["base_trial_id"],
                    "family": record["family"],
                    "token_position": int(record["token_position"]),
                    "rank": rank,
                    "pca_causal_subspace_overlap": overlap,
                    "mean_principal_angle_degrees": float(angles.mean()),
                    "variance_in_causal_subspace_vs_leading_pca": causal_variance
                    / max(leading_variance, 1e-20),
                    "causal_sensitivity_explained_by_leading_pca": causal_sensitivity_pca,
                    "low_variance_high_causal_direction_count": int(
                        np.sum(tail & high_causal)
                    ),
                    "recurrent_contribution": channel_contributions["recurrent"],
                    "conv_contribution": channel_contributions["conv"],
                    "kv_contribution": channel_contributions["kv"],
                }
            )
    output = pd.DataFrame(rows)
    path = context.root / VARIANCE_RECORDS
    output.to_parquet(path, index=False, compression="zstd")
    rank16 = output[output["rank"] == 16]
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "rank16_means": rank16.mean(numeric_only=True).to_dict(),
        "low_variance_high_causal_directions_present": bool(
            rank16["low_variance_high_causal_direction_count"].mean() >= 1
        ),
        "records": str(VARIANCE_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / VARIANCE_SUMMARY, summary)
    return summary


def _trajectory(context: Any) -> dict[str, Any]:
    source = context.root / CONFIRM_RECORDS
    split = "confirmatory"
    if not source.is_file():
        source = context.root / ORACLE_STAGE2_RECORDS
        split = "development"
    frame = pd.read_parquet(source)
    metrics = (
        "trajectory_error_norm",
        "trajectory_angle_degrees",
        "semantic_trajectory_cosine",
        "error_parallel_to_teacher",
        "error_orthogonal_to_teacher",
        "direction_cosine",
        "magnitude_ratio",
        "semantic_delta_agreement",
    )
    grouped = (
        frame.groupby(["method", "method_class", "horizon"])[list(metrics)]
        .mean()
        .reset_index()
    )
    path = context.root / TRAJECTORY_RECORDS
    grouped.to_parquet(path, index=False, compression="zstd")
    change = grouped.groupby("horizon").mean(numeric_only=True)
    norm_monotone = bool(
        np.all(np.diff(change["trajectory_error_norm"].to_numpy()) >= 0)
    )
    rotation_change = float(
        change["trajectory_angle_degrees"].iloc[-1]
        - change["trajectory_angle_degrees"].iloc[0]
    )
    semantic_change = float(
        change["semantic_trajectory_cosine"].iloc[-1]
        - change["semantic_trajectory_cosine"].iloc[0]
    )
    dominant = (
        "norm_amplification"
        if norm_monotone and rotation_change < 5
        else "direction_rotation"
    )
    if semantic_change < -0.15:
        dominant = (
            "semantic_divergence"
            if rotation_change < 10
            else "direction_rotation_and_semantic_divergence"
        )
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "source_split": split,
        "finite_horizon_ratios_are_jacobian_eigenvalues": False,
        "error_norm_monotone": norm_monotone,
        "angle_change_degrees": rotation_change,
        "semantic_cosine_change": semantic_change,
        "dominant_failure_mode": dominant,
        "records": str(TRAJECTORY_RECORDS),
        "records_sha256": sha256_file(path),
    }
    write_json_atomic(context.root / TRAJECTORY_SUMMARY, summary)
    return summary


def _scaling(context: Any) -> dict[str, Any]:
    offline = pd.read_parquet(context.root / SCALING_RECORDS)
    causal = pd.read_parquet(context.root / SCALING_CAUSAL_RECORDS)
    grouped = (
        causal.groupby(["method_class", "train_size", "dimension", "horizon"])
        .mean(numeric_only=True)
        .reset_index()
    )
    common = grouped[(grouped["dimension"] == 128) & (grouped["horizon"] == 1)]
    slopes = {}
    for method, values in common.groupby("method_class"):
        values = values.sort_values("train_size")
        if len(values) >= 2:
            slopes[str(method)] = float(
                np.polyfit(values["train_size"], values["direction_cosine"], 1)[0] * 450
            )
    identified_512_sizes = sorted(
        offline[(offline["dimension"] == 512) & (offline["status"] == "IDENTIFIED")][
            "train_size"
        ]
        .unique()
        .astype(int)
        .tolist()
    )
    summary = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "requested_train_sizes": [150, 300, 450, 600, 1000, 2000, 4000],
        "available_train_sizes": sorted(
            offline[offline["status"] == "IDENTIFIED"]["train_size"]
            .unique()
            .astype(int)
            .tolist()
        ),
        "unavailable_train_sizes": [1000, 2000, 4000],
        "unavailable_status": "NOT_IDENTIFIED_BANK_SIZE",
        "identified_512_train_sizes": identified_512_sizes,
        "fixed_512_saturation_identified": len(identified_512_sizes) >= 3,
        "direction_improvement_n150_to_n600_at_d128": slopes,
        "more_data_required": True,
        "reason": "only N=600 has enough centered empirical rank for 512D, so fixed-512 saturation cannot be estimated",
        "offline_records_sha256": sha256_file(context.root / SCALING_RECORDS),
        "causal_records_sha256": sha256_file(context.root / SCALING_CAUSAL_RECORDS),
    }
    write_json_atomic(context.root / SCALING_SUMMARY, summary)
    return summary


def _adjudicate(context: Any, base: dict[str, Any]) -> dict[str, Any]:
    measurement = json.loads((context.root / MEASUREMENT_SUMMARY).read_text())
    scaling = json.loads((context.root / SCALING_SUMMARY).read_text())
    jacobian = json.loads((context.root / JVP_SUMMARY).read_text())
    tangent = json.loads((context.root / TANGENT_SUMMARY).read_text())
    variance = json.loads((context.root / VARIANCE_SUMMARY).read_text())
    replacement = json.loads((context.root / REPLACEMENT_SUMMARY).read_text())
    confirm = json.loads((context.root / CONFIRM_SUMMARY).read_text())
    any_authorized = bool(confirm.get("authorized_methods"))
    full_replacement = bool(
        replacement.get("strict_full_state_replacement_pass", False)
    )
    if any_authorized and full_replacement:
        outcome = "V12-E"
        conclusion = "CANDIDATE_CAUSAL_SUFFICIENT_STATE"
    elif scaling["more_data_required"]:
        outcome = "V12-A"
        conclusion = "MORE_DATA_REQUIRED"
    elif any_authorized and tangent["large_tangent_rotation"]:
        outcome = "V12-C"
        conclusion = "LOCALLY_LOW_DIMENSIONAL_CAUSAL_MANIFOLD_SUPPORTED"
    elif any_authorized:
        outcome = "V12-B"
        conclusion = "GLOBAL_LOW_DIMENSIONAL_CAUSAL_STATE_SUPPORTED"
    else:
        outcome = "V12-D"
        conclusion = "WRITABLE_CAUSAL_STATE_HIGH_DIMENSIONAL_WITHIN_TESTED_PROTOCOL"
    payload = {
        "schema_version": SCHEMA_VERSION_V12,
        "protocol_version": PROTOCOL_V12,
        "source_freeze_digest": base["freeze_digest"],
        "decision_tree_outcome": outcome,
        "formal_conclusion": conclusion,
        "semantic_metric_instability_material": measurement[
            "semantic_metric_instability_material"
        ],
        "fixed_512_saturation_identified": scaling["fixed_512_saturation_identified"],
        "local_causal_rank_scope": jacobian["operator_scope"],
        "median_restricted_rank_90": jacobian["median_rank_90"],
        "median_restricted_rank_95": jacobian["median_rank_95"],
        "median_restricted_rank_99": jacobian["median_rank_99"],
        "tangent_rotation_large": tangent["large_tangent_rotation"],
        "low_variance_high_causal_directions_present": variance[
            "low_variance_high_causal_directions_present"
        ],
        "causal_authorized_methods": confirm.get("authorized_methods", []),
        "strict_full_state_replacement_pass": full_replacement,
        "smallest_causally_validated_dimension": min(
            [
                int(value.split("_d")[-1])
                for value in confirm.get("authorized_methods", [])
                if "_d" in value
            ],
            default=None,
        ),
        "hypothesis": "H3" if outcome == "V12-E" else "H2",
        "h3_upgrade_authorized": outcome == "V12-E",
        "autonomous_controller_authorized": outcome == "V12-E",
        "learned_state_dependent_decoder_authorized": outcome in {"V12-B", "V12-C"},
        "scientific_scope": "Qwen3.5 model and frozen v12 empirical operator/splits only",
    }
    write_json_atomic(context.root / ADJUDICATION, payload)
    return payload


def main() -> None:
    parser = standard_parser(
        "analyze causal geometry v12", "configs/causal_geometry_v12.yaml"
    )
    args = parser.parse_args()
    context = initialize_context("analyze-v12", args)
    try:
        base = verify_base_freeze(context.root, context.config)
        result: dict[str, Any] = {
            "tangent": _tangent(context),
            "variance": _variance(context),
            "trajectory": _trajectory(context),
            "scaling": _scaling(context),
        }
        result["adjudication"] = _adjudicate(context, base)
        context.finish("COMPLETED_V12_ANALYSIS", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
