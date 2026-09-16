"""Post-hoc frozen analyses and report dispatch for protocol v11."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.sufficiency_v9 import (
    _apply_ridge_weights,
    _cosine_rows,
    _ridge_weights,
)
from jclosure.protocol_v11 import (
    CONFIRM_FREEZE_PATH,
    PROTOCOL_V11,
    SCHEMA_VERSION_V11,
    verify_base_freeze,
    verify_derived_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.reporting_v11 import write_reports
from jclosure.state_models_v11 import CHANNELS

from .causal_v11 import (
    CHANNEL_RECORDS,
    CHANNEL_SUMMARY,
    CONFIRM_RECORDS,
    CONFIRM_SUMMARY,
    FACTOR_STAGE1_RECORDS,
    FACTOR_STAGE1_SUMMARY,
    FACTOR_STAGE2_RECORDS,
    ORACLE_RECORDS,
    ORACLE_SUMMARY,
)
from .prepare_v11 import build_bank

MANIFOLD_RECORDS = Path("results/v11/processed/causal_state_manifold_v11.parquet")
MANIFOLD_SUMMARY = Path("results/v11/processed/causal_state_manifold_v11.json")
COMPATIBILITY_RECORDS = Path("results/v11/processed/channel_compatibility_v11.parquet")
AMPLIFICATION_RECORDS = Path(
    "results/v11/processed/causal_error_amplification_v11.parquet"
)
AMPLIFICATION_SUMMARY = Path(
    "results/v11/processed/causal_error_amplification_v11.json"
)
CEILING_RECORDS = Path("results/v11/processed/architecture_ceiling_v11.parquet")
CEILING_SUMMARY = Path("results/v11/processed/architecture_ceiling_v11.json")
ADJUDICATION = Path("results/v11/processed/adjudication_v11.json")


def _reference_geometry(train_values: np.ndarray) -> dict[str, np.ndarray]:
    mean = train_values.mean(axis=0, keepdims=True)
    scale = train_values.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    fit = (train_values - mean) / scale
    _, singular, vh = np.linalg.svd(fit, full_matrices=False)
    return {
        "mean": mean,
        "scale": scale,
        "fit": fit,
        "singular": singular,
        "vh": vh,
    }


def _distances(
    query: np.ndarray,
    geometry: dict[str, np.ndarray],
    *,
    neighbors: int,
    local_rank: int,
    global_rank: int,
) -> dict[str, Any]:
    fit = geometry["fit"]
    held = (query[None] - geometry["mean"]) / geometry["scale"]
    distances = np.linalg.norm(fit - held, axis=1) / np.sqrt(fit.shape[1])
    nearest = np.argsort(distances)[:neighbors]
    local = fit[nearest]
    local_mean = local.mean(axis=0, keepdims=True)
    _, _, local_vh = np.linalg.svd(local - local_mean, full_matrices=False)
    use_rank = min(local_rank, len(local_vh))
    local_centered = held - local_mean
    local_projection = (local_centered @ local_vh[:use_rank].T) @ local_vh[:use_rank]
    local_error = float(
        np.linalg.norm(local_centered - local_projection)
        / max(float(np.linalg.norm(local_centered)), 1e-20)
    )
    singular = geometry["singular"]
    vh = geometry["vh"]
    rank = min(global_rank, len(vh))
    centered = held - fit.mean(axis=0, keepdims=True)
    projected = (centered @ vh[:rank].T) @ vh[:rank]
    outside = float(
        np.sum((centered - projected) ** 2) / max(float(np.sum(centered**2)), 1e-20)
    )
    score = centered @ vh.T
    variance = (singular**2) / max(len(fit) - 1, 1)
    mahalanobis = float(np.sqrt(np.sum(score**2 / np.maximum(variance[None], 1e-6))))
    return {
        "nearest_neighbor_index": int(nearest[0]),
        "nearest_neighbor_distance": float(distances[nearest[0]]),
        "mean_k_neighbor_distance": float(distances[nearest].mean()),
        "mahalanobis_distance": mahalanobis,
        "local_pca_reconstruction_error": local_error,
        "outside_global_subspace_energy_fraction": outside,
    }


def _cycle_error(query: np.ndarray, model: Any) -> float:
    standardized = (query[None] - model.mean) / model.scale
    scores = standardized @ model.vh.T
    reconstructed = (scores @ model.vh) * model.scale + model.mean
    return float(
        np.linalg.norm(reconstructed - query[None])
        / max(float(np.linalg.norm(query)), 1e-20)
    )


def _manifold(
    context: Any, base: dict[str, Any], confirm: dict[str, Any]
) -> dict[str, Any]:
    bank, data, train = build_bank(context)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = [str(value) for value in base["confirmatory"]["base_trial_ids"]]
    selected = np.asarray([lookup[value] for value in ids], dtype=int)
    method_values: dict[str, dict[str, np.ndarray]] = {}
    for spec in confirm["method_specs"]:
        inputs, _, _ = bank.decode_inputs(spec)
        method_values[str(spec["method"])] = {
            name: inputs[name][1] for name in CHANNELS
        }
    section = context.config["causal_geometry_v11"]["manifold"]
    geometries = {
        name: _reference_geometry(bank.blocks[name][train]) for name in CHANNELS
    }
    records: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    state_names = ["clean_zero", "teacher_raw", *method_values]
    for base_id, source_index in zip(ids, selected, strict=True):
        for state_name in state_names:
            nearest: dict[str, int] = {}
            for channel in CHANNELS:
                if state_name == "clean_zero":
                    query = np.zeros(bank.rank, dtype=np.float32)
                elif state_name == "teacher_raw":
                    query = bank.blocks[channel][source_index]
                else:
                    query = method_values[state_name][channel][source_index]
                values = _distances(
                    query,
                    geometries[channel],
                    neighbors=int(section["nearest_neighbors"]),
                    local_rank=int(section["local_pca_rank"]),
                    global_rank=int(section["global_subspace_rank"]),
                )
                nearest[channel] = values["nearest_neighbor_index"]
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION_V11,
                        "protocol_version": PROTOCOL_V11,
                        "record_type": "natural_state_manifold",
                        "base_trial_id": base_id,
                        "state": state_name,
                        "channel": channel,
                        "cycle_relative_error": _cycle_error(
                            query, bank.models[channel]
                        ),
                        **values,
                    }
                )
            pair_agreement = np.mean(
                [
                    nearest["recurrent"] == nearest["conv"],
                    nearest["recurrent"] == nearest["kv"],
                    nearest["conv"] == nearest["kv"],
                ]
            )
            compatibility.append(
                {
                    "base_trial_id": base_id,
                    "state": state_name,
                    "all_channels_same_neighbor": float(
                        len(set(nearest.values())) == 1
                    ),
                    "pairwise_neighbor_agreement": float(pair_agreement),
                    **{f"{name}_neighbor": value for name, value in nearest.items()},
                }
            )
    frame = pd.DataFrame(records)
    compatibility_frame = pd.DataFrame(compatibility)
    frame.to_parquet(context.root / MANIFOLD_RECORDS, index=False, compression="zstd")
    compatibility_frame.to_parquet(
        context.root / COMPATIBILITY_RECORDS, index=False, compression="zstd"
    )
    metrics = [
        "nearest_neighbor_distance",
        "mean_k_neighbor_distance",
        "mahalanobis_distance",
        "local_pca_reconstruction_error",
        "outside_global_subspace_energy_fraction",
        "cycle_relative_error",
    ]
    aggregates: dict[str, Any] = {}
    for state, values in frame.groupby("state", sort=True):
        aggregates[str(state)] = {}
        for channel, current in values.groupby("channel", sort=True):
            aggregates[str(state)][str(channel)] = {
                metric: float(current[metric].mean()) for metric in metrics
            }
        current_compatibility = compatibility_frame[
            compatibility_frame["state"] == state
        ]
        aggregates[str(state)]["joint_compatibility"] = {
            "all_channels_same_neighbor": float(
                current_compatibility["all_channels_same_neighbor"].mean()
            ),
            "pairwise_neighbor_agreement": float(
                current_compatibility["pairwise_neighbor_agreement"].mean()
            ),
        }
    summary = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "source_freeze_digest": confirm["freeze_digest"],
        "split_role": "independent_confirmatory",
        "natural_reference": "training teacher raw intervention deltas only",
        "aggregates": aggregates,
        "records": str(MANIFOLD_RECORDS),
        "records_sha256": sha256_file(context.root / MANIFOLD_RECORDS),
        "compatibility_records": str(COMPATIBILITY_RECORDS),
        "compatibility_records_sha256": sha256_file(
            context.root / COMPATIBILITY_RECORDS
        ),
    }
    write_json_atomic(context.root / MANIFOLD_SUMMARY, summary)
    return summary


def _amplification(context: Any, confirm: dict[str, Any]) -> dict[str, Any]:
    sources = [
        ("channel", CHANNEL_RECORDS, "condition"),
        ("oracle", ORACLE_RECORDS, "method"),
        ("factorized_stage1", FACTOR_STAGE1_RECORDS, "method"),
        ("factorized_stage2", FACTOR_STAGE2_RECORDS, "method"),
        ("confirmatory", CONFIRM_RECORDS, "method"),
    ]
    records: list[dict[str, Any]] = []
    for source, path, label_column in sources:
        full_path = context.root / path
        if not full_path.is_file() or full_path.stat().st_size == 0:
            continue
        frame = pd.read_parquet(full_path)
        if frame.empty:
            continue
        teacher = frame["teacher_j_effect_norm"].astype(float).to_numpy()
        decoded = frame["decoded_j_effect_norm"].astype(float).to_numpy()
        cosine = frame["direction_cosine"].astype(float).to_numpy()
        frame = frame.assign(
            error_norm=np.sqrt(
                np.maximum(teacher**2 + decoded**2 - 2 * teacher * decoded * cosine, 0)
            ),
            direction_error=1.0 - cosine,
            semantic_error=1.0 - frame["semantic_delta_agreement"].astype(float),
            analysis_source=source,
            analysis_label=frame[label_column].astype(str),
        )
        for (label, base_id), values in frame.groupby(
            ["analysis_label", "base_trial_id"], sort=True
        ):
            values = values.sort_values("horizon")
            previous = None
            for _, row in values.iterrows():
                current = {
                    "schema_version": SCHEMA_VERSION_V11,
                    "protocol_version": PROTOCOL_V11,
                    "record_type": "causal_error_amplification",
                    "analysis_source": source,
                    "analysis_label": str(label),
                    "base_trial_id": str(base_id),
                    "family": str(row["family"]),
                    "horizon": int(row["horizon"]),
                    "error_norm": float(row["error_norm"]),
                    "direction_error": float(row["direction_error"]),
                    "semantic_error": float(row["semantic_error"]),
                    "amplification_from_horizon": None,
                    "error_amplification_ratio": None,
                }
                if previous is not None:
                    current["amplification_from_horizon"] = int(previous["horizon"])
                    current["error_amplification_ratio"] = float(
                        row["error_norm"] / max(float(previous["error_norm"]), 1e-20)
                    )
                records.append(current)
                previous = row
    frame = pd.DataFrame(records)
    frame.to_parquet(
        context.root / AMPLIFICATION_RECORDS, index=False, compression="zstd"
    )
    aggregates: dict[str, Any] = {}
    for (source, label, horizon), values in frame.groupby(
        ["analysis_source", "analysis_label", "horizon"], sort=True
    ):
        key = f"{source}:{label}"
        aggregates.setdefault(key, {})[str(int(horizon))] = {
            "mean_error_norm": float(values["error_norm"].mean()),
            "mean_direction_error": float(values["direction_error"].mean()),
            "mean_semantic_error": float(values["semantic_error"].mean()),
            "mean_error_amplification_ratio": (
                float(values["error_amplification_ratio"].dropna().mean())
                if values["error_amplification_ratio"].notna().any()
                else None
            ),
            "count": int(len(values)),
        }
    summary = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "source_freeze_digest": confirm["freeze_digest"],
        "error_definition": (
            "sqrt(||teacher||^2+||decoded||^2-2||teacher||||decoded||cosine)"
        ),
        "jacobian_status": (
            "NOT_ESTIMATED; finite teacher-forced horizon ratios are the frozen "
            "local dynamical amplification proxy"
        ),
        "aggregates": aggregates,
        "records": str(AMPLIFICATION_RECORDS),
        "records_sha256": sha256_file(context.root / AMPLIFICATION_RECORDS),
    }
    write_json_atomic(context.root / AMPLIFICATION_SUMMARY, summary)
    return summary


def _prediction_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if target.shape[1] == 1:
        left = predicted[:, 0]
        right = target[:, 0]
        correlation = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else 0.0
        return {
            "correlation": correlation,
            "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        }
    return {
        "direction_cosine": float(_cosine_rows(predicted, target).mean()),
        "normalized_rmse": float(
            np.sqrt(np.mean((predicted - target) ** 2))
            / max(float(np.sqrt(np.mean(target**2))), 1e-20)
        ),
    }


def _ceiling(
    context: Any, base: dict[str, Any], confirm: dict[str, Any]
) -> dict[str, Any]:
    bank, data, train = build_bank(context)
    lookup = {str(value): index for index, value in enumerate(data["ids"].tolist())}
    ids = [str(value) for value in base["confirmatory"]["base_trial_ids"]]
    evaluation = np.asarray([lookup[value] for value in ids], dtype=int)
    representations: dict[str, tuple[str, np.ndarray]] = {
        "ceiling_a_combined_599": ("combined_reference", data["features"]),
        "ceiling_b_architecture_1797": (
            "architecture_resolved",
            data["layerwise"],
        ),
    }
    for spec in confirm["method_specs"]:
        _, latent, _ = bank.decode_inputs(spec)
        representations[f"compact_{spec['method']}"] = ("compact", latent)
    targets = {
        "h1_j_effect": data["targets"]["next_delta"],
        "h4_j_effect": data["targets"]["future_delta_h4"],
        "output_effect": data["targets"]["output_delta"],
    }
    records: list[dict[str, Any]] = []
    current = data["targets"]["current_j"]
    for name, (ceiling_class, values) in representations.items():
        x = np.concatenate((current, values), axis=1)
        weights = _ridge_weights(x[train], x[evaluation], 1.0)
        for target_name, target in targets.items():
            predicted = _apply_ridge_weights(weights, target[train])
            records.append(
                {
                    "schema_version": SCHEMA_VERSION_V11,
                    "protocol_version": PROTOCOL_V11,
                    "record_type": "architecture_resolved_predictive_ceiling",
                    "representation": name,
                    "ceiling_class": ceiling_class,
                    "dimension": int(values.shape[1]),
                    "target": target_name,
                    **_prediction_metrics(predicted, target[evaluation]),
                }
            )
    for target_name in targets:
        records.append(
            {
                "schema_version": SCHEMA_VERSION_V11,
                "protocol_version": PROTOCOL_V11,
                "record_type": "architecture_resolved_predictive_ceiling",
                "representation": "ceiling_c_raw_full_intervention",
                "ceiling_class": "raw_full_persistent_intervention",
                "dimension": None,
                "target": target_name,
                **(
                    {"correlation": 1.0, "rmse": 0.0}
                    if target_name == "output_effect"
                    else {"direction_cosine": 1.0, "normalized_rmse": 0.0}
                ),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_parquet(context.root / CEILING_RECORDS, index=False, compression="zstd")
    rows = frame.to_dict(orient="records")
    lookup_rows = {(row["representation"], row["target"]): row for row in rows}
    gaps = {}
    for target_name in targets:
        a = lookup_rows[("ceiling_a_combined_599", target_name)]
        b = lookup_rows[("ceiling_b_architecture_1797", target_name)]
        metric = "correlation" if target_name == "output_effect" else "direction_cosine"
        gaps[target_name] = {
            "metric": metric,
            "combined_599": float(a[metric]),
            "architecture_1797": float(b[metric]),
            "architecture_minus_combined": float(b[metric] - a[metric]),
        }
    summary = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "source_freeze_digest": confirm["freeze_digest"],
        "split_role": "independent_confirmatory",
        "definitions": {
            "ceiling_a": "combined block-normalized raw dual-PCA, 599D",
            "ceiling_b": "separate REC+conv+KV raw dual-PCA coordinates, 1797D",
            "ceiling_c": "teacher raw full persistent intervention; causal identity ceiling",
        },
        "architecture_gaps": gaps,
        "records": rows,
        "record_artifact": str(CEILING_RECORDS),
        "record_artifact_sha256": sha256_file(context.root / CEILING_RECORDS),
    }
    write_json_atomic(context.root / CEILING_SUMMARY, summary)
    return summary


def _adjudicate(context: Any, confirm: dict[str, Any]) -> dict[str, Any]:
    channel = json.loads((context.root / CHANNEL_SUMMARY).read_text())
    oracle = json.loads((context.root / ORACLE_SUMMARY).read_text())
    factor1 = json.loads((context.root / FACTOR_STAGE1_SUMMARY).read_text())
    final = json.loads((context.root / CONFIRM_SUMMARY).read_text())
    oracle_success = bool(oracle.get("authorized_methods"))
    factorized_h1 = any(
        value["horizon_all_family_gate_pass"].get("1", False)
        for method, value in factor1.get("authorization", {}).items()
    )
    final_authorized = [
        method
        for method in final.get("authorized_methods", [])
        if method != "unified_causal_d512"
    ]
    factorized_final = [
        method for method in final_authorized if method.startswith("factorized_")
    ]
    singles = ["decoded_rec", "decoded_conv", "decoded_kv"]
    single_h1 = all(
        channel["authorization"][name]["horizon_all_family_gate_pass"].get("1", False)
        for name in singles
    )
    joint_h1 = channel["authorization"]["decoded_all"][
        "horizon_all_family_gate_pass"
    ].get("1", False)
    if factorized_final:
        outcome = "B"
        conclusion = "factorized compact state passed the full frozen causal gate"
    elif single_h1 and not joint_h1:
        outcome = "D"
        conclusion = "single-channel writes work but joint compatibility fails"
    elif oracle_success:
        outcome = "A"
        conclusion = (
            "low-dimensional oracle causal state exists; decoder learning fails"
        )
    else:
        outcome = "C"
        conclusion = (
            "tested low-rank oracle projections do not establish writable sufficiency"
        )
    controller = bool(final_authorized)
    value = {
        "schema_version": SCHEMA_VERSION_V11,
        "protocol_version": PROTOCOL_V11,
        "source_freeze_digest": confirm["freeze_digest"],
        "decision_tree_outcome": outcome,
        "strongest_warranted_conclusion": conclusion,
        "oracle_long_horizon_authorized": oracle_success,
        "factorized_h1_development_pass": factorized_h1,
        "independent_confirmatory_authorized_methods": final_authorized,
        "smallest_causally_validated_dimension": (
            min(
                int(spec["dimension"])
                for spec in confirm["method_specs"]
                if spec["method"] in final_authorized
            )
            if final_authorized
            else None
        ),
        "hypothesis_status": "H3" if controller else "H2",
        "autonomous_controller_training_authorized": controller,
        "free_continuation_executed": False,
    }
    write_json_atomic(context.root / ADJUDICATION, value)
    return value


def main() -> None:
    parser = standard_parser(
        "analyze architecture-resolved causal geometry v11",
        "configs/causal_geometry_v11.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("analyze", "report"))
    args = parser.parse_args()
    context = initialize_context("analyze-v11", args)
    try:
        base = verify_base_freeze(context.root, context.config)
        confirm = verify_derived_freeze(
            context.root, context.config, CONFIRM_FREEZE_PATH
        )
        if args.stage == "analyze":
            summary = {
                "manifold": _manifold(context, base, confirm),
                "amplification": _amplification(context, confirm),
                "ceiling": _ceiling(context, base, confirm),
                "adjudication": _adjudicate(context, confirm),
            }
        else:
            summary = write_reports(context.root, context.config)
        context.finish("COMPLETED_V11_ANALYSIS_STAGE", summary=summary)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
