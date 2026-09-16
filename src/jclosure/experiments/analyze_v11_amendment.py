"""Correct the clean-zero cycle metric and quantify channel nonadditivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.prepare_v11 import build_bank
from jclosure.protocol_v11_manifold_amendment import (
    PROTOCOL,
    build_freeze,
    verify_freeze,
)
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.reporting_v11_amendment import write_amended_reports
from jclosure.state_models_v11 import CHANNELS

MANIFOLD_INPUT = Path("results/v11/processed/causal_state_manifold_v11.parquet")
MANIFOLD_RECORDS = Path(
    "results/v11/processed/causal_state_manifold_v11_amendment_1.parquet"
)
MANIFOLD_SUMMARY = Path(
    "results/v11/processed/causal_state_manifold_v11_amendment_1.json"
)
CHANNEL_INPUT = Path("results/v11/processed/channelwise_causal_v11.parquet")
INTERACTION_RECORDS = Path(
    "results/v11/processed/channel_interaction_nonadditivity_v11.parquet"
)
INTERACTION_SUMMARY = Path(
    "results/v11/processed/channel_interaction_nonadditivity_v11.json"
)


def _clean_cycle_absolute(context: Any) -> dict[str, float]:
    bank, _, _ = build_bank(context)
    output = {}
    for channel in CHANNELS:
        model = bank.models[channel]
        query = np.zeros((1, bank.rank), dtype=np.float32)
        standardized = (query - model.mean) / model.scale
        reconstructed = (
            standardized @ model.vh.T @ model.vh
        ) * model.scale + model.mean
        output[channel] = float(np.linalg.norm(reconstructed - query))
    return output


def _manifold(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(context.root / MANIFOLD_INPUT)
    absolute = _clean_cycle_absolute(context)
    frame["cycle_metric_identified"] = frame["state"] != "clean_zero"
    frame["cycle_absolute_error"] = np.nan
    for channel, value in absolute.items():
        mask = (frame["state"] == "clean_zero") & (frame["channel"] == channel)
        frame.loc[mask, "cycle_absolute_error"] = value
    frame.loc[frame["state"] == "clean_zero", "cycle_relative_error"] = np.nan
    frame.to_parquet(context.root / MANIFOLD_RECORDS, index=False, compression="zstd")
    aggregates: dict[str, Any] = {}
    for state, values in frame.groupby("state", sort=True):
        aggregates[str(state)] = {}
        for channel, current in values.groupby("channel", sort=True):
            relative = current["cycle_relative_error"].dropna()
            absolute_values = current["cycle_absolute_error"].dropna()
            aggregates[str(state)][str(channel)] = {
                "cycle_relative_error": (
                    float(relative.mean()) if len(relative) else None
                ),
                "cycle_absolute_error": (
                    float(absolute_values.mean()) if len(absolute_values) else None
                ),
                "identified": bool(current["cycle_metric_identified"].all()),
            }
    summary = {
        "protocol_version": PROTOCOL,
        "source_freeze_digest": freeze["freeze_digest"],
        "declared_correction": freeze["declared_issue"],
        "original_records_preserved": str(MANIFOLD_INPUT),
        "aggregates": aggregates,
        "records": str(MANIFOLD_RECORDS),
        "records_sha256": sha256_file(context.root / MANIFOLD_RECORDS),
    }
    write_json_atomic(context.root / MANIFOLD_SUMMARY, summary)
    return summary


def _error_norm(frame: pd.DataFrame) -> np.ndarray:
    teacher = frame["teacher_j_effect_norm"].astype(float).to_numpy()
    decoded = frame["decoded_j_effect_norm"].astype(float).to_numpy()
    cosine = frame["direction_cosine"].astype(float).to_numpy()
    return np.sqrt(
        np.maximum(teacher**2 + decoded**2 - 2 * teacher * decoded * cosine, 0)
    )


def _interaction(context: Any, freeze: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(context.root / CHANNEL_INPUT)
    frame = frame.assign(error_norm=_error_norm(frame))
    pivot = frame.pivot_table(
        index=["base_trial_id", "family", "horizon"],
        columns="condition",
        values="error_norm",
        aggfunc="first",
    )
    definitions = {
        "rec_conv": ("decoded_rec_conv", ["decoded_rec", "decoded_conv"]),
        "rec_kv": ("decoded_rec_kv", ["decoded_rec", "decoded_kv"]),
        "conv_kv": ("decoded_conv_kv", ["decoded_conv", "decoded_kv"]),
        "all": (
            "decoded_all",
            ["decoded_rec", "decoded_conv", "decoded_kv"],
        ),
    }
    records = []
    for label, (joint, singles) in definitions.items():
        rss = np.sqrt(sum(pivot[name] ** 2 for name in singles))
        additive = sum(pivot[name] for name in singles)
        current = pd.DataFrame(
            {
                "joint_condition": label,
                "joint_error_norm": pivot[joint],
                "single_error_rss": rss,
                "single_error_sum": additive,
                "joint_to_rss_ratio": pivot[joint] / np.maximum(rss, 1e-20),
                "joint_minus_single_sum": pivot[joint] - additive,
            },
            index=pivot.index,
        ).reset_index()
        records.append(current)
    output = pd.concat(records, ignore_index=True)
    output.insert(0, "protocol_version", PROTOCOL)
    output.to_parquet(
        context.root / INTERACTION_RECORDS, index=False, compression="zstd"
    )
    aggregates: dict[str, Any] = {}
    for (condition, horizon), values in output.groupby(
        ["joint_condition", "horizon"], sort=True
    ):
        aggregates.setdefault(str(condition), {})[str(int(horizon))] = {
            "mean_joint_error_norm": float(values["joint_error_norm"].mean()),
            "mean_single_error_rss": float(values["single_error_rss"].mean()),
            "mean_joint_to_rss_ratio": float(values["joint_to_rss_ratio"].mean()),
            "mean_joint_minus_single_sum": float(
                values["joint_minus_single_sum"].mean()
            ),
            "count": int(len(values)),
        }
    summary = {
        "protocol_version": PROTOCOL,
        "source_freeze_digest": freeze["freeze_digest"],
        "proxy_definition": freeze["interaction_proxy"],
        "interpretation_limit": (
            "ratio >1 is compatible with constructive cross-channel error interaction; "
            "ratio <1 with cancellation. It is not an exact vector interaction because "
            "per-example error vectors were not retained."
        ),
        "aggregates": aggregates,
        "records": str(INTERACTION_RECORDS),
        "records_sha256": sha256_file(context.root / INTERACTION_RECORDS),
    }
    write_json_atomic(context.root / INTERACTION_SUMMARY, summary)
    return summary


def main() -> None:
    parser = standard_parser(
        "v11 manifold metric and channel interaction amendment",
        "configs/causal_geometry_v11.yaml",
    )
    parser.add_argument(
        "--stage", choices=("freeze", "analyze", "report"), required=True
    )
    args = parser.parse_args()
    context = initialize_context("v11-manifold-amendment", args)
    try:
        if args.stage == "freeze":
            summary = build_freeze(context.root, context.config)
        else:
            freeze = verify_freeze(context.root, context.config)
            if args.stage == "analyze":
                summary = {
                    "manifold": _manifold(context, freeze),
                    "interaction": _interaction(context, freeze),
                }
            else:
                summary = write_amended_reports(context.root)
        context.finish("COMPLETED_V11_MANIFOLD_AMENDMENT", summary=summary)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
