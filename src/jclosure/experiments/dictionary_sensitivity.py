"""Analyze paired closure effects across nested measured-J dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.experiments.common import (
    initialize_context,
    require_closure_eligible_layers,
    standard_parser,
)
from jclosure.provenance import write_json_atomic
from jclosure.statistics import clustered_bootstrap_ci


def _records(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.json_normalize(rows, sep=".") if rows else pd.DataFrame()


def _effect(
    data: pd.DataFrame,
    *,
    condition: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    subset = data[data["condition"] == condition]
    if condition == "non_j":
        subset = subset[subset["clamp_condition"] == "single"]
    if len(subset) < 2 or subset["prompt_id"].nunique() < 2:
        return None
    return clustered_bootstrap_ci(
        subset,
        cluster_col="prompt_id",
        value_col="metrics.js_divergence",
        n_resamples=int(config["statistics"]["bootstrap_resamples"]),
        seed=int(config["reproducibility"]["bootstrap_seed"]),
    ).__dict__


def summarize_dictionary_effects(
    data: pd.DataFrame,
    *,
    dictionary_sizes: list[int],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return per-M effects using only trials valid at every declared M."""

    required = {
        "paired_trial_id",
        "dictionary_size",
        "dictionary_hash",
        "valid",
        "condition",
        "clamp_condition",
        "prompt_id",
        "metrics.js_divergence",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"closure records lack paired dictionary fields: {missing}")
    duplicate = data.duplicated(["paired_trial_id", "dictionary_size"], keep=False)
    if duplicate.any():
        raise ValueError("paired_trial_id is not unique within dictionary size")
    expected = set(dictionary_sizes)
    sizes_by_trial = data.groupby("paired_trial_id")["dictionary_size"].agg(
        lambda values: {int(value) for value in values}
    )
    common = set(
        sizes_by_trial[sizes_by_trial.map(lambda value: value == expected)].index
    )
    common_valid = set(
        data[
            data["paired_trial_id"].isin(common) & (data["valid"] == True)  # noqa: E712
        ]
        .groupby("paired_trial_id")["dictionary_size"]
        .agg(lambda values: {int(value) for value in values})
        .loc[lambda series: series.map(lambda value: value == expected)]
        .index
    )
    records: list[dict[str, Any]] = []
    for size in dictionary_sizes:
        all_at_size = data[data["dictionary_size"] == size]
        paired = all_at_size[all_at_size["paired_trial_id"].isin(common_valid)]
        e_r = _effect(paired, condition="non_j", config=config)
        e_j = _effect(paired, condition="j_positive", config=config)
        records.append(
            {
                "schema_version": 2,
                "protocol_version": "phase0_protocol_v2",
                "dictionary_size": size,
                "dictionary_hashes": sorted(all_at_size["dictionary_hash"].unique()),
                "attempted": int(len(all_at_size)),
                "valid_at_size": int(all_at_size["valid"].sum()),
                "common_valid_trials": int(len(common_valid)),
                "e_r": e_r,
                "e_j": e_j,
            }
        )
    return records, len(common_valid)


def main() -> None:
    parser = standard_parser(
        "Analyze paired dictionary-size closure sensitivity",
        "configs/confirm_v2.yaml",
    )
    args = parser.parse_args()
    context = initialize_context("dictionary-sensitivity", args)
    try:
        require_closure_eligible_layers(context)
        if args.dry_run:
            context.finish("DRY_RUN")
            return
        paths = sorted(context.raw_dir.glob("closure-*/trials/**/*.jsonl"))
        data = _records(paths)
        if data.empty:
            raise RuntimeError("no closure records exist for dictionary sensitivity")
        sizes = [int(value) for value in context.config["closure"]["dictionary_sizes"]]
        summaries, common_count = summarize_dictionary_effects(
            data,
            dictionary_sizes=sizes,
            config=context.config,
        )
        output = context.processed_dir / "dictionary_sensitivity.json"
        write_json_atomic(
            output,
            {
                "schema_version": 2,
                "protocol_version": "phase0_protocol_v2",
                "run_id": context.run_id,
                "common_valid_trials": common_count,
                "records": summaries,
            },
        )
        pd.json_normalize(summaries, sep=".").to_parquet(
            context.processed_dir / "dictionary_sensitivity.parquet", index=False
        )
        context.finish("COMPLETED", common_valid_trials=common_count)
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
