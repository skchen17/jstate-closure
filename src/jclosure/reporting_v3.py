"""Generate exploratory-v3 reports and figures only from saved records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.experiments.common import repository_root
from jclosure.geometry import record_passes_state_equality
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.statistics import (
    benjamini_hochberg,
    clustered_bootstrap_ci,
    clustered_sign_flip_p_value,
    normalized_remainder_effect,
    numerical_null_threshold,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_completed(root: Path, prefix: str) -> tuple[Path, dict[str, Any]] | None:
    manifests = sorted((root / "results/v3/raw").glob(f"{prefix}-*/manifest.json"))
    for path in reversed(manifests):
        payload = _load_json(path)
        if payload.get("status") in {"COMPLETED", "BANK_COMPLETED"}:
            return path, payload
    return None


def _read_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _latest_completed_closure_sources(
    root: Path,
) -> tuple[pd.DataFrame, list[Path], str | None]:
    groups: dict[str, dict[int, tuple[str, Path, dict[str, Any]]]] = {}
    for manifest_path in sorted(
        (root / "results/v3/raw").glob("closure-v3-*/manifest.json")
    ):
        payload = _load_json(manifest_path)
        if payload.get("status") != "COMPLETED":
            continue
        group = str(payload.get("shard_group_id", payload.get("run_id", "")))
        shard = int(payload.get("shard_index", 0))
        created = str(payload.get("created_at", ""))
        groups.setdefault(group, {})[shard] = (created, manifest_path, payload)
    complete: list[tuple[str, str, dict[int, tuple[str, Path, dict[str, Any]]]]] = []
    for group, shards in groups.items():
        counts = {int(value[2].get("shard_count", 1)) for value in shards.values()}
        if len(counts) != 1:
            continue
        count = counts.pop()
        if set(shards) != set(range(count)):
            continue
        complete.append((max(value[0] for value in shards.values()), group, shards))
    if not complete:
        return pd.DataFrame(), [], None
    _, group, shards = max(complete, key=lambda value: (value[0], value[1]))
    paths = [
        path
        for shard in sorted(shards)
        for path in sorted(shards[shard][1].parent.glob("trials/**/*.jsonl"))
    ]
    frames = [pd.read_json(path, lines=True) for path in paths if path.stat().st_size]
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        paths,
        group,
    )


def summarize_closure_v3(
    records: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 2_026_090_1,
) -> pd.DataFrame:
    """Summarize clean-relative effects without breaking paired base trials."""

    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    if "protocol_key" not in frame:
        frame["protocol_key"] = (
            frame["state_definition"].astype(str)
            + "-"
            + frame["dictionary_size"].astype(str)
        )
    if "js_divergence" not in frame:
        if "metrics" not in frame:
            raise ValueError("closure records do not contain metrics")
        frame["js_divergence"] = frame["metrics"].map(
            lambda value: value.get("js_divergence")
            if isinstance(value, dict)
            else np.nan
        )
    required = {
        "prompt_id",
        "protocol_key",
        "task_family",
        "position_scope",
        "source",
        "strength",
        "condition",
        "clamp_mode",
        "valid",
        "js_divergence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"closure records missing columns: {missing}")
    group_keys = [
        "protocol_key",
        "state_definition",
        "dictionary_size",
        "task_family",
        "position_scope",
        "source",
        "strength",
        "condition",
        "clamp_mode",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_keys, sort=True, dropna=False):
        valid = group[group["valid"] & group["js_divergence"].notna()]
        row = dict(zip(group_keys, key, strict=True))
        row.update(
            attempted=int(len(group)),
            valid=int(len(valid)),
            excluded=int(len(group) - len(valid)),
            estimate=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_clusters=0,
            p_value=np.nan,
        )
        if not valid.empty:
            ci = clustered_bootstrap_ci(
                valid,
                cluster_col="prompt_id",
                value_col="js_divergence",
                n_resamples=n_resamples,
                seed=seed,
            )
            row.update(
                estimate=ci.estimate,
                ci_lower=ci.lower,
                ci_upper=ci.upper,
                n_clusters=ci.n_clusters,
                p_value=clustered_sign_flip_p_value(
                    valid,
                    cluster_col="prompt_id",
                    value_col="js_divergence",
                    n_resamples=n_resamples,
                    seed=seed,
                ),
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    finite_p = summary["p_value"].notna()
    summary["p_value_bh"] = np.nan
    if finite_p.any():
        summary.loc[finite_p, "p_value_bh"] = benjamini_hochberg(
            summary.loc[finite_p, "p_value"].to_numpy(dtype=float)
        )
    for column in (
        "positive_control_estimate",
        "positive_control_ci_lower",
        "null_threshold",
        "normalized_remainder_eta",
        "normalized_remainder_eta_ci_lower",
        "normalized_remainder_eta_ci_upper",
    ):
        summary[column] = np.nan
    summary["positive_control_usable"] = False
    match_keys = [
        "protocol_key",
        "state_definition",
        "dictionary_size",
        "task_family",
        "position_scope",
        "source",
        "strength",
    ]
    positive = summary[
        (summary["condition"] == "j_positive")
        & (summary["clamp_mode"] == "single")
    ]
    positive_lookup = {
        tuple(row[column] for column in match_keys): row
        for _, row in positive.iterrows()
    }
    identity = frame[
        (frame["condition"] == "identity")
        & frame["valid"]
        & frame["js_divergence"].notna()
    ]
    identity_lookup = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in identity.groupby(match_keys, sort=True, dropna=False)
    }
    remainder_indices = summary.index[summary["condition"] == "state_preserving"]
    for index in remainder_indices:
        key = tuple(summary.at[index, column] for column in match_keys)
        control = positive_lookup.get(key)
        null = numerical_null_threshold(
            identity_lookup.get(key, pd.DataFrame()).get(
                "js_divergence", pd.Series(dtype=float)
            ),
            floor=1e-4,
        )
        summary.at[index, "null_threshold"] = null
        if control is None or pd.isna(control["estimate"]):
            continue
        usable = bool(float(control["ci_lower"]) > null)
        summary.at[index, "positive_control_estimate"] = float(control["estimate"])
        summary.at[index, "positive_control_ci_lower"] = float(control["ci_lower"])
        summary.at[index, "positive_control_usable"] = usable
        eta = normalized_remainder_effect(
            float(summary.at[index, "estimate"]),
            float(control["estimate"]),
            j_effect_lower=float(control["ci_lower"]),
            null_threshold=null,
        )
        if eta is not None:
            summary.at[index, "normalized_remainder_eta"] = eta
            if "base_trial_id" not in frame:
                continue
            remainder_mask = np.logical_and.reduce(
                [frame[column] == summary.at[index, column] for column in match_keys]
            )
            remainder_rows = frame[
                remainder_mask
                & (frame["condition"] == "state_preserving")
                & (frame["clamp_mode"] == summary.at[index, "clamp_mode"])
                & frame["valid"]
            ][["base_trial_id", "prompt_id", "js_divergence"]]
            positive_rows = frame[
                remainder_mask
                & (frame["condition"] == "j_positive")
                & (frame["clamp_mode"] == "single")
                & frame["valid"]
            ][["base_trial_id", "js_divergence"]]
            paired = remainder_rows.merge(
                positive_rows,
                on="base_trial_id",
                suffixes=("_remainder", "_positive"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            clustered = paired.groupby("prompt_id", sort=True)[
                ["js_divergence_remainder", "js_divergence_positive"]
            ].sum()
            generator = np.random.default_rng(seed)
            sampled = generator.integers(
                0, len(clustered), size=(n_resamples, len(clustered))
            )
            remainder_sums = clustered["js_divergence_remainder"].to_numpy()[
                sampled
            ].sum(axis=1)
            positive_sums = clustered["js_divergence_positive"].to_numpy()[
                sampled
            ].sum(axis=1)
            ratios = remainder_sums / np.maximum(positive_sums, 1e-12)
            summary.at[index, "normalized_remainder_eta"] = float(
                paired["js_divergence_remainder"].mean()
                / max(paired["js_divergence_positive"].mean(), 1e-12)
            )
            lower, upper = np.quantile(ratios, [0.025, 0.975])
            summary.at[index, "normalized_remainder_eta_ci_lower"] = float(lower)
            summary.at[index, "normalized_remainder_eta_ci_upper"] = float(upper)
    return summary


def _save_figure(
    target: Path,
    *,
    sources: list[Path],
    draw: Any,
) -> dict[str, Any]:
    if not sources or any(not source.is_file() for source in sources):
        raise RuntimeError(f"{target.name} lacks machine-readable source records")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    draw(axis)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return {
        "figure": str(target),
        "sha256": sha256_file(target),
        "sources": [
            {"path": str(source), "sha256": sha256_file(source)}
            for source in sources
        ],
        "manual_values": False,
    }


def _rank_value(value: Any, key: str) -> int | None:
    if isinstance(value, dict):
        result = value.get(key)
        return None if result is None else int(result)
    return None


def _latest_completed_shard_dirs(root: Path, stage: str) -> set[Path]:
    latest: dict[int, tuple[str, Path]] = {}
    for manifest_path in sorted((root / "results/v3/raw").glob("geometry-v3-*/manifest.json")):
        payload = _load_json(manifest_path)
        if payload.get("status") != "COMPLETED" or payload.get("stage") != stage:
            continue
        if stage == "pareto" and payload.get("limit") is not None:
            continue
        shard = int(payload.get("shard_index", 0))
        created = str(payload.get("created_at", ""))
        if shard not in latest or created > latest[shard][0]:
            latest[shard] = (created, manifest_path.parent)
    return {value[1] for value in latest.values()}


def _geometry_sources(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], str]:
    all_paths = sorted(
        (root / "results/v3/raw").glob("geometry-v3-*/map_spectra-*.parquet")
    )
    all_local_paths = sorted(
        (root / "results/v3/raw").glob("geometry-v3-*/local_spectra-*.parquet")
    )
    formal_paths = [path for path in all_paths if not path.name.endswith("-smoke.parquet")]
    formal_local_paths = [
        path for path in all_local_paths if not path.name.endswith("-smoke.parquet")
    ]
    completed_dirs = _latest_completed_shard_dirs(root, "spectrum")
    if completed_dirs:
        formal_paths = [path for path in formal_paths if path.parent in completed_dirs]
        formal_local_paths = [
            path for path in formal_local_paths if path.parent in completed_dirs
        ]
    if formal_paths and formal_local_paths:
        paths = formal_paths
        local_paths = formal_local_paths
        execution_scope = "formal"
    elif formal_paths or formal_local_paths:
        paths = formal_paths
        local_paths = formal_local_paths
        execution_scope = "formal_incomplete"
    else:
        paths = all_paths
        local_paths = all_local_paths
        execution_scope = "smoke" if paths or local_paths else "none"
    return (
        _read_parquets(paths),
        _read_parquets(local_paths),
        [*paths, *local_paths],
        execution_scope,
    )


def _pareto_sources(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((root / "results/v3/raw").glob("geometry-v3-*/pareto_records*.parquet"))
    paths = [
        path
        for path in paths
        if "preflight" not in path.name and "part-" not in path.name
    ]
    completed_dirs = _latest_completed_shard_dirs(root, "pareto")
    if completed_dirs:
        paths = [path for path in paths if path.parent in completed_dirs]
    return _read_parquets(paths), paths


def _pareto_state_equality_mask(records: pd.DataFrame) -> pd.Series:
    if records.empty:
        return pd.Series(dtype=bool, index=records.index)
    return pd.Series(
        [
            record_passes_state_equality(
                row,
                state_definition=str(row["state_definition"]),
            )
            for row in records.to_dict("records")
        ],
        index=records.index,
        dtype=bool,
    )


def summarize_pareto_v3(
    records: pd.DataFrame,
    *,
    formal_displacement: float = 0.20,
) -> pd.DataFrame:
    """Summarize v3 candidates using state-definition-specific hard gates."""

    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    frame["construction_valid"] = frame["valid"].fillna(False).astype(bool)
    frame["state_equal"] = (
        frame["construction_valid"] & _pareto_state_equality_mask(frame)
    )
    frame["rms_valid"] = frame["rms_drift"] <= 0.02
    frame["natural_equal"] = (
        frame["state_equal"]
        & frame["rms_valid"]
        & frame["natural"].fillna(False).astype(bool)
    )
    frame["formal_valid"] = (
        frame["natural_equal"]
        & (frame["displacement_fraction"] >= formal_displacement)
    )
    frame["null_tolerance_label"] = frame["null_tolerance"].map(
        lambda value: "not_applicable"
        if pd.isna(value)
        else f"{float(value):.0e}"
    )
    rows: list[dict[str, Any]] = []
    keys = [
        "layer",
        "dictionary_size",
        "state_definition",
        "method",
        "null_tolerance_label",
    ]
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        natural_equal = group[group["natural_equal"]]
        formal = group[group["formal_valid"]]
        identifier = "prompt_id" if "prompt_id" in group else "paired_trial_id"
        exclusions = Counter(
            str(value)
            for value in group.loc[~group["construction_valid"], "exclusion_reason"]
            if pd.notna(value)
        )
        statuses = Counter(str(value) for value in group["optimization_status"])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "attempted_rows": int(len(group)),
                "attempted_anchors": int(group[identifier].nunique()),
                "construction_valid_rows": int(group["construction_valid"].sum()),
                "state_equal_rows": int(group["state_equal"].sum()),
                "natural_equal_rows": int(group["natural_equal"].sum()),
                "formal_valid_rows": int(group["formal_valid"].sum()),
                "formal_valid_anchors": int(formal[identifier].nunique()),
                "max_natural_equal_displacement": (
                    float(natural_equal["displacement_fraction"].max())
                    if not natural_equal.empty
                    else np.nan
                ),
                "exclusion_counts": json.dumps(
                    dict(sorted(exclusions.items())), sort_keys=True
                ),
                "optimization_status_counts": json.dumps(
                    dict(sorted(statuses.items())), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _formal_pareto_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    dense = (
        (summary["state_definition"] == "V3-Dense")
        & summary["method"].isin(
            ["norm_tangent_dense_null", "hard_constrained"]
        )
        & (summary["null_tolerance_label"] == "1e-04")
    )
    sparse = (
        (summary["state_definition"] == "V3-Sparse")
        & (summary["method"] == "sparse_remainder")
        & (summary["null_tolerance_label"] == "not_applicable")
    )
    return summary[dense | sparse].copy()


def _format_pareto_summary(summary: pd.DataFrame) -> str:
    selected = _formal_pareto_rows(summary)
    if selected.empty:
        return "Formal Pareto construction records were not available."
    labels = {
        "norm_tangent_dense_null": "Dense local-null",
        "hard_constrained": "Dense optimized",
        "sparse_remainder": "Sparse same-definition",
    }
    lines = [
        "| M | Layer | Method | constructed/attempted rows | "
        "state-equal rows | natural+equal rows | formal anchors | max displacement |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(
        ["dictionary_size", "layer", "method"]
    ).iterrows():
        maximum = row["max_natural_equal_displacement"]
        maximum_text = "none" if pd.isna(maximum) else f"{float(maximum):.6f}"
        lines.append(
            f"| {int(row['dictionary_size'])} | {int(row['layer'])} | "
            f"{labels.get(str(row['method']), str(row['method']))} | "
            f"{int(row['construction_valid_rows'])}/{int(row['attempted_rows'])} | "
            f"{int(row['state_equal_rows'])} | "
            f"{int(row['natural_equal_rows'])} | "
            f"{int(row['formal_valid_anchors'])}/{int(row['attempted_anchors'])} | "
            f"{maximum_text} |"
        )
    return "\n".join(lines)


def _format_pareto_attrition(summary: pd.DataFrame) -> str:
    selected = _formal_pareto_rows(summary)
    if selected.empty:
        return "Formal construction attrition was not available."
    rows = []
    for (size, method), group in selected.groupby(
        ["dictionary_size", "method"], sort=True
    ):
        rows.append(
            {
                "dictionary_size": int(size),
                "method": str(method),
                "attempted": int(group["attempted_rows"].sum()),
                "constructed": int(group["construction_valid_rows"].sum()),
                "equal": int(group["state_equal_rows"].sum()),
                "natural_equal": int(group["natural_equal_rows"].sum()),
                "formal": int(group["formal_valid_rows"].sum()),
            }
        )
    return "\n".join(
        f"- M={row['dictionary_size']}, {row['method']}: "
        f"constructed {row['constructed']}/{row['attempted']}, "
        f"state-equal {row['equal']}, natural+equal {row['natural_equal']}, "
        f"formal rows {row['formal']}"
        for row in rows
    )


def build_geometry_figures(root: Path) -> list[dict[str, Any]]:
    maps, local, spectrum_paths, execution_scope = _geometry_sources(root)
    pareto, pareto_paths = _pareto_sources(root)
    figures: list[dict[str, Any]] = []
    if execution_scope != "formal":
        return figures
    target_root = root / "results/v3/figures"
    if not maps.empty:
        maps = maps.copy()
        maps["rank_1e4"] = maps["tolerance_ranks"].map(
            lambda value: _rank_value(value, "relative_1e-04")
        )

        def draw_rank(axis: plt.Axes) -> None:
            for (size, kind), group in maps.groupby(["dictionary_size", "map_kind"]):
                ordered = group.sort_values("layer")
                axis.plot(
                    ordered["layer"],
                    ordered["rank_1e4"] / 2560,
                    marker="o",
                    label=f"M={size} {kind}",
                )
            axis.set(
                xlabel="Layer",
                ylabel="rank at 1e-4 σmax / d_model",
                title="Raw and centered dense-map numerical rank",
            )
            axis.legend(fontsize=7, ncol=2)

        figures.append(
            _save_figure(
                target_root / "15_geometry_map_rank.png",
                sources=spectrum_paths,
                draw=draw_rank,
            )
        )
    if not local.empty:
        local = local.copy()
        local["tangent_null_1e4"] = local["tangent_null_dimensions"].map(
            lambda value: _rank_value(value, "relative_1e-04")
        )

        def draw_null(axis: plt.Axes) -> None:
            grouped = (
                local.groupby(["layer", "dictionary_size"])["tangent_null_1e4"]
                .median()
                .reset_index()
            )
            for size, group in grouped.groupby("dictionary_size"):
                axis.plot(
                    group["layer"],
                    group["tangent_null_1e4"],
                    marker="o",
                    label=f"M={size}",
                )
            axis.set(
                xlabel="Layer",
                ylabel="Median tangent-null dimension",
                title="Local normalized-state tangent null at 1e-4 σmax",
            )
            axis.legend(fontsize=8)

        figures.append(
            _save_figure(
                target_root / "16_local_tangent_null.png",
                sources=spectrum_paths,
                draw=draw_null,
            )
        )
    if not pareto.empty:
        pareto = pareto.copy()
        pareto["state_equal"] = _pareto_state_equality_mask(pareto)
        formal_method = (
            (
                (pareto["state_definition"] == "V3-Dense")
                & pareto["method"].isin(
                    ["norm_tangent_dense_null", "hard_constrained"]
                )
                & np.isclose(pareto["null_tolerance"], 1e-4)
            )
            | (
                (pareto["state_definition"] == "V3-Sparse")
                & (pareto["method"] == "sparse_remainder")
            )
        )
        feasible = pareto[
            pareto["valid"].fillna(False).astype(bool)
            & pareto["state_equal"]
            & (pareto["rms_drift"] <= 0.02)
            & pareto["natural"].fillna(False).astype(bool)
            & formal_method
        ]

        def draw_displacement(axis: plt.Axes) -> None:
            grouped = (
                feasible.groupby(["dictionary_size", "layer", "method"])[
                    "displacement_fraction"
                ]
                .max()
                .reset_index()
            )
            styles = {
                "norm_tangent_dense_null": "--",
                "hard_constrained": "-",
                "sparse_remainder": ":",
            }
            for (size, method), group in grouped.groupby(
                ["dictionary_size", "method"], sort=True
            ):
                ordered = group.sort_values("layer")
                axis.plot(
                    ordered["layer"],
                    ordered["displacement_fraction"],
                    marker="o",
                    linestyle=styles.get(str(method), "-"),
                    label=f"M={int(size)} {method}",
                )
            axis.axhline(0.20, color="black", linestyle="-.", linewidth=1)
            axis.set(
                xlabel="Layer",
                ylabel="Maximum feasible displacement / natural scale",
                title="State-definition-specific Pareto feasibility",
            )
            axis.legend(fontsize=6, ncol=2)

        figures.append(
            _save_figure(
                target_root / "17_pareto_max_displacement.png",
                sources=pareto_paths,
                draw=draw_displacement,
            )
        )

        def draw_mismatch(axis: plt.Axes) -> None:
            for method, group in pareto.groupby("method"):
                axis.scatter(
                    group["displacement_fraction"],
                    1 - group["dense_cosine"],
                    s=8,
                    alpha=0.35,
                    label=str(method),
                )
            axis.axhline(1 - 0.995, color="red", linestyle="--")
            axis.axvline(0.20, color="black", linestyle=":")
            axis.set(
                xlabel="Displacement / natural scale",
                ylabel="Dense-state error (1 − cosine)",
                yscale="log",
                title="Sparse/dense construction mismatch",
            )
            axis.legend(fontsize=7)

        figures.append(
            _save_figure(
                target_root / "18_sparse_dense_mismatch.png",
                sources=pareto_paths,
                draw=draw_mismatch,
            )
        )
    return figures


def _format_map_summary(maps: pd.DataFrame) -> str:
    if maps.empty:
        return "Map spectra were not executed."
    values = []
    for _, row in maps.sort_values(["dictionary_size", "layer", "map_kind"]).iterrows():
        rank = _rank_value(row["tolerance_ranks"], "relative_1e-04")
        values.append(
            f"- M={int(row['dictionary_size'])}, layer {int(row['layer'])}, "
            f"{row['map_kind']}: rank@1e-4={rank}/2560, "
            f"stable rank={float(row['stable_rank']):.3f}, status={row['rank_status']}"
        )
    return "\n".join(values)


def _geometry_diagnosis(maps: pd.DataFrame, local: pd.DataFrame, pareto: pd.DataFrame) -> str:
    if maps.empty or local.empty:
        return "D — geometry audit incomplete; no state-definition diagnosis is warranted."
    local_ranks = local["tolerance_ranks"].map(
        lambda value: _rank_value(value, "relative_1e-04")
    )
    local_null = local["tangent_null_dimensions"].map(
        lambda value: _rank_value(value, "relative_1e-04")
    )
    near_injective = bool(
        local_ranks.median() >= 0.99 * (2560 - 1) or local_null.median() <= 25
    )
    max_displacement = None
    if not pareto.empty:
        state_equal = _pareto_state_equality_mask(pareto)
        feasible = pareto[
            pareto["valid"].fillna(False).astype(bool)
            & state_equal
            & (pareto["rms_drift"] <= 0.02)
            & pareto["natural"].fillna(False).astype(bool)
        ]
        if not feasible.empty:
            max_displacement = float(feasible["displacement_fraction"].max())
    if near_injective:
        return (
            "Dense state-definition feasibility warning: the median local rank at "
            "1e-4 is "
            f"{float(local_ranks.median()):.0f}/2560 and the median tangent-null "
            f"dimension is {float(local_null.median()):.0f}. The normalized dense "
            "profile is therefore operationally near-injective under the frozen rule. "
            "This is not compact H1 evidence and triggers low-dimensional search."
        )
    if max_displacement is None or max_displacement < 0.20:
        value = "none" if max_displacement is None else f"{max_displacement:.6f}"
        return (
            "Dense state-definition feasibility failure: maximum natural, "
            f"state-equal displacement was {value}, below the frozen 0.20 threshold. "
            "This is not H1 evidence and triggers low-dimensional search."
        )
    return (
        "At least one strict natural dense-preserving candidate reached the frozen "
        "0.20 displacement. Behavioral closure still requires calibration authorization."
    )


def build_reports(root: Path) -> dict[str, Any]:
    maps, local, spectrum_paths, execution_scope = _geometry_sources(root)
    pareto, pareto_paths = _pareto_sources(root)
    pareto_summary = summarize_pareto_v3(pareto)
    pareto_summary_path = root / "results/v3/processed/pareto_formal_summary_v3.parquet"
    if not pareto_summary.empty:
        pareto_summary.to_parquet(pareto_summary_path, index=False)
    calibration_path = root / "results/v3/processed/clamp_v3_calibration.json"
    calibration = _load_json(calibration_path) if calibration_path.is_file() else None
    closure_records, closure_paths, closure_group = _latest_completed_closure_sources(
        root
    )
    closure_summary = summarize_closure_v3(closure_records)
    closure_summary_path = root / "results/v3/processed/closure_v3_effects.parquet"
    if not closure_summary.empty:
        closure_summary.to_parquet(closure_summary_path, index=False)
    figures = build_geometry_figures(root)
    figure_manifest = {
        "schema_version": 3,
        "protocol_version": "exploratory_protocol_v3",
        "figures": figures,
    }
    write_json_atomic(
        root / "results/v3/processed/figure_manifest_v3.json", figure_manifest
    )
    diagnosis = (
        _geometry_diagnosis(maps, local, pareto)
        if execution_scope == "formal"
        else "D — only GPU smoke diagnostics were completed; no formal geometry or "
        "state-definition diagnosis is warranted."
        if execution_scope == "smoke"
        else "D — geometry audit incomplete; no state-definition diagnosis is warranted."
    )
    verification_status = "ANALYZED" if execution_scope == "formal" else "UNVERIFIED"
    run_manifests = sorted((root / "results/v3/raw").glob("*/manifest.json"))
    failed_runs = [
        {
            "run_id": payload.get("run_id"),
            "kind": payload.get("kind"),
            "error": payload.get("error"),
        }
        for payload in (_load_json(path) for path in run_manifests)
        if payload.get("status") == "FAILED"
    ]
    smoke_runs = [
        payload.get("run_id")
        for payload in (_load_json(path) for path in run_manifests)
        if payload.get("status") == "COMPLETED"
        and any(
            str(value).endswith("-smoke.parquet")
            for value in payload.get("outputs", {}).values()
        )
    ]
    jvp_count = 0
    jvp_failures = 0
    radial_max = None
    if not local.empty:
        jvp_count = int(local["jvp_passed"].notna().sum())
        jvp_failures = int((local["jvp_passed"] == False).sum())  # noqa: E712
        radial_max = float(local["radial_residual"].max())
    geometry_report = f"""# J-state geometry audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Date: 2026-08-29
- Verification Status: {verification_status}
- Protocol: exploratory protocol v3
- Baseline: d504eaa14af45f9df32101cf4599c55d3fac8707

## Status

This report is generated from saved Parquet records. Phase 0 v2 and its 0/1400
strict clamp result were not modified or re-adjudicated.

- Execution scope: {execution_scope}
- Successful smoke run IDs: {smoke_runs or 'none'}

{diagnosis}

## Map spectra

{_format_map_summary(maps)}

## Local normalized-state checks

- Local rows: {len(local)}
- Analytic/autograd JVP/VJP checked rows: {jvp_count}
- Rows failing the frozen 1e-4 relative-error check: {jvp_failures}
- Maximum normalized radial residual: {"not executed" if radial_max is None else f"{radial_max:.6g}"}

## Pareto audit

- Candidate rows: {len(pareto)}
- Source files: {len(pareto_paths)} Pareto and {len(spectrum_paths)} spectrum Parquet files
- Failed run manifests retained: {len(failed_runs)}
- Canonical state-definition-aware summary: {str(pareto_summary_path.relative_to(root)) if pareto_summary_path.is_file() else "not generated"}

The dense formal rows below use the frozen `1e-4 × sigma_max` tolerance. Sparse
rows use their independent support/coefficient/reconstruction equality gate;
dense cosine is only a sensitivity metric for V3-Sparse.

### Formal construction attrition

{_format_pareto_attrition(pareto_summary)}

### Layer-by-layer formal feasibility

`formal anchors` counts anchors with at least one natural, state-equal candidate
at displacement `>=0.20`; `max displacement` is computed before imposing that
minimum so infeasible cells remain visible.

{_format_pareto_summary(pareto_summary)}

All thresholds are protocol constants. No behavioral H1/H2/H3 conclusion is drawn
from geometry or construction feasibility alone.
"""
    (root / "reports/JSTATE_GEOMETRY_AUDIT.md").write_text(
        geometry_report, encoding="utf-8"
    )
    if calibration is None:
        clamp_text = "Calibration was not executed; behavioral v3 remains gated."
        authorized: list[str] = []
        attempted = 0
        valid = 0
    else:
        authorized = list(calibration.get("behavioral_authorized_protocols", []))
        attempted = int(calibration.get("attempted", 0))
        valid = int(calibration.get("formal_valid", 0))
        reason_counts: Counter[str] = Counter(
            reason
            for row in calibration.get("layers", [])
            for reason in row.get("reasons", [])
        )
        clamp_text = (
            f"Calibration saved {valid}/{attempted} formal-valid candidate records. "
            f"Authorized protocols: {authorized or 'none'}. Gate reasons: "
            f"{dict(reason_counts)}."
        )
    calibration_report = f"""# Clamp v3 calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Date: 2026-08-29
- Verification Status: {"ANALYZED" if calibration is not None else "UNVERIFIED"}
- Protocol: exploratory protocol v3

## Gate

{clamp_text}

Formal validity requires the state-definition-specific equality gate, RMS drift
at most 0.02, displacement at least 0.20 of the natural scale, and the frozen
naturality envelope. Candidates between 0.05 and 0.20 are sensitivity records
only and cannot support H1/H2/H3.
"""
    (root / "reports/CLAMP_V3_CALIBRATION.md").write_text(
        calibration_report, encoding="utf-8"
    )
    if closure_records.empty:
        closure_text = (
            "No completed, shard-complete v3 behavioral run was found. "
            "Closure and mediation remain unexecuted."
        )
        base_attempted = 0
        base_valid = 0
    else:
        base_attempted = int(closure_records["base_trial_id"].nunique())
        base_mask = (
            closure_records["base_formal_valid"].astype(bool)
            if "base_formal_valid" in closure_records
            else pd.Series(False, index=closure_records.index)
        )
        base_valid = int(
            closure_records.loc[
                base_mask, "base_trial_id"
            ].nunique()
        )
        remainder = closure_summary[
            closure_summary["condition"] == "state_preserving"
        ]
        usable = int(remainder["positive_control_usable"].sum())
        closure_text = (
            f"Completed shard group `{closure_group}` contains {len(closure_records)} "
            f"records from {base_valid}/{base_attempted} valid/attempted base trials. "
            f"Positive-control-gated eta is available for {usable}/{len(remainder)} "
            "state-preserving summary cells."
        )
    causal_report = f"""# Closure causal report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Date: 2026-08-30
- Verification Status: {"ANALYZED" if not closure_records.empty else "UNVERIFIED"}
- Protocol: exploratory protocol v3

## Status

{closure_text}

Effects are clean-relative full-vocabulary Jensen–Shannon divergence with
10,000 prompt-clustered bootstrap resamples. `eta` is emitted only where the
positive-control CI lower bound exceeds both the identity 99th-percentile null
and `1e-4`. One-shot, final-persistent, and all-position-persistent results
remain separate; no binary mediation label is substituted for those arms.

Machine-readable summaries: {str(closure_summary_path.relative_to(root)) if closure_summary_path.is_file() else "not generated"}.
"""
    (root / "reports/CLOSURE_CAUSAL_REPORT.md").write_text(
        causal_report, encoding="utf-8"
    )
    execution = {
        "schema_version": 3,
        "protocol_version": "exploratory_protocol_v3",
        "geometry": (
            "COMPLETED"
            if execution_scope == "formal"
            else "SMOKE_COMPLETED"
            if execution_scope == "smoke" and not maps.empty and not local.empty
            else "INCOMPLETE"
            if execution_scope == "formal_incomplete"
            else "FAILED"
            if failed_runs
            else "INCOMPLETE"
        ),
        "pareto": "COMPLETED" if not pareto.empty else "UNEXECUTED",
        "calibration": "COMPLETED" if calibration is not None else "UNEXECUTED",
        "behavioral_authorized_protocols": authorized,
        "behavioral_closure": (
            "COMPLETED"
            if not closure_records.empty
            else "AUTHORIZED_NOT_EXECUTED"
            if authorized
            else "GATED"
        ),
        "behavioral_shard_group": closure_group,
        "behavioral_base_trials": {
            "attempted": base_attempted,
            "valid": base_valid,
        },
        "lowdim_search": (
            "UNEXECUTED"
            if execution_scope != "formal" or maps.empty or local.empty
            else "REQUIRED_OR_PENDING"
            if "warning" in diagnosis.casefold()
            else "NOT_TRIGGERED"
        ),
        "strongest_warranted_conclusion": "D",
        "failed_runs": failed_runs,
        "source_hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                *spectrum_paths,
                *pareto_paths,
                *([pareto_summary_path] if pareto_summary_path.is_file() else []),
                *closure_paths,
                *run_manifests,
            ]
        },
    }
    write_json_atomic(root / "results/v3/processed/execution_status_v3.json", execution)
    final_path = root / "reports/FINAL_REPORT.md"
    marker = "\n## Exploratory protocol v3 update\n"
    existing = final_path.read_text(encoding="utf-8")
    existing = existing.split(marker, 1)[0].rstrip()
    failed_text = (
        "\n".join(
            f"- `{item['run_id']}`: {item['error']}" for item in failed_runs
        )
        or "- None"
    )
    appendix = f"""
{marker}
The v1/v2 records, thresholds, reports, and 0/1400 calibration result remain
byte-identical under the committed SHA-256 regression guard.

- Geometry status: **{execution['geometry']}**
- Pareto status: **{execution['pareto']}**
- V3 clamp calibration: **{execution['calibration']}**
- Behavioral protocols authorized: **{authorized or 'none'}**
- Strongest warranted classification after v3: **D**

{diagnosis}

Failed v3 runs are evidence about execution only and are not interpreted as
model behavior:

{failed_text}

No H1-Dense, H1-Sparse, H2, or H3 claim is permitted unless a frozen operational
state passes calibration and the paired behavioral, mediation, rollout, and
causal-fidelity gates. Small-perturbation records below 0.20 cannot support those
claims.
"""
    final_path.write_text(existing + appendix, encoding="utf-8")
    return execution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry_v3.yaml")
    parser.parse_args()
    root = repository_root()
    result = build_reports(root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
