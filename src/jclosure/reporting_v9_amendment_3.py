"""Correct protocol-v9 figures after visual QA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.provenance import write_json_atomic
from jclosure.reporting_v9_amendment_1 import corrected_curve_rows
from jclosure.reporting_v9_amendment_2 import build_reports as build_reports_2


def _insert_after(path: Path, marker: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    if value in text:
        return
    marker_position = text.find(marker)
    if marker_position < 0:
        raise RuntimeError(f"report marker missing in {path}: {marker}")
    position = text.find("\n", marker_position)
    path.write_text(text[: position + 1] + "\n" + value + text[position + 1 :])


def _figures(root: Path, config: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    sweep = pd.read_parquet(
        root / "results/v9/processed/sufficiency_dimension_sweep_v9.parquet"
    )
    residual = pd.read_parquet(
        root / "results/v9/processed/residual_information_localization_v9.parquet"
    )
    feature_manifest = json.loads(
        (root / "results/v8/processed/structured_features_v8.json").read_text()
    )
    raw_elements = sum(
        int(value["elements"])
        for value in feature_manifest["definition"]["selected_blocks"]
        if "elements" in value
    )
    curves = corrected_curve_rows(sweep, rank=599, raw_bytes=2 * raw_elements)
    section = config["sufficiency_v9"]

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for method, values in curves.groupby("method", sort=True):
        values = values.sort_values("dimension")
        predictive = values[values["status"] != "NOT_IDENTIFIED_RANK_LIMIT"]
        axes[0].plot(
            predictive["dimension"],
            predictive["predictive_gap_closed"],
            marker="o",
            label=method,
        )
        conditional = values[values["status"] == "COMPLETED_HELD_OUT"]
        axes[1].plot(
            conditional["dimension"],
            conditional["conditional_residual_gain"],
            marker="o",
            label=method,
        )
        semantic = sweep[
            (sweep["family"] == "pooled")
            & (sweep["endpoint"] == "next_semantic")
            & (sweep["method"] == method)
        ].sort_values("dimension")
        if not semantic.empty:
            axes[2].plot(
                semantic["dimension"],
                semantic["semantic_agreement"],
                marker="o",
                label=method,
            )
        axes[3].plot(
            values["dimension"], values["state_bytes_fp32"], marker="o", label=method
        )
    axes[0].axhline(
        float(section["predictive_gap_closed_minimum"]),
        color="black",
        linestyle="--",
    )
    axes[1].axhline(
        float(section["conditional_residual_gain_maximum"]),
        color="black",
        linestyle="--",
    )
    axes[2].axhline(
        float(section["semantic_agreement_minimum"]),
        color="black",
        linestyle="--",
    )
    axes[0].set_ylabel("predictive gap closed")
    axes[1].set_ylabel("v9 conditional residual gain")
    axes[1].set_title("v9 comparator only (512D+)", fontsize=10)
    axes[2].set_ylabel("semantic top-10 agreement")
    axes[3].set_ylabel("compact state bytes (FP32)")
    for axis in axes:
        axis.set_xlabel("dimension")
        axis.set_xscale("log", base=2)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(
        root / "results/v9/figures/sufficiency_dimension_elbow_v9.png", dpi=180
    )
    plt.close(figure)

    completed = residual[residual["dimension"] == 512].copy()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(completed["source"], completed["conditional_gain"], color="#4472C4")
    axis.errorbar(
        completed["source"],
        completed["conditional_gain"],
        yerr=[
            completed["conditional_gain"] - completed["lower"],
            completed["upper"] - completed["conditional_gain"],
        ],
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("standalone / interaction conditional gain")
    axis.set_title("512D residual information localization")
    figure.tight_layout()
    figure.savefig(
        root / "results/v9/figures/residual_information_localization_v9.png",
        dpi=180,
    )
    plt.close(figure)


def build_reports(
    root: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    amendment_1: dict[str, Any],
    amendment_2: dict[str, Any],
    amendment_3: dict[str, Any],
) -> dict[str, str]:
    reports = build_reports_2(
        root, config, freeze, amendment_1, amendment_2
    )
    _figures(root, config)
    note = (
        f"Reporting amendment 3: `{amendment_3['amendment_digest']}`. "
        "After visual QA, the conditional curve is restricted to the comparable v9 "
        "512/576/599D results and the residual-axis label is corrected; frozen analysis "
        "values are unchanged.\n"
    )
    for relative in reports.values():
        path = root / relative
        marker = (
            "## Protocol v9 conditional-sufficiency"
            if path == root / "reports/FINAL_REPORT.md"
            else "# "
        )
        _insert_after(path, marker, note)

    dimension_path = root / "reports/SUFFICIENCY_DIMENSION_SWEEP_V9.md"
    figure_note = (
        "The conditional-gain figure intentionally excludes the v8 points below 512D "
        "because they used the older duplicated-coordinate comparator.\n\n"
    )
    text = dimension_path.read_text(encoding="utf-8")
    anchor = "## Rank-aware pooled next-J sweep"
    if figure_note not in text:
        text = text.replace(anchor, figure_note + anchor, 1)
        dimension_path.write_text(text, encoding="utf-8")

    final_path = root / "reports/FINAL_REPORT.md"
    text = final_path.read_text(encoding="utf-8")
    old = (
        "scripts/run_sufficiency_v9_reporting_amendment_2.sh report "
        "--run-suffix final-v9"
    )
    new = (
        "scripts/run_sufficiency_v9_reporting_amendment_3.sh freeze "
        "--run-suffix figure-qa-freeze\n"
        "scripts/run_sufficiency_v9_reporting_amendment_3.sh report "
        "--run-suffix final-v9"
    )
    if old not in text:
        raise RuntimeError("v9 amendment-2 report command missing")
    final_path.write_text(text.replace(old, new, 1), encoding="utf-8")

    manifest_path = root / "results/v9/processed/report_manifest_v9.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_amendment_3_version"] = amendment_3["amendment_version"]
    manifest["reporting_amendment_3_digest"] = amendment_3["amendment_digest"]
    write_json_atomic(manifest_path, manifest)
    return reports
