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
from jclosure.provenance import sha256_file, write_json_atomic


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
        feasible = pareto[
            (pareto["dense_cosine"] >= 0.995)
            & (pareto["top10_overlap"] >= 0.8)
            & (pareto["rms_drift"] <= 0.02)
            & pareto["natural"]
        ]

        def draw_displacement(axis: plt.Axes) -> None:
            grouped = (
                feasible.groupby(["dictionary_size", "method"])[
                    "displacement_fraction"
                ]
                .max()
                .reset_index()
            )
            methods = sorted(grouped["method"].unique())
            sizes = sorted(grouped["dictionary_size"].unique())
            x = np.arange(len(methods))
            width = 0.8 / max(len(sizes), 1)
            for offset, size in enumerate(sizes):
                values = [
                    grouped[
                        (grouped["dictionary_size"] == size)
                        & (grouped["method"] == method)
                    ]["displacement_fraction"].max()
                    for method in methods
                ]
                axis.bar(x + offset * width, values, width, label=f"M={size}")
            axis.set(
                xticks=x + width * (len(sizes) - 1) / 2,
                xticklabels=methods,
                ylabel="Maximum feasible displacement / natural scale",
                title="Dense-equality Pareto feasibility",
            )
            axis.tick_params(axis="x", rotation=25)
            axis.legend(fontsize=8)

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
        feasible = pareto[
            (pareto["dense_cosine"] >= 0.995)
            & (pareto["top10_overlap"] >= 0.8)
            & (pareto["rms_drift"] <= 0.02)
            & pareto["natural"]
        ]
        if not feasible.empty:
            max_displacement = float(feasible["displacement_fraction"].max())
    if near_injective or max_displacement is None or max_displacement < 0.20:
        return (
            "Dense state-definition feasibility warning: the local dense profile is "
            "near-injective or no natural strict candidate reached the frozen 0.20 "
            "displacement. This is not H1 evidence and triggers low-dimensional search."
        )
    return (
        "At least one strict natural dense-preserving candidate reached the frozen "
        "0.20 displacement. Behavioral closure still requires calibration authorization."
    )


def build_reports(root: Path) -> dict[str, Any]:
    maps, local, spectrum_paths, execution_scope = _geometry_sources(root)
    pareto, pareto_paths = _pareto_sources(root)
    calibration_path = root / "results/v3/processed/clamp_v3_calibration.json"
    calibration = _load_json(calibration_path) if calibration_path.is_file() else None
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
        "behavioral_closure": "AUTHORIZED" if authorized else "GATED",
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
            for path in [*spectrum_paths, *pareto_paths, *run_manifests]
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
