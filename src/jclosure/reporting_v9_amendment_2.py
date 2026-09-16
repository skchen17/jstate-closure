"""Final presentation amendment for protocol v9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure import reporting_v9 as base
from jclosure.provenance import write_json_atomic
from jclosure.reporting_v9_amendment_1 import build_reports as build_reports_1


def corrected_status_table(frame: pd.DataFrame) -> str:
    columns = [
        "endpoint",
        "method",
        "dimension",
        "status",
        "baseline_score",
        "candidate_score",
        "ceiling_score",
        "candidate_baseline_delta",
        "ceiling_baseline_delta",
        "predictive_gap_closed",
        "conditional_residual_gain",
        "conditional_lower",
        "conditional_upper",
        "semantic_agreement",
        "causal_status",
    ]
    return frame[[column for column in columns if column in frame.columns]].to_markdown(
        index=False
    )


def _insert_after(path: Path, marker: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    if value in text:
        return
    marker_position = text.find(marker)
    if marker_position < 0:
        raise RuntimeError(f"report marker missing in {path}: {marker}")
    position = text.find("\n", marker_position)
    path.write_text(text[: position + 1] + "\n" + value + text[position + 1 :])


def build_reports(
    root: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    amendment_1: dict[str, Any],
    amendment_2: dict[str, Any],
) -> dict[str, str]:
    original = base._status_table
    base._status_table = corrected_status_table
    try:
        reports = build_reports_1(root, config, freeze, amendment_1)
    finally:
        base._status_table = original

    note = (
        f"Reporting amendment 2: `{amendment_2['amendment_digest']}`. "
        "It adds endpoint labels, comparator documentation, and the executed report command; "
        "frozen analysis values are unchanged.\n"
    )
    for relative in reports.values():
        path = root / relative
        marker = "## Protocol v9 conditional-sufficiency" if path == root / "reports/FINAL_REPORT.md" else "# "
        _insert_after(path, marker, note)

    dimension_path = root / "reports/SUFFICIENCY_DIMENSION_SWEEP_V9.md"
    methodology = (
        "## Conditional-comparator amendment\n\n"
        "For v9, `J + C_d + residual(raw | C_d)` is evaluated by the frozen full-information "
        "ceiling, which is informationally equivalent to compact coordinates plus their "
        "orthogonal complement and avoids entering the compact coordinates twice in a ridge "
        "model. The v8 value near `0.063` used duplicated compact and full coordinates; it is "
        "therefore not directly comparable to the v9 512/576/599D values. The v9 results "
        "establish a near-zero conditional gain within the tested 512D+ range, but do not "
        "localize a transition below 512D under the corrected comparator.\n\n"
    )
    text = dimension_path.read_text(encoding="utf-8")
    anchor = "## Rank-aware pooled next-J sweep"
    if methodology not in text:
        text = text.replace(anchor, methodology + anchor, 1)
        dimension_path.write_text(text, encoding="utf-8")

    final_path = root / "reports/FINAL_REPORT.md"
    text = final_path.read_text(encoding="utf-8")
    old = "scripts/run_sufficiency_v9.sh report --run-suffix final-v9"
    new = (
        "scripts/run_sufficiency_v9_reporting_amendment_1.sh freeze "
        "--run-suffix label-filter-freeze\n"
        "scripts/run_sufficiency_v9_reporting_amendment_2.sh freeze "
        "--run-suffix final-presentation-freeze\n"
        "scripts/run_sufficiency_v9_reporting_amendment_2.sh report "
        "--run-suffix final-v9"
    )
    if old not in text:
        raise RuntimeError("v9 base report command missing")
    final_path.write_text(text.replace(old, new, 1), encoding="utf-8")

    manifest_path = root / "results/v9/processed/report_manifest_v9.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_amendment_2_version"] = amendment_2["amendment_version"]
    manifest["reporting_amendment_2_digest"] = amendment_2["amendment_digest"]
    manifest["reports"] = reports
    write_json_atomic(manifest_path, manifest)
    return reports
