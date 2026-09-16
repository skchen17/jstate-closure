"""Reporting-only amendment for protocol-v9 endpoint labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure import reporting_v9 as base
from jclosure.provenance import write_json_atomic


def corrected_curve_rows(
    sweep: pd.DataFrame, rank: int, raw_bytes: int
) -> pd.DataFrame:
    frame = sweep[
        (sweep["family"] == "pooled")
        & sweep["endpoint"].isin(["next", "next_j"])
        & sweep["method"].isin(
            [
                "pca",
                "predictive_bottleneck",
                "causal_bottleneck",
                "semantic_causal_bottleneck",
            ]
        )
    ].copy()
    frame["state_bytes_fp32"] = frame["dimension"] * 4
    frame["joint_J_plus_C_bytes_fp32"] = (4096 + frame["dimension"]) * 4
    frame["encoder_parameter_equivalent"] = frame["dimension"] * rank
    frame["raw_to_compact_state_ratio"] = raw_bytes / frame["state_bytes_fp32"]
    return frame


def _insert_note(path: Path, marker: str, note: str) -> None:
    text = path.read_text(encoding="utf-8")
    if note in text:
        return
    position = text.find("\n", text.find(marker))
    if position < 0:
        raise RuntimeError(f"report marker missing in {path}")
    path.write_text(text[: position + 1] + "\n" + note + text[position + 1 :])


def build_reports(
    root: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    amendment: dict[str, Any],
) -> dict[str, str]:
    original = base._curve_rows
    base._curve_rows = corrected_curve_rows
    try:
        reports = base.build_reports(root, config, freeze)
    finally:
        base._curve_rows = original

    note = (
        f"Reporting amendment: `{amendment['amendment_digest']}`. "
        "It corrects only the next-J endpoint label filter; frozen analysis values are unchanged.\n"
    )
    for relative in reports.values():
        path = root / relative
        _insert_note(path, "# ", note)
    final = root / "reports/FINAL_REPORT.md"
    _insert_note(final, "## Protocol v9 conditional-sufficiency", note)
    reports["final"] = "reports/FINAL_REPORT.md"

    manifest_path = root / "results/v9/processed/report_manifest_v9.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_amendment_version"] = amendment["amendment_version"]
    manifest["reporting_amendment_digest"] = amendment["amendment_digest"]
    manifest["reports"] = reports
    write_json_atomic(manifest_path, manifest)
    return reports
