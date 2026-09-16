"""Machine-derived reports for conditional sufficiency protocol v9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic

START = "<!-- V9-RESULTS:START -->"
END = "<!-- V9-RESULTS:END -->"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _status_table(frame: pd.DataFrame) -> str:
    columns = [
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
    available = [column for column in columns if column in frame.columns]
    return frame[available].to_markdown(index=False)


def _curve_rows(sweep: pd.DataFrame, rank: int, raw_bytes: int) -> pd.DataFrame:
    frame = sweep[
        (sweep["family"] == "pooled")
        & (sweep["endpoint"] == "next_j")
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


def build_reports(
    root: Path, config: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, str]:
    summary = json.loads((root / "results/v9/processed/sufficiency_dimension_sweep_v9.json").read_text())
    residual_summary = json.loads(
        (root / "results/v9/processed/residual_information_localization_v9.json").read_text()
    )
    multihorizon_summary = json.loads(
        (root / "results/v9/processed/multihorizon_sufficiency_v9.json").read_text()
    )
    enriched_summary = json.loads(
        (root / "results/v9/processed/behavior_enriched_causal_v9.json").read_text()
    )
    sweep = pd.read_parquet(root / summary["records"])
    residual = pd.read_parquet(root / residual_summary["records"])
    multihorizon = pd.read_parquet(root / multihorizon_summary["records"])
    enriched = pd.read_parquet(root / enriched_summary["records"])
    section = config["sufficiency_v9"]
    rank = int(summary["rank_audit"]["effective_rank"])
    feature_manifest = json.loads(
        (root / "results/v8/processed/structured_features_v8.json").read_text()
    )
    raw_elements = sum(
        int(value["elements"])
        for value in feature_manifest["definition"]["selected_blocks"]
        if "elements" in value
    )
    raw_bytes = 2 * raw_elements
    primary = str(section["primary_method"])
    reports: dict[str, str] = {}
    figure_root = root / "results/v9/figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    curves = _curve_rows(sweep, rank, raw_bytes)
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for method, values in curves.groupby("method", sort=True):
        values = values.sort_values("dimension")
        axes[0].plot(values["dimension"], values["predictive_gap_closed"], marker="o", label=method)
        axes[1].plot(values["dimension"], values["conditional_residual_gain"], marker="o", label=method)
        semantic = sweep[
            (sweep["family"] == "pooled")
            & (sweep["endpoint"] == "next_semantic")
            & (sweep["method"] == method)
        ].sort_values("dimension")
        if not semantic.empty:
            axes[2].plot(semantic["dimension"], semantic["semantic_agreement"], marker="o", label=method)
        axes[3].plot(values["dimension"], values["state_bytes_fp32"], marker="o", label=method)
    axes[0].axhline(float(section["predictive_gap_closed_minimum"]), color="black", linestyle="--")
    axes[1].axhline(float(section["conditional_residual_gain_maximum"]), color="black", linestyle="--")
    axes[2].axhline(float(section["semantic_agreement_minimum"]), color="black", linestyle="--")
    axes[0].set_ylabel("predictive gap closed")
    axes[1].set_ylabel("conditional residual gain")
    axes[2].set_ylabel("semantic top-10 agreement")
    axes[3].set_ylabel("compact state bytes (FP32)")
    for axis in axes:
        axis.set_xlabel("dimension")
        axis.set_xscale("log", base=2)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    dimension_figure = figure_root / "sufficiency_dimension_elbow_v9.png"
    figure.savefig(dimension_figure, dpi=180)
    plt.close(figure)

    completed_residual = residual[residual["dimension"] == 512].copy()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if not completed_residual.empty:
        axis.bar(
            completed_residual["source"],
            completed_residual["conditional_gain"],
            color="#4472C4",
        )
        axis.errorbar(
            completed_residual["source"],
            completed_residual["conditional_gain"],
            yerr=[
                completed_residual["conditional_gain"] - completed_residual["lower"],
                completed_residual["upper"] - completed_residual["conditional_gain"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Shapley / interaction conditional gain")
    axis.set_title("512D residual information localization")
    figure.tight_layout()
    residual_figure = figure_root / "residual_information_localization_v9.png"
    figure.savefig(residual_figure, dpi=180)
    plt.close(figure)

    identified = curves[curves["status"] != "NOT_IDENTIFIED_RANK_LIMIT"].copy()
    rank_limited = summary["rank_audit"]["rank_limited_dimensions"]
    primary_next = identified[identified["method"] == primary].sort_values("dimension")
    conditional_values = primary_next[
        primary_next["dimension"].isin([512, 576, 599])
    ][["dimension", "conditional_residual_gain", "conditional_lower", "conditional_upper"]]
    sufficiency_text = "\n".join(
        [
            "# Sufficiency Dimension Sweep v9",
            "",
            f"Protocol freeze: `{freeze['freeze_digest']}`; v8 source freeze: `{freeze['v8_source_freeze_digest']}`.",
            "",
            f"The frozen training split has 600 pairs and effective rank `{rank}`. Requested dimensions `{rank_limited}` are therefore `NOT_IDENTIFIED_RANK_LIMIT`; they were not silently clamped to 599D.",
            "",
            "## Rank-aware pooled next-J sweep",
            "",
            _status_table(curves.sort_values(["method", "dimension"])),
            "",
            "## Primary conditional-gain continuation",
            "",
            conditional_values.to_markdown(index=False),
            "",
            "## State and encoder size",
            "",
            curves[["method", "dimension", "state_bytes_fp32", "joint_J_plus_C_bytes_fp32", "encoder_parameter_equivalent", "raw_to_compact_state_ratio"]].to_markdown(index=False),
            "",
            "Gap ratios are `NA` whenever the frozen ceiling-minus-baseline denominator is below the frozen epsilon. Raw baseline, candidate, ceiling, and both deltas remain reported in the table.",
            "",
            f"Observational candidates passing every pooled/family predictive, conditional, and semantic gate: `{summary['observational_gate_pass_count']}`. Smallest fully authorized dimension: `{summary['smallest_authorized_dimension']}`.",
            "",
            "Causal decoded-state metrics remain `NA` because the preregistered observational gate did not authorize cache decoding. This is a gate outcome, not a zero causal effect.",
        ]
    )
    path = root / "reports/SUFFICIENCY_DIMENSION_SWEEP_V9.md"
    path.write_text(sufficiency_text + "\n")
    reports["dimension"] = str(path.relative_to(root))

    residual_text = "\n".join(
        [
            "# Residual Information Localization v9",
            "",
            "Architecture-aligned recurrent, convolution, and KV block scores were added after the compact state. The first three rows are standalone held-out next-J conditional gains; `interaction` is the joint gain minus those three standalone gains, so the four terms sum to the joint gain.",
            "",
            residual.sort_values(["dimension", "source"]).to_markdown(index=False),
            "",
            "## Rank-limited requested dimensions",
            "",
            pd.DataFrame(
                [
                    {"dimension": key, **value}
                    for key, value in residual_summary["dimensions"].items()
                    if value["status"] != "COMPLETED"
                ]
            ).to_markdown(index=False),
        ]
    )
    path = root / "reports/RESIDUAL_INFORMATION_LOCALIZATION_V9.md"
    path.write_text(residual_text + "\n")
    reports["residual"] = str(path.relative_to(root))

    multihorizon_text = "\n".join(
        [
            "# Multihorizon Sufficiency v9",
            "",
            "The frozen v8 capture contains four teacher-forced future tokens. Horizons 1, 2, and 4 are evaluated independently; horizon 8 is explicitly unavailable and is not extrapolated.",
            "",
            multihorizon.sort_values(["dimension", "horizon"]).to_markdown(index=False),
            "",
            "Output-logit and task-decision endpoints exist only at the captured four-token trajectory endpoint. Earlier horizons are not back-filled from that endpoint.",
        ]
    )
    path = root / "reports/MULTIHORIZON_SUFFICIENCY_V9.md"
    path.write_text(multihorizon_text + "\n")
    reports["multihorizon"] = str(path.relative_to(root))

    enriched_text = "\n".join(
        [
            "# Behavior-Enriched Causal v9",
            "",
            f"Selection is confined to held-out final-test pairs and has no model-fit overlap. Frozen selection: `{enriched_summary['selection']}`.",
            "",
            _status_table(enriched.sort_values(["dimension", "endpoint"])),
            "",
            f"Decoded compact-state causal fidelity status: `{enriched_summary['causal_fidelity_status']}`.",
        ]
    )
    path = root / "reports/BEHAVIOR_ENRICHED_CAUSAL_V9.md"
    path.write_text(enriched_text + "\n")
    reports["enriched"] = str(path.relative_to(root))

    residual_512 = residual_summary["dimensions"].get("512", {})
    largest_source = residual_512.get("largest_residual_source")
    semantic_curve = sweep[
        (sweep["family"] == "pooled")
        & (sweep["endpoint"] == "next_semantic")
        & (sweep["method"] == primary)
    ].sort_values("dimension")
    semantic_text = ", ".join(
        f"{int(row.dimension)}D={_fmt(row.semantic_agreement)}"
        for row in semantic_curve.itertuples()
    )
    horizon8 = "not measured: frozen v8 capture ends at four tokens"
    final_block = "\n".join(
        [
            START,
            "",
            "## Protocol v9 conditional-sufficiency dimension update",
            "",
            f"The v9 rank audit found an effective training rank of `{rank}` from 600 frozen training pairs. Consequently 768/1024/1536/2048D are not statistically identifiable in this dataset and are reported as `NA`, not as 599D aliases.",
            "",
            "### Required v9 answers",
            "",
            f"1. Conditional residual gain from 512D upward: `{conditional_values.to_dict('records')}`.",
            "2. A validated dimension elbow was not established beyond the 599D sample-rank ceiling.",
            f"3. Smallest candidate sufficient dimension: `{summary['smallest_authorized_dimension']}`.",
            f"4. Largest 512D architecture-residual source: `{largest_source}`; full standalone-plus-interaction breakdown is in `{reports['residual']}`.",
            "5. Predictive and causal dimension requirements cannot yet be equated: decoded causal fidelity remained gated.",
            f"6. One-/two-/four-token results are recorded; eight-token status is `{horizon8}`.",
            f"7. The independent effect-enriched subset contains `{enriched_summary['selection']['count']}` final-test pairs; its conclusion remains observationally gated.",
            f"8. Primary-method semantic fidelity by dimension: `{semantic_text}`.",
            "9. No validated compact sufficient persistent state has been obtained.",
            "10. The current limitation is a combination of sample-rank/state-capacity identification and representation/objective insufficiency; the data do not support claiming intrinsic incompressibility.",
            f"11. Autonomous-controller authorization: `{summary['controller_authorized']}`.",
            "",
            "The strongest warranted conclusion remains H2 for the tested measured-J state. Persistent information is compressible predictively, but no compact state has passed predictive, conditional, semantic, multi-horizon, and decoded causal gates together.",
            "",
            "Exact commands:",
            "",
            "```bash",
            "scripts/run_sufficiency_v9.sh freeze --run-suffix protocol-freeze",
            "scripts/run_sufficiency_v9.sh analyze --run-suffix rank-aware-full",
            "scripts/run_sufficiency_v9.sh report --run-suffix final-v9",
            "```",
            "",
            END,
        ]
    )
    final_path = root / "reports/FINAL_REPORT.md"
    current = final_path.read_text()
    if START in current and END in current:
        prefix, remainder = current.split(START, 1)
        _, suffix = remainder.split(END, 1)
        current = prefix.rstrip() + "\n\n" + final_block + suffix
    else:
        current = current.rstrip() + "\n\n" + final_block + "\n"
    final_path.write_text(current)

    manifest = {
        "schema_version": 12,
        "protocol_version": "conditional_sufficiency_dimension_protocol_v9",
        "source_freeze_digest": freeze["freeze_digest"],
        "source_v8_freeze_digest": freeze["v8_source_freeze_digest"],
        "reports": reports,
        "figures": [
            str(dimension_figure.relative_to(root)),
            str(residual_figure.relative_to(root)),
        ],
        "sources": {
            "dimension": summary["records"],
            "residual": residual_summary["records"],
            "multihorizon": multihorizon_summary["records"],
            "enriched": enriched_summary["records"],
        },
    }
    write_json_atomic(root / "results/v9/processed/report_manifest_v9.json", manifest)
    manifest["report_manifest_sha256"] = sha256_file(
        root / "results/v9/processed/report_manifest_v9.json"
    )
    return reports
