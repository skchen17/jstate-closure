"""Machine-derived reports for corrected causal-sufficiency protocol v10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.protocol_v10 import SCHEMA_VERSION_V10
from jclosure.provenance import sha256_file, write_json_atomic

START = "<!-- V10-RESULTS:START -->"
END = "<!-- V10-RESULTS:END -->"


def _ci_text(value: dict[str, Any] | None) -> str:
    if value is None:
        return "NA"
    return f"{value['estimate']:.4f} [{value['lower']:.4f}, {value['upper']:.4f}]"


def _aggregate_frame(causal: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dimension, horizons in causal["aggregates"].items():
        for horizon, groups in horizons.items():
            for group, metrics in groups.items():
                rows.append(
                    {
                        "dimension": int(dimension),
                        "horizon": int(horizon),
                        "group": group,
                        "count": metrics["count"],
                        "direction": _ci_text(metrics["direction_cosine"]),
                        "magnitude": _ci_text(metrics["magnitude_ratio"]),
                        "semantic_delta": _ci_text(
                            metrics["semantic_delta_agreement"]
                        ),
                        "output_direction": _ci_text(
                            metrics["output_direction_cosine"]
                        ),
                        "task_sign": _ci_text(
                            metrics["task_decision_sign_agreement"]
                        ),
                        "effect_weighted_direction": metrics[
                            "effect_weighted_direction_cosine"
                        ],
                        "gate_pass": metrics["causal_gate_pass"],
                    }
                )
    return pd.DataFrame(rows)


def build_reports(
    root: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    candidates: dict[str, Any],
) -> dict[str, str]:
    sweep_summary = json.loads(
        (root / "results/v10/processed/corrected_sub512_sufficiency_v10.json").read_text()
    )
    residual_summary = json.loads(
        (root / "results/v10/processed/residual_localization_audit_v10.json").read_text()
    )
    semantic_summary = json.loads(
        (root / "results/v10/processed/semantic_sufficiency_audit_v10.json").read_text()
    )
    causal = json.loads(
        (root / "results/v10/processed/decoded_causal_state_validation_v10.json").read_text()
    )
    sweep = pd.read_parquet(root / sweep_summary["records"])
    residual = pd.read_parquet(root / residual_summary["records"])
    semantic = pd.read_parquet(root / semantic_summary["records"])
    causal_records = pd.read_parquet(root / causal["records"])
    aggregate = _aggregate_frame(causal)
    primary = config["causal_sufficiency_v10"]["primary_method"]
    reports: dict[str, str] = {}

    residual_final = residual[
        (residual["split"] == "final_test") & (residual["family"] == "pooled")
    ][["dimension", "source", "conditional_gain", "lower", "upper"]]
    residual_text = "\n".join(
        [
            "# Residual Localization Audit v10",
            "",
            f"Base freeze: `{freeze['freeze_digest']}`; candidate freeze: `{candidates['freeze_digest']}`.",
            "",
            "The v9 architecture-localization comparator was not mathematically identical to the corrected joint comparator. It appended raw block scores after compact coordinates and therefore retained compact information inside each block. v10 uses five-fold out-of-fold residualization on training rows and train-only residualization on held-out rows.",
            "",
            "`combined_reference` is the corrected residual inside the same 599D combined ceiling. `architecture_joint` uses the richer 3x599 block-specific score space, so it may remain non-zero even when the combined-reference residual is near zero.",
            "",
            residual_final.to_markdown(index=False),
        ]
    )
    path = root / "reports/RESIDUAL_LOCALIZATION_AUDIT_V10.md"
    path.write_text(residual_text + "\n", encoding="utf-8")
    reports["residual"] = str(path.relative_to(root))

    curve = sweep[
        (sweep["split"] == "final_test") & (sweep["family"] == "pooled")
    ][
        [
            "method",
            "dimension",
            "baseline_score",
            "candidate_score",
            "ceiling_score",
            "predictive_gap_closed",
            "conditional_residual_gain",
            "conditional_lower",
            "conditional_upper",
            "observational_pass",
        ]
    ].sort_values(["method", "dimension"])
    corrected_text = "\n".join(
        [
            "# Corrected Sub-512 Sufficiency v10",
            "",
            "All dimensions use the same corrected full-information comparator. Candidate selection used validation only; final-test rows below are confirmatory and did not change the dimensions.",
            "",
            curve.to_markdown(index=False),
            "",
            f"Validation all-family eligible dimensions: `{sweep_summary['validation_all_family_eligible_dimensions']}`.",
            f"Smallest observationally sufficient dimension: `{sweep_summary['smallest_observationally_sufficient_dimension']}`.",
            f"Frozen decoded-causal candidates: `{candidates['candidate_dimensions']}`.",
        ]
    )
    path = root / "reports/CORRECTED_SUB512_SUFFICIENCY_V10.md"
    path.write_text(corrected_text + "\n", encoding="utf-8")
    reports["sub512"] = str(path.relative_to(root))

    semantic_table = semantic[
        (semantic["split"] == "final_test")
        & (semantic["family"] == "pooled")
        & (semantic["method"] == primary)
    ][
        [
            "dimension",
            "semantic_baseline",
            "semantic_compact",
            "semantic_full_ceiling",
            "semantic_retention_relative_to_full",
            "semantic_residual_gain",
            "semantic_residual_lower",
            "semantic_residual_upper",
            "semantic_relative_pass",
        ]
    ].sort_values("dimension")
    semantic_text = "\n".join(
        [
            "# Semantic Sufficiency Audit v10",
            "",
            "Absolute semantic quality and relative semantic sufficiency are separate gates. The absolute probe ceiling is reported without requiring compact state to exceed it; relative sufficiency uses compact/full retention and full-minus-compact residual gain.",
            "",
            semantic_table.to_markdown(index=False),
        ]
    )
    path = root / "reports/SEMANTIC_SUFFICIENCY_AUDIT_V10.md"
    path.write_text(semantic_text + "\n", encoding="utf-8")
    reports["semantic"] = str(path.relative_to(root))

    causal_table = aggregate[aggregate["group"].isin(["pooled", "effect_enriched"])]
    causal_text = "\n".join(
        [
            "# Decoded Causal State Validation v10",
            "",
            f"Strict interface audit: `{causal['strict_interface']}`.",
            "",
            causal_table.to_markdown(index=False),
            "",
            f"Authorization: `{causal['authorization']}`.",
            f"Smallest candidate causal sufficient dimension: `{causal['smallest_candidate_causal_sufficient_dimension']}`.",
        ]
    )
    path = root / "reports/DECODED_CAUSAL_STATE_VALIDATION_V10.md"
    path.write_text(causal_text + "\n", encoding="utf-8")
    reports["causal"] = str(path.relative_to(root))

    horizon_text = "\n".join(
        [
            "# Long-Horizon State Sufficiency v10",
            "",
            "All listed results are newly captured teacher-forced trajectories; h8/h16 are measured rather than extrapolated from h4. Limited free continuation remains separately gated.",
            "",
            aggregate[aggregate["group"] == "pooled"].to_markdown(index=False),
            "",
            f"Free-continuation executed: `{causal['free_continuation_executed']}`.",
        ]
    )
    path = root / "reports/LONG_HORIZON_STATE_SUFFICIENCY_V10.md"
    path.write_text(horizon_text + "\n", encoding="utf-8")
    reports["horizon"] = str(path.relative_to(root))

    enriched_records = causal_records[causal_records["effect_enriched"]]
    enriched_text = "\n".join(
        [
            "# Behavior-Enriched Confirmatory v10",
            "",
            f"Frozen independent selection contains `{len(candidates['effect_enriched_base_trial_ids'])}` pairs. IDs were frozen before decoded outcomes; encoder fitting and dimension selection use no final-test rows.",
            "",
            aggregate[aggregate["group"] == "effect_enriched"].to_markdown(index=False),
            "",
            f"Machine-record rows: `{len(enriched_records)}`.",
        ]
    )
    path = root / "reports/BEHAVIOR_ENRICHED_CONFIRMATORY_V10.md"
    path.write_text(enriched_text + "\n", encoding="utf-8")
    reports["enriched"] = str(path.relative_to(root))

    dim512 = residual_final[residual_final["dimension"] == 512]
    residual_values = {
        str(row.source): {
            "estimate": float(row.conditional_gain),
            "lower": float(row.lower),
            "upper": float(row.upper),
        }
        for row in dim512.itertuples()
    }
    causal_dimension = causal["smallest_candidate_causal_sufficient_dimension"]
    adjudication = (
        "H3 receives direct causal-interface support"
        if causal_dimension is not None
        else "H2 remains strongest; H3 is not established"
    )
    final_block = "\n".join(
        [
            START,
            "",
            "## Protocol v10 corrected causal-sufficiency update",
            "",
            f"Base freeze `{freeze['freeze_digest']}`; candidate freeze `{candidates['freeze_digest']}`.",
            "",
            "### Required v10 answers",
            "",
            "1. v9 architecture residual localization had coordinate-duplication/comparator mismatch; it was not the corrected joint conditional estimand.",
            f"2. Corrected 512D residual breakdown: `{residual_values}`.",
            f"3. Smallest observationally sufficient dimension: `{sweep_summary['smallest_observationally_sufficient_dimension']}`.",
            "4. 512D is retained as a strong reference, not assumed to be the minimum.",
            "5. Semantic failure is adjudicated relative to the full semantic ceiling; absolute probe quality and representation retention are reported separately.",
            f"6. Frozen compact candidates decoded to architecture-aligned recurrent/conv/KV state: `{candidates['candidate_dimensions']}`.",
            f"7. Decoded teacher-intervention authorization: `{causal['authorization']}`.",
            f"8. Newly measured h1/h2/h4/h8/h16 results are in `{reports['horizon']}`.",
            f"9. Effect-enriched results are in `{reports['enriched']}`.",
            f"10. Candidate causal sufficient state: `{causal_dimension}`.",
            "11. Failure modes, where present, are identified by decoded direction/magnitude/semantic/output gates rather than inferred from observational prediction.",
            f"12. Autonomous-controller authorization: `{causal['controller_authorized']}`.",
            "",
            f"H2/H3 adjudication: **{adjudication}**.",
            "",
            "Exact commands:",
            "",
            "```bash",
            "scripts/run_causal_sufficiency_v10.sh freeze --run-suffix protocol-freeze",
            "scripts/run_causal_sufficiency_v10.sh audit --run-suffix corrected-audit",
            "scripts/run_causal_sufficiency_v10.sh freeze-candidates --run-suffix candidate-freeze",
            "scripts/run_causal_sufficiency_v10.sh prepare-decoder --device 0 --run-suffix decoder",
            "scripts/run_causal_sufficiency_v10.sh causal --device 0 --run-suffix causal-confirmatory",
            "scripts/run_causal_sufficiency_v10.sh report --run-suffix final-v10",
            "```",
            "",
            END,
        ]
    )
    final_path = root / "reports/FINAL_REPORT.md"
    current = final_path.read_text(encoding="utf-8")
    if START in current and END in current:
        prefix, remainder = current.split(START, 1)
        _, suffix = remainder.split(END, 1)
        current = prefix.rstrip() + "\n\n" + final_block + suffix
    else:
        current = current.rstrip() + "\n\n" + final_block + "\n"
    final_path.write_text(current, encoding="utf-8")
    reports["final"] = "reports/FINAL_REPORT.md"

    figure_root = root / "results/v10/figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, values in curve.groupby("method", sort=True):
        axes[0].plot(
            values["dimension"], values["predictive_gap_closed"], marker="o", label=method
        )
        axes[1].plot(
            values["dimension"], values["conditional_residual_gain"], marker="o", label=method
        )
    axes[0].axhline(
        config["causal_sufficiency_v10"]["predictive_gap_closed_minimum"],
        color="black",
        linestyle="--",
    )
    axes[1].axhline(
        config["causal_sufficiency_v10"]["conditional_residual_gain_maximum"],
        color="black",
        linestyle="--",
    )
    axes[0].set_ylabel("predictive gap closed")
    axes[1].set_ylabel("corrected conditional residual gain")
    for axis in axes:
        axis.set_xlabel("dimension")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    curve_figure = figure_root / "corrected_sub512_curve_v10.png"
    figure.savefig(curve_figure, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    pooled = aggregate[aggregate["group"] == "pooled"]
    for dimension, values in pooled.groupby("dimension", sort=True):
        direction = [
            causal["aggregates"][str(dimension)][str(horizon)]["pooled"][
                "direction_cosine"
            ]["estimate"]
            for horizon in values["horizon"]
        ]
        axis.plot(values["horizon"], direction, marker="o", label=f"{dimension}D")
    axis.axhline(
        config["causal_sufficiency_v10"]["causal"]["direction_cosine_minimum"],
        color="black",
        linestyle="--",
    )
    axis.set_xlabel("teacher-forced horizon")
    axis.set_ylabel("decoded/teacher J-effect direction cosine")
    axis.legend()
    figure.tight_layout()
    causal_figure = figure_root / "decoded_causal_horizon_v10.png"
    figure.savefig(causal_figure, dpi=180)
    plt.close(figure)

    manifest = {
        "schema_version": SCHEMA_VERSION_V10,
        "protocol_version": freeze["protocol_version"],
        "source_freeze_digest": freeze["freeze_digest"],
        "candidate_freeze_digest": candidates["freeze_digest"],
        "reports": reports,
        "figures": [
            str(curve_figure.relative_to(root)),
            str(causal_figure.relative_to(root)),
        ],
        "sources": {
            "sweep": sweep_summary["records"],
            "residual": residual_summary["records"],
            "semantic": semantic_summary["records"],
            "causal": causal["records"],
        },
    }
    manifest_path = root / "results/v10/processed/report_manifest_v10.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return reports
