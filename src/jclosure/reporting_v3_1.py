"""Build v3.1 reports and figures exclusively from saved machine records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _calibration(root: Path, figures: list[dict[str, Any]]) -> str:
    summary_path = root / "results/v3_1/processed/closure_v3_1_calibration.json"
    summary = _json(summary_path)
    if summary is None:
        return "# Closure v3.1 Calibration\n\nStatus: NOT EXECUTED. No calibration summary exists.\n"
    rows = pd.DataFrame(summary.get("layers", []))
    if not rows.empty:
        figure = root / "results/v3_1/figures/closure_v3_1_calibration_validity.png"
        for scope, group in rows.groupby("position_scope"):
            plt.plot(group["l1"], group["intervention_valid"], marker="o", label=scope)
        plt.axhline(160, color="black", linestyle="--", label="frozen 160/200 gate")
        plt.xlabel("Direct intervention layer L1")
        plt.ylabel("Strict-valid initial interventions")
        plt.legend()
        _save_figure(figure)
        figures.append(
            {
                "figure": str(figure.relative_to(root)),
                "source": str(summary_path.relative_to(root)),
                "source_sha256": sha256_file(summary_path),
            }
        )
    status = "AUTHORIZED" if summary.get("behavioral_authorized") else "GATED"
    lines = [
        "# Closure v3.1 Calibration",
        "",
        f"Status: {status}.",
        "",
        f"Attempted layer/scope records: {summary.get('attempted', 0)}.",
        f"Selected protocols: `{json.dumps(summary.get('authorized_protocols', []), sort_keys=True)}`.",
        "",
        "Intervention eligibility retains the frozen ≥0.20 displacement requirement. Later restoration eligibility does not impose a minimum correction magnitude.",
        "",
    ]
    if not rows.empty:
        lines.extend([rows.to_markdown(index=False), ""])
    return "\n".join(lines)


def _causal(root: Path, figures: list[dict[str, Any]]) -> str:
    summaries = []
    for domain in ("pilot", "confirmation"):
        path = root / f"results/v3_1/processed/closure_v3_1_{domain}.json"
        value = _json(path)
        if value is not None:
            summaries.append((domain, path, value))
    if not summaries:
        return "# Closure v3.1 Causal Results\n\nStatus: NOT EXECUTED OR NOT AUTHORIZED. No paired causal summary exists.\n"
    records = []
    for domain, _, value in summaries:
        for condition, effect in value.get("effects", {}).items():
            records.append({"domain": domain, "condition": condition, **effect})
    frame = pd.DataFrame(records)
    if not frame.empty:
        figure = root / "results/v3_1/figures/closure_v3_1_js_effects.png"
        labels = frame["domain"] + ":" + frame["condition"]
        plt.bar(range(len(frame)), frame["estimate"])
        plt.errorbar(
            range(len(frame)),
            frame["estimate"],
            yerr=[
                frame["estimate"] - frame["lower"],
                frame["upper"] - frame["estimate"],
            ],
            fmt="none",
            color="black",
            capsize=2,
        )
        plt.xticks(range(len(frame)), labels, rotation=70, ha="right")
        plt.ylabel("Clean-relative full-vocabulary JS divergence")
        _save_figure(figure)
        for _, path, _ in summaries:
            figures.append(
                {
                    "figure": str(figure.relative_to(root)),
                    "source": str(path.relative_to(root)),
                    "source_sha256": sha256_file(path),
                }
            )
    lines = [
        "# Closure v3.1 Causal Results",
        "",
        "All effects below are paired, clean-relative, and restricted to common-valid trials.",
        "",
    ]
    for domain, _, value in summaries:
        lines.extend(
            [
                f"## {domain.title()}",
                "",
                f"Complete paired base trials: {value.get('complete_paired_base_trials', 0)}.",
                f"Numerical-null threshold: {value.get('null_threshold')}.",
                f"Mediation: `{json.dumps(value.get('mediation'), sort_keys=True)}`.",
                "",
            ]
        )
    if not frame.empty:
        lines.extend([frame.to_markdown(index=False), ""])
    return "\n".join(lines)


def _memory(root: Path, figures: list[dict[str, Any]]) -> str:
    trace_path = root / "results/v3_1/processed/compact_memory_trace_summary.json"
    screen_path = (
        root / "results/v3_1/processed/compact_memory_representation_screen.json"
    )
    trace = _json(trace_path)
    screen = _json(screen_path)
    lines = ["# Compact-Memory Exploratory Results", ""]
    if trace is None:
        lines.append("Status: NOT EXECUTED. No teacher-trace summary exists.")
        return "\n".join(lines) + "\n"
    count_frame = pd.DataFrame(trace.get("counts", []))
    lines.extend(
        [
            f"Representation screen authorized by trace gate: {trace.get('representation_screen_authorized', False)}.",
            "",
        ]
    )
    if not count_frame.empty:
        lines.extend([count_frame.to_markdown(index=False), ""])
        figure = root / "results/v3_1/figures/compact_memory_trace_attrition.png"
        labels = count_frame["split"] + ":" + count_frame["family"]
        plt.bar(range(len(count_frame)), count_frame["attempted"], label="attempted")
        plt.bar(range(len(count_frame)), count_frame["parseable"], label="parseable")
        plt.xticks(range(len(count_frame)), labels, rotation=60, ha="right")
        plt.legend()
        _save_figure(figure)
        figures.append(
            {
                "figure": str(figure.relative_to(root)),
                "source": str(trace_path.relative_to(root)),
                "source_sha256": sha256_file(trace_path),
            }
        )
    if screen is None:
        lines.append("Representation screen: NOT EXECUTED or gated by trace attrition.")
        return "\n".join(lines) + "\n"
    screen_records = root / screen["records"]
    frame = pd.read_parquet(screen_records)
    lines.extend(
        [
            f"Temporal training authorized: {screen.get('temporal_training_authorized', False)}.",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )
    if not frame.empty:
        figure = root / "results/v3_1/figures/compact_memory_representation_screen.png"
        for family, group in frame.groupby("representation_family"):
            plt.plot(
                group["dimension"],
                group["validation_next_state_cosine"],
                marker="o",
                label=family,
            )
        plt.xlabel("State dimension")
        plt.ylabel("Validation one-step decoded cosine")
        plt.legend()
        _save_figure(figure)
        figures.append(
            {
                "figure": str(figure.relative_to(root)),
                "source": str(screen_records.relative_to(root)),
                "source_sha256": sha256_file(screen_records),
            }
        )
    results = []
    for manifest_path in sorted(
        (root / "results/v3_1/raw").glob("compact-memory-v3-1-*/manifest.json")
    ):
        manifest = _json(manifest_path)
        if (
            not manifest
            or manifest.get("status") != "COMPLETED"
            or manifest.get("stage") != "train"
        ):
            continue
        result_path = root / manifest["result"]
        value = _json(result_path)
        if value:
            for horizon in value.get("test", {}).get("horizons", []):
                results.append(
                    {
                        "model_family": value["model_family"],
                        "state_dimension": value["state_dimension"],
                        "memory_dimension": value.get("memory_dimension"),
                        "history_length": value.get("history_length"),
                        "seed": value["seed"],
                        **horizon,
                    }
                )
    if results:
        result_frame = pd.DataFrame(results)
        lines.extend(
            ["## Autonomous rollout", "", result_frame.to_markdown(index=False), ""]
        )
        figure = root / "results/v3_1/figures/compact_memory_rollout_horizon.png"
        for key, group in result_frame.groupby(
            ["model_family", "state_dimension", "memory_dimension", "history_length"],
            dropna=False,
        ):
            curve = group.groupby("horizon")["decoded_cosine_median"].median()
            plt.plot(curve.index, curve.values, marker="o", label=str(key))
        plt.xlabel("Autonomous horizon")
        plt.ylabel("Decoded dense-J cosine")
        plt.legend(fontsize=6)
        _save_figure(figure)
        figures.append(
            {
                "figure": str(figure.relative_to(root)),
                "source": "controller_result.json files",
                "source_sha256": "see per-run manifests",
            }
        )
    else:
        lines.append("Autonomous controller training: NOT EXECUTED or gated.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(repository_root()))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    figures: list[dict[str, Any]] = []
    calibration = _calibration(root, figures)
    causal = _causal(root, figures)
    memory = _memory(root, figures)
    (root / "reports/CLOSURE_V3_1_CALIBRATION.md").write_text(
        calibration, encoding="utf-8"
    )
    (root / "reports/CLOSURE_V3_1_CAUSAL.md").write_text(causal, encoding="utf-8")
    (root / "reports/COMPACT_MEMORY_EXPLORATORY.md").write_text(
        memory, encoding="utf-8"
    )
    final_path = root / "reports/FINAL_REPORT.md"
    final = final_path.read_text(encoding="utf-8")
    status_line = "> **No operational compact state has yet passed the complete causal/behavioral criteria required to distinguish H1/H2/H3.**"
    if status_line not in final:
        heading, remainder = final.split("\n", 1)
        final = f"{heading}\n\n{status_line}\n{remainder}"
    start = "<!-- V3.1 STATUS START -->"
    end = "<!-- V3.1 STATUS END -->"
    calibration_summary = _json(
        root / "results/v3_1/processed/closure_v3_1_calibration.json"
    )
    trace_summary = _json(
        root / "results/v3_1/processed/compact_memory_trace_summary.json"
    )
    v31_block = "\n".join(
        [
            start,
            "## Protocol v3.1 status",
            "",
            f"Part A behavioral authorization: {bool(calibration_summary and calibration_summary.get('behavioral_authorized'))}.",
            f"Part B representation-screen authorization: {bool(trace_summary and trace_summary.get('representation_screen_authorized'))}.",
            "Exact counts, effects, confidence intervals, and attrition are generated in the three v3.1 protocol reports. No classification is upgraded when a required gate is absent.",
            end,
        ]
    )
    if start in final and end in final:
        prefix, remainder = final.split(start, 1)
        _, suffix = remainder.split(end, 1)
        final = prefix.rstrip() + "\n\n" + v31_block + suffix
    else:
        final = final.rstrip() + "\n\n" + v31_block + "\n"
    final_path.write_text(final, encoding="utf-8")
    write_json_atomic(
        root / "results/v3_1/processed/figure_manifest_v3_1.json",
        {"schema_version": 4, "figures": figures},
    )


if __name__ == "__main__":
    main()
