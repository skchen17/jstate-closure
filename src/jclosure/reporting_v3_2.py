"""Machine-sourced reports and figures for causal/memory protocol v3.2."""

from __future__ import annotations

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


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _calibration(root: Path, figures: list[dict[str, Any]]) -> str:
    path = root / "results/v3_2/processed/closure_v3_2_calibration.json"
    value = _json(path)
    if value is None:
        return "# Closure v3.2 Calibration\n\nStatus: NOT EXECUTED.\n"
    rows = []
    for layer in value.get("layers", []):
        for restoration_layer, result in layer.get("runtime_conditional", {}).items():
            rows.append({
                "l1": layer["l1"], "method": layer["restoration_method"],
                "scope": layer["restoration_scope"], "restoration_layer": int(restoration_layer),
                "applicable": result["applicable"], "successes": result["successes"],
                "conditional_rate": result["rate"], "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"], "eligible": result["eligible"],
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        figure = root / "results/v3_2/figures/restoration_conditional_rates.png"
        for key, group in frame.groupby(["method", "scope"]):
            curve = group.groupby("restoration_layer")["conditional_rate"].mean()
            plt.plot(curve.index, curve.values, marker="o", label=f"{key[0]}:{key[1]}")
        plt.axhline(0.8, color="black", linestyle="--", label="frozen conditional gate")
        plt.xlabel("Restoration layer")
        plt.ylabel("P(valid restoration | valid initial intervention)")
        plt.legend(fontsize=7)
        _save(figure)
        figures.append({"figure": str(figure.relative_to(root)), "source": str(path.relative_to(root)), "source_sha256": sha256_file(path)})
    status = "AUTHORIZED" if value.get("behavioral_authorized") else "GATED"
    lines = [
        "# Closure v3.2 Calibration", "", f"Status: {status}.", "",
        f"Calibration records: {value.get('attempted', 0)}.",
        f"Authorized protocol: `{json.dumps(value.get('authorized_protocols', []), sort_keys=True)}`.", "",
        "Initial interventions use final-token scope only. Restoration rates use initial-valid trials as the conditional denominator.", "",
    ]
    if not frame.empty:
        lines.extend([frame.to_markdown(index=False), ""])
    return "\n".join(lines)


def _causal(root: Path, figures: list[dict[str, Any]]) -> str:
    summaries = []
    for domain in ("pilot", "confirmation"):
        path = root / f"results/v3_2/processed/closure_v3_2_{domain}.json"
        value = _json(path)
        if value is not None:
            summaries.append((domain, path, value))
    if not summaries:
        return "# Closure v3.2 Causal Results\n\nStatus: NOT EXECUTED OR NOT AUTHORIZED.\n"
    rows = []
    for domain, _, value in summaries:
        for condition, result in value.get("effects", {}).get("js_divergence", {}).items():
            rows.append({"domain": domain, "condition": condition, **result})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        figure = root / "results/v3_2/figures/closure_v3_2_js_effects.png"
        labels = frame["domain"] + ":" + frame["condition"]
        plt.bar(range(len(frame)), frame["estimate"])
        plt.errorbar(range(len(frame)), frame["estimate"], yerr=[frame["estimate"] - frame["lower"], frame["upper"] - frame["estimate"]], fmt="none", color="black")
        plt.xticks(range(len(frame)), labels, rotation=70, ha="right")
        plt.ylabel("Clean-relative full-vocabulary JS")
        _save(figure)
        figures.append({"figure": str(figure.relative_to(root)), "source": str(summaries[-1][1].relative_to(root)), "source_sha256": sha256_file(summaries[-1][1])})
    lines = ["# Closure v3.2 Causal Results", "", "Effects use common-valid paired base trials.", ""]
    for domain, _, value in summaries:
        lines.extend([
            f"## {domain.title()}", "",
            f"Complete paired base trials: {value.get('complete_paired_base_trials', 0)}.",
            f"Instrumentation gate: `{json.dumps(value.get('instrumentation_gate'), sort_keys=True)}`.",
            f"Mediation: `{json.dumps(value.get('mediation'), sort_keys=True)}`.", "",
        ])
    if not frame.empty:
        lines.extend([frame.to_markdown(index=False), ""])
    return "\n".join(lines)


def _trace_audit(root: Path, figures: list[dict[str, Any]]) -> str:
    path = root / "results/v3_2/processed/compact_memory_trace_audit_v3_2.json"
    value = _json(path)
    if value is None:
        return "# Compact-Memory Trace Audit\n\nStatus: NOT EXECUTED.\n"
    frame = pd.DataFrame(value.get("counts", []))
    if not frame.empty:
        figure = root / "results/v3_2/figures/compact_memory_trace_quality.png"
        grouped = frame.groupby("family")[["attempted", "parseable", "teacher_correct"]].sum()
        grouped.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Trajectories")
        _save(figure)
        figures.append({"figure": str(figure.relative_to(root)), "source": str(path.relative_to(root)), "source_sha256": sha256_file(path)})
    lines = [
        "# Compact-Memory Trace Audit", "",
        f"Canonical records: {value.get('total_records', 0)}.",
        f"Representation screen authorized: {value.get('representation_screen_authorized', False)}.",
        f"Duplicate IDs: {len(value.get('duplicate_example_ids', []))}; split overlaps: {len(value.get('split_overlap_ids', []))}; invalid tensors: {value.get('corrupted_or_invalid_traces', 0)}.", "",
        "`teacher_correct` is ground-truth trajectory correctness; parseable-but-wrong traces remain available only for teacher-dynamics imitation.", "",
    ]
    if not frame.empty:
        lines.extend([frame.to_markdown(index=False), ""])
    return "\n".join(lines)


def _memory(root: Path, figures: list[dict[str, Any]]) -> str:
    screen_path = root / "results/v3_2/processed/compact_memory_representation_screen_v3_2.json"
    screen = _json(screen_path)
    lines = ["# Compact-Memory Exploratory Results", ""]
    if screen is None:
        lines.append("Status: representation screen NOT EXECUTED.")
        return "\n".join(lines) + "\n"
    records = pd.read_parquet(root / screen["records"])
    lines.extend([
        f"Temporal training authorized: {screen.get('temporal_training_authorized', False)}.",
        f"Selected representation: `{json.dumps(screen.get('overall_selected'), sort_keys=True)}`.", "",
        records.to_markdown(index=False), "",
    ])
    if not records.empty:
        figure = root / "results/v3_2/figures/compact_memory_representation_screen.png"
        for family, group in records.groupby("representation_family"):
            plt.plot(group["dimension"], group["validation_next_state_cosine"], marker="o", label=family)
        plt.xlabel("State dimension")
        plt.ylabel("Validation next-state cosine")
        plt.legend()
        _save(figure)
        figures.append({"figure": str(figure.relative_to(root)), "source": str(screen_path.relative_to(root)), "source_sha256": sha256_file(screen_path)})
    rollout_rows = []
    for manifest_path in sorted((root / "results/v3_2/raw").glob("compact-memory-v3-2-*/manifest.json")):
        manifest = _json(manifest_path)
        if not manifest or manifest.get("status") != "COMPLETED" or manifest.get("stage") != "train":
            continue
        result = _json(root / manifest["result"])
        if result:
            for row in result.get("test", {}).get("horizons", []):
                rollout_rows.append({
                    "model_family": result["model_family"], "history_length": result.get("history_length"),
                    "memory_dimension": result.get("memory_dimension"), "seed": result["seed"],
                    "training_subset": result["training_subset"], **row,
                })
    if rollout_rows:
        rollout = pd.DataFrame(rollout_rows)
        lines.extend(["## Autonomous rollout", "", rollout.to_markdown(index=False), ""])
        figure = root / "results/v3_2/figures/compact_memory_autonomous_rollout.png"
        for key, group in rollout.groupby(["model_family", "history_length", "memory_dimension"], dropna=False):
            curve = group.groupby("horizon")["decoded_cosine_median"].median()
            plt.plot(curve.index, curve.values, marker="o", label=str(key))
        plt.xlabel("Autonomous horizon")
        plt.ylabel("Decoded dense-J cosine")
        plt.legend(fontsize=6)
        _save(figure)
    else:
        lines.append("Autonomous controller results: NOT EXECUTED or gated.")
    return "\n".join(lines) + "\n"


def _update_final(root: Path, calibration: dict[str, Any] | None, memory: dict[str, Any] | None) -> None:
    path = root / "reports/FINAL_REPORT.md"
    text = path.read_text(encoding="utf-8")
    headline = "> **J measurement is validated, but no causal restoration protocol or compact recurrent state has yet satisfied the complete behavioral criteria required to distinguish H1/H2/H3.**"
    lines = text.splitlines()
    if len(lines) > 2 and lines[2].startswith("> **"):
        lines[2] = headline
    marker_start = "<!-- V3.2 STATUS START -->"
    marker_end = "<!-- V3.2 STATUS END -->"
    block = "\n".join([
        marker_start, "## Protocol v3.2 status", "",
        f"Part A behavioral authorization: {'AUTHORIZED' if calibration and calibration.get('behavioral_authorized') else 'GATED OR NOT EVALUATED'}.",
        f"Part B temporal training authorization: {'AUTHORIZED' if memory and memory.get('temporal_training_authorized') else 'GATED OR NOT EVALUATED'}.",
        "No H1/H2/H3 classification is upgraded without paired causal and autonomous-rollout evidence.", marker_end,
    ])
    joined = "\n".join(lines)
    if marker_start in joined and marker_end in joined:
        prefix, rest = joined.split(marker_start, 1)
        _, suffix = rest.split(marker_end, 1)
        joined = prefix.rstrip() + "\n\n" + block + suffix
    else:
        joined = joined.rstrip() + "\n\n" + block + "\n"
    path.write_text(joined, encoding="utf-8")


def build(root: Path) -> None:
    figures: list[dict[str, Any]] = []
    calibration_text = _calibration(root, figures)
    causal_text = _causal(root, figures)
    trace_text = _trace_audit(root, figures)
    memory_text = _memory(root, figures)
    (root / "reports/CLOSURE_V3_2_CALIBRATION.md").write_text(calibration_text, encoding="utf-8")
    (root / "reports/CLOSURE_V3_2_CAUSAL.md").write_text(causal_text, encoding="utf-8")
    (root / "reports/COMPACT_MEMORY_TRACE_AUDIT.md").write_text(trace_text, encoding="utf-8")
    (root / "reports/COMPACT_MEMORY_EXPLORATORY.md").write_text(memory_text, encoding="utf-8")
    calibration = _json(root / "results/v3_2/processed/closure_v3_2_calibration.json")
    memory = _json(root / "results/v3_2/processed/compact_memory_representation_screen_v3_2.json")
    _update_final(root, calibration, memory)
    write_json_atomic(root / "results/v3_2/processed/figure_manifest_v3_2.json", {"schema_version": 5, "protocol_version": "v3_2", "figures": figures})


def main() -> None:
    build(repository_root())


if __name__ == "__main__":
    main()
