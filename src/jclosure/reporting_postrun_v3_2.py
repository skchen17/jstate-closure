"""Post-run, provenance-complete reports for protocol v3.2.

The frozen :mod:`jclosure.reporting_v3_2` remains unchanged.  This additive
builder consumes its outputs plus the controller/reference summaries and adds
the final adjudication tables and provenance-complete figures.
"""

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
from jclosure.reporting_v3_2 import build as build_frozen_report


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def figure_provenance(root: Path, figure: Path, source: Path) -> dict[str, str]:
    if not figure.is_file() or not source.is_file():
        raise FileNotFoundError("figure provenance requires existing files")
    return {
        "figure": str(figure.relative_to(root)),
        "figure_sha256": sha256_file(figure),
        "source": str(source.relative_to(root)),
        "source_sha256": sha256_file(source),
    }


def _save(root: Path, figure: Path, source: Path, entries: list[dict[str, str]]) -> None:
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    entries.append(figure_provenance(root, figure, source))


def _controller_figures(
    root: Path, summary: pd.DataFrame, source: Path, entries: list[dict[str, str]]
) -> None:
    all_parseable = summary[summary["training_subset"] == "all_parseable"]
    history = all_parseable[
        all_parseable["model_family"].isin(["markov", "history"])
        & (all_parseable["horizon"] == 8)
    ].copy()
    if not history.empty:
        history.loc[history["model_family"] == "markov", "history_length"] = 0
        curve = history.groupby("history_length")["decoded_cosine_median"].median()
        plt.plot(curve.index, curve.values, marker="o")
        plt.xlabel("History length (0 = Markov)")
        plt.ylabel("Median horizon-8 decoded dense-J cosine")
        _save(
            root,
            root / "results/v3_2/figures/compact_memory_history_order.png",
            source,
            entries,
        )
    gru = all_parseable[all_parseable["model_family"] == "gru"]
    if not gru.empty:
        horizon8 = gru[gru["horizon"] == 8]
        curve = horizon8.groupby("memory_dimension")["decoded_cosine_median"].median()
        plt.plot(curve.index, curve.values, marker="o")
        plt.xlabel("GRU memory dimension")
        plt.ylabel("Median horizon-8 decoded dense-J cosine")
        _save(
            root,
            root / "results/v3_2/figures/compact_memory_dimension.png",
            source,
            entries,
        )
        for dimension, group in gru.groupby("memory_dimension", sort=True):
            curve = group.groupby("horizon")["decoded_cosine_median"].median()
            plt.plot(curve.index, curve.values, marker="o", label=str(int(dimension)))
        plt.xlabel("Autonomous rollout horizon")
        plt.ylabel("Median decoded dense-J cosine")
        plt.legend(title="Memory dim", fontsize=7)
        _save(
            root,
            root / "results/v3_2/figures/compact_memory_rollout_by_horizon.png",
            source,
            entries,
        )
        semantic = horizon8.groupby("memory_dimension")[[
            "teacher_action_fidelity", "ground_truth_action_accuracy"
        ]].median()
        semantic.plot(kind="bar", ax=plt.gca())
        plt.ylabel("Horizon-8 action metric")
        plt.xlabel("GRU memory dimension")
        _save(
            root,
            root / "results/v3_2/figures/teacher_vs_ground_truth_actions.png",
            source,
            entries,
        )


def _reference_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(
        (root / "results/v3_2/raw").glob(
            "compact-memory-references-v3-2-*/manifest.json"
        )
    ):
        manifest = _json(manifest_path)
        if not manifest or manifest.get("status") != "COMPLETED":
            continue
        if manifest.get("stage") != "references":
            continue
        result = _json(root / str(manifest["result"]))
        if not result:
            continue
        autonomous = result["autonomous_pca512_recurrent"]
        for value in autonomous["test"]:
            rows.append(
                {
                    "seed": result["seed"],
                    "reference_type": autonomous["reference_type"],
                    **value,
                }
            )
    return rows


def _memory_report(root: Path) -> str:
    screen_path = root / "results/v3_2/processed/compact_memory_representation_screen_v3_2.json"
    screen = _json(screen_path)
    analysis = _json(
        root / "results/v3_2/processed/compact_memory_controller_analysis_v3_2.json"
    )
    lines = ["# Compact-Memory Exploratory Results", ""]
    if not screen:
        lines.extend(["Status: representation screen NOT COMPLETED.", ""])
        return "\n".join(lines)
    selected = screen["overall_selected"]
    lines.extend(
        [
            "## Representation screen",
            "",
            f"Temporal training authorized: {screen['temporal_training_authorized']}.",
            (
                f"Selected candidate: {selected['dimension']}D "
                f"`{selected['representation_family']}`; validation next-state cosine "
                f"{selected['validation_next_state_cosine']:.6f}, reconstruction cosine "
                f"{selected['validation_reconstruction_cosine']:.6f}, Phase-0 pass@10 "
                f"retention {selected['phase0_pass10_retention']:.6f}, causal direction "
                f"retention {selected['causal_direction_retention']:.6f} over "
                f"{selected['causal_trials']} trials."
            ),
            "64D, 128D, and 256D candidates did not pass the frozen semantic/causal retention gates.",
            "",
        ]
    )
    if analysis:
        required_completed = analysis.get(
            "completed_all_parseable_controller_results",
            analysis["completed_controller_results"],
        )
        lines.extend(
            [
                "## Controller adjudication",
                "",
                f"Status: {analysis['status']}; completed {required_completed}/{analysis['expected_controller_results']} required all-parseable controller runs. "
                f"Completed sensitivity runs: {analysis.get('completed_sensitivity_controller_results', 0)}.",
                f"Minimum memory dimension passing the complete utility gate: {analysis['minimum_useful_memory_dimension']}.",
                f"H3 follow-up authorized: {analysis['h3_followup_authorized']} ({analysis['h3_followup_reason']}).",
                "",
                pd.DataFrame(analysis["memory_utility"]).to_markdown(index=False),
                "",
            ]
        )
        summary_path = root / analysis["summary_records"]
        summary = pd.read_parquet(summary_path)
        compact = summary[
            (summary["horizon"].isin([8, 16]))
            & (summary["training_subset"] == "all_parseable")
        ][
            [
                "model_family",
                "history_length",
                "memory_dimension",
                "seed",
                "horizon",
                "decoded_cosine_median",
                "trajectory_distance_mean",
                "teacher_action_fidelity",
                "ground_truth_action_accuracy",
                "time_to_divergence_median",
                "rollout_failure_rate",
            ]
        ]
        lines.extend(["## Autonomous rollout", "", compact.to_markdown(index=False), ""])
        reference = analysis.get("remainder_reference", {})
        if reference.get("per_seed"):
            lines.extend(
                [
                    "## Remainder-reference gap",
                    "",
                    "Teacher-current one-step references and the autonomous recurrent reference are distinct endpoints.",
                    "",
                    pd.DataFrame(reference["per_seed"]).to_markdown(index=False),
                    "",
                    f"Positive Markov-to-autonomous-reference gap: {reference['positive_markov_to_reference_gap']}; median reference-minus-baseline cosine: {reference['median_reference_minus_baseline']}.",
                    "",
                ]
            )
        sensitivity = summary[
            summary["training_subset"] == "teacher_correct_only"
        ]
        if not sensitivity.empty:
            lines.extend(
                [
                    "## Teacher-correct-only sensitivity",
                    "",
                    "This comparison is severely underpowered and is reported only as a quality sensitivity.",
                    "",
                    sensitivity.to_markdown(index=False),
                    "",
                ]
            )
    else:
        lines.extend(["Controller analysis: INCOMPLETE or not yet executed.", ""])
    references = _reference_rows(root)
    if references:
        lines.extend(
            [
                "## Autonomous remainder-aware reference",
                "",
                pd.DataFrame(references).to_markdown(index=False),
                "",
            ]
        )
    else:
        lines.extend(["Autonomous remainder-aware reference: NOT YET COMPLETED.", ""])
    counterfactual = root / "results/v3_2/processed/compact_memory_counterfactual_trajectories_v3_2.parquet"
    lines.extend(
        [
            "## Causal fidelity",
            "",
            (
                "Available for adjudication."
                if counterfactual.is_file()
                else "Unavailable: no validated token-time teacher J-swap trajectory exists. Observational rollout is not causal-fidelity evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _final_status(root: Path) -> str:
    calibration = _json(root / "results/v3_2/processed/closure_v3_2_calibration.json")
    causal = _json(root / "results/v3_2/processed/closure_v3_2_pilot.json")
    analysis = _json(
        root / "results/v3_2/processed/compact_memory_controller_analysis_v3_2.json"
    )
    authorized = bool(calibration and calibration.get("behavioral_authorized"))
    paired = int(causal.get("complete_paired_base_trials", 0)) if causal else 0
    useful = analysis.get("minimum_useful_memory_dimension") if analysis else None
    summary = None
    if analysis:
        summary_path = root / str(analysis["summary_records"])
        if summary_path.is_file():
            summary = pd.read_parquet(summary_path)
    history_answer = "Controller analysis is incomplete."
    teacher_answer = "Teacher imitation and ground-truth accuracy are reported separately."
    if summary is not None and not summary.empty:
        horizon8 = summary[
            (summary["training_subset"] == "all_parseable")
            & (summary["horizon"] == 8)
        ]
        markov = horizon8[horizon8["model_family"] == "markov"]
        histories = horizon8[horizon8["model_family"] == "history"]
        if not markov.empty and not histories.empty:
            markov_cosine = float(markov["decoded_cosine_median"].median())
            best_history = float(
                histories.groupby("history_length")["decoded_cosine_median"]
                .median()
                .max()
            )
            history_answer = (
                "No clear non-Markov advantage was observed: the median Markov "
                f"horizon-8 cosine was {markov_cosine:.6f}, versus a best "
                f"history-run value of {best_history:.6f}."
            )
        if not horizon8.empty:
            teacher = float(horizon8["teacher_action_fidelity"].median())
            ground_truth = float(horizon8["ground_truth_action_accuracy"].median())
            teacher_answer = (
                "They are distinct endpoints: across horizon-8 controller summaries, "
                f"median teacher-action fidelity was {teacher:.6f} and median "
                f"ground-truth action accuracy was {ground_truth:.6f}; only 13 "
                "teacher trajectories were teacher-correct."
            )
    questions = [
        ("1. Does a final-token same-J perturbation change the future?", "Not estimable without an authorized paired causal pilot." if not causal else "See the machine-recorded E_single estimates in CLOSURE_V3_2_CAUSAL.md."),
        ("2. How much effect does persistent-final remove?", "Not estimable." if not causal else "See M_final and its paired raw effects."),
        ("3. Does persistent-all remove additional effect?", "Not estimable." if not causal else "See M_all and the persistent-all raw effect."),
        ("4. Does measured-J act as the main mediation workspace?", "Undetermined; no mediation claim is made without a non-null E_single and valid restoration chains."),
        ("5. Is the current compact J state visibly non-Markov?", history_answer),
        ("6. Does compact recurrent memory improve autonomous rollout?", "No tested GRU memory dimension passed the frozen utility gate." if analysis and useful is None else f"Minimum fully passing memory dimension: {useful}."),
        ("7. What is the smallest useful memory dimension?", str(useful) if useful is not None else "None established."),
        ("8. Do teacher imitation and ground-truth accuracy agree?", teacher_answer),
        ("9. Which hypothesis is best supported?", "D. Paired causal restoration and causal-fidelity gates are not both complete."),
        ("10. Is 1M-100M controller scaling warranted next?", "No. The frozen recurrent-memory utility gate did not pass." if analysis and useful is None else "Only if the complete memory-utility and H3 follow-up gates pass."),
    ]
    lines = [
        "<!-- V3.2 POSTRUN START -->",
        "## Protocol v3.2 post-run adjudication",
        "",
        f"Causal calibration authorized: {authorized}. Paired causal base trials: {paired}.",
        f"Compact-memory utility minimum dimension: {useful}.",
        "Strongest warranted classification: **D** unless a later machine record completes every paired causal, autonomous-rollout, and causal-fidelity gate.",
        "",
    ]
    for question, answer in questions:
        lines.extend([f"### {question}", "", answer, ""])
    lines.append("<!-- V3.2 POSTRUN END -->")
    return "\n".join(lines)


def build(root: Path) -> None:
    build_frozen_report(root)
    entries: list[dict[str, str]] = []
    analysis = _json(
        root / "results/v3_2/processed/compact_memory_controller_analysis_v3_2.json"
    )
    if analysis:
        source = root / str(analysis["summary_records"])
        _controller_figures(root, pd.read_parquet(source), source, entries)
    (root / "reports/COMPACT_MEMORY_EXPLORATORY.md").write_text(
        _memory_report(root), encoding="utf-8"
    )
    final_path = root / "reports/FINAL_REPORT.md"
    text = final_path.read_text(encoding="utf-8")
    start, end = "<!-- V3.2 POSTRUN START -->", "<!-- V3.2 POSTRUN END -->"
    block = _final_status(root)
    if start in text and end in text:
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        text = prefix.rstrip() + "\n\n" + block + suffix
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    final_path.write_text(text, encoding="utf-8")
    base = _json(root / "results/v3_2/processed/figure_manifest_v3_2.json") or {}
    existing = [value for value in base.get("figures", []) if "figure" in value]
    for value in existing:
        figure = root / str(value["figure"])
        if figure.is_file():
            value["figure_sha256"] = sha256_file(figure)
    by_figure = {value["figure"]: value for value in [*existing, *entries]}
    write_json_atomic(
        root / "results/v3_2/processed/figure_manifest_v3_2.json",
        {
            "schema_version": 5,
            "protocol_version": "v3_2",
            "figures": [by_figure[key] for key in sorted(by_figure)],
        },
    )


def main() -> None:
    build(repository_root())


if __name__ == "__main__":
    main()
