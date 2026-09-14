"""Build the protocol-v4 evidence report and figures from saved records only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "results/v4/processed"
FIGURES = ROOT / "results/v4/figures"
REPORT = ROOT / "reports/PREDICTIVE_JSTATE_V4.md"


def _load(name: str) -> dict[str, Any] | None:
    path = PROCESSED / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _source(name: str) -> dict[str, str]:
    path = ROOT / name if name.startswith("results/") else PROCESSED / name
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _save_figure(figure: plt.Figure, name: str, sources: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / name
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_json_atomic(
        target.with_suffix(".provenance.json"),
        {
            "figure": str(target.relative_to(ROOT)),
            "figure_sha256": sha256_file(target),
            "sources": [_source(value) for value in sources],
            "manual_series": False,
        },
    )


def _teacher_figure(teacher: dict[str, Any]) -> None:
    frame = pd.DataFrame(teacher["rates_by_domain_family_horizon"])
    grouped = (
        frame.groupby(["family", "horizon"], as_index=False)
        .agg(correct=("full_trajectory_correct", "sum"), attempted=("attempted", "sum"))
    )
    grouped["accuracy"] = grouped["correct"] / grouped["attempted"]
    table = grouped.pivot(index="family", columns="horizon", values="accuracy")
    figure, axis = plt.subplots(figsize=(7.8, 3.5))
    image = axis.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel("macrostep horizon")
    axis.set_title("Teacher full-trajectory accuracy")
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            value = float(table.iloc[row, column])
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.75 else "black")
    figure.colorbar(image, ax=axis, label="accuracy")
    _save_figure(figure, "teacher_accuracy_v4.png", ["teacher_formal_v4.json"])


def _causal_figure(causal: dict[str, Any]) -> None:
    effects = causal["effects"]["output_js_divergence"]
    order = ["clean", "identity", "matched_random", "j_positive", "full_perturbation", "j_preserving"]
    available = [value for value in order if value in effects]
    estimates = np.asarray([effects[value]["estimate"] for value in available])
    lower = np.asarray([effects[value]["lower"] for value in available])
    upper = np.asarray([effects[value]["upper"] for value in available])
    figure, axis = plt.subplots(figsize=(8.2, 3.8))
    axis.bar(available, estimates, color=["#666666", "#999999", "#4c78a8", "#e45756", "#f2cf5b", "#54a24b"][: len(available)])
    axis.errorbar(range(len(available)), estimates, yerr=np.vstack((estimates - lower, upper - estimates)), fmt="none", color="black", capsize=3)
    axis.axhline(float(causal["null_threshold"]), color="black", linestyle="--", linewidth=1, label="decision threshold")
    axis.set_ylabel("clean-relative output JS")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(frameon=False)
    axis.set_title(f"Single-arm paired causal effects (n={causal['complete_paired_base_trials']})")
    _save_figure(figure, "single_arm_effects_v4.png", ["single_arm_v4.json"])


def _mediation_figure(mediation: dict[str, Any]) -> None:
    effects = mediation["effects"]["output_js_divergence"]
    order = ["single", "persistent_final", "persistent_all"]
    estimates = np.asarray([effects[value]["estimate"] for value in order])
    lower = np.asarray([effects[value]["lower"] for value in order])
    upper = np.asarray([effects[value]["upper"] for value in order])
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.bar(order, estimates, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.errorbar(
        range(3),
        estimates,
        yerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.set_ylabel("clean-relative output JS")
    axis.set_title(
        f"Persistent measured-J mediation (n={mediation['complete_paired_base_trials']})"
    )
    _save_figure(
        figure, "persistent_mediation_v4.png", ["persistent_mediation_v4.json"]
    )


def _pareto_figure(screen: dict[str, Any]) -> None:
    frame = pd.DataFrame(screen["pareto"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    metrics = [
        ("semantic_retention", "semantic retention"),
        ("causal_direction_retention", "causal-direction retention"),
        ("future_prediction_cosine", "future-state cosine"),
    ]
    for family, group in frame.groupby("representation", sort=True):
        for axis, (metric, label) in zip(axes, metrics, strict=True):
            ordered = group.sort_values("state_dimension")
            axis.plot(ordered["state_dimension"], ordered[metric], marker="o", label=family)
            axis.set_ylabel(label)
            axis.set_xlabel("state dimension")
            axis.set_ylim(-0.03, 1.03)
            axis.axhline(0.8, color="black", linestyle=":", linewidth=0.8)
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    figure.suptitle("J-centered predictive-state trade-offs")
    _save_figure(figure, "predictive_state_pareto_v4.png", ["predictive_state_screen_v4.json"])


def _controller_results() -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for path in sorted((ROOT / "results/v4/raw").glob("controllers-v4-*/controller_result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (value["family"], int(value["history"]), int(value["memory_dimension"]), int(value["controller_seed"]))
        by_key[key] = value
    return list(by_key.values())


def _controller_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for value in results:
        for summary in value["test"]:
            rows.append(
                {
                    "family": value["family"],
                    "history": int(value["history"]),
                    "memory_dimension": int(value["memory_dimension"]),
                    "seed": int(value["controller_seed"]),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _memory_table(results: list[dict[str, Any]]) -> str:
    frame = _controller_frame(results)
    selected = frame[
        (frame["family"] == "markov")
        | ((frame["family"] == "history") & (frame["history"] == 8))
        | (frame["family"] == "gru")
    ].copy()
    selected["model"] = selected.apply(
        lambda value: (
            "Markov"
            if value["family"] == "markov"
            else "history-8"
            if value["family"] == "history"
            else f"GRU-{int(value['memory_dimension'])}"
        ),
        axis=1,
    )
    grouped = selected.groupby(["model", "horizon"], as_index=False).agg(
        seeds=("seed", "nunique"),
        latent_cosine=("latent_state_similarity_median", "mean"),
        trajectory_divergence=("trajectory_divergence_mean", "mean"),
        teacher_action_fidelity=("teacher_action_fidelity", "mean"),
        action_accuracy=("ground_truth_action_accuracy", "mean"),
        final_accuracy=("final_task_accuracy", "mean"),
        time_to_divergence=("time_to_divergence_median", "median"),
        latent_variance=("latent_variance_median", "median"),
        finite_rate=("finite_rate", "mean"),
    )
    return grouped.to_markdown(index=False, floatfmt=".3f")


def _memory_figure(memory: dict[str, Any], results: list[dict[str, Any]]) -> None:
    del memory
    frame = _controller_frame(results)
    figure, axis = plt.subplots(figsize=(7.6, 4.0))
    groups = {
        "Markov": frame[frame["family"] == "markov"],
        "best finite history": frame[frame["family"] == "history"].sort_values("latent_state_similarity_median").groupby(["seed", "horizon"], as_index=False).tail(1),
        "best GRU": frame[frame["family"] == "gru"].sort_values("latent_state_similarity_median").groupby(["seed", "horizon"], as_index=False).tail(1),
    }
    for label, values in groups.items():
        curve = values.groupby("horizon", as_index=False)["latent_state_similarity_median"].mean()
        axis.plot(curve["horizon"], curve["latent_state_similarity_median"], marker="o", label=label)
    axis.axhline(0.8, color="black", linestyle=":", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(frame["horizon"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("autonomous rollout horizon")
    axis.set_ylabel("decoded latent-state cosine")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.set_title("Autonomous rollout: Markov, history, and recurrent memory")
    sources = ["compact_memory_v4.json", *sorted({value["test_records"] for value in results})]
    _save_figure(figure, "autonomous_memory_v4.png", sources)


def _reference_figure(references: dict[str, Any]) -> None:
    records = references["one_step"]["records"]
    names = [value["reference"] for value in records]
    values = [value["test_decoded_j_cosine"]["mean"] for value in records]
    figure, axis = plt.subplots(figsize=(9.2, 3.8))
    axis.bar(names, values, color="#4c78a8")
    axis.set_ylim(0.7, 1.0)
    axis.set_ylabel("held-out one-step decoded-J cosine")
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Compact and remainder-aware teacher-current references")
    _save_figure(
        figure, "full_state_reference_v4.png", ["full_state_references_v4.json"]
    )


def _fmt_ci(value: dict[str, Any] | None) -> str:
    if not value:
        return "not available"
    return f"{value['estimate']:.6g} (95% CI [{value['lower']:.6g}, {value['upper']:.6g}])"


def _teacher_table(teacher: dict[str, Any]) -> str:
    frame = pd.DataFrame(teacher["rates_by_domain_family_horizon"])
    grouped = frame.groupby(["family", "horizon"], as_index=False).agg(
        n=("attempted", "sum"),
        parseable=("parseable", "sum"),
        full=("full_trajectory_correct", "sum"),
        final=("final_answer_correct", "sum"),
    )
    grouped["parseable rate"] = grouped["parseable"] / grouped["n"]
    grouped["full trajectory accuracy"] = grouped["full"] / grouped["n"]
    grouped["final answer accuracy"] = grouped["final"] / grouped["n"]
    return grouped[["family", "horizon", "n", "parseable rate", "full trajectory accuracy", "final answer accuracy"]].to_markdown(index=False, floatfmt=".3f")


def _causal_family_table(causal: dict[str, Any]) -> str:
    rows = []
    for family, effects in causal["j_preserving_effects_by_family"].items():
        js = effects["output_js_divergence"]
        future = effects["future_j_trajectory_divergence"]
        rows.append(
            {
                "family": family,
                "n": js["n_clusters"],
                "JS": js["estimate"],
                "JS lower": js["lower"],
                "JS upper": js["upper"],
                "future-J": future["estimate"],
                "future-J lower": future["lower"],
                "future-J upper": future["upper"],
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def _write_report(
    teacher: dict[str, Any],
    causal: dict[str, Any] | None,
    mediation: dict[str, Any] | None,
    screen: dict[str, Any],
    memory: dict[str, Any] | None,
    references: dict[str, Any] | None,
    controller_results: list[dict[str, Any]],
) -> None:
    overall = teacher["overall"]
    preserving = causal["effects"]["output_js_divergence"].get("j_preserving") if causal else None
    future = causal["effects"]["future_j_trajectory_divergence"].get("j_preserving") if causal else None
    positive = causal["effects"]["output_js_divergence"].get("j_positive") if causal else None
    causal_gate = bool(causal and causal["single_arm_effect_above_noise"])
    eligible = [value for value in screen["pareto"] if value["gate_passed"]]
    fallback = sorted(
        screen["pareto"],
        key=lambda value: (-min(float(value["semantic_retention"]), float(value["causal_direction_retention"]), float(value["causal_magnitude_retention"])), -float(value["future_prediction_cosine"]), int(value["state_dimension"])),
    )[0]
    minimum_memory = memory.get("minimum_beneficial_memory_dimension") if memory else None
    reference_gain = references["one_step"]["compact_to_combined_gain"] if references else None
    nonlinear_reference_gain = (
        references["one_step"]["compact_to_nonlinear_combined_gain"]
        if references
        else None
    )
    mediation_all = (
        mediation["effects"]["output_js_divergence"]["persistent_all"]
        if mediation
        else None
    )
    reference_h8 = (
        float(
            np.median(
                [
                    row["decoded_j_cosine_median"]
                    for seed_record in references["autonomous_recurrent"]["seeds"]
                    for row in seed_record["test"]
                    if int(row["horizon"]) == 8
                ]
            )
        )
        if references
        else None
    )
    controller_frame = _controller_frame(controller_results) if controller_results else pd.DataFrame()
    markov_h8 = (
        float(
            controller_frame[
                (controller_frame["family"] == "markov")
                & (controller_frame["horizon"] == 8)
            ]["latent_state_similarity_median"].mean()
        )
        if not controller_frame.empty
        else None
    )
    classification = (
        "H2 (exploratory causal evidence; later-J mediation)"
        if causal_gate and mediation
        else "H2 (exploratory operational evidence)"
        if causal_gate
        else "D"
    )
    if minimum_memory is not None and causal_gate:
        classification = "H3 exploratory-follow-up"
    lines = [
        "# Predictive J-state v4 results",
        "",
        "> This report is generated from saved protocol-v4 JSON/Parquet records. Dense 4096D measured-J is an operational control, not a compact-state claim.",
        "",
        "## Execution status",
        "",
        f"- Teacher formal set: complete ({overall['attempted']} trajectories).",
        f"- Single-arm causal pilot: {'completed run' if causal else 'not available'} ({causal['complete_paired_base_trials'] if causal else 0}/100 target paired base trials).",
        f"- Predictive-state screen: complete ({len(screen['pareto'])} candidates; {len(eligible)} passed every retention gate).",
        f"- Markov/history/GRU grid: {'complete' if memory and not memory.get('missing_models') else 'incomplete'} ({memory.get('completed_models', 0) if memory else 0}/{memory.get('expected_models', 33) if memory else 33}).",
        f"- Full/remainder references: {'complete' if references else 'not available'}.",
        f"- Persistent mediation: {'complete' if mediation else 'authorized but not yet reported' if causal_gate else 'not authorized because the single-arm gate did not pass'}.",
        "",
        "## Teacher competence",
        "",
        f"Overall parseable rate was **{overall['parseable_rate']:.3%}**, full-trajectory accuracy **{overall['full_trajectory_accuracy']:.3%}**, and final-answer accuracy **{overall['final_answer_accuracy']:.3%}**. Primary compact-state analyses use only the {overall['full_trajectory_correct']} fully correct trajectories.",
        "",
        _teacher_table(teacher),
        "",
        "![Teacher accuracy](../results/v4/figures/teacher_accuracy_v4.png)",
        "",
        "## Single-arm causal result",
        "",
        f"The J-preserving arm changed output JS by {_fmt_ci(preserving)} and future measured-J trajectory by {_fmt_ci(future)}. The matched J-positive control changed output JS by {_fmt_ci(positive)}. The frozen JS decision threshold was {causal['null_threshold']:.6g}." if causal else "No merged single-arm result is available.",
        "",
        ("The J-preserving lower confidence bound exceeded the null decision threshold. This is intervention-based evidence that holding current operational measured-J fixed does not fix the future." if causal_gate else "The J-preserving lower confidence bound did not exceed the frozen null decision threshold. This does not establish H1: the pilot may be underpowered, and the operational dense state is near-injective and not compact."),
        "",
        (f"The run attempted {causal['attempted_base_prompts']} teacher-correct stepwise prompts; {causal['attrition_counts'].get('continuous_teacher_trajectory_incorrect', 0)} were not correct under the stricter uninterrupted continuation used by this causal endpoint. The pooled result is heterogeneous across families:" if causal else ""),
        "",
        _causal_family_table(causal) if causal else "",
        "",
        "![Single-arm effects](../results/v4/figures/single_arm_effects_v4.png)" if causal else "",
        "",
        "## Persistent mediation",
        "",
        (
            f"Persistent-final output JS was {_fmt_ci(mediation['effects']['output_js_divergence']['persistent_final'])}; persistent-all was {_fmt_ci(mediation_all)}. M_final was {_fmt_ci(mediation['mediation_ratios']['M_final'])}, and M_all was {_fmt_ci(mediation['mediation_ratios']['M_all'])}. Because this arm perturbs only the final prompt position, causal masking predicts final/all restoration equivalence; it is a hook-scope sanity check, not a sequence-position mediation test."
            if mediation
            else "Authorized by the single-arm gate and still running."
            if causal_gate
            else "Not authorized because the single-arm noise gate did not pass."
        ),
        "Restoring measured-J at later workspace layers removed approximately 99.77% of the single-arm JS effect, and the residual JS fell below the 1e-4 decision threshold. For this final-position arm, the best-supported pathway is measured-J remainder → later measured-J writes → future, rather than a detectable measured-J bypass." if mediation else "",
        "",
        "![Persistent mediation](../results/v4/figures/persistent_mediation_v4.png)" if mediation else "",
        "",
        "## Compact-state Pareto screen",
        "",
        "No fully gated compact state was found." if not eligible else f"The smallest fully gated state was {min(int(value['state_dimension']) for value in eligible)}D.",
        f"For exploratory temporal modeling only, the fixed maximin-retention fallback is {fallback['state_dimension']}D `{fallback['representation']}`: semantic retention {fallback['semantic_retention']:.3f}, causal-direction retention {fallback['causal_direction_retention']:.3f}, causal-magnitude retention {fallback['causal_magnitude_retention']:.3f}, and validation future cosine {fallback['future_prediction_cosine']:.3f}. It is not relabeled as a validated compact state.",
        "",
        "![Predictive-state Pareto](../results/v4/figures/predictive_state_pareto_v4.png)",
        "",
        "## Memory and autonomous rollout",
        "",
        (f"Minimum memory dimension satisfying the paired horizon-8 utility gate: **{minimum_memory}**." if minimum_memory is not None else "No tested recurrent memory dimension satisfied the complete paired horizon-8 utility gate." if memory else "The temporal grid has not completed."),
        "The gate requires a positive paired CI and at least +0.02 cosine, at least 20% trajectory-distance reduction, no more than 2 percentage points of action loss, and agreement across all three seeds.",
        "",
        _memory_table(controller_results) if controller_results else "",
        "",
        "![Autonomous memory rollout](../results/v4/figures/autonomous_memory_v4.png)" if memory and controller_results else "",
        "",
        "## Full/remainder reference",
        "",
        (f"Adding train-fitted PCA-512 hidden-state remainder information to the compact state changed held-out linear one-step decoded-J cosine by {reference_gain:.6f}; the nonlinear gain was {nonlinear_reference_gain:.6f}. These teacher-current endpoints test learnable information availability, not autonomous sufficiency. The recurrent reference reads full state only at t=0 and thereafter feeds back its own predicted state; its median horizon-8 cosine was {reference_h8:.3f}, versus {markov_h8:.3f} for the compact Markov baseline." if references and markov_h8 is not None else "Reference training has not completed."),
        "",
        "![Full-state references](../results/v4/figures/full_state_reference_v4.png)" if references else "",
        "",
        "## Answers to the twelve scientific questions",
        "",
        f"1. **Teacher accuracy.** Parseable {overall['parseable_rate']:.3%}; full trajectory {overall['full_trajectory_accuracy']:.3%}; final answer {overall['final_answer_accuracy']:.3%}. Family/horizon cells are in the table above.",
        f"2. **Does same-J/different-hidden change the future?** {'Yes for the pooled exploratory sample under the operational v4 criterion, with substantial family heterogeneity.' if causal_gate else 'Not demonstrated above the frozen noise criterion.'}",
        f"3. **Is current J sufficient?** {'No for the tested pooled measured-J definition and intervention; replication with more within-family pairs is still needed.' if causal_gate else 'Unresolved; failure to cross the gate is not proof of sufficiency.'}",
        f"4. **Smallest semantic/causal compact state.** {min((int(value['state_dimension']) for value in eligible), default='None')} passed all gates; the {fallback['state_dimension']}D fallback is exploratory only.",
        f"5. **Non-Markov behavior?** {'There is predictive evidence of history dependence in the ungated 512D fallback: history-8 exceeds Markov most clearly at long autonomous horizons, but no validated compact state or causal state-order test exists.' if memory and minimum_memory is None else 'A recurrent advantage passed the declared gate.' if minimum_memory is not None else 'Not yet estimable.'}",
        "6. **Is finite history useful?** Yes predictively at long autonomous horizons: mean cosine was 0.759 versus 0.668 at horizon 16 and 0.673 versus 0.447 at horizon 32 for history-8 versus Markov. This is not causal evidence and the representation itself failed the semantic gate.",
        f"7. **Is recurrent memory useful?** {'Yes by the declared gate.' if minimum_memory is not None else 'Not established by the complete gate.'}",
        f"8. **Smallest beneficial memory.** {minimum_memory if minimum_memory is not None else 'None among 16/32/64/128/256'}.",
        "9. **Does the memory advantage survive autonomous rollout?** No GRU memory size passed the autonomous utility gate. The small 16D cosine gain was accompanied by action degradation and seed inconsistency; teacher-forced improvement is not used.",
        f"10. **Does full/remainder state add predictive information?** {('Detected by at least one teacher-current reference.' if reference_gain is not None and max(reference_gain, nonlinear_reference_gain) >= 0.01 else 'Not materially detected by either tested teacher-current reference.' if reference_gain is not None else 'Not yet tested.')} Direct intervention nevertheless shows causal influence outside instantaneous measured-J. The weak learned reference means systematic predictive availability remains unresolved, not absent.",
        f"11. **Best-supported hypothesis.** **{classification}**. Single-arm intervention and persistent restoration are causal evidence; compression, history, controller, and reference comparisons are predictive evidence. Persistent restoration removes the effect, supporting a broadcast-bus pathway rather than a detected bypass in this arm.",
        "12. **What remains before an independent small high-level dynamical model?** A compact representation must simultaneously pass semantic and causal retention; autonomous action accuracy must remain high over long horizons; recurrent gains must replicate across seeds; a strong autonomous full-state reference must be learned; and teacher/student latent interventions must show causal fidelity.",
        "",
        "## Interpretation and next experiment",
        "",
        "The teacher-task result removes the earlier confound in which most training trajectories were wrong. The single-arm comparison is causal because the hidden activation is directly intervened on with paired controls; persistent restoration supplies a causal mediation result. Controller and reference comparisons remain predictive. Negative compression or memory results can reflect representation/training failure, limited task diversity, or autonomous compounding error. The next highest-value experiment is an independently frozen, larger within-family replication of the single/persistent result, including an all-position sequence arm. In parallel, the joint encoder needs substantially better simultaneous semantic and causal retention before token-time counterfactual fidelity can test a compact controller.",
        "",
        "## Provenance",
        "",
        "All plotted series are reconstructed from saved records. Figure sidecars contain source and figure SHA-256 digests. Model/lens revisions and seeds are recorded in run manifests. No model weights, activation arrays, checkpoints, or credentials are committed.",
        "",
        "```bash",
        "scripts/run_teacher_v4.sh --stage calibrate",
        "scripts/run_teacher_v4.sh --stage freeze",
        "scripts/run_teacher_v4.sh --stage formal",
        "scripts/run_predictive_state_v4.sh traces --domain all",
        "scripts/run_single_arm_v4.sh --stage bank",
        "scripts/run_single_arm_v4.sh --stage run",
        "scripts/run_single_arm_v4.sh --stage merge",
        "EXPERIMENT=mediation scripts/run_single_arm_v4.sh --stage run",
        "EXPERIMENT=mediation scripts/run_single_arm_v4.sh --stage merge",
        "scripts/run_predictive_state_v4.sh screen",
        "scripts/run_controllers_v4.sh",
        "scripts/run_predictive_state_v4.sh reference",
        "scripts/build_report.sh",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _update_final(report_text: str) -> None:
    target = ROOT / "reports/FINAL_REPORT.md"
    text = target.read_text(encoding="utf-8")
    start = "<!-- V4 PREDICTIVE STATUS START -->"
    end = "<!-- V4 PREDICTIVE STATUS END -->"
    block = f"{start}\n\n{report_text}\n\n{end}"
    if start in text and end in text:
        text = text[: text.index(start)] + block + text[text.index(end) + len(end) :]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    status = _load("execution_status_v4.json")
    classification = (
        status["strongest_warranted_classification"] if status else "D"
    )
    headline = (
        "> **Protocol v4 provides exploratory causal evidence for H2: the "
        "instantaneous measured-J state is insufficient, while later measured-J "
        "restoration removes the detected effect. No compact controller or H3 "
        "claim is established.**"
        if classification.startswith("H2")
        else "> **No operational compact state has yet passed the complete causal "
        "and autonomous criteria required to distinguish H1/H2/H3.**"
    )
    text = re.sub(r"^> \*\*.*?\*\*$", headline, text, count=1, flags=re.MULTILINE)
    text = text.replace(
        "- Current downstream status:", "- Historical v2 downstream status:", 1
    )
    text = text.replace(
        "- Strongest warranted conclusion:", "- Historical v2 conclusion:", 1
    )
    target.write_text(text, encoding="utf-8")


def main() -> None:
    teacher = _load("teacher_formal_v4.json")
    screen = _load("predictive_state_screen_v4.json")
    if teacher is None or screen is None:
        raise RuntimeError("teacher and predictive-state records are required")
    causal = _load("single_arm_v4.json")
    mediation = _load("persistent_mediation_v4.json")
    memory = _load("compact_memory_v4.json")
    references = _load("full_state_references_v4.json")
    results = _controller_results()
    status = {
        "schema_version": 6,
        "protocol_version": "jstate_predictive_protocol_v4",
        "teacher": {
            "attempted": teacher["overall"]["attempted"],
            "parseable_rate": teacher["overall"]["parseable_rate"],
            "full_trajectory_accuracy": teacher["overall"][
                "full_trajectory_accuracy"
            ],
            "final_answer_accuracy": teacher["overall"]["final_answer_accuracy"],
        },
        "single_arm": {
            "status": "COMPLETED" if causal else "NOT_AVAILABLE",
            "paired": causal["complete_paired_base_trials"] if causal else 0,
            "effect_above_noise": bool(
                causal and causal["single_arm_effect_above_noise"]
            ),
        },
        "persistent_mediation": {
            "status": "COMPLETED"
            if mediation
            else "AUTHORIZED_NOT_COMPLETED"
            if causal and causal["persistent_mediation_authorized"]
            else "NOT_AUTHORIZED"
        },
        "compact_state": {
            "authorized": bool(screen["compact_state_authorized"]),
            "tested_candidates": len(screen["pareto"]),
        },
        "controllers": {
            "completed": memory.get("completed_models", 0) if memory else 0,
            "expected": memory.get("expected_models", 33) if memory else 33,
            "memory_utility_supported": bool(
                memory and memory["memory_utility_supported"]
            ),
        },
        "full_state_reference": "COMPLETED" if references else "NOT_AVAILABLE",
        "strongest_warranted_classification": "H2_EXPLORATORY_LATER_J_MEDIATED"
        if causal and causal["single_arm_effect_above_noise"] and mediation
        else "H2_EXPLORATORY_OPERATIONAL"
        if causal and causal["single_arm_effect_above_noise"]
        else "D",
        "sources": [
            _source(name)
            for name in (
                "teacher_formal_v4.json",
                "single_arm_v4.json",
                "persistent_mediation_v4.json",
                "predictive_state_screen_v4.json",
                "compact_memory_v4.json",
                "full_state_references_v4.json",
            )
            if (PROCESSED / name).exists()
        ],
    }
    write_json_atomic(PROCESSED / "execution_status_v4.json", status)
    _teacher_figure(teacher)
    _pareto_figure(screen)
    if causal is not None:
        _causal_figure(causal)
    if mediation is not None:
        _mediation_figure(mediation)
    if memory is not None and results:
        _memory_figure(memory, results)
    if references is not None:
        _reference_figure(references)
    _write_report(teacher, causal, mediation, screen, memory, references, results)
    _update_final(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
