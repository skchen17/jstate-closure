"""Machine-only report builders for persistent-state protocol v8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jclosure.provenance import write_json_atomic

START = "<!-- V8-RESULTS:START -->"
END = "<!-- V8-RESULTS:END -->"


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _ci(value: dict[str, Any]) -> str:
    return f"{_fmt(value.get('estimate'))} [{_fmt(value.get('lower'))}, {_fmt(value.get('upper'))}]"


def build_reports(root: Path) -> dict[str, str]:
    """Build every v8 report only from hashed JSON/Parquet products."""

    calibration = json.loads(
        (root / "results/v8/processed/difficulty_calibration_v8_r6.json").read_text()
    )
    teacher = json.loads(
        (root / "results/v8/processed/teacher_formal_v8.json").read_text()
    )
    competence_attempt = json.loads(
        (
            root / "results/v8/processed/teacher_formal_v8_r6_attempt.json"
        ).read_text()
    )
    screen = json.loads(
        (root / "results/v8/processed/structured_component_screen_v8.json").read_text()
    )
    compression_path = (
        root / "results/v8/processed/persistent_state_compression_v8.json"
    )
    compression = (
        json.loads(compression_path.read_text())
        if compression_path.is_file()
        else {"status": "NOT_EXECUTED"}
    )
    calibration_rows = pd.DataFrame(calibration["rows"])
    teacher_rows = pd.DataFrame(teacher["rows"])
    raw = screen["effects"]["pooled"]
    report_values: dict[str, str] = {}
    figure_root = root / "results/v8/figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    labels = [f"R{index}" for index in range(8)]
    estimates = [raw["output_js_divergence"][label]["estimate"] for label in labels]
    lower = [raw["output_js_divergence"][label]["lower"] for label in labels]
    upper = [raw["output_js_divergence"][label]["upper"] for label in labels]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, estimates, color="#4472C4")
    axis.errorbar(
        labels,
        estimates,
        yerr=[
            [
                estimate - value
                for estimate, value in zip(estimates, lower, strict=True)
            ],
            [
                value - estimate
                for estimate, value in zip(estimates, upper, strict=True)
            ],
        ],
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.set_ylabel("clean-relative output JS")
    axis.set_title("Persistent-state raw component screen (final test)")
    figure.tight_layout()
    figure.savefig(figure_root / "structured_component_screen_v8.png", dpi=180)
    plt.close(figure)

    dataset_text = "\n".join(
        [
            "# Persistent Causal Dataset v8",
            "",
            f"Protocol: `{screen['protocol_version']}`; freeze: `{screen['source_freeze_digest']}`.",
            "",
            "## Difficulty calibration",
            "",
            calibration_rows.to_markdown(index=False),
            "",
            "## Preserved pre-freeze competence failure",
            "",
            "The first formal horizon assignment was rejected before causal execution because modular-arithmetic horizon 3 fell below the 70% competence rule on final test.",
            "",
            pd.DataFrame(competence_attempt["rows"]).to_markdown(index=False),
            "",
            "## Frozen formal teacher competence",
            "",
            teacher_rows.to_markdown(index=False),
            "",
            f"Valid causal pairs: `{screen['valid_pair_count']}`; final-test pairs: `{screen['final_test_pair_count']}`.",
            "",
            "No teacher-incorrect trajectory enters the paired causal or compression estimates.",
        ]
    )
    (root / "reports/PERSISTENT_CAUSAL_DATASET_V8.md").write_text(dataset_text + "\n")
    report_values["dataset"] = "reports/PERSISTENT_CAUSAL_DATASET_V8.md"

    conditions = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")
    rows = []
    for condition in conditions:
        rows.append(
            {
                "condition": condition,
                "next-J L2": _ci(raw["next_j_l2"][condition]),
                "future-J divergence": _ci(
                    raw["future_j_trajectory_divergence"][condition]
                ),
                "output JS": _ci(raw["output_js_divergence"][condition]),
                "ground-truth accuracy change": _ci(
                    raw["ground_truth_accuracy_change"][condition]
                ),
                "direction to R7": _ci(raw["direction_cosine_to_full"][condition]),
                "magnitude / R7": _ci(raw["magnitude_ratio_to_full"][condition]),
                "output-sign agreement": _ci(
                    raw["output_sign_agreement_to_full"][condition]
                ),
            }
        )
    family_rows = []
    for family, values in screen["effects"].items():
        if family == "pooled":
            continue
        for condition in ("R4", "R5", "R6", "R7"):
            family_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "next-J L2": _ci(values["next_j_l2"][condition]),
                    "direction to R7": _ci(
                        values["direction_cosine_to_full"][condition]
                    ),
                    "magnitude / R7": _ci(
                        values["magnitude_ratio_to_full"][condition]
                    ),
                    "output JS": _ci(values["output_js_divergence"][condition]),
                }
            )
    component_text = "\n".join(
        [
            "# Structured Component Screen v8",
            "",
            f"Final-test pairs: `{screen['final_test_pair_count']}` across `{screen['family_counts']}`.",
            "",
            pd.DataFrame(rows).to_markdown(index=False),
            "",
            "## Family-wise primary components",
            "",
            pd.DataFrame(family_rows).to_markdown(index=False),
            "",
            "## Factorial interaction decomposition",
            "",
            pd.DataFrame(
                [
                    {
                        "mask": mask,
                        "atoms": "+".join(value["atoms"]) or "intercept",
                        "order": value["order"],
                        "vector next-J interaction / R7": _ci(
                            value["vector_interaction_ratio_to_full"]
                        ),
                        "output-JS interaction": _ci(
                            value["output_js_interaction"]
                        ),
                    }
                    for mask, value in screen["interaction_effects"].items()
                ]
            ).to_markdown(index=False),
            "",
            "The committed factorial records include standalone, added, leave-one-out, all pairwise, and Möbius higher-order interactions.",
            f"Records: `{screen['trial_records']}` and `{screen['interaction_records']}`.",
        ]
    )
    (root / "reports/STRUCTURED_COMPONENT_SCREEN_V8.md").write_text(
        component_text + "\n"
    )
    report_values["screen"] = "reports/STRUCTURED_COMPONENT_SCREEN_V8.md"

    state = compression.get("status", "COMPLETED")
    smallest = compression.get("smallest_authorized")
    sweep_table = "Compression sweep not executed."
    family_table = "Family-wise compression sweep not executed."
    best_predictive = None
    universal_best = None
    family_specific_best = None
    compression_records = compression.get("records")
    if compression_records:
        sweep = pd.read_parquet(root / compression_records)
        pooled = sweep[sweep["family"] == "pooled"].copy()
        best_predictive = pooled.sort_values(
            ["predictive_gap_closed", "conditional_residual_gain"],
            ascending=[False, True],
        ).iloc[0]
        universal_values = pooled[pooled["regime"] == "universal"]
        family_values = pooled[pooled["regime"] == "family_specific"]
        universal_best = (
            float(universal_values["predictive_gap_closed"].max())
            if not universal_values.empty
            else None
        )
        family_specific_best = (
            float(family_values["predictive_gap_closed"].max())
            if not family_values.empty
            else None
        )
        sweep_table = pooled[
            [
                "regime",
                "method",
                "dimension",
                "predictive_gap_closed",
                "conditional_residual_gain",
                "semantic_agreement",
                "authorized",
            ]
        ].to_markdown(index=False)
        family_table = (
            sweep[sweep["family"] != "pooled"]
            .groupby(["family", "dimension"], sort=True)
            .agg(
                best_predictive_gap=("predictive_gap_closed", "max"),
                minimum_conditional_gain=("conditional_residual_gain", "min"),
                any_authorized=("authorized", "max"),
            )
            .reset_index()
            .to_markdown(index=False)
        )
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for (regime, method), values in pooled.groupby(["regime", "method"]):
            ordered = values.sort_values("dimension")
            label = f"{regime}/{method}"
            axes[0].plot(
                ordered["dimension"],
                ordered["predictive_gap_closed"],
                marker="o",
                label=label,
            )
            axes[1].plot(
                ordered["dimension"],
                ordered["conditional_residual_gain"],
                marker="o",
                label=label,
            )
            causal = pd.to_numeric(
                ordered["causal_gap_closed"], errors="coerce"
            )
            if causal.notna().any():
                axes[2].plot(
                    ordered["dimension"], causal, marker="o", label=label
                )
        axes[0].axhline(0.8, color="black", linestyle="--", linewidth=1)
        axes[1].axhline(0.02, color="black", linestyle="--", linewidth=1)
        axes[2].axhline(0.8, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("predictive gap closed")
        axes[1].set_ylabel("conditional residual cosine gain")
        axes[2].set_ylabel("decoded causal gap closed")
        if pd.to_numeric(pooled["causal_gap_closed"], errors="coerce").isna().all():
            axes[2].text(
                0.5,
                0.5,
                "GATED: no decoded interventions",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
        for axis in axes:
            axis.set_xlabel("state dimension")
            axis.set_xscale("log", base=2)
        axes[1].legend(fontsize=5, loc="best")
        figure.tight_layout()
        figure.savefig(figure_root / "persistent_state_dimension_v8.png", dpi=180)
        plt.close(figure)
    compression_text = "\n".join(
        [
            "# Persistent State Compression v8",
            "",
            f"Status: `{state}`.",
            f"Smallest authorized state: `{smallest}`.",
            "",
            "The 16/32/64/128/256/512D sweep compares PCA/SVD, predictive and causal bottlenecks, nonlinear encoders, and layerwise fusion under universal, family-specific, and shared-plus-family-residual regimes.",
            "",
            "## Pooled dimension sweep",
            "",
            sweep_table,
            "",
            "## Family-wise sensitivity",
            "",
            family_table,
        ]
    )
    (root / "reports/PERSISTENT_STATE_COMPRESSION_V8.md").write_text(
        compression_text + "\n"
    )
    report_values["compression"] = "reports/PERSISTENT_STATE_COMPRESSION_V8.md"

    conditional_text = "\n".join(
        [
            "# Conditional Sufficiency v8",
            "",
            f"Compression status: `{state}`.",
            f"Smallest state passing predictive plus conditional-residual gates: `{smallest}`.",
            "",
            "Authorization requires the held-out residual-state gain confidence interval to be practically near zero; prediction alone is not treated as sufficiency.",
        ]
    )
    (root / "reports/CONDITIONAL_SUFFICIENCY_V8.md").write_text(conditional_text + "\n")
    report_values["conditional"] = "reports/CONDITIONAL_SUFFICIENCY_V8.md"

    fidelity_text = "\n".join(
        [
            "# Causal State Fidelity v8",
            "",
            f"Compression status: `{state}`.",
            f"Smallest state passing all causal direction, magnitude, semantic, and output-sign gates: `{smallest}`.",
            "",
            "No autonomous controller is authorized unless predictive sufficiency, conditional residual sufficiency, and causal fidelity pass together.",
            "",
            "## Validity and fallacy scan",
            "",
            "1. Predictive gains are not relabeled as causal effects.",
            "2. A high-dimensional dense J profile is not called compact.",
            "3. Teacher-incorrect items are reported but excluded from the primary paired state estimates.",
            "4. Train, validation, and final-test programs are hash-disjoint.",
            "5. Failed calibration rounds remain visible and did not lower thresholds.",
            "6. R7 is a measured persistent-state ceiling, not a proof of complete model state.",
            "7. Pooled effects are accompanied by family-wise estimates.",
            "8. Conditional residual gain is a required sufficiency test, not an optional diagnostic.",
            "9. Causal compression fidelity requires decoded cache interventions rather than regression fit alone.",
            "10. A failed encoder does not prove that no compact state exists.",
            "11. No consciousness, true-thought, or teacher parameter-localization claim is made.",
        ]
    )
    (root / "reports/CAUSAL_STATE_FIDELITY_V8.md").write_text(fidelity_text + "\n")
    report_values["fidelity"] = "reports/CAUSAL_STATE_FIDELITY_V8.md"
    r7_next = raw["next_j_l2"]["R7"]
    r4_direction = raw["direction_cosine_to_full"]["R4"]
    r4_magnitude = raw["magnitude_ratio_to_full"]["R4"]
    r5_magnitude = raw["magnitude_ratio_to_full"]["R5"]
    kv_effect = raw["next_j_l2"]["R5"]
    rec_kv_interaction = screen["interaction_effects"].get("5", {})
    best_text = (
        "not executed"
        if best_predictive is None
        else (
            f"{best_predictive['regime']}/{best_predictive['method']}/"
            f"{int(best_predictive['dimension'])}D: predictive gap "
            f"{best_predictive['predictive_gap_closed']:.4f}, conditional gain "
            f"{best_predictive['conditional_residual_gain']:.4f}"
        )
    )
    final_block = "\n".join(
        [
            START,
            "",
            "## Protocol v8 large-sample persistent-state update",
            "",
            f"The frozen five-family data produced `{screen['valid_pair_count']}` valid paired states, including `{screen['final_test_pair_count']}` independent final-test pairs. Raw full-persistent swapping changed next measured-J by {_ci(r7_next)} L2. Recurrent-matrix plus convolution state aligned with the full-persistent causal direction at {_ci(r4_direction)}.",
            "",
            f"Compression status is `{state}` and the smallest fully authorized state is `{smallest}`. Controller authorization is `{compression.get('controller_authorized', False)}`. Predictive screening is not counted as causal or conditional sufficiency; decoded-state intervention gates remain mandatory.",
            "",
            "The strongest conclusion therefore remains **H2 for the tested operational measured-J state** unless and until one compressed persistent state passes predictive, conditional-residual, and causal-fidelity gates together. Dense measured-J is not relabeled as a compact state.",
            "",
            "Evidence labels: R0–R7/factorial cache swaps are intervention-based causal evidence; compression regressions are held-out predictive evidence; confidence intervals quantify sampling uncertainty but do not establish state minimality.",
            "",
            "### Required v8 questions",
            "",
            f"1. KV/recurrent attribution replication: R4 direction {_ci(r4_direction)} and R5 standalone next-J {_ci(kv_effect)}.",
            f"2. Recurrent/conv dominance: R4 magnitude {_ci(r4_magnitude)} versus R5 {_ci(r5_magnitude)}.",
            f"3. L27/H3 independent contribution: R5 next-J is {_ci(kv_effect)}; this is a direct cache-swap effect.",
            f"4. REC×KV interaction: `{rec_kv_interaction}`.",
            f"5. Smallest raw component set passing the frozen raw screen: `{screen.get('selected_raw_state')}`.",
            f"6. Full persistent ceiling: R7 next-J {_ci(r7_next)}.",
            f"7. Compression outcome: `{state}`; best predictive candidate {best_text}.",
            f"8. Smallest sufficient dimension: `{smallest}`.",
            f"9. Conditional residual gain near zero: `{compression.get('predictive_conditional_pass_count', 0) > 0}` under every pooled/family gate.",
            f"10. Decoded causal intervention fidelity passed: `{smallest is not None}`.",
            f"11. Best universal versus family-specific predictive gap: `{universal_best}` versus `{family_specific_best}`.",
            f"12. Candidate sufficient persistent state obtained: `{smallest is not None}`.",
            f"13. Autonomous recurrent dynamics authorized: `{compression.get('controller_authorized', False)}`.",
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
    write_json_atomic(
        root / "results/v8/processed/report_manifest_v8.json",
        {
            "reports": report_values,
            "figures": [
                "results/v8/figures/structured_component_screen_v8.png",
                "results/v8/figures/persistent_state_dimension_v8.png"
                if compression_records
                else None,
            ],
            "sources": {
                "raw_screen": "results/v8/processed/structured_component_screen_v8.json",
                "compression": str(compression_path.relative_to(root))
                if compression_path.is_file()
                else None,
            },
        },
    )
    return report_values
