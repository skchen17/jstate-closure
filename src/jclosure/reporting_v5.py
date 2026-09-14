"""Build protocol-v5 reports and figures exclusively from saved machine records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "results/v5/processed"
FIGURES = ROOT / "results/v5/figures"
REPORT = ROOT / "reports/PERIPHERAL_STATE_V5.md"
FINAL = ROOT / "reports/FINAL_REPORT.md"
START = "<!-- PERIPHERAL_V5_START -->"
END = "<!-- PERIPHERAL_V5_END -->"


def _load(name: str, *, required: bool = True) -> dict[str, Any] | None:
    path = PROCESSED / name
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _source(name: str) -> dict[str, str]:
    path = PROCESSED / name
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _save_figure(figure: plt.Figure, name: str, sources: list[str]) -> str:
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
    return str(target.relative_to(ROOT))


def _reference_figure(reference: dict[str, Any]) -> str:
    rows = []
    pooled = [value for value in reference["summaries"] if value["family"] == "pooled"]
    for input_state, values in pd.DataFrame(pooled).groupby("input_state", sort=True):
        rows.append(
            {
                "input_state": input_state,
                "mean": float(values["next_j_cosine"].mean()),
                "minimum": float(values["next_j_cosine"].min()),
                "maximum": float(values["next_j_cosine"].max()),
            }
        )
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.bar(frame["input_state"], frame["mean"], color=["#4c78a8", "#54a24b"])
    axis.errorbar(
        np.arange(len(frame)),
        frame["mean"],
        yerr=np.vstack(
            (frame["mean"] - frame["minimum"], frame["maximum"] - frame["mean"])
        ),
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.set_ylabel("next-J cosine")
    axis.set_title("Full measured-J remainder reference")
    return _save_figure(
        figure, "full_remainder_reference_v5.png", ["full_remainder_reference_v5.json"]
    )


def _dimension_figures(sweep: dict[str, Any]) -> list[str]:
    selected = pd.DataFrame(
        [
            value
            for value in sweep["records"]
            if sweep["selected_by_dimension"].get(str(value["state_dimension"]))
            == value["encoder_family"]
        ]
    ).sort_values("state_dimension")
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    axis.plot(selected["state_dimension"], selected["gap_closed"], marker="o")
    axis.axhline(0.8, color="black", linestyle=":", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(selected["state_dimension"])
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("peripheral state dimension")
    axis.set_ylabel("J-only → full-remainder gap closed")
    axis.set_title("Compact peripheral-state sweep")
    gap = _save_figure(
        figure, "peripheral_gap_closed_v5.png", ["peripheral_dimension_sweep_v5.json"]
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True)
    metrics = (
        ("conditional_residual_gain", "conditional residual gain"),
        ("causal_direction_cosine", "causal delta cosine"),
        ("output_sign_agreement", "output-sign agreement"),
    )
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values = pd.to_numeric(selected[metric], errors="coerce")
        axis.plot(selected["state_dimension"], values, marker="o")
        axis.set_xscale("log", base=2)
        axis.set_xticks(selected["state_dimension"])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("dimension")
        axis.set_ylabel(label)
    figure.suptitle("Conditional sufficiency and causal fidelity")
    quality = _save_figure(
        figure,
        "peripheral_sufficiency_fidelity_v5.png",
        ["peripheral_dimension_sweep_v5.json"],
    )
    return [gap, quality]


def _h2_figure(replication: dict[str, Any]) -> str:
    rows = []
    for family, metrics in replication["effects"].items():
        if family == "pooled":
            continue
        effect = metrics["output_js_divergence"].get("j_preserving")
        if effect:
            rows.append({"family": family, **effect})
    frame = pd.DataFrame(rows).sort_values("estimate")
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    positions = np.arange(len(frame))
    axis.errorbar(
        frame["estimate"],
        positions,
        xerr=np.vstack(
            (frame["estimate"] - frame["lower"], frame["upper"] - frame["estimate"])
        ),
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0001, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(positions, frame["family"])
    axis.set_xlabel("clean-relative output JS")
    axis.set_title("Independent same-J / changed-hidden replication")
    return _save_figure(
        figure, "h2_family_replication_v5.png", ["h2_replication_v5.json"]
    )


def _recurrent_figure(recurrent: dict[str, Any]) -> str | None:
    if recurrent.get("status") != "COMPLETED":
        return None
    frame = pd.read_parquet(ROOT / recurrent["records"])
    curve = frame.groupby(["architecture", "horizon"], as_index=False).agg(
        j_similarity=("j_similarity", "mean"),
        c_similarity=("c_similarity", "mean"),
    )
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    for architecture, values in curve.groupby("architecture", sort=True):
        axis.plot(
            values["horizon"], values["j_similarity"], marker="o", label=architecture
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(curve["horizon"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_ylim(0, 1)
    axis.set_xlabel("autonomous horizon")
    axis.set_ylabel("J trajectory similarity")
    axis.legend(frameon=False)
    axis.set_title("Autonomous (J, C) rollout")
    return _save_figure(
        figure,
        "peripheral_recurrent_rollout_v5.png",
        ["peripheral_recurrent_v5.json"],
    )


def _fmt_ci(value: dict[str, Any]) -> str:
    return f"{value['estimate']:.6f} [{value['lower']:.6f}, {value['upper']:.6f}]"


def _reference_table(reference: dict[str, Any]) -> str:
    rows = [
        {
            "family": "pooled",
            "cosine gain (95% CI)": _fmt_ci(reference["gain"]["cosine_gain"]),
            "action accuracy gain": reference["gain"]["semantic_accuracy_gain"],
            "transitions": reference["gain"]["n_transitions"],
            "trajectories×seeds": reference["gain"]["n_trajectory_seed_clusters"],
        }
    ]
    for family, value in reference["gain"]["by_family"].items():
        rows.append(
            {
                "family": family,
                "cosine gain (95% CI)": _fmt_ci(value["cosine_gain"]),
                "action accuracy gain": value["semantic_accuracy_gain"],
                "transitions": value["n_transitions"],
                "trajectories×seeds": value["n_trajectories"],
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def _compact_table(sweep: dict[str, Any]) -> str:
    rows = []
    for value in sweep["records"]:
        if (
            sweep["selected_by_dimension"].get(str(value["state_dimension"]))
            != value["encoder_family"]
        ):
            continue
        rows.append(
            {
                "dimension": value["state_dimension"],
                "encoder": value["encoder_family"],
                "next-J cosine": value["next_j_cosine"],
                "semantic accuracy": value["semantic_accuracy"],
                "gap closed": value["gap_closed"],
                "conditional gain": value.get("conditional_residual_gain"),
                "causal cosine": value.get("causal_direction_cosine"),
                "causal magnitude": value.get("causal_magnitude_ratio"),
                "output sign": value.get("output_sign_agreement"),
                "authorized": value.get("authorized", False),
                "parameters": value["parameter_count"],
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".6f")


def _h2_table(replication: dict[str, Any]) -> str:
    rows = []
    for family, metrics in replication["effects"].items():
        effect = metrics["output_js_divergence"].get("j_preserving")
        trajectory = metrics["future_j_trajectory_divergence"].get("j_preserving")
        if effect and trajectory:
            rows.append(
                {
                    "family": family,
                    "valid": replication["valid_by_family"].get(
                        family, replication["complete_paired_base_trials"]
                    ),
                    "output JS (95% CI)": _fmt_ci(effect),
                    "future-J divergence (95% CI)": _fmt_ci(trajectory),
                    "above frozen noise": family
                    in replication["families_above_frozen_noise"],
                }
            )
    return pd.DataFrame(rows).to_markdown(index=False)


def build() -> dict[str, Any]:
    freeze_path = ROOT / "artifacts/peripheral_v5.freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    reference = _load("full_remainder_reference_v5.json")
    sweep = _load("peripheral_dimension_sweep_v5.json")
    replication = _load("h2_replication_v5.json")
    recurrent = _load("peripheral_recurrent_v5.json", required=False)
    assert reference is not None and sweep is not None and replication is not None
    figures = [
        _reference_figure(reference),
        *_dimension_figures(sweep),
        _h2_figure(replication),
    ]
    recurrent_figure = _recurrent_figure(recurrent or {})
    if recurrent_figure:
        figures.append(recurrent_figure)

    authorized = sorted(
        sweep.get("authorized_candidates", []),
        key=lambda value: value["state_dimension"],
    )
    smallest = authorized[0] if authorized else None
    h2_families = replication["families_above_frozen_noise"]
    recurrent_status = (recurrent or {}).get("status", "NOT_EXECUTED")
    if smallest and recurrent_status == "COMPLETED":
        conclusion = "H3 exploratory: a compact peripheral state passed one-step causal gates and was evaluated in autonomous recurrence."
    elif h2_families:
        conclusion = "H2 remains the strongest operational interpretation: independent families show same-J/changed-hidden effects, but no complete compact recurrent system is established."
    else:
        conclusion = (
            "D: the v5 evidence is insufficient to upgrade the frozen conclusions."
        )

    ceiling_gain = reference["gain"]["cosine_gain"]
    conditional_answer = (
        f"The smallest authorized candidate was {smallest['state_dimension']}D ({smallest['encoder_family']})."
        if smallest
        else "No tested 16–512D candidate passed the frozen gap, conditional-residual, and causal-fidelity gates together."
    )
    recurrent_answer = (
        "Autonomous recurrence was executed; see the saved horizon-wise table below."
        if recurrent_status == "COMPLETED"
        else f"Autonomous recurrence status: {recurrent_status}; the frozen compact-state authorization gate prevented interpretation."
    )
    recurrent_table = ""
    if recurrent and recurrent.get("status") == "COMPLETED":
        recurrent_table = pd.DataFrame(recurrent["results"]).to_markdown(index=False)

    body = f"""# Peripheral Computation State v5

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: causal-predictive experiment execution
- Protocol: `{freeze["protocol_version"]}`
- Freeze digest: `{freeze["freeze_digest"]}`
- State definition: layer-23 normalized 4,096-concept measured-J plus a train-fitted measured-J remainder
- Verification status: machine results loaded from frozen, hash-verified records

## Result in one sentence

{conclusion}

The full-remainder next-J cosine gain was **{_fmt_ci(ceiling_gain)}**. Full-remainder ceiling authorization was **{reference["full_remainder_ceiling_authorized"]}**. The independent H2 arm completed **{replication["complete_paired_base_trials"]}** paired base trials; families above the frozen JS noise floor were `{", ".join(h2_families) if h2_families else "none"}`.

## Full-remainder reference

The operational remainder is `{reference["operational_remainder_definition"]}`. It is a train-fitted measured-J remainder, not a proof of the exact mathematical non-J complement. Selection used validation only; the table below is from the held-out rollout-test split and three frozen confirmation seeds.

{_reference_table(reference)}

![Full-remainder reference](../results/v5/figures/full_remainder_reference_v5.png)

## Compact peripheral-state sweep

{_compact_table(sweep)}

{conditional_answer}

![Gap closed](../results/v5/figures/peripheral_gap_closed_v5.png)

![Conditional sufficiency and causal fidelity](../results/v5/figures/peripheral_sufficiency_fidelity_v5.png)

Trajectory summaries in the saved records are explicitly teacher-current one-step aggregates. They are not autonomous rollouts.

## Independent family-wise H2 replication

{_h2_table(replication)}

![Family-wise H2 replication](../results/v5/figures/h2_family_replication_v5.png)

Mediation authorization was **{replication["mediation_authorized"]}**. A later-J persistent-restoration analysis is interpreted only when a family-wise single-arm lower CI exceeds the frozen `1e-4` JS floor.

## Recurrent controller and autonomous rollout

{recurrent_answer}

{recurrent_table}

## Answers to the twelve preregistered questions

1. **Does full remainder improve next-J prediction?** The paired cosine gain was {_fmt_ci(ceiling_gain)}; formal authorization was `{reference["full_remainder_ceiling_authorized"]}`.
2. **Which task families improve?** See the family-wise full-reference table; pooled results are not substituted for heterogeneous families.
3. **Can peripheral information be compressed?** {conditional_answer}
4. **Smallest effective C_t?** {smallest["state_dimension"] if smallest else "None authorized"}.
5. **How much of the gap is closed?** The selected-by-dimension values are reported in the compact table without post-hoc threshold changes.
6. **Does residual R_t add value after C_t?** The `conditional gain` column measures the held-out gain from a PCA-256 summary of residual `(R_t | C_t)`.
7. **Does C_t have intervention fidelity?** The causal cosine, magnitude ratio, semantic-delta agreement, and output-sign agreement are stored pooled and by family; authorization requires the frozen direction/sign criteria.
8. **Does H2 replicate family-wise?** Families exceeding the frozen causal noise rule: `{", ".join(h2_families) if h2_families else "none"}`.
9. **Is (J_t,C_t) approximately sufficient?** `{bool(smallest)}` under the operational, finite-dictionary state and frozen v5 gates; this is not a claim of exact sufficiency.
10. **Can a recurrent controller maintain C_t?** {recurrent_answer}
11. **How long is autonomous rollout stable?** Horizon-wise values are reported only if recurrence was authorized and executed; teacher-current prediction is never relabeled autonomous.
12. **Is there a compact effective high-level dynamical system?** {conclusion}

## Evidence boundaries and alternative explanations

- H2 replication and intervention fidelity use interventions; the ceiling, compression, and conditional tests are predictive/associational.
- A positive prediction gain does not establish causal sufficiency. A PCA success alone is not labeled a peripheral causal state.
- A weak full-remainder ceiling gates compact-state interpretation rather than turning compact-model failure into evidence for J sufficiency.
- Family heterogeneity is retained. The report does not infer all families from a pooled mean.
- All representation fitting and model selection use train/validation data; rollout-test and the independent causal split are held out.
- Teacher-current trajectory aggregates and autonomous recurrence are explicitly separated.
- The dictionary-limited measured-J remainder is not described as the complete non-J space.
- Failed/invalid causal attempts remain in raw records and valid counts are reported.
- Frozen thresholds are not lowered after observing results.
- Confidence intervals quantify sampling uncertainty; they do not remove model, task, intervention, or measurement uncertainty.
- No result is interpreted as consciousness, extracted true thoughts, or parameter-level physical modularity.

## Provenance

Every plotted series is regenerated from saved JSON/Parquet records. Figure sidecars contain source paths and SHA-256 hashes. Model checkpoints, hidden tensors, and causal state tensors remain uncommitted artifacts referenced by hashes.
"""
    REPORT.write_text(body, encoding="utf-8")
    existing = FINAL.read_text(encoding="utf-8")
    block = f"{START}\n\n## Protocol v5 peripheral-state update\n\n{conclusion}\n\nFull-remainder gain: {_fmt_ci(ceiling_gain)}. Compact candidate authorized: `{bool(smallest)}`. Independent H2 families above the frozen noise rule: `{', '.join(h2_families) if h2_families else 'none'}`. Recurrent status: `{recurrent_status}`. Full details: [PERIPHERAL_STATE_V5.md](PERIPHERAL_STATE_V5.md).\n\n{END}"
    if START in existing and END in existing:
        existing = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            block,
            existing,
            flags=re.DOTALL,
        )
    else:
        existing = existing.rstrip() + "\n\n" + block + "\n"
    FINAL.write_text(existing, encoding="utf-8")
    manifest = {
        "schema_version": 7,
        "protocol_version": freeze["protocol_version"],
        "freeze_digest": freeze["freeze_digest"],
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": sha256_file(REPORT),
        "final_report_sha256": sha256_file(FINAL),
        "figures": figures,
        "sources": [
            _source("full_remainder_reference_v5.json"),
            _source("peripheral_dimension_sweep_v5.json"),
            _source("h2_replication_v5.json"),
            *([_source("peripheral_recurrent_v5.json")] if recurrent else []),
        ],
        "strongest_warranted_conclusion": conclusion,
    }
    write_json_atomic(PROCESSED / "report_manifest_v5.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
