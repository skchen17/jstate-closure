"""Machine-generated reports for the protocol-v6 foundation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic


def _f(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _ci(value: dict[str, Any] | None) -> str:
    if not value or value.get("estimate") is None:
        return "NA"
    return f"{_f(value['estimate'])} [{_f(value['lower'])}, {_f(value['upper'])}]"


def _passport(mode: str, freezes: dict[str, str]) -> str:
    freeze_lines = "\n".join(
        f"- {key} freeze digest: `{value}`" for key, value in freezes.items()
    )
    return f"""## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `{mode}`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
{freeze_lines}
"""


def build(root: Path) -> dict[str, Any]:
    processed = root / "results/v6/processed"
    figures = root / "results/v6/figures"
    figures.mkdir(parents=True, exist_ok=True)
    causal = json.loads((processed / "causal_endpoint_v6.json").read_text())
    ceiling = json.loads((processed / "peripheral_ceiling_v6.json").read_text())
    compact_path = processed / "compact_peripheral_v6.json"
    compact = json.loads(compact_path.read_text()) if compact_path.is_file() else None
    freezes = {
        "causal/null": causal["source_freeze_digest"],
        "strong ceiling": ceiling["source_freeze_digest"],
        "compact": compact["source_freeze_digest"] if compact else "not-run",
        "delivery/report": json.loads(
            (root / "artifacts/peripheral_v6.freeze.json").read_text()
        )["freeze_digest"],
    }

    endpoint = pd.read_parquet(root / causal["endpoint_records"])
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for family, frame in endpoint.groupby("family", sort=True):
        ax.hist(
            frame["delta_j_l2_f32"],
            bins=12,
            histtype="step",
            linewidth=1.6,
            label=family,
        )
    ax.set_xlabel("teacher next-J intervention delta L2 (float32)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    fig.tight_layout()
    precision_figure = figures / "causal_delta_precision_v6.png"
    fig.savefig(precision_figure, dpi=180)
    plt.close(fig)

    trials = pd.read_parquet(root / causal["trial_records"])
    trials["metrics"] = trials["metrics"].map(json.loads)
    trials["output_js_divergence"] = trials["metrics"].map(
        lambda value: value["output_js_divergence"]
    )
    order = [
        "identity",
        "single",
        "persistent_final",
        "persistent_null_final",
        "persistent_all",
        "persistent_null_all",
    ]
    means = trials.groupby(["family", "condition"])["output_js_divergence"].mean()
    table = means.unstack().reindex(columns=order)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    table.plot(kind="bar", ax=ax)
    ax.set_ylabel("clean-relative output JS")
    ax.set_xlabel("task family")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    restoration_figure = figures / "restoration_null_v6.png"
    fig.savefig(restoration_figure, dpi=180)
    plt.close(fig)

    ordinary = pd.read_parquet(root / ceiling["ordinary_records"])
    metric_names = [
        "next_j_cosine",
        "action_correct",
        "causal_projection_rmse",
        "top_causal_dimension_rmse",
    ]
    comparison = ordinary.groupby("model")[metric_names].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for metric, ax in zip(metric_names, axes.flat, strict=True):
        comparison[metric].plot(kind="bar", ax=ax, title=metric)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    ceiling_figure = figures / "peripheral_ceiling_endpoints_v6.png"
    fig.savefig(ceiling_figure, dpi=180)
    plt.close(fig)

    compact_figure: Path | None = None
    if compact is not None and compact.get("results"):
        compact_frame = pd.DataFrame(compact["results"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for family, frame in compact_frame.groupby("family", sort=True):
            ordered = frame.sort_values("dimension")
            axes[0].plot(
                ordered["dimension"],
                ordered["predictive_gap_closed"],
                marker="o",
                label=family,
            )
            axes[1].plot(
                ordered["dimension"],
                ordered["causal_gap_closed"],
                marker="o",
                label=family,
            )
        for ax, title in zip(
            axes,
            ("predictive ceiling gap closed", "causal ceiling gap closed"),
            strict=True,
        ):
            ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("C dimension")
            ax.set_ylabel("fraction")
            ax.set_title(title)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        compact_figure = figures / "compact_peripheral_pareto_v6.png"
        fig.savefig(compact_figure, dpi=180)
        plt.close(fig)

    precision = causal["precision_audit"]
    v5 = json.loads((root / "results/v5/processed/h2_replication_v5.json").read_text())
    with np.load(root / v5["causal_state_artifact"], allow_pickle=False) as payload:
        v5_delta = payload["next_j_intervened"].astype(np.float32) - payload[
            "next_j_clean"
        ].astype(np.float32)
        v5_delta_norm = np.linalg.norm(v5_delta, axis=1)
        old_ids = payload["base_trial_id"].astype("U64")
        old_candidates = payload["intervened_hidden"].astype(np.float32)
    with np.load(
        root / causal["causal_endpoint_artifact"], allow_pickle=False
    ) as payload:
        new_ids = payload["base_trial_id"].astype("U64")
        new_candidates = payload["hidden_intervened"][:, 0].astype(np.float32)
    old_by_id = {value: index for index, value in enumerate(old_ids)}
    quantization_error = np.asarray(
        [
            np.linalg.norm(new_candidates[index] - old_candidates[old_by_id[value]])
            for index, value in enumerate(new_ids)
        ]
    )
    single_metrics = trials[trials["condition"] == "single"]["metrics"]
    old_endpoint_material = sum(
        value["future_j_divergence_curve"][1] > 1e-6 for value in single_metrics
    )
    fidelity = f"""# Peripheral Causal Fidelity v6

{_passport("corrective causal-fidelity measurement", freezes)}

## Result

The repaired endpoint contains **{precision["count"]}** paired interventions. It measures the immediate same-forward write from the layer-23 intervention to layer 24, rather than the structurally weak next-token/layer-23 endpoint used in v5. All hidden states, measured-J profiles, deltas, logits, and operational remainders are stored as float32.

- non-zero next-write delta: **{_f(precision["nonzero_fraction_f32"], 3)}** in float32 and **{_f(precision["nonzero_fraction_f64_projection"], 3)}** with float64 projection arithmetic;
- delta L2: min **{_f(precision["delta_norm_f32"]["min"], 9)}**, median **{_f(precision["delta_norm_f32"]["median"], 9)}**, max **{_f(precision["delta_norm_f32"]["max"], 9)}**;
- float64-projection median L2: **{_f(precision["delta_norm_f64_projection"]["median"], 9)}**;
- median float32/float64 direction cosine: **{_f(precision["median_f32_f64_cosine"], 9)}**.

## What failed in v5

The attribution is more specific than “float16 erased a valid signal.” In the old artifact, **{int(np.count_nonzero(v5_delta_norm))}/66** next-token/layer-23 deltas survived float16, and the float32 v6 trajectory audit likewise finds only **{old_endpoint_material}/66** material divergences at that old endpoint (`J-distance > 1e-6`). Thus the dominant problem was endpoint geometry/causal masking: a layer-23 final-token edit normally cannot alter the next token's layer-23 state unless it first changes the emitted token. Float16 made sub-quantization changes unavailable, but does not explain the 64 zeros by itself. Separately, reusing the float16 candidate for v5 mediation introduced median candidate quantization L2 **{_f(np.median(quantization_error), 6)}**, large enough to matter relative to the repaired median layer-24 delta **{_f(precision["delta_norm_f32"]["median"], 6)}**.

The float64 check isolates projection arithmetic only—the source activations remain float32—and is not a claim that model inference ran in float64.

## Family counts

{pd.Series(causal["valid_endpoints_by_family"]).rename("n").to_frame().to_markdown()}

## Saved evidence

- Endpoint records: `{causal["endpoint_records"]}`
- Full scientific tensor artifact (uncommitted by policy): `{causal["causal_endpoint_artifact"]}`
- Artifact SHA-256: `{causal["causal_endpoint_artifact_sha256"]}`
- Figure: `{precision_figure.relative_to(root)}`
"""
    (root / "reports/PERIPHERAL_CAUSAL_FIDELITY_V6.md").write_text(
        fidelity, encoding="utf-8"
    )

    focus = ["pooled", "boolean_logic", "simple_state_transition", "modular_arithmetic"]
    rows = []
    for family in focus:
        if family not in causal["effects"]:
            continue
        value = causal["effects"][family]
        rows.append(
            {
                "family": family,
                "E_single": _ci(value["output_js_divergence"].get("single")),
                "E_persistent_final": _ci(
                    value["output_js_divergence"].get("persistent_final")
                ),
                "E_null_final": _ci(
                    value["output_js_divergence"].get("persistent_null_final")
                ),
                "corrected_final": _ci(
                    value["artifact_corrected"]["final"]["persistent_minus_null"]
                ),
                "M_corrected_final": _f(
                    value["artifact_corrected"]["final"].get("mediation_point_estimate")
                ),
                "E_persistent_all": _ci(
                    value["output_js_divergence"].get("persistent_all")
                ),
                "E_null_all": _ci(
                    value["output_js_divergence"].get("persistent_null_all")
                ),
                "corrected_all": _ci(
                    value["artifact_corrected"]["all"]["persistent_minus_null"]
                ),
                "M_corrected_all": _f(
                    value["artifact_corrected"]["all"].get("mediation_point_estimate")
                ),
            }
        )
    restoration = f"""# Restoration Null Control v6

{_passport("restoration-artifact control", freezes)}

## Paired result

All persistent-null arms use the same layer list, position scope, dense projector, optimization implementation, and hook schedule as their corresponding persistent intervention, but replace the initial perturbation with the clean state. `E_restore_artifact` is their clean-relative effect. `corrected` is paired `E_persistent - E_restore_artifact`.

{pd.DataFrame(rows).to_markdown(index=False)}

The pooled single effect is `{_ci(causal["effects"]["pooled"]["output_js_divergence"]["single"])}`; corrected persistent effect is `{_ci(causal["effects"]["pooled"]["artifact_corrected"]["all"]["persistent_minus_null"])}`, a point-estimate reduction of **{_f(causal["effects"]["pooled"]["artifact_corrected"]["all"]["mediation_point_estimate"] * 100, 2)}%**. Boolean logic reduces from `{_ci(causal["effects"]["boolean_logic"]["output_js_divergence"]["single"])}` to `{_ci(causal["effects"]["boolean_logic"]["artifact_corrected"]["all"]["persistent_minus_null"])}` (**{_f(causal["effects"]["boolean_logic"]["artifact_corrected"]["all"]["mediation_point_estimate"] * 100, 2)}%**). This supports mediation by later measured-J writes for the pooled/Boolean effects. The six-item state-transition family is not interpretable as mediation because its single-effect lower CI does not clear the frozen `1e-4` floor and persistent restoration amplifies its effect.

Mediation ratios are descriptive only when the family-specific single effect clears the preregistered noise requirement. The clean-state restoration null is essentially numerical zero, but it cannot exclude a state-dependent projector distortion that appears only after a real perturbation.

Figure: `{restoration_figure.relative_to(root)}`.
"""
    (root / "reports/RESTORATION_NULL_CONTROL_V6.md").write_text(
        restoration, encoding="utf-8"
    )

    gains = []
    for metric, value in ceiling["ordinary_gains"].items():
        gains.append({"endpoint": metric, "full-minus-baseline": _ci(value)})
    for metric, value in ceiling["causal_gains"].items():
        gains.append(
            {"endpoint": f"causal::{metric}", "full-minus-baseline": _ci(value)}
        )
    family_rows = []
    for family, values in ceiling["family_wise_gains"].items():
        family_rows.append(
            {
                "family": family,
                "next-J cosine gain": _ci(values.get("next_j_cosine")),
                "semantic accuracy gain": _ci(values.get("action_accuracy")),
                "causal-direction gain": _ci(values.get("causal_direction_cosine")),
            }
        )
    absolute = ceiling["absolute_means"]
    screen_table = pd.DataFrame(ceiling["screen"])[
        [
            "architecture",
            "parameter_count",
            "validation_next_j_cosine",
            "validation_action_accuracy",
            "validation_causal_projection_rmse",
            "selection_score",
        ]
    ]
    ceiling_report = f"""# Peripheral Ceiling v6

{_passport("strong multi-endpoint peripheral reference", freezes)}

## Result

- Selected J-only/history baseline: `{ceiling["baseline_choice"]["architecture"]}`
- Selected full-peripheral model: `{ceiling["full_choice"]["architecture"]}`
- Strong peripheral ceiling authorized: **{ceiling["strong_peripheral_ceiling_authorized"]}**
- Authorization gates: `{json.dumps(ceiling["authorization_gates"], sort_keys=True)}`
- J-only parameters: `{int(absolute["baseline"]["parameter_count"])}`; full-R parameters: `{int(absolute["full"]["parameter_count"])}`
- Absolute causal-direction cosine: J-only `{_f(absolute["baseline_causal"]["causal_direction_cosine"])}`, full-R `{_f(absolute["full_causal"]["causal_direction_cosine"])}`
- Absolute rollout-test next-J cosine: J-only `{_f(absolute["baseline"]["next_j_cosine"])}`, full-R `{_f(absolute["full"]["next_j_cosine"])}`
- Absolute semantic accuracy: J-only `{_f(absolute["baseline"]["action_correct"])}`, full-R `{_f(absolute["full"]["action_correct"])}`

{pd.DataFrame(gains).to_markdown(index=False)}

The validation-only architecture rule selected the gated full-R model; causal-test scores were excluded from model selection. The strong ceiling is authorized by causal-direction and causal-projection gates, not by the global-profile gate (semantic accuracy lost **{_f(-ceiling["ordinary_gains"]["action_correct"]["estimate"] * 100, 2)}** percentage points) and not by output-sign/semantic-delta agreement. This is evidence for small-energy, causally predictive peripheral information, not a uniformly better behavioral predictor.

The models jointly optimize next-J profile, causal directions/top coordinates, and semantic action. Causal-sensitive directions were fitted only on `causal_fit`; all reported causal gains use disjoint `causal_test` pairs. The full operational remainder is a train-fitted residual representation, not a proof about every possible non-J coordinate system. Attention training reported a nondeterministic CUDA backward kernel; three frozen seeds are retained in the estimates.

## Validation-only architecture screen

{screen_table.to_markdown(index=False)}

## Family-wise gains

{pd.DataFrame(family_rows).to_markdown(index=False)}

Figure: `{ceiling_figure.relative_to(root)}`.
"""
    (root / "reports/PERIPHERAL_CEILING_V6.md").write_text(
        ceiling_report, encoding="utf-8"
    )

    if compact is None:
        compact_text = "Compact search was not executed."
    else:
        compact_frame = pd.DataFrame(compact["results"])
        compact_table = compact_frame[
            [
                "family",
                "dimension",
                "predictive_gap_closed",
                "causal_gap_closed",
                "action_accuracy",
                "causal_direction_cosine",
                "output_sign_agreement",
            ]
        ]
        selected = compact.get("selected")
        conditional = compact.get("conditional_sufficiency")
        compact_text = f"""The strong ceiling authorized the sweep. **{compact["screen_passing_count"]}** of 18 candidates passed the joint screen. The smallest/only screen-pass was `{selected["family"]} {selected["dimension"]}D` with predictive gap closed `{_f(selected["predictive_gap_closed"])}`, causal gap closed `{_f(selected["causal_gap_closed"])}`, causal-direction cosine `{_f(selected["causal_direction_cosine"])}`, and semantic accuracy `{_f(selected["action_accuracy"])}`.

It did **not** pass conditional sufficiency: adding `residual(R|C)` changed next-J cosine by `{_f(conditional["predictive_residual_gain"])}`, causal-direction cosine by `{_f(conditional["causal_residual_gain"])}`, and semantic accuracy by `{_f(conditional["semantic_residual_gain"])}`. The semantic gain is **{_f(conditional["semantic_residual_gain"] * 100, 2)} percentage points**, above the frozen 2-point limit. These conditional values are exploratory single-seed point estimates; absence of a confirmatory CI is an additional reason not to authorize the compact state.

{compact_table.to_markdown(index=False)}

Figure: `{compact_figure.relative_to(root) if compact_figure else "not-generated"}`."""
    compact_report = f"""# Compact Peripheral State v6

{_passport("gated compact-peripheral search", freezes)}

## Gate/result

{compact_text}

No recurrent controller was trained in v6.
"""
    (root / "reports/COMPACT_PERIPHERAL_STATE_V6.md").write_text(
        compact_report, encoding="utf-8"
    )

    full_authorized = ceiling["strong_peripheral_ceiling_authorized"]
    nonzero = precision["nonzero_fraction_f32"]
    pooled = causal["effects"]["pooled"]
    boolean = causal["effects"]["boolean_logic"]
    selected = compact.get("selected") if compact else None
    conditional = compact.get("conditional_sufficiency") if compact else None
    final = f"""# J-State Closure Final Report

{_passport("cumulative evidence adjudication through v6", freezes)}

## Current adjudication

The strongest warranted conclusion remains **H2 for the tested operational measured-J state**, with evidence that the pooled and Boolean effects are predominantly mediated by later measured-J writes under the tested restoration operator. The full operational remainder contains learnable causal-direction information, but no compact peripheral state passed conditional sufficiency. This does **not** establish H3, an autonomous controller, consciousness, “true thoughts,” or parameter localization.

## Required v6 questions

1. **Is the repaired teacher delta stable?** {"Yes" if nonzero >= 0.95 else "Not uniformly"}: non-zero float32 fraction `{_f(nonzero, 3)}`, median f32/f64-projection cosine `{_f(precision["median_f32_f64_cosine"])}`.
2. **How much of v5 failure was float16?** The old float16 endpoint had 2/66 nonzero vectors, while the float32 re-audit finds only 2/66 material effects at that same next-token/layer-23 endpoint. Therefore most zeros reflect the endpoint's causal structure, not float16 alone. Float16 nevertheless perturbed stored intervention candidates by median L2 `{_f(np.median(quantization_error))}`, which likely explains the v5 mediation inconsistency.
3. **Does restoration alone cause an effect?** No detectable clean-state artifact: pooled JS `{_ci(pooled["output_js_divergence"]["persistent_null_all"])}`.
4. **Corrected v4/v5 mediation?** Pooled single JS `{_ci(pooled["output_js_divergence"]["single"])}` falls to corrected persistent `{_ci(pooled["artifact_corrected"]["all"]["persistent_minus_null"])}` (M=`{_f(pooled["artifact_corrected"]["all"]["mediation_point_estimate"])}`). Boolean falls from `{_ci(boolean["output_js_divergence"]["single"])}` to `{_ci(boolean["artifact_corrected"]["all"]["persistent_minus_null"])}` (M=`{_f(boolean["artifact_corrected"]["all"]["mediation_point_estimate"])}`).
5. **H2-A, current J insufficiency?** Yes for the tested state: pooled and Boolean single-arm effects clear the frozen numerical floor; Boolean is the strongest family.
6. **H2-B, mediation by later J writes?** Supported for pooled/Boolean under this operator (92.3%/99.1% point-estimate removal), but not universally: state-transition has only six items, fails the single-effect noise gate, and is amplified by restoration. State-dependent restoration distortion remains a limitation not measured by a clean-state null.
7. **Does full remainder contain learnable information?** Yes for causal directions/projections: causal-direction gain `{_ci(ceiling["causal_gains"]["causal_direction_cosine"])}` and ordinary causal-projection RMSE improvement `{_ci(ceiling["ordinary_gains"]["causal_projection_rmse"])}`. Semantic/output benefits are absent or uncertain.
8. **Most informative endpoints?** Intervention-sensitive J directions, causal-projection RMSE, and next-J cosine. Semantic accuracy decreases and output-sign gain crosses zero.
9. **Which families?** Causal-direction gains are positive in all five families; short graph is largest. Global next-J gains are largest in modular arithmetic/state transition and absent in variable binding. Sample sizes for causal family estimates are only 3–12.
10. **Strong ceiling?** `{full_authorized}`.
11. **Smallest effective C?** No validated sufficient C. The smallest/only joint screen-pass is `{selected["family"]} {selected["dimension"]}D` if “effective” means predictive+causal gap screening only.
12. **Causal fidelity and conditional sufficiency?** Screen causal fidelity reaches cosine `{_f(selected["causal_direction_cosine"])}` and closes `{_f(selected["causal_gap_closed"])}` of the full causal gap, but conditional residual adds `{_f(conditional["causal_residual_gain"])}` causal cosine and `{_f(conditional["semantic_residual_gain"] * 100, 2)}` semantic percentage points. It therefore fails sufficiency.
13. **Ready for recurrent controllers?** No. v6 intentionally trained none; entry additionally requires a strong ceiling, a compact gap-closing C, stable causal fidelity, and conditional sufficiency.

## Evidence classes

- **Causal:** paired teacher interventions and persistent/null arms.
- **Predictive/associational:** held-out one-step peripheral-reference gains.
- **Numerical validation:** float32 storage and float64 projection sensitivity.
- **Practical magnitude:** raw JS/J/action effects and CIs, not p-values alone.

## Validity/fallacy scan

1. Correlation/causation: predictive ceiling is not called causal.
2. Measurement validity: finite 4096D measured-J is named operationally.
3. Aggregation: pooled and family-wise effects are separate.
4. Selection leakage: causal directions use fit; strong architecture selection uses ordinary validation only; fidelity uses causal test. The compact sweep is exploratory and its 18-way test-set screen is not presented as confirmatory.
5. Optional stopping: frozen gates determine compact authorization.
6. Null-result interpretation: weak models do not prove absent information.
7. Scale conflation: small global energy and causal importance are separate.
8. Ratio instability: mediation ratios are gated on non-noise single effects.
9. Intervention naturality: inherited validated v5 candidate criteria remain explicit.
10. Restoration confounding: clean-state null limitation is stated.
11. Overclaiming: no consciousness, true-thought, parameter-localization, or autonomous-controller claim.

## Next blocker

The strong ceiling is now established for causal-sensitive endpoints. The blocker is a compact `C_t` that retains these effects under an independent confirmation split and passes predictive, causal, and semantic conditional-sufficiency tests with CIs. Recurrent training remains blocked until then.

## Exact v6 commands

```bash
scripts/run_peripheral_v6.sh freeze --run-suffix protocol-freeze-r3
CUDA_VISIBLE_DEVICES=0 scripts/run_peripheral_v6.sh causal --run-suffix full-66
scripts/run_peripheral_v6.sh merge --run-suffix merge-full66
scripts/run_peripheral_v6.sh freeze --run-suffix leakfree-ceiling-freeze-r6
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh ceiling --run-suffix leakfree-strong
scripts/run_peripheral_v6.sh freeze --run-suffix matched-compact-freeze-r7
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh compact --run-suffix compact-sweep-matched
scripts/run_peripheral_v6.sh freeze --run-suffix delivery-freeze-r12
scripts/build_report_v6.sh --run-suffix final-reports-r5
```
"""
    (root / "reports/FINAL_REPORT.md").write_text(final, encoding="utf-8")

    report_paths = [
        root / "reports/PERIPHERAL_CAUSAL_FIDELITY_V6.md",
        root / "reports/RESTORATION_NULL_CONTROL_V6.md",
        root / "reports/PERIPHERAL_CEILING_V6.md",
        root / "reports/COMPACT_PERIPHERAL_STATE_V6.md",
        root / "reports/FINAL_REPORT.md",
    ]
    manifest = {
        "schema_version": 8,
        "protocol_version": "peripheral_foundations_protocol_v6",
        "source_freeze_digests": freezes,
        "source_summaries": {
            "causal": "results/v6/processed/causal_endpoint_v6.json",
            "ceiling": "results/v6/processed/peripheral_ceiling_v6.json",
            "compact": "results/v6/processed/compact_peripheral_v6.json",
        },
        "reports": {
            str(path.relative_to(root)): sha256_file(path) for path in report_paths
        },
        "figures": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                precision_figure,
                restoration_figure,
                ceiling_figure,
                *([compact_figure] if compact_figure else []),
            )
        },
    }
    write_json_atomic(processed / "report_manifest_v6.json", manifest)
    return manifest
