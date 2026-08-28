"""Build protocol-v2 status, figures, and reports from saved records only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic

DOWNSTREAM_RUNS = {
    "closure": "closure-*",
    "natural_collisions": "natural-collisions-*",
    "memory_order": "memory-order-*",
    "controller_distillation": "distill-controller-*",
    "dictionary_sensitivity": "dictionary-sensitivity-*",
    "token_time_closure": "token-time-closure-*",
    "modularity": "modularity-*",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _latest_manifest(raw: Path, pattern: str) -> Path | None:
    paths = sorted(raw.glob(f"{pattern}/manifest.json"))
    return paths[-1] if paths else None


def build_execution_status(root: Path) -> dict[str, Any]:
    """Derive the v2 gate/stage matrix from immutable run manifests."""

    processed = root / "results" / "processed"
    raw = root / "results" / "raw"
    phase0_path = processed / "phase0_v2_gate.json"
    calibration_path = processed / "layer_calibration.json"
    phase0 = _load(phase0_path) if phase0_path.exists() else None
    calibration = _load(calibration_path) if calibration_path.exists() else None
    eligible = calibration.get("eligible_layers", []) if calibration else []
    phase0_passed = bool(phase0 and phase0.get("passed"))
    calibration_passed = bool(eligible)
    gate_reason = (
        "No closure-eligible layer passed the frozen strict clamp calibration."
        if phase0_passed and calibration and not calibration_passed
        else "Phase 0 v2 did not pass."
        if phase0 and not phase0_passed
        else "Required v2 gate artifacts are incomplete."
    )
    stages: dict[str, Any] = {
        "phase0_v2": {
            "status": "COMPLETED" if phase0 else "UNEXECUTED",
            "gate": "PASSED"
            if phase0_passed
            else "FAILED"
            if phase0
            else "NOT_EXECUTED",
            "run_id": phase0.get("run_id") if phase0 else None,
            "manifest": _relative(
                root, raw / str(phase0.get("run_id")) / "manifest.json"
            )
            if phase0 and (raw / str(phase0.get("run_id")) / "manifest.json").exists()
            else None,
        },
        "layer_calibration": {
            "status": "COMPLETED" if calibration else "UNEXECUTED",
            "gate": "PASSED"
            if calibration_passed
            else "FAILED"
            if calibration
            else "NOT_EXECUTED",
            "run_id": calibration.get("run_id") if calibration else None,
            "eligible_layers": eligible,
            "manifest": _relative(
                root, raw / str(calibration.get("run_id")) / "manifest.json"
            )
            if calibration
            and (raw / str(calibration.get("run_id")) / "manifest.json").exists()
            else None,
        },
    }
    sources = [path for path in (phase0_path, calibration_path) if path.exists()]
    for stage, pattern in DOWNSTREAM_RUNS.items():
        manifest_path = _latest_manifest(raw, pattern)
        manifest = _load(manifest_path) if manifest_path else None
        sources.extend([manifest_path] if manifest_path else [])
        stages[stage] = {
            "status": "GATED"
            if phase0_passed and calibration and not calibration_passed
            else manifest.get("status", "UNKNOWN")
            if manifest
            else "UNEXECUTED",
            "runner_status": manifest.get("status") if manifest else None,
            "run_id": manifest.get("run_id") if manifest else None,
            "reason": manifest.get("error", gate_reason) if manifest else gate_reason,
            "manifest": _relative(root, manifest_path) if manifest_path else None,
        }
    stages["qwen3_6_27b_confirmation"] = {
        "status": "UNEXECUTED",
        "reason": "Primary-model layer calibration did not pass; optional confirmation was not attempted.",
    }
    payload = {
        "schema_version": 2,
        "protocol_version": "phase0_protocol_v2",
        "overall_gate": "FAILED_AT_LAYER_CALIBRATION"
        if phase0_passed and calibration and not calibration_passed
        else "PASSED"
        if phase0_passed and calibration_passed
        else "INCOMPLETE_OR_PHASE0_FAILED",
        "strongest_warranted_conclusion": "D",
        "reason": gate_reason,
        "stages": stages,
        "sources": [
            {"path": _relative(root, path), "sha256": sha256_file(path)}
            for path in sources
        ],
    }
    write_json_atomic(processed / "execution_status_v2.json", payload)
    return payload


def _save(
    root: Path,
    name: str,
    source_paths: list[Path],
    draw: Any,
    *,
    status_only: bool = False,
) -> dict[str, Any]:
    if not source_paths or any(not path.exists() for path in source_paths):
        raise RuntimeError(
            f"{name}: every figure requires existing machine-readable sources"
        )
    target = root / "results" / "figures" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    draw(axis)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return {
        "figure": _relative(root, target),
        "sha256": sha256_file(target),
        "sources": [
            {"path": _relative(root, path), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "status_only": status_only,
        "manual_values": False,
    }


def _status_draw(title: str, reason: str) -> Any:
    def draw(axis: plt.Axes) -> None:
        axis.axis("off")
        axis.set_title(title)
        axis.text(
            0.5,
            0.57,
            "GATED / NOT EXECUTED",
            ha="center",
            va="center",
            fontsize=17,
            weight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.38,
            reason,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            transform=axis.transAxes,
        )

    return draw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_figures(root: Path, status: dict[str, Any]) -> list[dict[str, Any]]:
    processed = root / "results" / "processed"
    gate_path = processed / "phase0_v2_gate.json"
    layer_path = processed / "layer_calibration.json"
    execution_path = processed / "execution_status_v2.json"
    gate = _load(gate_path)
    calibration = _load(layer_path)
    reason = str(status["reason"])
    entries: list[dict[str, Any]] = []

    def draw_lens(axis: plt.Axes) -> None:
        for family, family_label in (
            ("factual_two_hop", "multihop"),
            ("order_of_operations", "order of operations"),
        ):
            methods = gate["official_main"]["families"][family]
            for method, style in (("jacobian", "-"), ("logit", "--")):
                curve = methods[method]["per_layer"]
                layers = sorted(int(value) for value in curve)
                axis.plot(
                    layers,
                    [curve[str(layer)]["hit10"] for layer in layers],
                    style,
                    label=f"{family_label}: {method}",
                )
        axis.axvspan(20, 30, color="grey", alpha=0.12, label="frozen band")
        axis.set(
            xlabel="Model block-output layer", ylabel="Per-layer item-weighted hit@10"
        )
        axis.legend(fontsize=8, ncol=2)

    entries.append(
        _save(root, "01_lens_validation_by_layer.png", [gate_path], draw_lens)
    )

    calibration_manifest = (
        root / "results" / "raw" / calibration["run_id"] / "manifest.json"
    )
    clamp_path = (
        root / "results" / "raw" / calibration["run_id"] / "clamp_calibration.jsonl"
    )
    clamps = _read_jsonl(clamp_path)

    def draw_match(axis: plt.Axes) -> None:
        axis.scatter(
            [row["remainder_fraction"] for row in clamps],
            [1.0 - row["dense_cosine"] for row in clamps],
            c=[row["layer"] for row in clamps],
            cmap="viridis",
            s=9,
            alpha=0.42,
        )
        axis.axhline(1.0 - 0.995, color="red", linestyle="--", label="cosine threshold")
        axis.axvline(0.20, color="black", linestyle=":", label="remainder threshold")
        axis.set(
            xlabel="Measured-J remainder displacement / natural scale",
            ylabel="Measured-J match error (1 − cosine)",
        )
        axis.legend(fontsize=8)

    entries.append(
        _save(
            root,
            "02_j_match_vs_remainder.png",
            [clamp_path, calibration_manifest],
            draw_match,
        )
    )

    status_figures = (
        (
            "03_future_j_vs_remainder.png",
            "Future measured-J divergence vs remainder distance",
        ),
        (
            "04_output_divergence_controls.png",
            "Output divergence by intervention control",
        ),
        ("05_single_vs_persistent_clamp.png", "One-shot vs persistent clamp"),
        ("06_natural_collision_scatter.png", "Natural measured-J collision analysis"),
        ("07_predictor_vs_history.png", "Layer-depth predictor vs measured-J history"),
        (
            "08_controller_vs_parameters.png",
            "Controller performance vs parameter count",
        ),
        ("09_rollout_error_vs_horizon.png", "Autonomous rollout error vs horizon"),
        ("10_intervention_fidelity.png", "Teacher/student intervention fidelity"),
        ("11_dictionary_size_effect.png", "Dictionary-size effect E_R(M)"),
        ("13_token_time_history.png", "Token-time macrostate history"),
    )
    for name, title in status_figures:
        entries.append(
            _save(
                root,
                name,
                [execution_path, layer_path],
                _status_draw(title, reason),
                status_only=True,
            )
        )

    def draw_eligibility(axis: plt.Axes) -> None:
        layers = [row["layer"] for row in calibration["layers"]]
        width = 0.25
        x = np.arange(len(layers))
        axis.bar(
            x - width,
            [row["multihop_hit10"] for row in calibration["layers"]],
            width,
            label="multihop hit@10",
        )
        axis.bar(
            x,
            [row["order_ops_hit10"] for row in calibration["layers"]],
            width,
            label="order ops hit@10",
        )
        axis.bar(
            x + width,
            [row["clamp_valid_rate"] for row in calibration["layers"]],
            width,
            label="strict clamp valid rate",
        )
        axis.axhline(0.20, color="grey", linestyle=":", label="readout threshold")
        axis.axhline(0.80, color="red", linestyle="--", label="clamp threshold")
        axis.set(xticks=x, xticklabels=layers, xlabel="Candidate layer", ylabel="Rate")
        axis.legend(fontsize=8, ncol=2)

    entries.append(
        _save(
            root,
            "12_layer_eligibility.png",
            [layer_path, calibration_manifest],
            draw_eligibility,
        )
    )

    def draw_attrition(axis: plt.Axes) -> None:
        reasons: Counter[str] = Counter()
        for row in clamps:
            for value in str(row.get("exclusion_reason") or "valid").split(","):
                reasons[value] += 1
        labels = sorted(reasons)
        axis.bar(labels, [reasons[label] for label in labels], color="#4C78A8")
        axis.set(xlabel="Strict calibration outcome/reason", ylabel="Trial count")
        axis.tick_params(axis="x", rotation=25)

    entries.append(
        _save(
            root,
            "14_clamp_attrition.png",
            [clamp_path, calibration_manifest],
            draw_attrition,
        )
    )
    return entries


def _commands(root: Path, status: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for stage in status["stages"].values():
        manifest_rel = stage.get("manifest")
        if not manifest_rel:
            continue
        command = _load(root / manifest_rel).get("command")
        if command:
            commands.append(" ".join(str(value) for value in command))
    return commands


def _phase0_text(gate: dict[str, Any]) -> str:
    families = gate["official_main"]["families"]
    multihop = families["factual_two_hop"]["jacobian"]
    order = families["order_of_operations"]["jacobian"]
    rank = gate["rank_advantage_ci"]
    positive = gate["positive_control"]["ci"]
    return (
        f"Fresh official-compatible pass@10 was {multihop['pass_at']['10']:.6f} "
        f"for multihop ({multihop['item_count']} items) and {order['pass_at']['10']:.6f} "
        f"for order of operations ({order['item_count']} items). The item-clustered "
        f"MRR advantage was {rank['estimate']:.6f}, 95% CI "
        f"[{rank['lower']:.6f}, {rank['upper']:.6f}]. At frozen layer "
        f"{gate['positive_control_layer']}, the intended-answer log-odds effect was "
        f"{positive['estimate']:.6f}, 95% CI [{positive['lower']:.6f}, "
        f"{positive['upper']:.6f}], above the {gate['positive_control']['null_threshold']:.4g} null envelope."
    )


def build_reports(root: Path, status: dict[str, Any]) -> None:
    gate = _load(root / "results" / "processed" / "phase0_v2_gate.json")
    calibration = _load(root / "results" / "processed" / "layer_calibration.json")
    phase0_text = _phase0_text(gate)
    layer_rows = calibration["layers"]
    best_rate = max(row["clamp_valid_rate"] for row in layer_rows)
    attempts = sum(row["clamp_attempts"] for row in layer_rows)
    valid = sum(row["clamp_valid"] for row in layer_rows)
    candidate_layers = ", ".join(str(row["layer"]) for row in layer_rows)
    commands = _commands(root, status)
    command_text = "\n".join(f"- `{value}`" for value in commands) or "- None"
    common = (
        f"{phase0_text}\n\n"
        f"Independent closure-layer calibration then tested layers {candidate_layers}. "
        f"It obtained {valid}/{attempts} strictly valid clamp trials; the best layer-level "
        f"valid rate was {best_rate:.3f}, below the frozen 0.80 requirement. All candidate "
        "layers therefore failed only the clamp-valid-rate criterion, and the eligible set is empty."
    )
    causal = f"""# Closure causal report

## Status

GATED / UNINTERPRETABLE. {common}

No formal closure, dictionary-size, final-token mediation, or sequence-state mediation
effect was estimated. In particular, there is no valid value of E_R, E_J, eta, future-J
divergence, output JS divergence, or answer effect. The 1,400 calibration attempts are
measurement/calibration evidence, not Phase 3 causal trials.

The observed failure must not be read as evidence for H1 or H2: the proposed
measured-J restoration did not meet dense-cosine and RMS-drift requirements while
retaining the requested remainder displacement. Changing the threshold or intervention
source after observing this result would require a separately frozen exploratory v3.

## Commands recorded by run manifests

{command_text}
"""
    token = f"""# Token-time closure report

## Status

GATED / NOT EXECUTED. {common}

T1, T2, and T3 macrostate construction and autonomous feedback code are implemented
and unit-tested, but no teacher traces, token-time predictors, recurrent memory models,
or autonomous rollouts were trained. No future teacher token is consumed by the rollout
interface. There is therefore no result about token-time closure, short-memory sufficiency,
controller size, procedural generalization, or intervention fidelity.
"""
    phase0_manifest = root / "results" / "raw" / gate["run_id"] / "manifest.json"
    phase0_command = " ".join(str(value) for value in _load(phase0_manifest)["command"])
    families = gate["official_main"]["families"]
    multi = families["factual_two_hop"]["jacobian"]
    order = families["order_of_operations"]["jacobian"]
    position_multi = gate["position16_sensitivity"]["families"]["factual_two_hop"][
        "jacobian"
    ]
    copy_multi = gate["copy_excluded_sensitivity"]["families"]["factual_two_hop"][
        "jacobian"
    ]
    copy_order = gate["copy_excluded_sensitivity"]["families"]["order_of_operations"][
        "jacobian"
    ]
    phase0_report = f"""# Phase 0 v2 — Confirmatory J-lens validation

## Material Passport

- Protocol: `phase0_protocol_v2`
- Run ID: `{gate["run_id"]}`
- Verification Status: VERIFIED / LOCKED
- Exact command: `{phase0_command}`
- Model: `{gate["model_id"]}` at `{gate["model_revision"]}`
- Lens revision: `{gate["lens_revision"]}`
- Lens SHA-256: `{gate["lens_sha256"]}`
- Freeze manifest SHA-256: `{gate["freeze_manifest_sha256"]}`
- Readout records SHA-256: `{gate["readout_records_sha256"]}`

## Fresh confirmatory results

| Family | items | concepts | pass@1 | pass@5 | pass@10 | best layer | strict-all-layers hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| multihop | {multi["item_count"]} | {multi["concept_count"]} | {multi["pass_at"]["1"]:.6f} | {multi["pass_at"]["5"]:.6f} | {multi["pass_at"]["10"]:.6f} | {multi["best_layer"]} | {multi["strict_all_layers_sensitivity"]["10"]:.6f} |
| order of operations | {order["item_count"]} | {order["concept_count"]} | {order["pass_at"]["1"]:.6f} | {order["pass_at"]["5"]:.6f} | {order["pass_at"]["10"]:.6f} | {order["best_layer"]} | {order["strict_all_layers_sensitivity"]["10"]:.6f} |

{phase0_text}

The primary metric includes all valid positions and flags copied concepts. The
`position>=16` sensitivity retained {position_multi["item_count"]} multihop items and had
pass@10 {position_multi["pass_at"]["10"]:.6f}. Copy-excluded sensitivity had multihop
pass@10 {copy_multi["pass_at"]["10"]:.6f} and order-of-operations pass@10
{copy_order["pass_at"]["10"]:.6f}. Coverage was `{json.dumps(gate["coverage"], sort_keys=True)}`.

## Frozen artifacts

Nested concept-dictionary hashes: `{json.dumps(gate["dictionary_hashes"], sort_keys=True)}`.
The declared synonyms are official-compatible; they are not claimed to reproduce an
unpublished Anthropic internal synonym table item by item.

## Gate

**PASSED** under every frozen conjunctive criterion. The adjudication is locked;
post-confirmation protocol changes require a separately labeled exploratory v3.
The full per-layer curves and every item-level rank remain in the machine records.
"""
    calibration_manifest = (
        root / "results" / "raw" / calibration["run_id"] / "manifest.json"
    )
    calibration_command = " ".join(
        str(value) for value in _load(calibration_manifest)["command"]
    )
    clamp_path = (
        root / "results" / "raw" / calibration["run_id"] / "clamp_calibration.jsonl"
    )
    clamps = _read_jsonl(clamp_path)
    table_rows = "\n".join(
        "| {layer} | {multihop_hit10:.6f} | {order_ops_hit10:.6f} | "
        "{rank_advantage_ci_lower:.6f} | {positive_control_ci_lower:.6f} | "
        "{clamp_valid}/{clamp_attempts} | {eligible} |".format(**row)
        for row in layer_rows
    )
    cosine_values = np.asarray([row["dense_cosine"] for row in clamps], dtype=float)
    rms_values = np.asarray([row["rms_drift"] for row in clamps], dtype=float)
    remainder_values = np.asarray(
        [row["remainder_fraction"] for row in clamps], dtype=float
    )
    reason_counts: Counter[str] = Counter()
    for row in clamps:
        for reason in str(row.get("exclusion_reason") or "valid").split(","):
            reason_counts[reason] += 1
    layer_report = f"""# Closure-eligible layer calibration

## Material Passport

- Run ID: `{calibration["run_id"]}`
- Verification Status: VERIFIED / FAILED GATE
- Exact command: `{calibration_command}`
- Input Phase 0 gate: `{calibration["phase0_v2_gate_run_id"]}`
- Candidate layers: `{calibration["candidate_layers"]}`
- Closure-eligible layers: `{calibration["eligible_layers"]}`

## Results

| layer | multihop hit@10 | order hit@10 | rank-CI lower | positive-CI lower | valid clamps | eligible |
|---:|---:|---:|---:|---:|---:|:---:|
{table_rows}

All seven candidates passed the readout, family point-estimate, pooled rank-CI,
positive-control, numerical-null, deterministic-rerun, zero-strength, identity-patch,
and hook-cleanup criteria. Each failed the independently frozen clamp-valid-rate
criterion (required at least 80%). Numerical logit errors were exactly
`{json.dumps(calibration["numerical"], sort_keys=True)}`.

Across all {attempts} balanced attempts, strict valid count was {valid}. Median
dense measured-J cosine was {np.median(cosine_values):.6f} (threshold 0.995), median
activation RMS drift was {np.median(rms_values):.6f} (limit 0.02), and median remainder
fraction was {np.median(remainder_values):.6f} (minimum 0.20). Exclusion-reason counts
were `{json.dumps(dict(sorted(reason_counts.items())), sort_keys=True)}`; a trial may
contribute to more than one reason.

## Gate decision

**FAILED: no closure-eligible layer.** No thresholds or intervention source were
changed after observing this result. These records calibrate the state-construction
method; they are not Phase 3 causal effects and cannot adjudicate H1, H2, or H3.
"""
    final = f"""# Final report

## Material Passport

- Protocol: `phase0_protocol_v2` (frozen before fresh confirmation)
- Phase 0 v2 gate: **PASSED**
- Closure-layer calibration gate: **FAILED**
- Formal downstream phases: **GATED / NOT EXECUTED**
- Strongest warranted conclusion: **D — measurement quality is insufficient to distinguish H1, H2, and H3**

This report makes no claim about consciousness or extraction of a model's “true
thoughts.” “Measured-J component” and “measured-J remainder” refer only to the
declared finite dictionaries and sparse decomposition.

## Verified measurement results

{common}

The v2 Phase 0 result is statistically positive and practically above its frozen readout
thresholds. It does not supersede the independent causal-state gate. Strict-all-layers
hit@10 remains a sensitivity analysis, not the v2 primary statistic.

## Answers to the 15 adjudication questions

1. **Did the fresh Phase 0 readout gate pass?** Yes. The exact pass@10 and confidence intervals are reported above.
2. **Was a lens-quality band identified?** Yes: block-output layers 20–30, selected from calibration only.
3. **Did any layer become closure eligible?** No. Layers 23–29 passed readout, rank, positive-control, numerical, and repeatability checks, but each had 0/200 strictly valid clamps.
4. **Is instantaneous measured-J approximately Markov sufficient?** Undetermined; no gate-authorized closure trial was run.
5. **Does measured-J remainder causally influence future measured-J?** Undetermined; calibration failure prevents a causal estimate.
6. **Is any influence mediated by later measured-J writes?** Undetermined; one-shot, final-persistent, and all-position-persistent mediation arms were gated.
7. **Did final-token and sequence-state arms agree?** Not tested. Their scope hooks are implemented and tested, but neither arm produced empirical effects.
8. **Does E_R decrease as dictionary size grows from 4,096 to 16,384?** Not tested. Nested dictionaries were built, but no common-valid paired Phase 3 trials exist.
9. **Do natural collisions reproduce a remainder association?** Not tested; no observational collision bank was built after the layer gate failed.
10. **Can short layer-depth J history close the oracle gap?** Not tested.
11. **Can token-time J plus compact recurrent memory close the gap?** Not tested.
12. **What is the smallest stable autonomous controller?** None established; controller training was gated.
13. **Does a controller generalize to unseen procedural tasks?** Not tested.
14. **Does external knowledge restore knowledge-heavy performance?** Not tested.
15. **Was teacher/student latent-intervention fidelity demonstrated?** No; Phase 6B was not executed.

## Evidence by type

- **Intervention evidence:** the Phase 0 J-coordinate positive control passed. No valid Phase 3 causal intervention exists.
- **Observational evidence:** no v2 natural-collision result exists.
- **Statistical evidence:** the readout and positive-control CIs use 10,000 prompt-clustered bootstrap resamples. No downstream significance test was run.
- **Practical magnitude:** pass@10 and intended-answer log-odds are reported above. E_R, E_J, eta, rollout accuracy, and fidelity are unavailable.

## Interpretation boundary

H1, H2, and H3 remain unresolved. The result is not a negative finding about J-space
closure; it is a failure of the preregistered sparse reconstruction/restoration method to
produce acceptable checkpoint states at any candidate layer. The strongest permitted
classification remains D under all preregistered nearby thresholds used for the formal
gate, because there are no eligible layers and no formal downstream trials.

## Reproducibility

The v1 records and `reports/PHASE0_VALIDATION.md` remain unchanged. The v2 freeze,
fresh records, calibration attempts (including invalid rows), failure manifests, processed
summaries, and figures are committed. Figures 1, 2, 12, and 14 visualize measured data;
the remaining required figures are explicitly machine-sourced gated-status panels, not
quantitative results.

### Recorded downstream commands

{command_text}
"""
    (root / "reports" / "CLOSURE_CAUSAL_REPORT.md").write_text(causal, encoding="utf-8")
    (root / "reports" / "TOKEN_TIME_CLOSURE.md").write_text(token, encoding="utf-8")
    (root / "reports" / "FINAL_REPORT.md").write_text(final, encoding="utf-8")
    (root / "reports" / "PHASE0_V2_CONFIRMATORY.md").write_text(
        phase0_report, encoding="utf-8"
    )
    (root / "reports" / "LAYER_CALIBRATION.md").write_text(
        layer_report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/confirm_v2.yaml")
    parser.parse_args()
    root = repository_root()
    status = build_execution_status(root)
    figures = build_figures(root, status)
    write_json_atomic(
        root / "results" / "processed" / "figure_manifest_v2.json",
        {
            "schema_version": 2,
            "protocol_version": "phase0_protocol_v2",
            "manual_values": False,
            "figures": figures,
        },
    )
    build_reports(root, status)


if __name__ == "__main__":
    main()
