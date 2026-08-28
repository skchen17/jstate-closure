"""Provenance-checked statistics, required figures, and final report generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jclosure.config import load_config
from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic
from jclosure.statistics import (
    benjamini_hochberg,
    clustered_bootstrap_ci,
    clustered_sign_flip_p_value,
)


def _read_jsonl(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return pd.json_normalize(records, sep=".") if records else pd.DataFrame()


def _read_json_records(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
    return pd.json_normalize(records, sep=".") if records else pd.DataFrame()


def write_execution_status(root: Path) -> dict[str, Any]:
    """Derive an auditable stage matrix without inventing downstream results."""

    processed = root / "results" / "processed"
    gate_path = processed / "phase0_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
    phase0_status = "COMPLETED" if gate else "UNEXECUTED"
    gate_decision = (
        "PASSED" if gate and gate.get("passed") else "FAILED" if gate else "NOT_EXECUTED"
    )
    blocked = bool(not gate or not gate.get("passed", False))
    reason = (
        "Phase 0 measurement gate failed; downstream empirical phases were not run."
        if gate
        else "Phase 0 measurement gate was not executed."
    )
    stages = {
        "phase0_validation": {
            "status": phase0_status,
            "gate": gate_decision,
            "run_id": gate.get("run_id") if gate else None,
        },
        **{
            name: {"status": "UNEXECUTED" if blocked else "PENDING", "reason": reason}
            for name in (
                "closure",
                "persistent_clamp",
                "natural_collisions",
                "memory_order",
                "controller_distillation",
                "controller_causal_fidelity",
                "modularity",
                "qwen3_6_27b_confirmation",
            )
        },
    }
    payload = {
        "schema_version": 1,
        "derived_from": str(gate_path) if gate_path.exists() else None,
        "phase0_gate": gate_decision,
        "stages": stages,
    }
    write_json_atomic(processed / "execution_status.json", payload)
    return payload


def _require_provenance(data: pd.DataFrame, name: str) -> None:
    if data.empty:
        raise ValueError(f"{name}: no records")
    if "run_id" not in data or data["run_id"].isna().any():
        raise ValueError(f"{name}: plotted series lacks run_id provenance")


def _save_figure(
    path: Path,
    source_paths: list[Path],
    plot: Callable[[plt.Axes], None],
) -> dict[str, Any]:
    if not source_paths:
        raise ValueError(f"{path.name}: no machine-readable source")
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    plot(axis)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "figure": str(path),
        "sha256": sha256_file(path),
        "sources": [
            {"path": str(source), "sha256": sha256_file(source)} for source in source_paths
        ],
    }


def _save_status_figure(
    path: Path,
    source_paths: list[Path],
    *,
    title: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    """Render an explicit no-result panel from a machine-readable run status."""

    def plot(axis: plt.Axes) -> None:
        axis.axis("off")
        axis.set_title(title)
        axis.text(
            0.5,
            0.56,
            status,
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.39,
            reason,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            transform=axis.transAxes,
        )

    entry = _save_figure(path, source_paths, plot)
    entry.update({"status_only": True, "status": status, "reason": reason})
    return entry


def _boxplot(axis: plt.Axes, data: pd.DataFrame, category: str, value: str) -> None:
    labels = list(dict.fromkeys(data[category].astype(str)))
    values = [data.loc[data[category].astype(str) == label, value].dropna() for label in labels]
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.tick_params(axis="x", rotation=25)


def generate_figures(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    processed = root / "results" / "processed"
    raw = root / "results" / "raw"
    figures = root / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}

    gate_path = processed / "phase0_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    gate_status = "FAILED" if gate and not gate.get("passed", False) else "NOT EXECUTED"
    gated_reason = (
        "Phase 0 gate failed; this downstream experiment was not executed."
        if gate_status == "FAILED"
        else "No completed machine-readable experiment records were found."
    )
    execution_status_path = processed / "execution_status.json"
    status_sources = (
        [execution_status_path]
        if execution_status_path.exists()
        else [gate_path]
        if gate_path.exists()
        else []
    )

    def add_status(name: str, title: str, fallback_reason: str) -> None:
        if not status_sources:
            unavailable[name] = fallback_reason
            return
        try:
            manifest.append(
                _save_status_figure(
                    figures / name,
                    status_sources,
                    title=title,
                    status=gate_status,
                    reason=gated_reason,
                )
            )
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"

    closure_paths = sorted(raw.glob("closure-*/trials/**/*.jsonl"))
    closure_paths.extend(sorted(raw.glob("closure-*/trials.jsonl")))
    collision_paths = sorted(raw.glob("natural-collisions-*/collisions.jsonl"))
    memory_paths = sorted(processed.glob("memory_order_*.parquet"))
    controller_paths = sorted(processed.glob("controllers_*.parquet"))
    closure = _read_jsonl(closure_paths)
    collisions = _read_jsonl(collision_paths)
    memory = pd.concat([pd.read_parquet(path) for path in memory_paths], ignore_index=True) if memory_paths else pd.DataFrame()
    controllers = pd.concat([pd.read_parquet(path) for path in controller_paths], ignore_index=True) if controller_paths else pd.DataFrame()

    jobs: list[tuple[str, list[Path], pd.DataFrame, Callable[[plt.Axes], None]]] = []
    layer_path = processed / "phase0_layer_scores.parquet"
    if layer_path.exists():
        layer = pd.read_parquet(layer_path)
        layer["run_id"] = gate.get("run_id")
        jobs.append(
            (
                "01_lens_validation_by_layer.png",
                [layer_path, gate_path],
                layer,
                lambda axis, d=layer: (
                    axis.plot(d["layer"], d["advantage"], label="MRR advantage"),
                    axis.plot(d["layer"], d["smoothed_advantage"], label="3-layer mean"),
                    axis.set(xlabel="Layer", ylabel="J-lens − logit-lens MRR"),
                    axis.legend(),
                ),
            )
        )
    else:
        add_status(
            "01_lens_validation_by_layer.png",
            "Lens validation by layer",
            "Phase 0 layer records absent",
        )

    if not closure.empty:
        valid = closure[closure["valid"] == True].copy()  # noqa: E712
        jobs.extend(
            [
                (
                    "02_j_match_vs_remainder.png",
                    closure_paths,
                    valid,
                    lambda axis, d=valid: (
                        axis.scatter(
                            1 - d["metrics.checkpoint_dense_cosine"],
                            d["metrics.remainder_fraction"],
                            s=9,
                            alpha=0.4,
                        ),
                        axis.set(xlabel="Checkpoint J error (1 − cosine)", ylabel="Remainder displacement / natural scale"),
                    ),
                ),
                (
                    "03_future_j_vs_remainder.png",
                    closure_paths,
                    valid,
                    lambda axis, d=valid: (
                        axis.scatter(
                            d["metrics.remainder_distance"],
                            d["metrics.mean_future_j_distance"],
                            s=9,
                            alpha=0.4,
                        ),
                        axis.set(xlabel="Checkpoint remainder distance", ylabel="Mean future J distance"),
                    ),
                ),
                (
                    "04_output_divergence_controls.png",
                    closure_paths,
                    valid,
                    lambda axis, d=valid: (
                        _boxplot(axis, d, "condition", "metrics.js_divergence"),
                        axis.set(xlabel="Condition", ylabel="Full-vocabulary JS divergence (nats)"),
                    ),
                ),
                (
                    "05_single_vs_persistent_clamp.png",
                    closure_paths,
                    valid[valid["condition"] == "non_j"],
                    lambda axis, d=valid[valid["condition"] == "non_j"]: (
                        _boxplot(axis, d, "clamp_condition", "metrics.js_divergence"),
                        axis.set(xlabel="Clamp", ylabel="Output JS divergence (nats)"),
                    ),
                ),
            ]
        )
    else:
        for name, title in (
            ("02_j_match_vs_remainder.png", "J-match error vs remainder perturbation"),
            ("03_future_j_vs_remainder.png", "Future J divergence vs remainder distance"),
            ("04_output_divergence_controls.png", "Output divergence by intervention control"),
            ("05_single_vs_persistent_clamp.png", "Single vs persistent clamp"),
        ):
            add_status(name, title, "closure records absent")

    if not collisions.empty:
        jobs.append(
            (
                "06_natural_collision_scatter.png",
                collision_paths,
                collisions,
                lambda axis, d=collisions: (
                    axis.scatter(d["current_j_distance"], d["future_j_distance"], c=d["remainder_distance"], s=10, alpha=0.5),
                    axis.set(xlabel="Current J distance", ylabel="Future J distance"),
                ),
            )
        )
    else:
        add_status(
            "06_natural_collision_scatter.png",
            "Natural J-collision analysis",
            "collision records absent",
        )

    if not memory.empty:
        history = memory[memory["condition"] == "j_history"]
        jobs.append(
            (
                "07_predictor_vs_history.png",
                memory_paths,
                history,
                lambda axis, d=history: (
                    axis.plot(d["history_length"], d["metrics.rollout_final_dense_cosine"], marker="o"),
                    axis.set(xlabel="J history length", ylabel="Final rollout dense cosine", xscale="log"),
                ),
            )
        )
    else:
        add_status(
            "07_predictor_vs_history.png",
            "Predictor performance vs J-history length",
            "memory-order records absent",
        )

    if not controllers.empty:
        jobs.extend(
            [
                (
                    "08_controller_vs_parameters.png",
                    controller_paths,
                    controllers,
                    lambda axis, d=controllers: (
                        *(
                            axis.plot(
                                group["parameter_count"],
                                group["metrics.answer_accuracy"],
                                marker="o",
                                linestyle="none",
                                label=str(name),
                            )
                            for name, group in d.groupby("family")
                        ),
                        axis.set(xlabel="Learned parameters", ylabel="Autonomous task accuracy", xscale="log"),
                        axis.legend(),
                    ),
                ),
                (
                    "09_rollout_error_vs_horizon.png",
                    controller_paths,
                    controllers,
                    lambda axis, d=controllers: (
                        *(
                            axis.plot(
                                np.arange(len(row["metrics.rollout_horizon_cosine"])),
                                1 - np.asarray(row["metrics.rollout_horizon_cosine"]),
                                alpha=0.25,
                                label=str(row["family"]) if index == 0 else None,
                            )
                            for index, (_, row) in enumerate(d.iterrows())
                        ),
                        axis.set(xlabel="Layer-depth horizon", ylabel="1 − dense cosine"),
                    ),
                ),
                (
                    "10_intervention_fidelity.png",
                    controller_paths,
                    controllers[controllers["causal_fidelity.available"] == True],  # noqa: E712
                    lambda axis, d=controllers[controllers["causal_fidelity.available"] == True]: (  # noqa: E712
                        axis.scatter(
                            d["causal_fidelity.trajectory_delta_cosine"],
                            d["causal_fidelity.intervention_direction_agreement"],
                            c=np.log10(d["parameter_count"]),
                        ),
                        axis.set(xlabel="Teacher/student trajectory-delta cosine", ylabel="Intervention-direction agreement"),
                    ),
                ),
            ]
        )
    else:
        for name, title in (
            ("08_controller_vs_parameters.png", "Controller performance vs parameter count"),
            ("09_rollout_error_vs_horizon.png", "Free-rollout error vs horizon"),
            ("10_intervention_fidelity.png", "Teacher vs student intervention fidelity"),
        ):
            add_status(name, title, "controller records absent")

    for name, paths, data, plot in jobs:
        try:
            _require_provenance(data, name)
            manifest.append(_save_figure(figures / name, paths, plot))
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    return manifest, unavailable


def _closure_summary(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = sorted((root / "results" / "raw").glob("closure-*/trials/**/*.jsonl"))
    paths.extend(sorted((root / "results" / "raw").glob("closure-*/trials.jsonl")))
    data = _read_jsonl(paths)
    if data.empty:
        return {"status": "NOT_EXECUTED"}
    valid = data[(data["valid"] == True) & (data["primary_layer_pair"] == True)].copy()  # noqa: E712
    if valid.empty:
        return {"status": "NO_VALID_PRIMARY_TRIALS", "attempted": len(data)}
    summaries: dict[str, Any] = {
        "status": "ANALYZED",
        "attempted": len(data),
        "valid": len(valid),
    }

    def effect_summary(frame: pd.DataFrame) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for condition, key in (("non_j", "e_r"), ("j_positive", "e_j")):
            subset = frame[frame["condition"] == condition]
            if condition == "non_j":
                subset = subset[subset["clamp_condition"] == "single"]
            if len(subset) < 2:
                continue
            ci = clustered_bootstrap_ci(
                subset,
                cluster_col="prompt_id",
                value_col="metrics.js_divergence",
                n_resamples=int(config["statistics"]["bootstrap_resamples"]),
                seed=int(config["reproducibility"]["bootstrap_seed"]),
            )
            output[key] = ci.__dict__
        non_j = frame[
            (frame["condition"] == "non_j")
            & (frame["clamp_condition"] == "single")
        ]
        if len(non_j) >= 2:
            future_ci = clustered_bootstrap_ci(
                non_j,
                cluster_col="prompt_id",
                value_col="metrics.mean_future_j_distance",
                n_resamples=int(config["statistics"]["bootstrap_resamples"]),
                seed=int(config["reproducibility"]["bootstrap_seed"]),
            )
            output["future_j_effect"] = future_ci.__dict__
        if "e_r" in output and "e_j" in output:
            positive_floor = float(config["statistics"]["null_js_floor"])
            if output["e_j"]["lower"] > positive_floor:
                output["eta"] = output["e_r"]["estimate"] / max(
                    output["e_j"]["estimate"], 1e-12
                )
                output["eta_conservative_upper"] = output["e_r"]["upper"] / max(
                    output["e_j"]["lower"], 1e-12
                )
            else:
                output["eta_status"] = "UNUSABLE_POSITIVE_CONTROL"
        return output

    summaries.update(effect_summary(valid))
    summaries["by_family"] = {
        str(family): effect_summary(group)
        for family, group in valid.groupby("task_family", sort=True)
    }
    secondary: list[dict[str, Any]] = []
    non_j_single = valid[
        (valid["condition"] == "non_j")
        & (valid["clamp_condition"] == "single")
    ]
    for family, family_frame in non_j_single.groupby("task_family", sort=True):
        for outcome, null in (
            ("metrics.js_divergence", float(config["statistics"]["null_js_floor"])),
            ("metrics.mean_future_j_distance", 0.0),
            ("metrics.answer_flip", 0.0),
        ):
            if outcome not in family_frame or family_frame[outcome].notna().sum() < 2:
                continue
            p_value = clustered_sign_flip_p_value(
                family_frame,
                cluster_col="prompt_id",
                value_col=outcome,
                null=null,
                alternative="greater",
                n_resamples=int(config["statistics"]["bootstrap_resamples"]),
                seed=int(config["reproducibility"]["bootstrap_seed"]),
            )
            secondary.append(
                {
                    "task_family": str(family),
                    "outcome": outcome,
                    "null": null,
                    "raw_p_value": p_value,
                }
            )
    if secondary:
        adjusted = benjamini_hochberg([row["raw_p_value"] for row in secondary])
        for row, value in zip(secondary, adjusted, strict=True):
            row["bh_adjusted_p_value"] = float(value)
    summaries["secondary_comparisons"] = secondary
    clamp = valid[valid["condition"] == "non_j"]
    summaries["single_clamp_mean_js"] = float(
        clamp.loc[clamp["clamp_condition"] == "single", "metrics.js_divergence"].mean()
    )
    summaries["persistent_clamp_mean_js"] = float(
        clamp.loc[clamp["clamp_condition"] == "persistent", "metrics.js_divergence"].mean()
    )
    single = summaries["single_clamp_mean_js"]
    persistent = summaries["persistent_clamp_mean_js"]
    positive = summaries.get("e_j", {}).get("estimate")
    if positive is None or not np.isfinite(single) or not np.isfinite(persistent):
        summaries["mediation_outcome"] = "UNDETERMINED"
    elif single / max(positive, 1e-12) < 0.2:
        summaries["mediation_outcome"] = "LITTLE_SINGLE_CLAMP_EFFECT"
    elif persistent / max(single, 1e-12) < 0.2:
        summaries["mediation_outcome"] = "EFFECT_MEDIATED_BY_LATER_J_WRITES"
    else:
        summaries["mediation_outcome"] = "EFFECT_SURVIVES_PERSISTENT_J_CLAMP"
    return summaries


def _latest_table(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted((root / "results" / "processed").glob(pattern))
    return pd.read_parquet(paths[-1]) if paths else pd.DataFrame()


def write_threshold_sensitivity(root: Path) -> dict[str, int]:
    """Re-evaluate preregistered clamp/controller decisions over nearby cutoffs."""

    raw = root / "results" / "raw"
    processed = root / "results" / "processed"
    closure_paths = sorted(raw.glob("closure-*/trials/**/*.jsonl"))
    closure = _read_jsonl(closure_paths)
    counts = {"clamp_rows": 0, "controller_rows": 0}
    if not closure.empty:
        trials = closure[closure["condition"] == "non_j"].copy()
        records: list[dict[str, Any]] = []
        for cosine in (0.990, 0.995, 0.999):
            for overlap in (0.6, 0.8, 1.0):
                for rms_limit in (0.01, 0.02, 0.05):
                    for remainder in (0.10, 0.20, 0.50):
                        accepted = trials[
                            (trials["metrics.checkpoint_dense_cosine"] >= cosine)
                            & (trials["metrics.checkpoint_top10_overlap"] >= overlap)
                            & (trials["metrics.checkpoint_rms_drift"] <= rms_limit)
                            & (trials["metrics.remainder_fraction"] >= remainder)
                        ]
                        records.append(
                            {
                                "schema_version": 1,
                                "run_id": ",".join(sorted(trials["run_id"].unique())),
                                "dense_cosine_threshold": cosine,
                                "top10_overlap_threshold": overlap,
                                "rms_drift_threshold": rms_limit,
                                "remainder_fraction_threshold": remainder,
                                "accepted": len(accepted),
                                "attempted": len(trials),
                                "acceptance_rate": len(accepted) / max(len(trials), 1),
                                "mean_js_divergence": float(
                                    accepted["metrics.js_divergence"].mean()
                                )
                                if len(accepted)
                                else None,
                            }
                        )
        pd.DataFrame(records).to_parquet(
            processed / "closure_threshold_sensitivity.parquet", index=False
        )
        counts["clamp_rows"] = len(records)
    controllers = _latest_table(root, "controllers_*.parquet")
    if not controllers.empty:
        records = []
        for cosine in (0.85, 0.90, 0.95):
            for support in (0.60, 0.70, 0.80):
                for accuracy in (0.85, 0.90, 0.95):
                    for fidelity in (0.70, 0.80, 0.90):
                        for _, row in controllers.iterrows():
                            fidelity_value = row.get(
                                "causal_fidelity.intervention_direction_agreement"
                            )
                            stable = bool(
                                row["metrics.rollout_final_dense_cosine"] >= cosine
                                and row["metrics.sparse_support_f1"] >= support
                                and row["metrics.answer_accuracy"] >= accuracy
                                and pd.notna(fidelity_value)
                                and fidelity_value >= fidelity
                            )
                            records.append(
                                {
                                    "schema_version": 1,
                                    "run_id": row["run_id"],
                                    "family": row["family"],
                                    "parameter_count": row["parameter_count"],
                                    "initialization": row["initialization"],
                                    "dense_cosine_threshold": cosine,
                                    "sparse_f1_threshold": support,
                                    "accuracy_threshold": accuracy,
                                    "fidelity_threshold": fidelity,
                                    "stable": stable,
                                }
                            )
        pd.DataFrame(records).to_parquet(
            processed / "controller_threshold_sensitivity.parquet", index=False
        )
        counts["controller_rows"] = len(records)
    return counts


def build_final_report(root: Path, config: dict[str, Any], figure_status: dict[str, str]) -> str:
    gate_path = root / "results" / "processed" / "phase0_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
    closure = _closure_summary(root, config)
    memory = _latest_table(root, "memory_order_*.parquet")
    controllers = _latest_table(root, "controllers_*.parquet")
    modularity = _latest_table(root, "modularity_*.parquet")
    collisions = _latest_table(root, "natural_collisions_*.parquet")
    regression_paths = sorted(
        (root / "results" / "processed").glob("natural_collision_regression_*.json")
    )
    collision_regression = (
        json.loads(regression_paths[-1].read_text(encoding="utf-8"))
        if regression_paths
        else None
    )

    conclusion = "D"
    rationale = "measurement or execution is incomplete"
    classification_sensitivity: dict[str, str] = {
        "eta_0.10": "D",
        "eta_0.20": "D",
        "eta_0.30": "D",
    }
    if gate and gate.get("passed") and closure.get("status") == "ANALYZED":
        threshold = float(config["statistics"]["eta_threshold"])
        family_summaries = closure.get("by_family", {})
        family_eta = [
            value.get("eta_conservative_upper") for value in family_summaries.values()
        ]
        family_eta = [value for value in family_eta if value is not None]
        stable = not controllers.empty and bool(
            controllers.get("operationally_stable", pd.Series(dtype=bool)).any()
        )
        causal_families = [
            family
            for family, value in family_summaries.items()
            if value.get("e_r", {}).get("lower", 0)
            > float(config["statistics"]["null_js_floor"])
            and value.get("future_j_effect", {}).get("lower", 0) > 0
        ]
        j_rows = (
            memory[memory["condition"] == "j_history"].sort_values("history_length")
            if not memory.empty
            else pd.DataFrame()
        )
        oracle = (
            memory[memory["condition"].str.contains("remainder_oracle", na=False)]
            if not memory.empty
            else pd.DataFrame()
        )
        h3_pass = False
        if not j_rows.empty and not oracle.empty and (j_rows["history_length"] <= 4).any():
            instant = float(j_rows.iloc[0]["metrics.rollout_final_dense_cosine"])
            short_rows = j_rows[j_rows["history_length"] <= 4]
            short = float(short_rows["metrics.rollout_final_dense_cosine"].max())
            oracle_score = float(oracle["metrics.rollout_final_dense_cosine"].max())
            gap_fraction = (short - instant) / max(oracle_score - instant, 1e-12)
            short_accuracy = float(short_rows["metrics.answer_accuracy"].max())
            oracle_accuracy = float(oracle["metrics.answer_accuracy"].max())
            h3_pass = (
                gap_fraction >= 0.8
                and short_accuracy >= oracle_accuracy - 0.05
                and stable
            )
        collision_positive = False
        if collision_regression and collision_regression.get("status") == "COMPLETED":
            collision_positive = (
                collision_regression.get("future_j_distance", {}).get(
                    "remainder_coefficient", 0
                )
                > 0
                and collision_regression.get("output_js_divergence", {}).get(
                    "remainder_coefficient", 0
                )
                > 0
            )
        oracle_advantage = (
            not j_rows.empty
            and not oracle.empty
            and float(oracle["metrics.rollout_final_dense_cosine"].max())
            - float(j_rows["metrics.rollout_final_dense_cosine"].max())
            >= 0.05
        )
        if len(family_eta) >= 3 and all(value < threshold for value in family_eta) and stable:
            conclusion, rationale = (
                "A",
                "strict per-family closure and controller stability/fidelity criteria passed",
            )
        elif h3_pass and any(value >= threshold for value in family_eta):
            conclusion, rationale = (
                "C",
                "instantaneous closure failed but short J-history met the augmented-state criteria",
            )
        elif len(causal_families) >= 3 and collision_positive and oracle_advantage:
            conclusion, rationale = (
                "B",
                "three-family causal effects, natural collisions, and a remainder-oracle gap aligned",
            )
        for nearby in (0.10, 0.20, 0.30):
            if len(family_eta) >= 3 and all(value < nearby for value in family_eta) and stable:
                label = "A"
            elif h3_pass and any(value >= nearby for value in family_eta):
                label = "C"
            elif len(causal_families) >= 3 and collision_positive and oracle_advantage:
                label = "B"
            else:
                label = "D"
            classification_sensitivity[f"eta_{nearby:.2f}"] = label

    gate_status = "PASSED" if gate and gate.get("passed") else "FAILED" if gate else "NOT EXECUTED"
    gate_evidence = (
        "Not executed."
        if not gate
        else (
            f"Workspace band {gate.get('workspace_band')}; band-min hit@10 "
            f"{gate.get('hit10_by_family')}; held-out rank-advantage CI "
            f"{gate.get('rank_advantage_ci')}; positive-control CI "
            f"{gate.get('positive_control_ci')}."
        )
    )
    smallest = "Not established"
    if not controllers.empty and "operationally_stable" in controllers:
        stable_rows = controllers[controllers["operationally_stable"] == True]  # noqa: E712
        if not stable_rows.empty:
            row = stable_rows.sort_values("parameter_count").iloc[0]
            smallest = f"{int(row['parameter_count']):,} parameters ({row['family']}, {row['initialization']})"
    collision_text = (
        f"{len(collisions)} matched records; observational only."
        if not collisions.empty
        else "Not executed."
    )
    fact_text = "Not executed."
    if not modularity.empty:
        exact = modularity[modularity["fact_condition"] == "exact_relevant"]["metrics.answer_accuracy"].mean()
        no_fact = modularity[modularity["fact_condition"] == "no_fact"]["metrics.answer_accuracy"].mean()
        fact_text = f"Exact-fact mean accuracy {exact:.3f}; no-fact mean accuracy {no_fact:.3f}."
    missing_figures = ", ".join(sorted(figure_status)) or "None"
    return f"""# Final report

## Material Passport

- Origin: J-State Closure experiment pipeline
- Verification Status: {'ANALYZED' if gate and gate.get('passed') else 'UNVERIFIED / GATED'}
- Phase 0 gate: {gate_status}
- Strongest warranted conclusion: {conclusion} — {rationale}

No statement in this report concerns consciousness. J-state is an operational
measurement defined by the pinned Jacobian lens, not a claim to have extracted
the model's “true thoughts.”

## Measurement-gate result

{gate_evidence}

The gate is conjunctive. A positive rank advantage and a successful J-swap do
not override a failed hidden-intermediate hit@10 criterion.

## 1. Is instantaneous J-state approximately Markov sufficient?

{'Undetermined: the Phase 0 gate failed, so closure trials were not executed.' if gate_status != 'PASSED' else f'Closure summary: `{json.dumps(closure, sort_keys=True, default=str)}`.'}
This answer is intervention-based only when the gate passed and valid one-shot
clamp trials exist.

## 2. Does non-J state causally influence future J-space content?

{'Undetermined: no gate-authorized J-preserving remainder intervention was executed.' if closure.get('status') == 'NOT_EXECUTED' else 'See the valid J-preserving remainder trials and their future-J distances in the closure summary.'}
No causal claim is made when strict clamp validation failed.

## 3. Is this influence mediated by later writes into J-space?

{'Undetermined: neither single-clamp nor persistent-clamp mediation trials were executed.' if closure.get('status') == 'NOT_EXECUTED' else f"Single-clamp mean JS: `{closure.get('single_clamp_mean_js')}`; persistent-clamp mean JS: `{closure.get('persistent_clamp_mean_js')}`. The three-way mediation outcome is `{closure.get('mediation_outcome')}`."}
The three possible mediation outcomes are never reduced to a binary label.

## 4. Can J plus a small amount of memory form a sufficient state?

{'Memory-order predictors were not executed.' if memory.empty else f'{len(memory)} predictor conditions were evaluated; consult Figure 7 and the committed table for history/oracle gaps.'}

## 5. What is the smallest tested controller that maintains stable free rollout?

{smallest}. A model lacking validated intervention fidelity is not counted.

## 6. Does the controller generalize to unseen procedural reasoning tasks?

{'Not executed.' if controllers.empty else 'Held-out template/instance results are in the controller table; accuracy is reported alongside teacher retention and rollout metrics.'}

## 7. Does adding external knowledge restore knowledge-heavy task performance?

{fact_text}

## 8. Evidence for a small cognitive controller interpretation

{'No empirical support can be assessed because controller training was gated and not executed.' if controllers.empty else 'Stable autonomous rollout, held-out procedural accuracy, and teacher/student intervention agreement are the relevant evidence.'}
Ordinary teacher-forced prediction is secondary.

## 9. Evidence for a J-space broadcast-bus interpretation

{'No empirical support can be assessed because closure, mediation, remainder-oracle, and collision studies were gated and not executed.' if closure.get('status') == 'NOT_EXECUTED' else 'Valid non-J interventions that alter later J states, attenuation under persistent clamping, and remainder-oracle gaps bear on this hypothesis.'}
Natural collisions alone are observational. Collision status: {collision_text}

## 10. What remains ambiguous?

Missing or non-renderable required figures: {missing_figures}. Layer-depth is the
dynamical time coordinate, results are limited to tested prompts/layers, and a
failed lens gate is a measurement failure rather than evidence for H1, H2, or H3.
Downstream required figures are rendered as machine-sourced status panels when
the relevant experiment was gated; those panels are not quantitative evidence.

## Evidence boundary

- Observational: natural collision regressions and ordinary predictive fits.
- Intervention-based: only validated J swaps, strict J-preserving remainder
  perturbations, and persistent clamps.
- Statistical: prompt-clustered bootstrap intervals accompany effect estimates.
- Practical magnitude: raw JS divergence, answer effects, trajectory distances,
  and autonomous task accuracy are reported separately from significance.

The conclusion code uses preregistered operational thresholds, not universal
theoretical constants. Nearby-threshold sensitivity must be consulted before
generalizing beyond the tested setting. Classification sensitivity:
`{json.dumps(classification_sensitivity, sort_keys=True)}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/confirm.yaml")
    args = parser.parse_args()
    root = repository_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    write_execution_status(root)
    sensitivity_counts = write_threshold_sensitivity(root)
    figure_manifest, unavailable = generate_figures(root)
    write_json_atomic(
        root / "results" / "processed" / "figure_manifest.json",
        {
            "schema_version": 1,
            "figures": figure_manifest,
            "unavailable": unavailable,
            "manual_values": False,
            "threshold_sensitivity_rows": sensitivity_counts,
        },
    )
    report = build_final_report(root, config, unavailable)
    (root / "reports" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
