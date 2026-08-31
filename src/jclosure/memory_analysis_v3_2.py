"""Post-run aggregation for compact-memory protocol v3.2.

This module is deliberately downstream of the frozen training protocol.  It
only reads immutable controller records and produces paired, example-clustered
summaries; it never trains or selects a controller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jclosure.provenance import sha256_file, write_json_atomic


@dataclass(frozen=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    n_clusters: int


def paired_cluster_interval(
    frame: pd.DataFrame,
    *,
    cluster: str,
    value: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> PairedInterval:
    """Bootstrap the mean paired delta after averaging repeated seed rows."""

    if frame.empty:
        return PairedInterval(float("nan"), float("nan"), float("nan"), 0)
    clustered = frame.groupby(cluster, sort=True)[value].mean().to_numpy(dtype=float)
    if not np.isfinite(clustered).all():
        raise ValueError(f"nonfinite paired values in {value}")
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(clustered), size=(n_resamples, len(clustered)))
    estimates = clustered[samples].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    bounds = np.asarray(
        np.quantile(estimates, [alpha, 1.0 - alpha]), dtype=float
    ).reshape(2)
    lower = float(bounds[0])
    upper = float(bounds[1])
    return PairedInterval(
        estimate=float(clustered.mean()),
        lower=lower,
        upper=upper,
        n_clusters=int(len(clustered)),
    )


def memory_utility_reasons(
    *,
    cosine: PairedInterval,
    trajectory_reduction: float,
    teacher_fidelity_delta: float,
    seed_deltas: list[float],
    expected_seeds: int,
    config: dict[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if cosine.n_clusters == 0 or cosine.lower <= 0:
        reasons.append("cosine_ci_not_positive")
    if not np.isfinite(cosine.estimate) or cosine.estimate < float(
        config["memory_effect_min_cosine"]
    ):
        reasons.append("cosine_effect_below_threshold")
    if not np.isfinite(trajectory_reduction) or trajectory_reduction < float(
        config["trajectory_reduction_fraction"]
    ):
        reasons.append("trajectory_reduction_below_threshold")
    if not np.isfinite(teacher_fidelity_delta) or teacher_fidelity_delta < -float(
        config["semantic_noninferiority"]
    ):
        reasons.append("teacher_action_fidelity_inferior")
    if len(seed_deltas) != expected_seeds or not all(value > 0 for value in seed_deltas):
        reasons.append("seed_direction_inconsistent")
    return tuple(sorted(reasons))


def _controller_payloads(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(
        (root / "results/v3_2/raw").glob("compact-memory-v3-2-*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED" or manifest.get("stage") != "train":
            continue
        result_path = root / str(manifest["result"])
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        output.append((result_path, payload))
    return output


def _reference_payloads(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(
        (root / "results/v3_2/raw").glob(
            "compact-memory-references-v3-2-*/manifest.json"
        )
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED" or manifest.get("stage") != "references":
            continue
        result_path = root / str(manifest["result"])
        output.append(
            (result_path, json.loads(result_path.read_text(encoding="utf-8")))
        )
    return output


def _reference_summary(
    payloads: list[tuple[Path, dict[str, Any]]],
    summary: pd.DataFrame,
    baseline_keys: dict[int, str],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path, payload in payloads:
        seed = int(payload["seed"])
        horizon8 = next(
            (
                value
                for value in payload["autonomous_pca512_recurrent"]["test"]
                if int(value["horizon"]) == 8
            ),
            None,
        )
        if horizon8 is None:
            continue
        baseline = summary[
            (summary["model_key"] == baseline_keys.get(seed, ""))
            & (summary["horizon"] == 8)
        ]
        baseline_cosine = (
            float(baseline.iloc[0]["decoded_cosine_median"])
            if len(baseline) == 1
            else float("nan")
        )
        reference_cosine = float(horizon8["decoded_cosine_median"])
        rows.append(
            {
                "seed": seed,
                "baseline_key": baseline_keys.get(seed),
                "baseline_horizon8_cosine": baseline_cosine,
                "autonomous_reference_horizon8_cosine": reference_cosine,
                "reference_minus_baseline": reference_cosine - baseline_cosine,
                "linear_teacher_current_one_step_cosine": float(
                    payload["linear_current_one_step"]["test"][
                        "decoded_cosine_median"
                    ]
                ),
                "nonlinear_teacher_current_one_step_cosine": float(
                    payload["nonlinear_full_current_one_step"]["test"][
                        "decoded_cosine_median"
                    ]
                ),
                "source": str(path.relative_to(root)) if root else str(path),
                "source_sha256": sha256_file(path),
            }
        )
    differences = np.asarray(
        [value["reference_minus_baseline"] for value in rows], dtype=float
    )
    positive_gap = bool(
        len(differences) > 0
        and np.isfinite(differences).all()
        and float(np.median(differences)) > 0
    )
    return {
        "completed_seeds": sorted(value["seed"] for value in rows),
        "per_seed": rows,
        "positive_markov_to_reference_gap": positive_gap,
        "median_reference_minus_baseline": (
            float(np.median(differences)) if len(differences) else None
        ),
    }


def _model_key(payload: dict[str, Any]) -> str:
    history = payload.get("history_length")
    memory = payload.get("memory_dimension")
    return "-".join(
        (
            str(payload["model_family"]),
            f"h{int(history) if history is not None else 0}",
            f"m{int(memory) if memory is not None else 0}",
            f"s{int(payload['seed'])}",
            str(payload["training_subset"]),
        )
    )


def _frames(
    payloads: list[tuple[Path, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path, payload in payloads:
        model_key = _model_key(payload)
        history = payload.get("history_length")
        memory = payload.get("memory_dimension")
        common = {
            "model_key": model_key,
            "model_family": payload["model_family"],
            "history_length": int(history) if history is not None else 0,
            "memory_dimension": int(memory) if memory is not None else 0,
            "seed": int(payload["seed"]),
            "training_subset": payload["training_subset"],
            "parameter_count": int(payload["parameter_count"]),
        }
        for value in payload["test"]["horizons"]:
            summaries.append({**common, **value})
        for value in payload["test"]["rows"]:
            rows.append({**common, **value})
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    return pd.DataFrame(summaries), pd.DataFrame(rows), sources


def _baseline_keys(summary: pd.DataFrame) -> dict[int, str]:
    candidates = summary[
        (summary["training_subset"] == "all_parseable")
        & summary["model_family"].isin(["markov", "history"])
        & (summary["horizon"] == 8)
    ]
    output: dict[int, str] = {}
    for seed, group in candidates.groupby("seed", sort=True):
        best = group.sort_values(
            ["decoded_cosine_median", "model_family", "history_length"],
            ascending=[False, True, True],
        ).iloc[0]
        output[int(seed)] = str(best["model_key"])
    return output


def _paired_gru_rows(
    rows: pd.DataFrame, baseline_keys: dict[int, str]
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for seed, baseline_key in sorted(baseline_keys.items()):
        baseline = rows[
            (rows["model_key"] == baseline_key) & rows["horizon"].isin([8, 16])
        ]
        for dimension, gru in rows[
            (rows["seed"] == seed)
            & (rows["training_subset"] == "all_parseable")
            & (rows["model_family"] == "gru")
            & rows["horizon"].isin([8, 16])
        ].groupby("memory_dimension", sort=True):
            paired = gru.merge(
                baseline,
                on=["example_id", "horizon"],
                suffixes=("_gru", "_baseline"),
                validate="one_to_one",
            )
            paired["memory_dimension"] = int(dimension)
            paired["seed"] = int(seed)
            paired["baseline_key"] = baseline_key
            paired["cosine_delta"] = (
                paired["decoded_cosine_gru"] - paired["decoded_cosine_baseline"]
            )
            paired["trajectory_distance_delta"] = (
                paired["trajectory_distance_gru"]
                - paired["trajectory_distance_baseline"]
            )
            paired["teacher_action_fidelity_delta"] = (
                paired["teacher_action_fidelity_gru"]
                - paired["teacher_action_fidelity_baseline"]
            )
            paired["ground_truth_action_accuracy_delta"] = (
                paired["ground_truth_action_accuracy_gru"]
                - paired["ground_truth_action_accuracy_baseline"]
            )
            output.append(paired)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def analyze_controller_results(
    root: Path,
    *,
    config: dict[str, Any],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260901,
) -> dict[str, Any]:
    payloads = _controller_payloads(root)
    summary, rows, sources = _frames(payloads)
    expected_controller_seeds = [int(value) for value in config["controller_seeds"]]
    expected_models_per_seed = 1 + len(config["histories"]) + len(
        config["memory_dimensions"]
    )
    expected_keys = {
        f"markov-h0-m0-s{controller_seed}-all_parseable"
        for controller_seed in expected_controller_seeds
    }
    expected_keys.update(
        f"history-h{int(history)}-m0-s{controller_seed}-all_parseable"
        for controller_seed in expected_controller_seeds
        for history in config["histories"]
    )
    expected_keys.update(
        f"gru-h0-m{int(memory)}-s{controller_seed}-all_parseable"
        for controller_seed in expected_controller_seeds
        for memory in config["memory_dimensions"]
    )
    observed_all_parseable_keys = {
        _model_key(payload)
        for _, payload in payloads
        if payload.get("training_subset") == "all_parseable"
    }
    sensitivity_payloads = [
        payload
        for _, payload in payloads
        if payload.get("training_subset") != "all_parseable"
    ]
    sensitivity_runs = [
        {
            "model_key": _model_key(payload),
            "training_subset": payload["training_subset"],
            "train_trajectories": int(payload["train_trajectories"]),
            "validation_trajectories": int(payload["validation_trajectories"]),
            "test_trajectories": int(payload["test_trajectories"]),
            "validation_selection_horizon": payload.get(
                "validation_selection_horizon"
            ),
            "best_validation_selection_cosine": payload.get(
                "best_validation_selection_cosine"
            ),
            "sensitivity_underpowered": bool(
                payload.get("sensitivity_underpowered", False)
            ),
        }
        for payload in sensitivity_payloads
    ]
    baseline_keys = _baseline_keys(summary) if not summary.empty else {}
    paired = _paired_gru_rows(rows, baseline_keys) if not rows.empty else pd.DataFrame()
    reference_payloads = _reference_payloads(root)
    reference_summary = _reference_summary(
        reference_payloads, summary, baseline_keys, root=root
    )
    utilities: list[dict[str, Any]] = []
    for dimension in [int(value) for value in config["memory_dimensions"]]:
        group = paired[
            (paired["memory_dimension"] == dimension) & (paired["horizon"] == 8)
        ]
        cosine = paired_cluster_interval(
            group,
            cluster="example_id",
            value="cosine_delta",
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + dimension,
        )
        baseline_distance = float(group["trajectory_distance_baseline"].mean()) if len(group) else float("nan")
        gru_distance = float(group["trajectory_distance_gru"].mean()) if len(group) else float("nan")
        trajectory_reduction = (
            1.0 - gru_distance / baseline_distance
            if np.isfinite(baseline_distance) and baseline_distance > 0
            else float("nan")
        )
        teacher_delta = float(group["teacher_action_fidelity_delta"].mean()) if len(group) else float("nan")
        ground_truth_delta = float(group["ground_truth_action_accuracy_delta"].mean()) if len(group) else float("nan")
        seed_deltas = [
            float(value)
            for _, value in group.groupby("seed")["cosine_delta"].mean().items()
        ]
        reasons = memory_utility_reasons(
            cosine=cosine,
            trajectory_reduction=trajectory_reduction,
            teacher_fidelity_delta=teacher_delta,
            seed_deltas=seed_deltas,
            expected_seeds=len(expected_controller_seeds),
            config=config,
        )
        utilities.append(
            {
                "memory_dimension": dimension,
                "horizon": 8,
                "cosine_delta": asdict(cosine),
                "trajectory_reduction": trajectory_reduction,
                "teacher_action_fidelity_delta": teacher_delta,
                "ground_truth_action_accuracy_delta": ground_truth_delta,
                "seed_cosine_deltas": seed_deltas,
                "gate_passed": not reasons,
                "gate_reasons": list(reasons),
            }
        )
    missing_required_keys = sorted(expected_keys - observed_all_parseable_keys)
    unexpected_all_parseable_keys = sorted(observed_all_parseable_keys - expected_keys)
    complete = not missing_required_keys and set(baseline_keys) == set(
        expected_controller_seeds
    )
    passing = [value["memory_dimension"] for value in utilities if value["gate_passed"]]
    if not passing:
        h3_reason = "memory_utility_gate_failed"
    elif set(reference_summary["completed_seeds"]) != set(expected_controller_seeds):
        h3_reason = "autonomous_remainder_reference_incomplete"
    elif not reference_summary["positive_markov_to_reference_gap"]:
        h3_reason = "autonomous_reference_does_not_define_a_positive_markov_gap"
    else:
        h3_reason = "full_horizon_and_gap_closure_adjudication_not_passed"
    processed = root / "results/v3_2/processed"
    processed.mkdir(parents=True, exist_ok=True)
    summary_path = processed / "compact_memory_controller_summary_v3_2.parquet"
    paired_path = processed / "compact_memory_controller_pairs_v3_2.parquet"
    summary.to_parquet(summary_path, index=False)
    paired.to_parquet(paired_path, index=False)
    result = {
        "schema_version": 5,
        "protocol_version": "compact_memory_exploratory_v3_2",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "completed_controller_results": len(payloads),
        "completed_all_parseable_controller_results": len(
            observed_all_parseable_keys & expected_keys
        ),
        "completed_sensitivity_controller_results": len(sensitivity_payloads),
        "sensitivity_runs": sensitivity_runs,
        "expected_controller_results": len(expected_controller_seeds)
        * expected_models_per_seed,
        "missing_required_controller_keys": missing_required_keys,
        "unexpected_all_parseable_controller_keys": unexpected_all_parseable_keys,
        "baseline_keys": {str(key): value for key, value in baseline_keys.items()},
        "memory_utility": utilities,
        "minimum_useful_memory_dimension": min(passing) if passing else None,
        "h3_followup_authorized": False,
        "h3_followup_reason": h3_reason,
        "remainder_reference": reference_summary,
        "summary_records": str(summary_path.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "paired_records": str(paired_path.relative_to(root)),
        "paired_sha256": sha256_file(paired_path),
        "sources": sources,
    }
    write_json_atomic(
        processed / "compact_memory_controller_analysis_v3_2.json", result
    )
    return result
