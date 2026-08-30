"""Paired mediation inference and common-valid filtering for protocol v3.1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def common_valid_base_trials(
    frame: pd.DataFrame,
    *,
    required_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    valid = frame[frame["valid"].astype(bool)].copy()
    keep = []
    for base_trial_id, group in valid.groupby("base_trial_id", sort=True):
        observed = set(zip(group["condition"], group["mode"], strict=False))
        if required_pairs <= observed:
            keep.append(base_trial_id)
    return valid[valid["base_trial_id"].isin(keep)].copy()


def paired_mediation_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    n_resamples: int,
    confidence: float,
    seed: int,
    null_threshold: float,
) -> dict[str, Any]:
    preserving = frame[frame["condition"] == "state_preserving"]
    pivot = preserving.pivot_table(
        index=cluster_col,
        columns="mode",
        values=value_col,
        aggfunc="mean",
    ).dropna(subset=["single", "persistent_final", "persistent_all"])
    if pivot.empty:
        raise ValueError("no complete paired mediation clusters")
    values = pivot[["single", "persistent_final", "persistent_all"]].to_numpy(
        dtype=float
    )
    effects = values.mean(0)
    generator = np.random.default_rng(seed)
    samples = np.empty((n_resamples, 3), dtype=float)
    final_ratios = np.empty(n_resamples, dtype=float)
    all_ratios = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = values[generator.integers(0, len(values), size=len(values))]
        current = sampled.mean(0)
        samples[index] = current
        denominator = max(current[0], 1e-20)
        final_ratios[index] = 1 - current[1] / denominator
        all_ratios[index] = 1 - current[2] / denominator
    alpha = (1 - confidence) / 2
    single_ci = np.asarray(
        np.quantile(samples[:, 0], [alpha, 1 - alpha]), dtype=float
    )
    ratio_gate = bool(single_ci[0] > null_threshold and effects[0] > 1e-4)

    def interval(values: np.ndarray) -> dict[str, float]:
        bounds = np.asarray(
            np.quantile(values, [alpha, 1 - alpha]), dtype=float
        )
        return {"lower": float(bounds[0]), "upper": float(bounds[1])}

    return {
        "n_clusters": len(pivot),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "effects": {
            "single": float(effects[0]),
            "persistent_final": float(effects[1]),
            "persistent_all": float(effects[2]),
        },
        "single_ci": {
            "lower": float(single_ci[0]),
            "upper": float(single_ci[1]),
        },
        "ratio_interpretation_gate": ratio_gate,
        "M_final": (
            {
                "estimate": float(1 - effects[1] / max(effects[0], 1e-20)),
                **interval(final_ratios),
            }
            if ratio_gate
            else None
        ),
        "M_all": (
            {
                "estimate": float(1 - effects[2] / max(effects[0], 1e-20)),
                **interval(all_ratios),
            }
            if ratio_gate
            else None
        ),
        "null_threshold": float(null_threshold),
    }
