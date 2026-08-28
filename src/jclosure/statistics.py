"""Prompt-clustered inference and preregistered effect summaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_clusters: int
    n_observations: int
    n_resamples: int


def clustered_bootstrap_ci(
    data: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2_026_090_1,
) -> ConfidenceInterval:
    """Percentile bootstrap that samples entire prompt clusters."""

    frame = data[[cluster_col, value_col]].dropna()
    if frame.empty:
        raise ValueError("no finite observations")
    grouped = {
        cluster: group[value_col].to_numpy(dtype=float)
        for cluster, group in frame.groupby(cluster_col, sort=True)
    }
    clusters = list(grouped)
    if not clusters:
        raise ValueError("no clusters")
    observed = float(statistic(frame[value_col].to_numpy(dtype=float)))
    generator = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = generator.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([grouped[clusters[item]] for item in sampled])
        estimates[index] = statistic(values)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return ConfidenceInterval(
        estimate=observed,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_clusters=len(clusters),
        n_observations=len(frame),
        n_resamples=n_resamples,
    )


def paired_clustered_difference_ci(
    data: pd.DataFrame,
    *,
    cluster_col: str,
    condition_col: str,
    value_col: str,
    reference: str,
    treatment: str,
    **kwargs,
) -> ConfidenceInterval:
    pivot = data.pivot_table(
        index=cluster_col,
        columns=condition_col,
        values=value_col,
        aggfunc="mean",
    ).dropna(subset=[reference, treatment])
    differences = pd.DataFrame(
        {
            cluster_col: pivot.index.astype(str),
            "difference": pivot[treatment].to_numpy() - pivot[reference].to_numpy(),
        }
    )
    return clustered_bootstrap_ci(
        differences,
        cluster_col=cluster_col,
        value_col="difference",
        **kwargs,
    )


def clustered_sign_flip_p_value(
    data: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    null: float = 0.0,
    alternative: str = "two-sided",
    n_resamples: int = 10_000,
    seed: int = 2_026_090_1,
) -> float:
    """Cluster-level randomization p-value for a mean effect against a null."""

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    frame = data[[cluster_col, value_col]].dropna()
    cluster_means = (
        frame.groupby(cluster_col, sort=True)[value_col].mean().to_numpy(dtype=float)
        - null
    )
    if cluster_means.size == 0:
        raise ValueError("no finite clusters")
    observed = float(cluster_means.mean())
    generator = np.random.default_rng(seed)
    signs = generator.choice(
        np.asarray([-1.0, 1.0]), size=(n_resamples, cluster_means.size)
    )
    randomized = (signs * cluster_means).mean(axis=1)
    if alternative == "greater":
        extreme = randomized >= observed
    elif alternative == "less":
        extreme = randomized <= observed
    else:
        extreme = np.abs(randomized) >= abs(observed)
    return float((int(extreme.sum()) + 1) / (n_resamples + 1))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be between zero and one")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def numerical_null_threshold(
    values: Iterable[float], *, floor: float = 1e-4, quantile: float = 0.99
) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return floor
    return max(float(np.quantile(array, quantile)), floor)


def normalized_remainder_effect(
    remainder_effect: float,
    j_effect: float,
    *,
    epsilon: float = 1e-12,
    j_effect_lower: float | None = None,
    null_threshold: float = 1e-4,
) -> float | None:
    if j_effect_lower is not None and j_effect_lower <= null_threshold:
        return None
    if j_effect <= null_threshold:
        return None
    return remainder_effect / (j_effect + epsilon)


def gap_closed(j_only: float, candidate: float, oracle: float) -> float | None:
    denominator = j_only - oracle
    if denominator <= 1e-12:
        return None
    return (j_only - candidate) / denominator
