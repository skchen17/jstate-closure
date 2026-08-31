"""Conditional restoration gates and paired summaries for protocol v3.2."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from jclosure.records_v3_2 import ConditionalEligibilityV32


def conditional_success_summary(
    applicable: np.ndarray | pd.Series | list[bool],
    success: np.ndarray | pd.Series | list[bool],
    *,
    minimum_applicable: int,
    minimum_rate: float,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> ConditionalEligibilityV32:
    applicable_values = np.asarray(applicable, dtype=bool)
    success_values = np.asarray(success, dtype=bool)
    if applicable_values.shape != success_values.shape:
        raise ValueError("applicable and success arrays must have the same shape")
    observed = success_values[applicable_values]
    count = int(len(observed))
    successes = int(observed.sum())
    rate = float(observed.mean()) if count else 0.0
    if count:
        generator = np.random.default_rng(seed)
        # For a binary empirical distribution the non-parametric bootstrap
        # success count is exactly Binomial(n, p_hat).  Sampling it directly is
        # equivalent and avoids materializing n_resamples x n indices.
        samples = generator.binomial(count, rate, size=n_resamples) / count
        alpha = (1 - confidence) / 2
        bounds = np.asarray(np.quantile(samples, [alpha, 1 - alpha]), dtype=float)
        lower, upper = float(bounds[0]), float(bounds[1])
    else:
        lower = upper = 0.0
    return ConditionalEligibilityV32(
        applicable=count,
        successes=successes,
        rate=rate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        eligible=count >= minimum_applicable and rate >= minimum_rate,
    )


def conditional_summary_dict(*args, **kwargs) -> dict[str, object]:
    return asdict(conditional_success_summary(*args, **kwargs))


def exclude_invalid_restorations(
    frame: pd.DataFrame, *, required_modes: tuple[str, ...]
) -> pd.DataFrame:
    valid = frame[frame["valid"].astype(bool)].copy()
    keep: list[str] = []
    for base_id, group in valid.groupby("base_trial_id", sort=True):
        if set(required_modes) <= set(group["mode"]):
            keep.append(str(base_id))
    return valid[valid["base_trial_id"].astype(str).isin(keep)].copy()
