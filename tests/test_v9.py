from __future__ import annotations

import numpy as np

from jclosure.experiments.sufficiency_v9 import (
    _apply_ridge_weights,
    _ridge_weights,
    build_ordered_latents,
    safe_gap_ratio,
)


def test_v9_gap_ratio_marks_small_denominator_not_identified() -> None:
    value = safe_gap_ratio(0.50, 0.51, 0.5005, epsilon=0.002)
    assert value["identified"] is False
    assert value["gap_closed"] is None
    assert abs(value["candidate_baseline_delta"] - 0.01) < 1e-12
    assert abs(value["ceiling_baseline_delta"] - 0.0005) < 1e-12


def test_v9_gap_ratio_keeps_raw_scores_when_identified() -> None:
    value = safe_gap_ratio(0.50, 0.58, 0.60, epsilon=0.002)
    assert value["identified"] is True
    assert abs(value["gap_closed"] - 0.8) < 1e-12
    assert value["baseline_score"] == 0.50
    assert value["candidate_score"] == 0.58
    assert value["ceiling_score"] == 0.60


def test_v9_ordered_latents_are_rank_bounded_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(18, 9)).astype(np.float32)
    train = np.arange(12)
    targets = {
        "next_absolute": rng.normal(size=(18, 7)).astype(np.float32),
        "next_delta": rng.normal(size=(18, 7)).astype(np.float32),
        "future_delta_h4": rng.normal(size=(18, 7)).astype(np.float32),
    }
    weights = {
        "next_absolute": 0.4,
        "next_causal_delta": 0.4,
        "future_causal_delta": 0.2,
    }
    left, metadata = build_ordered_latents(features, train, targets, weights)
    right, metadata_again = build_ordered_latents(features, train, targets, weights)
    assert metadata == metadata_again
    assert metadata["rank"] <= len(train) - 1
    assert set(left) == {
        "pca",
        "predictive_bottleneck",
        "causal_bottleneck",
        "semantic_causal_bottleneck",
    }
    for method in left:
        np.testing.assert_allclose(left[method], right[method])
        assert left[method].shape == (18, metadata["rank"])


def test_v9_reused_ridge_weights_match_direct_dual_solution() -> None:
    rng = np.random.default_rng(31)
    train = rng.normal(size=(14, 7)).astype(np.float32)
    test = rng.normal(size=(5, 7)).astype(np.float32)
    target = rng.normal(size=(14, 11)).astype(np.float32)
    alpha = 0.7
    weights = _ridge_weights(train, test, alpha)
    actual = _apply_ridge_weights(weights, target)

    mean_x = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    fit = (train - mean_x) / scale
    held = (test - mean_x) / scale
    mean_y = target.mean(axis=0, keepdims=True)
    kernel = fit @ fit.T / fit.shape[1]
    cross = held @ fit.T / fit.shape[1]
    coefficients = np.linalg.solve(
        kernel + alpha * np.eye(len(fit)), target - mean_y
    )
    expected = (mean_y + cross @ coefficients).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
