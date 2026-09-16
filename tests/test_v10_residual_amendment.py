from __future__ import annotations

import numpy as np

from jclosure.experiments.residual_audit_v10_amendment import (
    frozen_base_residual_prediction,
)


def test_zero_residual_features_cannot_change_frozen_base() -> None:
    train = np.zeros((12, 3), dtype=np.float32)
    evaluate = np.zeros((5, 3), dtype=np.float32)
    target = np.arange(48, dtype=np.float32).reshape(12, 4)
    correction = frozen_base_residual_prediction(train, evaluate, target, 1.0)
    assert np.array_equal(correction, np.zeros_like(correction))


def test_residual_predictor_returns_only_increment() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(30, 2)).astype(np.float32)
    evaluate = rng.normal(size=(4, 2)).astype(np.float32)
    target = np.column_stack((train[:, 0], -2 * train[:, 1])).astype(np.float32)
    correction = frozen_base_residual_prediction(train, evaluate, target, 1e-6)
    centered = evaluate - train.mean(axis=0, keepdims=True)
    assert correction.shape == (4, 2)
    assert np.allclose(correction[:, 0], centered[:, 0], atol=2e-4)
    assert np.allclose(correction[:, 1], -2 * centered[:, 1], atol=2e-4)
