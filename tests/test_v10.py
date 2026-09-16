from __future__ import annotations

import numpy as np

from jclosure.experiments.decoded_causal_v10 import _dual_reconstruction_alpha
from jclosure.experiments.sufficiency_v10 import (
    cross_fitted_residuals,
    select_confirmatory_candidates,
    semantic_retention,
)


def test_v10_semantic_retention_is_relative_to_full_ceiling() -> None:
    value = semantic_retention(0.70, 0.714, epsilon=0.002)
    assert value["semantic_retention_identified"] is True
    assert abs(value["semantic_retention_relative_to_full"] - 0.70 / 0.714) < 1e-12
    assert abs(value["semantic_compact_minus_full"] + 0.014) < 1e-12


def test_v10_candidate_selection_uses_boundary_intermediate_and_reference() -> None:
    selected, status = select_confirmatory_candidates(
        [384, 416, 448, 480, 512],
        [64, 128, 256, 320, 384, 416, 448, 480, 512],
        512,
        [384, 448, 512],
    )
    assert selected == [384, 448, 512]
    assert status == "OBSERVATIONAL_GATE_PASS"


def test_v10_candidate_selection_has_frozen_diagnostic_fallback() -> None:
    selected, status = select_confirmatory_candidates(
        [], [64, 128, 256, 512], 512, [384, 448, 512]
    )
    assert selected == [384, 448, 512]
    assert status == "DIAGNOSTIC_ONLY_NO_ALL_FAMILY_PASS"


def test_v10_cross_fitted_residualization_removes_compact_linear_signal() -> None:
    rng = np.random.default_rng(17)
    compact = rng.normal(size=(30, 4)).astype(np.float32)
    mapping = rng.normal(size=(4, 6)).astype(np.float32)
    block = compact @ mapping
    train = np.arange(20)
    evaluation = np.arange(20, 30)
    folds = [train[train % 5 == value] for value in range(5)]
    residual_train, residual_evaluation, diagnostics = cross_fitted_residuals(
        compact, block, train, evaluation, folds, alpha=1e-6
    )
    assert float(np.mean(residual_train**2)) < 1e-8
    assert float(np.mean(residual_evaluation**2)) < 1e-8
    assert diagnostics["evaluation_residual_fraction"] < 1e-8


def test_v10_dual_alpha_reconstructs_training_score_geometry() -> None:
    rng = np.random.default_rng(23)
    q, _ = np.linalg.qr(rng.normal(size=(12, 5)))
    singular = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    train_scores = (q * singular).astype(np.float32)
    alpha, diagnostics = _dual_reconstruction_alpha(train_scores, train_scores)
    np.testing.assert_allclose(alpha @ train_scores, train_scores, atol=1e-5)
    assert diagnostics["score_reconstruction_max_abs_error"] < 1e-10
