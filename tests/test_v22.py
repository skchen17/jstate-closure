"""CPU invariants for V22 metric correction and action-model gates."""

import numpy as np

from jclosure.experiments.input_geometry_v22 import _angle, _operator, _ranks
from jclosure.experiments.analyze_v22 import _coverage, _kernel


def test_metric_whitening_recovers_physical_operator_rank():
    x = np.diag([4.0, 2.0, 1.0])
    gram = x @ x.T
    eigen, vectors = np.linalg.eigh(gram)
    keep = eigen > 1e-10
    whitener = vectors[:, keep] / np.sqrt(eigen[keep])[None, :]
    response = np.array([[4.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = _operator(response, whitener)
    assert result["r99"] == 2


def test_principal_angle_detects_input_rotation():
    left = np.eye(3)[:, :2]
    right = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    result = _angle(left, right)
    assert result["median_degrees"] == 45.0
    assert result["overlap"] == 0.5


def test_energy_ranks_are_monotone():
    r90, r95, r99, stable = _ranks(np.array([3.0, 2.0, 1.0, 0.5]))
    assert 1 <= r90 <= r95 <= r99 <= 4
    assert stable > 1.0


def test_coverage_residual_is_one_for_orthogonal_holdout():
    result = _coverage(np.eye(2, 3), np.array([[0.0, 0.0, 1.0]]))
    assert np.isclose(result["span_residual_median"], 1.0)
    assert np.isclose(result["nearest_abs_cosine_median"], 0.0)


def test_polynomial_action_kernels_are_symmetric():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    for kind in ("linear", "quadratic", "cubic", "rbf"):
        kernel = _kernel(features, kind)
        assert np.allclose(kernel, kernel.T)
        assert np.isfinite(kernel).all()
