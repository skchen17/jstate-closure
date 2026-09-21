"""CPU invariants for the V21 action/state and paired-geometry diagnostic."""

import numpy as np
import torch

from jclosure.experiments.action_representation_v21 import _coverage
from jclosure.experiments.factorial_v21 import _fit_predict, _spectrum
from jclosure.experiments.geometry_analysis_v21 import _angles, _spectrum as geometry_spectrum
from jclosure.experiments.geometry_analysis_amend_v21 import corrected_decompose
from jclosure.experiments.neural_models_v21 import LowRankHyper
from jclosure.experiments.z2_correction_v21 import _corrected_gram


def test_full_action_coverage_exposes_unseen_orthogonal_direction():
    train = np.eye(3, 5)[:2]
    validation = np.array([[0, 0, 1, 0, 0]], dtype=float)
    row = _coverage(train, validation)[0]
    assert row["train_span_relative_residual"] == 1.0
    assert row["nearest_train_abs_cosine"] == 0.0


def test_z2_zero_train_channel_uses_floor_without_nonfinite_gram():
    common = np.eye(18).tolist()
    descriptor = {"Z1_exact_linear_kernel": {"train_normalization_norm": 4.0},
                  "Z2_channel_exact_linear_kernels": {
                      "REC": {"train_normalization_norm": 2.0, "exact_requested_Gram": common},
                      "Conv": {"train_normalization_norm": 3.0, "exact_requested_Gram": common},
                      "KV": {"train_normalization_norm": 1e-12,
                             "exact_requested_Gram": (1e24*np.eye(18)).tolist()}}}
    gram, detail = _corrected_gram(descriptor)
    assert np.isfinite(gram).all()
    assert detail["channel_scales"]["KV"]["robust_train_only_floor"] == 1.0
    assert np.allclose(np.median(np.diag(gram[:12, :12])), 1.0)


def test_bilinear_kernel_predicts_separable_training_response():
    rng = np.random.default_rng(21)
    s = rng.normal(size=(6, 2))
    a = rng.normal(size=(4, 3))
    ks = s@s.T
    ka = a@a.T
    weights = rng.normal(size=(2, 3, 5))
    truth = np.einsum("ni,aj,ijd->nad", s, a, weights)
    prediction = _fit_predict(ks, ks, ka, ka, truth, 1e-8, "G1_bilinear", _spectrum(ks, ka))
    assert np.allclose(prediction, truth, atol=1e-4)


def test_paired_subspace_rotation_and_procrustes_separate_gain():
    a = np.array([[2.0, 0], [0, 1.0], [0, 0]], dtype=float)
    b = np.array([[0, 0], [2.0, 0], [0, 1.0]], dtype=float)
    sa, sb = geometry_spectrum(a), geometry_spectrum(b)
    pair = _angles(sa, sb)
    decomposition = corrected_decompose(a, b, sa, sb)
    assert pair["principal_angle_median_degrees"] > 20
    assert decomposition["pre_alignment_relative_error"] > 0.5
    assert decomposition["post_orthogonal_Procrustes_relative_error"] < 1e-7
    assert np.isclose(decomposition["frobenius_gain_ratio"], 1.0)


def test_low_rank_hypernetwork_is_odd_in_signed_action():
    torch.manual_seed(21)
    model = LowRankHyper(7, 4, 6)
    s = torch.randn(3, 7)
    a = torch.randn(3, 4)
    assert torch.allclose(model(s, -a), -model(s, a), atol=1e-6)
