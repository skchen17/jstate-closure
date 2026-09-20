"""CPU invariants for V20 operator coordinates, gates and finite geometry."""

import numpy as np

from jclosure.experiments.operator_geometry_v20 import _r95, _spectrum_dimension
from jclosure.experiments.operator_model_v20 import Fitted, _action_matrix, _design, _gate, _metric, _pca


def test_pca_uses_train_only_and_reconstructs_low_rank():
    rng = np.random.default_rng(20)
    latent = rng.normal(size=(24, 2))
    basis = rng.normal(size=(2, 10))
    train = (latent @ basis).astype(np.float32)
    validation = (rng.normal(size=(8, 2)) @ basis).astype(np.float32)
    pca = _pca(train, validation)
    assert float(pca["energy"][:2].sum()) > 0.999999
    reconstructed = pca["mean"] + pca["validation_coordinate"][:, :2] @ pca["basis"][:, :2].T
    assert np.allclose(reconstructed, validation, atol=1e-4)
    displaced = _pca(train, validation + 1000)
    assert np.allclose(displaced["mean"], pca["mean"])
    assert np.allclose(displaced["basis"], pca["basis"], atol=1e-4)


def test_continuous_action_design_has_no_identity_code():
    c = np.array([[1, 2], [3, 4]], dtype=np.float32)
    z = np.array([[5, 6], [7, 8]], dtype=np.float32)
    x = _design(c, z)
    assert x.shape == (4, 6)
    assert np.allclose(x[0], [5, 6, 5, 6, 10, 12])
    assert np.allclose(x[3], [7, 8, 21, 24, 28, 32])


def test_action_descriptor_sign_scale_and_composition():
    descriptor = {"positive_base_scale_z": {"1": [1.0, 2.0], "2": [3.0, 4.0]}}
    a = _action_matrix(descriptor, [1])[0]
    b = _action_matrix(descriptor, [2])[0]
    assert np.allclose(_action_matrix(descriptor, [1], -1)[0], -a)
    assert np.allclose(_action_matrix(descriptor, [1], multiplier=2)[0], 2 * a)
    assert np.allclose(a + b, [4, 6])


def test_gate_fails_bad_heldout_prediction():
    y = np.array([[[1.0] * 288]], dtype=np.float32)
    good = _metric(y, y, np.array(["boolean_logic"]), np.array(["base_0"]))
    bad = _metric(y, -y, np.array(["boolean_logic"]), np.array(["base_0"]))
    design = {"heldout_gates": {"relative_l2_max": 0.30, "cosine_min": 0.90,
                                "norm_ratio_min": 0.8, "norm_ratio_max": 1.2,
                                "family_relative_l2_max": 0.35}}
    assert _gate(good, design)
    assert not _gate(bad, design)


def test_r95_and_operator_manifold_rank():
    assert _r95(np.array([10.0, 1.0, 0.01]), 0.95) == 1
    matrix = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    summary = _spectrum_dimension(matrix, 0.95)
    assert summary["r95"] == 1


def test_each_frozen_operator_decoder_has_shared_continuous_action_interface():
    rng = np.random.default_rng(22)
    ctrain = rng.normal(size=(5, 2)).astype(np.float32)
    ztrain = rng.normal(size=(3, 4)).astype(np.float32)
    ytrain = rng.normal(size=(5, 3, 6)).astype(np.float32)
    mean = ytrain.reshape(5, -1).mean(axis=0)
    basis = np.linalg.qr(rng.normal(size=(18, 2)))[0]
    design = {"ridge_lambda": 1.0, "small_nonlinear_model": {"seed": 20, "hidden": 8,
             "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 2, "batch_size": 4}}
    for name in ("response_svd", "reduced_rank_regression", "tucker_factorization",
                 "bilinear_latent_operator", "small_nonlinear_latent_operator"):
        fitted = Fitted(name, 2, mean, basis, np.ones(2, dtype=np.float32),
                        ztrain, ytrain, ctrain, design)
        pred = fitted.predict(ctrain[:2], ztrain[:2])
        assert pred.shape == (2, 2, 6)
        assert np.isfinite(pred).all()
