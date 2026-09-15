from __future__ import annotations

import numpy as np

from jclosure.experiments import compress_persistent_v8 as v8
from jclosure.experiments.compress_persistent_v8_1 import corrected_latent


def test_corrected_latent_applies_global_fit_to_family_slice() -> None:
    rng = np.random.default_rng(20260828)
    train = rng.normal(size=(12, 8)).astype(np.float32)
    apply = rng.normal(size=(5, 8)).astype(np.float32)
    absolute = rng.normal(size=(12, 4)).astype(np.float32)
    delta = rng.normal(size=(12, 4)).astype(np.float32)
    v8._latent_original_v81 = v8._latent  # type: ignore[attr-defined]
    for method in ("pca", "predictive_bottleneck", "causal_bottleneck", "layerwise_fusion"):
        values, metadata = corrected_latent(
            method, 3, train, apply, absolute, delta
        )
        assert values.shape == (5, 3)
        assert np.isfinite(values).all()
        assert metadata["corrective_alignment"] is True


def test_corrected_latent_preserves_aligned_path() -> None:
    rng = np.random.default_rng(20260829)
    train = rng.normal(size=(10, 6)).astype(np.float32)
    absolute = rng.normal(size=(10, 3)).astype(np.float32)
    delta = rng.normal(size=(10, 3)).astype(np.float32)
    v8._latent_original_v81 = v8._latent  # type: ignore[attr-defined]
    expected, _ = v8._latent("pca", 2, train, train, absolute, delta)
    actual, _ = corrected_latent("pca", 2, train, train, absolute, delta)
    np.testing.assert_allclose(actual, expected)
