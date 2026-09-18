from __future__ import annotations

import numpy as np

from jclosure.experiments.geometry_v13 import (
    _moving_coefficients,
    _principal_angles,
)
from jclosure.protocol_v13 import _digest, _hash_ids


def test_v13_freeze_digest_ignores_own_digest() -> None:
    value = {"schema_version": 22, "value": [1, 2, 3]}
    digest = _digest(value)
    assert _digest({**value, "freeze_digest": digest}) == digest
    assert _hash_ids(["b", "a"]) == _hash_ids(["a", "b"])


def test_moving_tangent_relinearizes_across_atlas() -> None:
    atlas = [
        {"vh": np.asarray([[1.0, 0.0]], dtype=np.float32), "r95": 1},
        {"vh": np.asarray([[0.0, 1.0]], dtype=np.float32), "r95": 1},
    ]
    anchor_scores = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    value, visited = _moving_coefficients(
        np.asarray([1.0, 1.0], dtype=np.float32),
        atlas,
        anchor_scores,
        alpha=1.0,
        steps=3,
    )
    np.testing.assert_allclose(value, np.asarray([1.0, 1.0]), atol=1e-6)
    assert visited[:2] == [0, 1]


def test_principal_angles_identical_and_orthogonal() -> None:
    first = np.asarray([[1.0, 0.0]], dtype=np.float32)
    second = np.asarray([[0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(_principal_angles(first, first), [0.0], atol=1e-5)
    np.testing.assert_allclose(_principal_angles(first, second), [90.0], atol=1e-5)
