"""Focused V17 protocol and analysis invariants."""

from __future__ import annotations

import numpy as np

from jclosure.experiments.conditional_v17 import _feature
from jclosure.experiments.state_sufficiency_v17 import _predict_key, _score
from jclosure.protocol_v17 import digest


def test_v17_digest_excludes_only_digest_field() -> None:
    source = {"a": 1, "freeze_digest": "wrong"}
    assert digest(source) == digest({"a": 1})
    assert digest({"a": 2}) != digest({"a": 1})


def test_v17_action_only_predicts_train_mean() -> None:
    kernel = np.zeros((4, 4), dtype=np.float32)
    y = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = _predict_key(kernel, np.asarray([0, 1]), np.asarray([2, 3]), y, 0.1)
    np.testing.assert_allclose(pred, [[2.0, 3.0], [2.0, 3.0]])


def test_v17_raw_feature_train_rank_not_silently_clamped() -> None:
    kernel = np.asarray([[2.0, 0.0, 1.0], [0.0, 1.0, 0.5], [1.0, 0.5, 2.0]], dtype=np.float32)
    features, rank = _feature(kernel, 2, 32)
    assert rank == 2
    assert features.shape == (3, 2)


def test_v17_response_score_exact_fit() -> None:
    y = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    assert _score(y, y)["relative_l2"] == 0.0
