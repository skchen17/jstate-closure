from __future__ import annotations

import numpy as np

from jclosure.experiments.arch_compression_v7_corrective import (
    _predictive_comparison,
)
from jclosure.protocol_v7_corrective import PROTOCOL_V7_CORRECTIVE


def test_corrective_protocol_is_additive() -> None:
    assert PROTOCOL_V7_CORRECTIVE == "persistent_channel_compression_corrective_v7_1"


def test_predictive_comparison_uses_nonzero_corrected_target() -> None:
    generator = np.random.default_rng(5)
    current = generator.normal(size=(20, 4)).astype(np.float32)
    channel = generator.normal(size=(20, 3)).astype(np.float32)
    target = np.concatenate((channel[:, :2], channel[:, :2]), axis=1)
    result = _predictive_comparison(
        current,
        channel,
        target,
        list(range(15)),
        list(range(15, 20)),
        alpha=1.0,
        seed=9,
        resamples=100,
    )
    assert result["target_delta_l2"]["estimate"] > 0
    assert result["full_channel_delta_rmse"]["estimate"] >= 0
