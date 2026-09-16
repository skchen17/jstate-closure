from __future__ import annotations

import numpy as np

from jclosure.experiments.causal_v11 import _selection_score
from jclosure.state_models_v11 import proportional_allocations, semantic_effect_target


def test_semantic_effect_target_preserves_top_absolute_signs() -> None:
    delta = np.asarray([[1.0, -4.0, 3.0, 0.5]], dtype=np.float32)
    result = semantic_effect_target(delta, top_k=2)
    assert result.tolist() == [[0.0, -1.0, 1.0, 0.0]]


def test_proportional_allocations_preserve_budget() -> None:
    for budget in (64, 128, 192, 320, 448):
        allocations = proportional_allocations(budget)
        assert len(allocations) == 3
        assert all(sum(value) == budget for value in allocations)


def test_selection_score_rewards_better_direction() -> None:
    def metric(value: float) -> dict[str, float]:
        return {"estimate": value}

    common = {
        "magnitude_ratio": metric(1.0),
        "semantic_delta_agreement": metric(0.8),
        "output_direction_cosine": metric(0.8),
        "task_decision_sign_agreement": metric(0.8),
    }
    lower = _selection_score({**common, "direction_cosine": metric(0.4)})
    higher = _selection_score({**common, "direction_cosine": metric(0.9)})
    assert higher > lower
