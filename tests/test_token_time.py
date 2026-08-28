import torch

from jclosure.experiments.token_time_closure import (
    TokenMacroPredictor,
    autonomous_macro_rollout,
    pool_workspace_band,
)


def test_workspace_pooling_is_normalized_and_reports_dispersion():
    values = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    pooled, dispersion = pool_workspace_band(values)
    assert torch.allclose(torch.linalg.vector_norm(pooled), torch.tensor(1.0))
    assert dispersion > 0


def test_autonomous_rollout_feeds_predictions_back_and_supports_swap():
    torch.manual_seed(4)
    predictor = TokenMacroPredictor(8, history=2, hidden_dim=16, n_actions=3)
    initial = torch.nn.functional.normalize(torch.randn(2, 2, 8), dim=-1)
    clean, actions = autonomous_macro_rollout(predictor, initial, horizon=4)
    swapped, _ = autonomous_macro_rollout(
        predictor, initial, horizon=4, intervention=(0, 1)
    )
    assert clean.shape == (2, 4, 8)
    assert actions.shape == (2, 4, 3)
    assert not torch.equal(clean, swapped)
