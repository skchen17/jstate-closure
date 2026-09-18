from types import SimpleNamespace

import torch

from jclosure.experiments.jvp_v12 import _apply_direction
from jclosure.experiments.runtime_v13_jvp_batched import _apply_direction_batch


def _cache() -> SimpleNamespace:
    recurrent = SimpleNamespace(
        recurrent_states=torch.randn(1, 2, 3),
        conv_states=torch.randn(1, 2, 4),
    )
    attention = SimpleNamespace(
        keys=torch.randn(1, 2, 5, 3),
        values=torch.randn(1, 2, 5, 3),
    )
    return SimpleNamespace(layers=[recurrent, attention])


def _readout(cache: SimpleNamespace) -> torch.Tensor:
    parts = [
        cache.layers[0].recurrent_states.float().square().flatten(1).sum(1),
        cache.layers[0].conv_states.float().sin().flatten(1).sum(1),
        cache.layers[1].keys.float().square().flatten(1).sum(1),
        cache.layers[1].values.float().cos().flatten(1).sum(1),
    ]
    return torch.stack(parts, dim=-1)


def test_batched_jvp_matches_independent_scalar_jvps() -> None:
    torch.manual_seed(13)
    base = _cache()
    batch_size = 4
    rows = {
        "recurrent": torch.randn(batch_size, 1, 2, 3),
        "conv": torch.randn(batch_size, 1, 2, 4),
        "kv": torch.randn(batch_size, 1, 2, 2, 5, 3),
    }

    def batched_target(epsilon: torch.Tensor) -> torch.Tensor:
        amended = _apply_direction_batch(base, epsilon, rows, [0], [1])
        return _readout(amended)

    epsilon = torch.zeros(batch_size)
    _, batched = torch.autograd.functional.jvp(
        batched_target, epsilon, torch.ones_like(epsilon), strict=True
    )

    independent = []
    for index in range(batch_size):
        row = {name: value[index] for name, value in rows.items()}

        def scalar_target(
            epsilon: torch.Tensor, row: dict[str, torch.Tensor] = row
        ) -> torch.Tensor:
            amended = _apply_direction(base, epsilon, row, [0], [1])
            return _readout(amended)[0]

        scalar = torch.zeros(())
        _, derivative = torch.autograd.functional.jvp(
            scalar_target, scalar, torch.ones_like(scalar), strict=True
        )
        independent.append(derivative)

    torch.testing.assert_close(batched, torch.stack(independent), rtol=0, atol=0)
