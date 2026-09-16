from __future__ import annotations

import torch

from jclosure.experiments.strict_interface_v10 import recombination_metrics


def test_raw_residual_recombination_is_identity() -> None:
    generator = torch.Generator().manual_seed(17)
    teacher = torch.randn((3, 4, 5), generator=generator, dtype=torch.bfloat16)
    decoded = torch.randn((3, 4, 5), generator=generator, dtype=torch.bfloat16)
    metrics = recombination_metrics(decoded, teacher)
    assert metrics["recombined_full_raw_max_abs_error"] == 0.0
    assert metrics["raw_residual_relative_l2"] > 0.0
