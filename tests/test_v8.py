from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from jclosure.datasets_v8 import FAMILIES, generate_tasks
from jclosure.experiments.compress_persistent_v8 import (
    _dual_pca_scores,
    _raw_selection,
    structured_sketch,
)
from jclosure.persistent_state_v8 import (
    ATOM_ORDER,
    compose_condition,
    factorial_conditions,
    mobius_interactions,
)


def test_v8_generators_are_deterministic_and_unique() -> None:
    for family in FAMILIES:
        left = generate_tasks(family, 12, seed=17, horizon=2)
        right = generate_tasks(family, 12, seed=17, horizon=2)
        assert [task.program_hash for task in left] == [
            task.program_hash for task in right
        ]
        assert len({task.program_hash for task in left}) == 12
        assert all(len(task.semantic_actions) == 2 for task in left)


def test_v8_split_seed_domains_do_not_overlap() -> None:
    blocked: set[str] = set()
    left = generate_tasks(
        "modular_arithmetic", 30, seed=1, horizon=2, blocked_hashes=blocked
    )
    blocked.update(task.program_hash for task in left)
    right = generate_tasks(
        "modular_arithmetic", 30, seed=2, horizon=2, blocked_hashes=blocked
    )
    assert not (
        {task.program_hash for task in left} & {task.program_hash for task in right}
    )


def test_v8_factorial_and_mobius_are_complete() -> None:
    conditions = factorial_conditions()
    assert len(conditions) == 16
    effects = {
        mask: np.asarray([sum(index + 1 for index in range(4) if mask & (1 << index))])
        for mask in range(16)
    }
    interactions = mobius_interactions(effects)
    assert interactions[0].item() == 0
    assert all(
        abs(interactions[1 << index].item() - (index + 1)) < 1e-12 for index in range(4)
    )
    assert all(
        abs(interactions[mask].item()) < 1e-12
        for mask in range(16)
        if mask.bit_count() > 1
    )
    assert len(ATOM_ORDER) == 4


def test_v8_structured_sketch_has_declared_dimension() -> None:
    def state(offset: float) -> dict[str, object]:
        return {
            "recurrent": torch.full((6, 32, 128, 128), offset, dtype=torch.bfloat16),
            "conv": torch.full((6, 8192, 4), offset, dtype=torch.bfloat16),
            "kv": {
                "27": {
                    "keys": torch.full((4, 7, 256), offset, dtype=torch.bfloat16),
                    "values": torch.full((4, 7, 256), offset, dtype=torch.bfloat16),
                }
            },
        }

    result = structured_sketch(state(0.0), state(1.0))
    assert result.shape == (8192,)
    assert torch.isfinite(result).all()


def test_v8_factorial_cache_composition_preserves_primary_complement() -> None:
    def cache(value: float) -> SimpleNamespace:
        layers = []
        for _ in range(32):
            layers.append(
                SimpleNamespace(
                    recurrent_states=torch.full((1, 1, 2, 2), value),
                    conv_states=torch.full((1, 2, 2), value),
                    keys=torch.full((1, 4, 5, 2), value),
                    values=torch.full((1, 4, 5, 2), value),
                    has_previous_state=True,
                )
            )
        return SimpleNamespace(layers=layers)

    clean = cache(0.0)
    perturbed = cache(1.0)
    common = {
        "recurrent_layers": [28, 29, 30],
        "attention_layers": [27, 31],
        "primary_layer": 27,
        "primary_head": 3,
        "primary_position": 4,
    }
    complement = compose_condition(clean, perturbed, "kv_other", **common)
    assert complement.layers[27].keys[0, 3, 4].eq(0).all()
    assert complement.layers[27].keys[0, 2, 4].eq(1).all()
    full = compose_condition(
        clean,
        perturbed,
        "+".join(ATOM_ORDER),
        **common,
    )
    assert full.layers[27].keys.eq(1).all()
    assert full.layers[30].recurrent_states.eq(1).all()
    assert full.layers[30].conv_states.eq(1).all()
    top = compose_condition(clean, perturbed, "top_rec_l30", **common)
    assert top.layers[30].recurrent_states.eq(1).all()
    assert top.layers[30].conv_states.eq(0).all()
    assert clean.layers[27].keys.eq(0).all()
    assert perturbed.layers[27].keys.eq(1).all()


def test_v8_block_dual_pca_and_raw_selection() -> None:
    generator = torch.Generator().manual_seed(9)
    blocks = [
        torch.randn((10, 2, 3), generator=generator),
        torch.randn((10, 4), generator=generator),
    ]
    combined, structured, metadata = _dual_pca_scores(
        blocks, np.arange(6), device="cpu"
    )
    assert combined.shape[0] == 10
    assert structured.shape[0] == 10
    assert structured.shape[1] >= combined.shape[1]
    assert metadata[-1]["combined_rank"] == combined.shape[1]

    def estimate(value: float) -> dict[str, float]:
        return {"estimate": value, "lower": value, "upper": value}

    effects = {
        "direction_cosine_to_full": {
            f"R{index}": estimate(1.0 if index >= 4 else 0.5) for index in range(1, 8)
        },
        "magnitude_ratio_to_full": {
            f"R{index}": estimate(1.0 if index >= 4 else 0.5) for index in range(1, 8)
        },
        "output_sign_agreement_to_full": {
            f"R{index}": estimate(1.0 if index >= 4 else 0.5) for index in range(1, 8)
        },
    }
    selected, _ = _raw_selection(
        {"effects": {"pooled": effects}},
        {
            "minimum_causal_direction_cosine": 0.8,
            "minimum_full_gap_fraction": 0.9,
            "maximum_output_sign_loss": 0.02,
        },
    )
    assert selected == "R4"
