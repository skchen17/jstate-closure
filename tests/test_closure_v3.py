from types import SimpleNamespace

import pytest
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.experiments.closure_v3 import (
    _base_cell_key,
    _operational_clamp,
    _shard_source_target,
    _state_preserving_delta,
)
from jclosure.geometry import DenseJMap, SparseStateEquality
from jclosure.jstate import ConceptVocabulary, JStateEncoder


def _encoder() -> JStateEncoder:
    vocabulary = ConceptVocabulary((10, 11, 12), ("alpha", "beta", "gamma"))
    raw = {
        0: torch.tensor(
            [
                [2.0, 0, 0, 0],
                [0, 3.0, 0, 0],
                [0, 0, 4.0, 0],
            ]
        )
    }
    return JStateEncoder(raw, vocabulary, raw_directions=raw, k=3)


def test_base_trial_targets_are_split_across_shards_without_duplication():
    config = {
        "run": {
            "valid_per_cell": 100,
            "valid_base_trials_by_source": {"activation_difference": 167},
        }
    }
    targets = [
        _shard_source_target(
            config,
            "activation_difference",
            shard_index=index,
            shard_count=2,
            limit=None,
        )
        for index in range(2)
    ]
    assert targets == [84, 83]
    assert sum(targets) == 167
    assert _base_cell_key("dense-4096", "arithmetic", "activation_difference") == (
        "dense-4096:arithmetic:activation_difference"
    )


class _AlwaysNatural:
    def score(self, value):
        del value
        return SimpleNamespace(natural=True)


def test_dense_source_delta_uses_post_retraction_strength():
    encoder = _encoder()
    dense_map = DenseJMap.from_encoder(encoder)
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    difference = torch.tensor([0.0, 0.0, 0.0, 0.8])
    strength = 0.25
    delta = _state_preserving_delta(
        clean,
        difference,
        strength=strength,
        layer=0,
        state_definition="V3-Dense",
        encoder=encoder,
        dense_map=dense_map,
        tolerance=1e-6,
    )
    assert torch.linalg.vector_norm(delta).item() == pytest.approx(
        strength * torch.linalg.vector_norm(difference).item(), abs=1e-5
    )
    assert torch.nn.functional.cosine_similarity(
        dense_map.dense_state(clean, 0)[None],
        dense_map.dense_state(clean + delta, 0)[None],
    ).item() >= 0.995


def test_sparse_source_delta_preserves_sparse_state_by_definition():
    encoder = _encoder()
    dense_map = DenseJMap.from_encoder(encoder)
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    difference = torch.tensor([4.0, 3.0, 2.0, 0.8])
    delta = _state_preserving_delta(
        clean,
        difference,
        strength=0.5,
        layer=0,
        state_definition="V3-Sparse",
        encoder=encoder,
        dense_map=dense_map,
        tolerance=1e-6,
    )
    equality = SparseStateEquality.compare(
        encoder.decompose(clean, 0), encoder.decompose(clean + delta, 0)
    )
    assert equality.passed


def test_later_persistent_clamp_does_not_require_formal_displacement():
    encoder = _encoder()
    dense_map = DenseJMap.from_encoder(encoder)
    clean = torch.tensor([[2.0, 1.0, 0.5, 0.2]])
    donor = torch.tensor([[2.0, 1.0, 0.5, 1.2]])
    activation = clean.unsqueeze(0)
    capture = {}
    _operational_clamp(
        activation,
        0,
        positions=(0,),
        position_scope="final",
        clean_sequence=clean,
        donor_sequence=donor,
        state_definition="V3-Sparse",
        method="sparse_same_definition",
        encoder=encoder,
        dense_map=dense_map,
        naturality=_AlwaysNatural(),
        tolerance=1e-6,
        thresholds=V3ClampThresholds(),
        require_formal_displacement=False,
        capture=capture,
    )
    assert capture[(0, 0)]["clamp_valid"]
    assert not capture[(0, 0)]["validation"].formal_valid


def test_final_scope_uses_donor_final_position_when_lengths_differ():
    encoder = _encoder()
    dense_map = DenseJMap.from_encoder(encoder)
    clean = torch.tensor([[2.0, 1.0, 0.5, 0.2]])
    donor = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0],
            [2.0, 1.0, 0.5, 1.2],
        ]
    )
    activation = clean.unsqueeze(0).clone()
    activation[0, 0, 3] = 0.4
    capture = {}
    _operational_clamp(
        activation,
        0,
        positions=(0,),
        position_scope="final",
        clean_sequence=clean,
        donor_sequence=donor,
        state_definition="V3-Sparse",
        method="sparse_same_definition",
        encoder=encoder,
        dense_map=dense_map,
        naturality=_AlwaysNatural(),
        tolerance=1e-6,
        thresholds=V3ClampThresholds(),
        require_formal_displacement=False,
        capture=capture,
    )
    validation = capture[(0, 0)]["validation"]
    assert validation.displacement_fraction == pytest.approx(0.2, abs=1e-6)
