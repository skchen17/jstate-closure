import torch

from jclosure.clamp import (
    ClampThresholds,
    one_shot_clamp,
    one_shot_clamp_positions,
    persistent_clamp_transforms,
)
from jclosure.jstate import ConceptVocabulary, JStateEncoder


def encoder() -> JStateEncoder:
    vocabulary = ConceptVocabulary((10, 11, 12), ("alpha", "beta", "gamma"))
    directions = {0: torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])}
    return JStateEncoder(directions, vocabulary, k=3)


def test_one_shot_clamp_restores_j_and_keeps_remainder():
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    perturbed = torch.tensor([7.0, 4.0, 3.0, 0.8])
    result = one_shot_clamp(
        clean,
        perturbed,
        layer=0,
        encoder=encoder(),
        thresholds=ClampThresholds(rms_drift=0.25, min_remainder_fraction=0.1),
        natural_scale=1.0,
    )
    assert torch.allclose(result.activation[:3], clean[:3], atol=1e-5)
    assert torch.allclose(result.activation[3:], perturbed[3:], atol=1e-5)
    assert result.dense_cosine > 0.999
    assert result.top10_overlap == 1
    assert result.passed


def test_persistent_clamp_transform_operates_on_final_position():
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    transforms = persistent_clamp_transforms({0: clean}, encoder())
    activation = torch.zeros(1, 2, 4)
    activation[0, -1] = torch.tensor([9.0, 8.0, 7.0, 0.7])
    output = transforms[0](activation, 0)
    assert torch.allclose(output[0, -1, :3], clean[:3], atol=1e-5)
    assert output[0, -1, 3] == activation[0, -1, 3]
    assert torch.equal(output[0, 0], activation[0, 0])


def test_all_position_clamp_scope_and_persistent_transform():
    clean = torch.tensor(
        [[2.0, 1.0, 0.5, 0.2], [1.0, 2.0, 0.5, 0.3], [0.5, 1.0, 2.0, 0.4]]
    )
    perturbed = clean.clone()
    perturbed[:, :3] += 4
    perturbed[:, 3] += torch.tensor([0.1, 0.2, 0.3])
    result = one_shot_clamp_positions(
        clean,
        perturbed,
        layer=0,
        encoder=encoder(),
        positions=None,
        thresholds=ClampThresholds(rms_drift=0.5, min_remainder_fraction=0.0),
    )
    assert result.positions == (0, 1, 2)
    assert torch.allclose(result.activation[:, :3], clean[:, :3], atol=1e-5)
    assert torch.allclose(result.activation[:, 3], perturbed[:, 3], atol=1e-5)
    transforms = persistent_clamp_transforms(
        {0: clean}, encoder(), positions=None
    )
    output = transforms[0](perturbed.unsqueeze(0), 0)
    assert torch.allclose(output[0, :, :3], clean[:, :3], atol=1e-5)
