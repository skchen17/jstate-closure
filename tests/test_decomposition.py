import torch

from jclosure.decomposition import gradient_pursuit, strip_j_component
from jclosure.jstate import ConceptVocabulary, JStateEncoder


def test_nonnegative_sparse_recovery_and_remainder():
    dictionary = torch.eye(5)
    vector = torch.tensor([2.0, 0.0, 1.5, -3.0, 0.0])
    result = gradient_pursuit(vector, dictionary, k=3)
    assert result.atom_indices.tolist() == [0, 2]
    assert torch.all(result.coefficients >= 0)
    assert torch.allclose(result.reconstruction, torch.tensor([2.0, 0, 1.5, 0, 0]))
    assert torch.allclose(result.remainder, torch.tensor([0.0, 0, 0, -3.0, 0]))


def test_deterministic_tie_breaking():
    dictionary = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]])
    first = gradient_pursuit(torch.tensor([1.0, 0]), dictionary, k=1)
    second = gradient_pursuit(torch.tensor([1.0, 0]), dictionary, k=1)
    assert first.atom_indices.tolist() == [0]
    assert torch.equal(first.atom_indices, second.atom_indices)


def test_strip_j_component_is_orthogonal_for_orthonormal_frame():
    dictionary = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    stripped, _ = strip_j_component(torch.tensor([2.0, 3.0, 4.0]), dictionary, k=2)
    assert torch.allclose(dictionary @ stripped, torch.zeros(2), atol=1e-6)


def test_lazy_dictionary_materializes_only_requested_layer():
    vocabulary = ConceptVocabulary((1, 2), ("one", "two"))
    calls = []

    def build(layer):
        calls.append(layer)
        return torch.tensor([[1.0, 0], [0, 1.0]])

    encoder = JStateEncoder(
        None,
        vocabulary,
        raw_builder=build,
        available_layers=(0, 1),
        protocol_version="phase0_protocol_v2",
    )
    state = encoder.encode(torch.tensor([1.0, 2.0]), 1)
    assert calls == [1]
    assert state.dictionary_size == 2
    assert state.dictionary_hash == vocabulary.digest
    assert state.protocol_version == "phase0_protocol_v2"
