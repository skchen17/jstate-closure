import torch

from jclosure.interventions import (
    coordinate_swap_activation,
    matched_random_direction,
    non_j_direction,
    replace_activation,
    resolve_position_scope,
    steer_activation,
)


def test_zero_strength_and_identical_replacement_are_exact():
    activation = torch.randn(2, 5, 7)
    direction = torch.randn(7)
    assert torch.equal(
        steer_activation(activation, direction, strength=0, positions=(-1,)), activation
    )
    assert torch.equal(replace_activation(activation, activation), activation)


def test_steering_selected_position_only():
    activation = torch.zeros(1, 3, 4)
    output = steer_activation(
        activation, torch.tensor([1.0, 0, 0, 0]), strength=2, positions=(-1,)
    )
    assert output[0, -1, 0] == 2
    assert torch.count_nonzero(output[:, :-1]) == 0


def test_coordinate_swap_changes_two_coordinates():
    activation = torch.tensor([[[3.0, 1.0, 5.0]]])
    output = coordinate_swap_activation(
        activation,
        torch.tensor([1.0, 0, 0]),
        torch.tensor([0, 1.0, 0]),
        positions=(-1,),
    )
    assert torch.allclose(output, torch.tensor([[[1.0, 3.0, 5.0]]]), atol=1e-6)


def test_non_j_and_random_directions_are_deterministic():
    vector = torch.tensor([1.0, 2.0, 3.0])
    dictionary = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    stripped, _ = non_j_direction(vector, dictionary, k=2)
    assert torch.allclose(dictionary @ stripped, torch.zeros(2), atol=1e-6)
    first = matched_random_direction(vector, seed=12)
    second = matched_random_direction(vector, seed=12)
    assert torch.equal(first, second)
    assert torch.allclose(
        torch.linalg.vector_norm(first), torch.linalg.vector_norm(vector)
    )


def test_position_scopes_cover_final_padding_explicit_and_reasoning_span():
    assert resolve_position_scope(5, scope="final") == (4,)
    assert resolve_position_scope(5, scope="explicit", positions=(0, -1)) == (0, 4)
    assert resolve_position_scope(
        5,
        scope="all_non_padding",
        attention_mask=torch.tensor([0, 1, 1, 1, 1]),
    ) == (1, 2, 3, 4)
    assert resolve_position_scope(5, scope="reasoning_span", reasoning_span=(1, 4)) == (
        1,
        2,
        3,
    )
