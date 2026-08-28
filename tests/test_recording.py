import pytest
import torch
from torch import nn

from jclosure.recorder import ActivationRecorder, ResidualEditor


class Block(nn.Module):
    def __init__(self, delta: float, output_kind: str = "tensor") -> None:
        super().__init__()
        self.delta = delta
        self.output_kind = output_kind

    def forward(self, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        output = value + self.delta
        if self.output_kind == "tuple":
            return output, torch.tensor(1)
        if self.output_kind == "list":
            return [output, torch.tensor(1)]
        return output


class Toy(nn.Module):
    def __init__(self, kinds=("tensor", "tensor")) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(1, kinds[0]), Block(2, kinds[1])])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value[0] if isinstance(value, (tuple, list)) else value


@pytest.mark.parametrize("kind", ["tensor", "tuple", "list"])
def test_recording_preserves_output_structure(kind):
    model = Toy((kind, "tensor"))
    value = torch.zeros(1, 3, 4)
    with ActivationRecorder(model.layers, at=[0, 1]) as recorder:
        result = model(value)
    assert torch.equal(result, torch.full_like(value, 3))
    assert torch.equal(recorder.activations[0], torch.ones_like(value))
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks


def test_editor_applies_and_is_removed():
    model = Toy()
    value = torch.zeros(1, 3, 4)
    with ResidualEditor(model.layers, {0: lambda x, layer: x + 4 + 0 * layer}):
        assert torch.equal(model(value), torch.full_like(value, 7))
    assert torch.equal(model(value), torch.full_like(value, 3))
    assert not model.layers[0]._forward_hooks


def test_editor_cleanup_on_exception():
    model = Toy()

    def fail(value, layer):
        del value, layer
        raise RuntimeError("deliberate")

    with pytest.raises(RuntimeError, match="deliberate"):
        with ResidualEditor(model.layers, {0: fail}):
            model(torch.zeros(1, 1, 2))
    assert not model.layers[0]._forward_hooks

