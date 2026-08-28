import hashlib
import json

import pytest
import torch
from torch import nn

from jclosure.experiments.distill_controller import (
    build_budgeted_controller,
    controller_rollout,
)
from jclosure.experiments.memory_order import count_parameters
from jclosure.model import _model_source
from jclosure.recorder import ActivationRecorder, ResidualEditor


class Block(nn.Module):
    def __init__(self, delta: float, output_kind: str = "tensor") -> None:
        super().__init__()
        self.delta = delta
        self.output_kind = output_kind

    def forward(self, value):
        if isinstance(value, tuple | list):
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
        return value[0] if isinstance(value, tuple | list) else value


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


def test_local_model_source_requires_matching_hash_manifest(tmp_path, monkeypatch):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"verified")
    digest = hashlib.sha256(b"verified").hexdigest()
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "example/model",
                "revision": "abc",
                "files": {"weight.bin": digest},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JCLOSURE_MODEL_DIR", str(tmp_path))
    source, revision = _model_source({"id": "example/model", "revision": "abc"})
    assert source == str(tmp_path)
    assert revision == {}


def test_budgeted_controller_autonomous_intervention_rollout():
    model = build_budgeted_controller(
        "mlp",
        state_dim=16,
        n_layers=4,
        n_answers=3,
        feature_dim=8,
        budget=20_000,
    )
    assert abs(count_parameters(model) - 20_000) / 20_000 <= 0.05
    states = torch.nn.functional.normalize(torch.randn(2, 4, 16), dim=-1)
    features = torch.randn(2, 8)
    clean, _ = controller_rollout(
        model,
        states,
        features,
        standalone=False,
        feedback_probability=1.0,
    )
    changed, _ = controller_rollout(
        model,
        states,
        features,
        standalone=False,
        feedback_probability=1.0,
        intervention=(1, 0, 1),
    )
    assert clean.shape == states.shape
    assert not torch.equal(clean[:, 2:], changed[:, 2:])
