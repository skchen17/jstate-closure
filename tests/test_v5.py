from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jclosure.datasets_v5 import build_replication_tasks, v4_program_hashes
from jclosure.experiments.peripheral_v5 import _evaluate_recurrent
from jclosure.peripheral_v5 import (
    OneStepPredictor,
    PeripheralComposite,
    PeripheralEncoder,
    RecurrentPeripheralController,
    RemainderTransform,
    TransitionArrays,
    bottleneck_regularizer,
    build_u,
    make_history,
)
from jclosure.protocol_v5 import _digest
from jclosure.records_v5 import (
    H2ReplicationRecord,
    PeripheralReferenceRecord,
    PeripheralStateRecord,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v5_remainder_transform_is_train_fitted_and_finite(tmp_path: Path) -> None:
    generator = np.random.default_rng(3)
    j = generator.normal(size=(80, 12)).astype(np.float32)
    weights = generator.normal(size=(12, 9)).astype(np.float32)
    hidden = j @ weights + 0.2 * generator.normal(size=(80, 9)).astype(np.float32)
    transform = RemainderTransform.fit(j, hidden, rank=8, seed=5)
    remainder = transform.transform(j, hidden)
    assert remainder.shape == hidden.shape
    assert np.isfinite(remainder).all()
    assert np.max(np.abs(remainder.mean(axis=0))) < 1e-3
    path = tmp_path / "remainder.npz"
    transform.save(path)
    restored = RemainderTransform.load(path)
    np.testing.assert_allclose(restored.transform(j, hidden), remainder)


def test_v5_history_padding_and_mask_do_not_read_future() -> None:
    states = np.arange(20, dtype=np.float32).reshape(5, 4)
    values, mask = make_history(states, 3)
    np.testing.assert_array_equal(mask[0], [0, 0, 1])
    np.testing.assert_array_equal(values[0, -1], states[0])
    np.testing.assert_array_equal(values[2], states[:3])
    np.testing.assert_array_equal(values[3], states[1:4])


def test_v5_model_families_have_expected_shapes() -> None:
    batch, j_dim, u_dim, remainder_dim, compact_dim = 6, 24, 23, 13, 5
    j = torch.randn(batch, j_dim)
    u = torch.randn(batch, u_dim)
    remainder = torch.randn(batch, remainder_dim)
    for architecture in ("mlp", "gated_mlp", "attention"):
        model = OneStepPredictor(
            j_dim,
            u_dim,
            remainder_dim,
            32,
            16,
            architecture=architecture,
        )
        predicted, action = model(j, u, remainder)
        assert predicted.shape == (batch, j_dim)
        assert action.shape == (batch, 16)
        torch.testing.assert_close(
            torch.linalg.vector_norm(predicted, dim=1), torch.ones(batch)
        )
    encoder = PeripheralEncoder(
        remainder_dim, compact_dim, j_dim, u_dim, nonlinear=True
    )
    predictor = OneStepPredictor(
        j_dim,
        u_dim,
        compact_dim,
        32,
        16,
        architecture="gated_mlp",
    )
    composite = PeripheralComposite(encoder, predictor)
    predicted, action, compact = composite(j, u, remainder)
    assert predicted.shape == (batch, j_dim)
    assert action.shape == (batch, 16)
    assert compact.shape == (batch, compact_dim)
    assert torch.isfinite(bottleneck_regularizer(compact))


def test_v5_attention_history_respects_padding_mask() -> None:
    model = OneStepPredictor(
        8, 23, 0, 32, 16, architecture="attention", history_length=4
    )
    current = torch.randn(2, 8)
    u = torch.randn(2, 23)
    history = torch.randn(2, 4, 8)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.float32)
    predicted, action = model(current, u, history_j=history, history_mask=mask)
    assert predicted.shape == (2, 8)
    assert action.shape == (2, 16)


def test_v5_u_contains_action_family_and_clock() -> None:
    output = build_u(
        np.asarray([0, 15]),
        np.asarray(["boolean_logic", "variable_binding"]),
        np.asarray([0, 3]),
        np.asarray([4, 8]),
    )
    assert output.shape == (2, 23)
    assert output[0, 0] == 1
    assert output[1, 15] == 1
    assert output[0, 16] == 1
    assert output[1, 20] == 1
    assert output[0, -2] == 0
    assert output[1, -2] == 3 / 8


def test_v5_autonomous_rollout_feeds_back_predicted_action() -> None:
    class RecordingController(RecurrentPeripheralController):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.seen_u: list[torch.Tensor] = []

        def forward(
            self, measured_j: torch.Tensor, compact: torch.Tensor, u: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.seen_u.append(u.detach().cpu())
            logits = torch.zeros((len(u), 16), device=u.device)
            logits[:, 5] = 1
            return measured_j, compact, logits

    current_j = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (3, 1))
    compact = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (3, 1))
    actions = np.asarray([0, 1, 2], dtype=np.int64)
    families = np.asarray(["boolean_logic"] * 3)
    steps = np.arange(3, dtype=np.int16)
    horizons = np.asarray([3, 3, 3], dtype=np.int16)
    data = TransitionArrays(
        current_j=current_j,
        next_j=current_j,
        current_h=current_j,
        next_h=current_j,
        current_action=actions,
        next_action=np.asarray([1, 2, 3]),
        u=build_u(actions, families, steps, horizons),
        next_u=build_u(np.asarray([1, 2, 3]), families, steps + 1, horizons),
        history_j=current_j[:, None],
        history_mask=np.ones((3, 1), dtype=np.float32),
        example_id=np.asarray(["example"] * 3),
        family=families,
        step_index=steps,
        horizon=horizons,
    )
    model = RecordingController()
    _evaluate_recurrent(
        model,
        data,
        compact,
        compact,
        horizons=[1, 2, 3],
        device=torch.device("cpu"),
    )
    assert model.seen_u[0][0, 0] == 1
    assert model.seen_u[1][0, 5] == 1
    assert model.seen_u[2][0, 5] == 1
    assert model.seen_u[1][0, 1] == 0


def test_v5_replication_is_deterministic_and_disjoint_from_v4() -> None:
    from jclosure.config import load_config

    config = load_config(ROOT / "configs/peripheral_v5.yaml")
    first = build_replication_tasks(ROOT, config)
    second = build_replication_tasks(ROOT, config)
    assert [value.to_dict() for value in first] == [value.to_dict() for value in second]
    hashes = {value.program_hash for value in first}
    assert not hashes & v4_program_hashes(ROOT)
    assert len(first) == sum(config["h2_replication_v5"]["counts_by_family"].values())


def test_v5_freeze_digest_changes_on_protocol_change() -> None:
    left = {"protocol_version": "v5", "threshold": 0.8}
    right = {"protocol_version": "v5", "threshold": 0.81}
    assert _digest(left) != _digest(right)
    left["freeze_digest"] = "ignored"
    assert _digest(left) == _digest({"protocol_version": "v5", "threshold": 0.8})


def test_v5_schema_records_round_trip() -> None:
    values: list[
        PeripheralReferenceRecord | PeripheralStateRecord | H2ReplicationRecord
    ] = [
        PeripheralReferenceRecord("run", "mlp", "j_only", "test", 1, 0.9, 0.1, 0.8, 42),
        PeripheralStateRecord(
            "run", "pca", 64, "test", 0.9, 0.8, 0.7, None, None, None, None, False, 42
        ),
        H2ReplicationRecord(
            "run", "base", "prompt", "boolean_logic", 4, "clean", True, {}
        ),
    ]
    for value in values:
        payload = json.loads(json.dumps(value.to_dict()))
        assert payload["schema_version"] == 7
        assert payload["protocol_version"]
