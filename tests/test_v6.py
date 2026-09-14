from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jclosure.peripheral_v6 import (
    StrongPeripheralPredictor,
    fit_causal_directions,
    precision_audit,
    restoration_corrected_effects,
    strong_multitask_loss,
)
from jclosure.protocol_v6 import _digest
from jclosure.records_v6 import (
    RestorationNullRecord,
    TeacherCausalEndpointRecord,
)


def test_v6_precision_audit_preserves_small_nonzero_delta() -> None:
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(8, 12)).astype(np.float32)
    changed = hidden.copy()
    changed[:, 0] += np.float32(2e-6)
    matrix = rng.normal(size=(24, 12)).astype(np.float32)
    audit = precision_audit(hidden, changed, matrix)
    assert audit["nonzero_fraction_f32"] == 1.0
    assert audit["nonzero_fraction_f64"] == 1.0
    assert audit["median_norm_f32"] > 0
    assert audit["median_relative_norm_error"] < 0.1


def test_v6_restoration_correction_uses_common_pairs() -> None:
    frame = pd.DataFrame(
        {
            "base_trial_id": ["a", "a", "a", "b", "b", "b", "c"],
            "condition": [
                "single",
                "persistent_all",
                "persistent_null_all",
                "single",
                "persistent_all",
                "persistent_null_all",
                "single",
            ],
            "effect": [0.2, 0.12, 0.02, 0.4, 0.30, 0.10, 9.0],
        }
    )
    result = restoration_corrected_effects(
        frame,
        effect_column="effect",
        persistent_condition="persistent_all",
        null_condition="persistent_null_all",
    )
    assert list(result.index) == ["a", "b"]
    np.testing.assert_allclose(result["corrected_persistent"], [0.10, 0.20])


def test_v6_strong_models_and_multitask_targets_have_expected_shapes() -> None:
    batch, j_dim, r_dim, u_dim = 5, 32, 12, 23
    j = torch.nn.functional.normalize(torch.randn(batch, j_dim), dim=1)
    r = torch.randn(batch, r_dim)
    u = torch.randn(batch, u_dim)
    target = torch.nn.functional.normalize(torch.randn(batch, j_dim), dim=1)
    action = torch.randint(0, 16, (batch,))
    directions = torch.nn.functional.normalize(torch.randn(4, j_dim), dim=1)
    top = torch.tensor([1, 3, 5, 7])
    weights = {
        "full_profile": 1.0,
        "action": 1.0,
        "causal_projection": 2.0,
        "top_causal_dimensions": 2.0,
    }
    for architecture in (
        "j_only_residual",
        "full_remainder_linear",
        "full_remainder_gated",
        "full_remainder_residual",
        "full_remainder_attention",
    ):
        model = StrongPeripheralPredictor(
            j_dim, r_dim, u_dim, 32, 16, architecture=architecture
        )
        predicted, logits = model(j, u, r if model.uses_remainder else None)
        assert predicted.shape == (batch, j_dim)
        assert logits.shape == (batch, 16)
        loss = strong_multitask_loss(
            predicted, logits, target, action, directions, top, weights
        )
        assert torch.isfinite(loss.total)
    history_model = StrongPeripheralPredictor(
        j_dim, r_dim, u_dim, 32, 16, architecture="j_history_attention"
    )
    predicted, _ = history_model(
        j,
        u,
        history_j=j[:, None].repeat(1, 4, 1),
        history_mask=torch.ones(batch, 4),
    )
    assert predicted.shape == j.shape


def test_v6_causal_directions_are_deterministic() -> None:
    values = np.random.default_rng(8).normal(size=(20, 40)).astype(np.float32)
    left, top_left = fit_causal_directions(values, 8)
    right, top_right = fit_causal_directions(values, 8)
    np.testing.assert_allclose(np.abs(left), np.abs(right))
    np.testing.assert_array_equal(top_left, top_right)


def test_v6_records_round_trip_and_freeze_digest() -> None:
    endpoint = TeacherCausalEndpointRecord(
        "run",
        "base",
        "prompt",
        "boolean_logic",
        4,
        "causal_test",
        True,
        0.1,
        0.1,
        1.0,
        2.0,
        1.5,
        3.0,
        False,
        1,
    )
    restoration = RestorationNullRecord(
        "run", "base", "prompt", "boolean_logic", 4, "persistent_null_all", True, {}
    )
    for payload in (
        json.loads(json.dumps(endpoint.to_dict())),
        json.loads(json.dumps(restoration.to_dict())),
    ):
        assert payload["schema_version"] == 8
        assert payload["protocol_version"] == "peripheral_foundations_protocol_v6"
    assert _digest({"a": 1}) != _digest({"a": 2})


def test_v6_schema_is_additive() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas/protocol-v6-record.schema.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == 8
    assert (root / "artifacts/peripheral_v5.freeze.json").is_file()
