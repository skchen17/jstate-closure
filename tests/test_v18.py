"""Focused V18 freeze, action-model and distance invariants."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from jclosure.experiments.crossed_bank_v18 import _split
from jclosure.experiments.strict_match_v18 import _distance
from jclosure.experiments.strong_ceiling_v18 import _matrix, _summary
from jclosure.protocol_v18 import digest


def test_v18_digest_and_freeze_field() -> None:
    assert digest({"a": 1, "freeze_digest": "old"}) == digest({"a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_v18_unified_feature_hierarchy_shapes() -> None:
    state = np.ones((3, 128), dtype=np.float32)
    action = np.zeros((3, 8), dtype=np.float32)
    action[:, 0] = 0.5
    assert _matrix(state, action, "M0_additive_linear").shape == (3, 137)
    assert _matrix(state, action, "M1_bilinear").shape == (3, 1161)
    assert _matrix(state, action, "M2_quadratic_tensor").shape == (3, 2193)


def test_v18_response_score_exact() -> None:
    y = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    assert _summary(y, y)["relative_l2"] == 0.0


def test_v18_distance_zero_on_identity() -> None:
    kernel = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = _distance(kernel)
    np.testing.assert_allclose(np.diag(result), [0.0, 0.0])
    np.testing.assert_allclose(result[0, 1], np.sqrt(2.0))


def test_v18_frozen_split_and_amendment_chain() -> None:
    split = _split(Path(__file__).resolve().parents[1])
    assert len(split["train"]) == 2000
    assert len(split["validation"]) == 400
    assert len(split["actions"]) == 8
    assert {x["base_trial_id"] for x in split["train"]}.isdisjoint(
        {x["base_trial_id"] for x in split["validation"]}
    )


def test_v18_final_gates_and_bank_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/v18/processed"
    train = json.loads((output / "crossed_bank_train_v18.json").read_text())
    validation = json.loads((output / "crossed_bank_validation_v18.json").read_text())
    h1 = json.loads((output / "strong_h1_ceiling_v18.json").read_text())
    horizons = json.loads((output / "horizon_context_localization_v18.json").read_text())
    decision = json.loads((output / "v18_adjudication.json").read_text())
    assert (train["selected_states"], validation["selected_states"]) == (2000, 400)
    assert sum(x["rows"] for x in train["outputs"] if "crossed_response" in x["path"]) == 76800
    assert sum(x["rows"] for x in validation["outputs"] if "crossed_response" in x["path"]) == 15360
    assert [x["horizon"] for x in horizons["horizons"]] == [1, 2, 4, 8]
    assert decision["material_h1_raw_context"] == h1["material_h1_raw_context_gate_passed"]
    assert decision["compact_search_authorized"] == (
        bool(decision["material_h1_raw_context"] or decision["material_horizons"] or decision["strict_match_positive"])
    )
