from __future__ import annotations

from typing import Any

import pytest
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.datasets_v4 import generate_program_tasks, verify_disjoint_domains
from jclosure.experiments.mediation_v4 import _restoration_transform
from jclosure.experiments.teacher_tasks_v4 import parse_semantic_generation
from jclosure.geometry import DenseJMap
from jclosure.predictive_state_v4 import (
    PredictiveBottleneck,
    PredictiveLossWeights,
    predictive_bottleneck_loss,
)
from jclosure.single_arm_v4 import (
    aligned_rollout_metrics,
    matched_controls,
    shared_low_singular_basis,
)


class TinyTokenizer:
    all_special_ids = [99]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        value = int(ids[0])
        if value == 99 and skip_special_tokens:
            return ""
        return {0: " 0", 1: " 1", 2: "\n", 3: "bad"}.get(value, "")


def test_program_generation_is_deterministic_and_balanced() -> None:
    first = generate_program_tasks(
        "boolean_logic", "not_each", 8, seed=41, horizons=(4, 8)
    )
    second = generate_program_tasks(
        "boolean_logic", "not_each", 8, seed=41, horizons=(4, 8)
    )
    assert first == second
    assert [task.horizon for task in first] == [4, 8] * 4
    assert len({task.program_hash for task in first}) == len(first)


def test_program_domains_reject_ast_overlap() -> None:
    tasks = generate_program_tasks(
        "variable_binding", "four_way_lookup", 2, seed=7, horizons=(4,)
    )
    with pytest.raises(ValueError, match="example overlap"):
        verify_disjoint_domains({"train": tasks, "validation": [tasks[0]]})


def test_semantic_parser_accepts_whitespace_and_exact_prefix() -> None:
    task = generate_program_tasks(
        "boolean_logic", "not_each", 1, seed=4, horizons=(4,)
    )[0]
    expected = task.semantic_actions
    token_ids = [int(value) for value in expected] + [2, 99]
    parsed, valid, error = parse_semantic_generation(TinyTokenizer(), token_ids, task)
    assert valid
    assert error is None
    assert parsed == expected


def test_semantic_parser_rejects_special_before_completion() -> None:
    task = generate_program_tasks(
        "boolean_logic", "not_each", 1, seed=3, horizons=(4,)
    )[0]
    parsed, valid, error = parse_semantic_generation(TinyTokenizer(), [0, 99], task)
    assert not valid
    assert len(parsed) == 1
    assert error == "special_token_before_complete"


def test_single_arm_controls_are_norm_matched() -> None:
    clean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    preserving = clean + torch.tensor([0.2, -0.1, 0.3, -0.2])
    controls = matched_controls(
        clean,
        torch.tensor([4.0, 1.0, 2.0, 8.0]),
        preserving,
        torch.tensor([1.0, -1.0, 0.0, 0.5]),
        seed=19,
    )
    target = torch.linalg.vector_norm(preserving - clean)
    for name in ("matched_random", "j_positive", "full_perturbation"):
        assert torch.allclose(
            torch.linalg.vector_norm(controls[name] - clean), target, atol=1e-6
        )
    assert torch.equal(controls["identity"], clean)


def test_shared_low_singular_basis_uses_declared_tolerance() -> None:
    matrix = torch.diag(torch.tensor([4.0, 2.0, 0.01, 0.0001]))
    dense = DenseJMap({3: matrix})
    summary = shared_low_singular_basis(dense, 3, relative_tolerance=1e-2, device="cpu")
    assert summary.rank == 2
    assert summary.null_dimension == 2


def test_aligned_metrics_detect_future_divergence() -> None:
    clean = {
        "logits": [torch.tensor([4.0, 0.0]).numpy()],
        "j_states": [torch.tensor([1.0, 0.0]).numpy()],
        "target_log_odds": [2.0],
        "actions": ["0"],
        "expected_actions": ["0"],
        "parseable": True,
    }
    changed = {
        "logits": [torch.tensor([0.0, 4.0]).numpy()],
        "j_states": [torch.tensor([0.0, 1.0]).numpy()],
        "target_log_odds": [-2.0],
        "actions": ["1"],
        "expected_actions": ["0"],
        "parseable": True,
    }
    metrics = aligned_rollout_metrics(clean, changed)
    assert metrics["output_js_divergence"] > 0
    assert metrics["future_j_trajectory_divergence"] == pytest.approx(1.0)
    assert metrics["answer_flip"]
    assert metrics["task_accuracy_change"] == -1.0


def test_predictive_objective_uses_all_four_terms() -> None:
    torch.manual_seed(0)
    model = PredictiveBottleneck(12, 4, 3, nonlinear=False)
    current = torch.nn.functional.normalize(torch.randn(6, 12), dim=-1)
    future = torch.nn.functional.normalize(torch.randn(6, 12), dim=-1)
    actions = torch.arange(6) % 3
    causal_clean = current[:2]
    causal_swapped = torch.nn.functional.normalize(current[:2] + 0.1, dim=-1)
    loss, metrics = predictive_bottleneck_loss(
        model,
        current,
        future,
        actions,
        causal_clean=causal_clean,
        causal_swapped=causal_swapped,
        weights=PredictiveLossWeights(),
    )
    loss.backward()
    assert all(
        key in metrics
        for key in ("reconstruction", "future_prediction", "semantic", "causal")
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_persistent_modes_record_distinct_position_scopes() -> None:
    clean = torch.randn(3, 4)
    activation = clean[None].clone()
    dense = DenseJMap({2: torch.eye(4)})
    thresholds = V3ClampThresholds(
        dense_cosine=0.995,
        dense_top10_overlap=0.8,
        rms_drift=0.02,
        formal_displacement=0.2,
        sensitivity_displacement=0.05,
    )
    final_capture: dict[int, dict[str, Any]] = {}
    all_capture: dict[int, dict[str, Any]] = {}
    final = _restoration_transform(
        activation,
        2,
        clean=clean,
        dense_map=dense,
        mode="persistent_final",
        capture=final_capture,
        thresholds=thresholds,
    )
    all_positions = _restoration_transform(
        activation,
        2,
        clean=clean,
        dense_map=dense,
        mode="persistent_all",
        capture=all_capture,
        thresholds=thresholds,
    )
    assert torch.equal(final, activation)
    assert torch.equal(all_positions, activation)
    assert len(final_capture[2]["events"]) == 1
    assert len(all_capture[2]["events"]) == 3
