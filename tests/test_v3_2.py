import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from jclosure.clamp_v3_2 import build_v32_schedule, schedules_share_initial_perturbation
from jclosure.compact_memory_v3_2 import action_metrics, representation_gate_reasons
from jclosure.memory_analysis_v3_2 import (
    PairedInterval,
    memory_utility_reasons,
    paired_cluster_interval,
)
from jclosure.runtime_v3_2 import restoration_is_optimized
from jclosure.statistics_v3_2 import (
    conditional_success_summary,
    exclude_invalid_restorations,
)


def test_initial_and_restoration_scopes_are_independent_and_maps_differ():
    common = {
        "initial_layer": 23,
        "restoration_layers": [24, 25, 26],
        "sequence_length": 5,
        "initial_scope": "final",
    }
    single = build_v32_schedule(mode="single", restoration_scope="none", **common)
    final = build_v32_schedule(
        mode="persistent_final", restoration_scope="final", **common
    )
    all_positions = build_v32_schedule(
        mode="persistent_all", restoration_scope="all_non_padding", **common
    )
    assert schedules_share_initial_perturbation((single, final, all_positions))
    assert single.initial_events == final.initial_events == all_positions.initial_events
    assert len({single.events, final.events, all_positions.events}) == 3
    assert [(event.layer, event.position, event.operation_type) for event in single.events] == [
        (23, 4, "initial")
    ]
    assert (24, 0, "restoration") not in [
        (event.layer, event.position, event.operation_type) for event in final.events
    ]
    assert (24, 0, "restoration") in [
        (event.layer, event.position, event.operation_type) for event in all_positions.events
    ]


def test_mode_rejects_mismatched_restoration_scope():
    with pytest.raises(ValueError):
        build_v32_schedule(
            mode="persistent_final",
            initial_layer=23,
            restoration_layers=[24],
            sequence_length=4,
            initial_scope="final",
            restoration_scope="all_non_padding",
        )


def test_restoration_method_wiring_distinguishes_optimizer():
    assert not restoration_is_optimized("dense_local")
    assert restoration_is_optimized("dense_optimized")
    with pytest.raises(ValueError):
        restoration_is_optimized("declared_but_unwired")


def test_restoration_gate_uses_conditional_denominator():
    applicable = np.array([True, True, False, False, False])
    success = np.array([True, True, False, False, False])
    result = conditional_success_summary(
        applicable,
        success,
        minimum_applicable=2,
        minimum_rate=0.8,
        n_resamples=200,
        confidence=0.95,
        seed=7,
    )
    assert result.applicable == 2
    assert result.successes == 2
    assert result.rate == 1.0
    assert result.eligible


def test_invalid_restoration_is_absent_from_paired_aggregation():
    rows = []
    for base in ("a", "b"):
        for mode in ("single", "persistent_final", "persistent_all"):
            rows.append(
                {
                    "base_trial_id": base,
                    "mode": mode,
                    "valid": not (base == "a" and mode == "persistent_all"),
                }
            )
    result = exclude_invalid_restorations(
        pd.DataFrame(rows),
        required_modes=("single", "persistent_final", "persistent_all"),
    )
    assert set(result["base_trial_id"]) == {"b"}


def test_v32_schema_declares_final_primary_initial_scope():
    schema = json.loads(open("schemas/trial-record-v3-2.schema.json").read())
    assert schema["properties"]["schema_version"]["const"] == 5
    assert schema["properties"]["initial_scope"]["const"] == "final"
    config = json.loads(json.dumps({"initial_scope": "final"}))
    assert config["initial_scope"] == "final"


def test_hook_events_are_machine_serializable():
    schedule = build_v32_schedule(
        mode="persistent_all",
        initial_layer=23,
        restoration_layers=[24],
        sequence_length=3,
        initial_scope="final",
        restoration_scope="all_non_padding",
    )
    payload = asdict(schedule)
    assert json.loads(json.dumps(payload))["events"][0]["operation_type"] == "initial"


def test_teacher_fidelity_and_ground_truth_accuracy_are_distinct():
    metrics = action_metrics(
        np.asarray([1, 1]),
        np.asarray([1, 1]),
        np.asarray([0, 0]),
    )
    assert metrics["teacher_action_fidelity"] == 1.0
    assert metrics["ground_truth_action_accuracy"] == 0.0


def test_configured_semantic_and_causal_gates_are_enforced():
    config = {
        "minimum_reconstruction_cosine": 0.8,
        "minimum_phase0_pass10_retention": 0.9,
        "minimum_causal_direction_retention": 0.8,
        "minimum_causal_magnitude_retention": 0.8,
        "minimum_causal_trials": 10,
    }
    reasons = representation_gate_reasons(
        {
            "validation_reconstruction_cosine": 0.95,
            "phase0_pass10_retention": 0.89,
            "causal_direction_retention": 0.79,
            "causal_magnitude_retention": 0.81,
            "causal_trials": 12,
        },
        config,
    )
    assert reasons == ("causal_direction_retention", "semantic_retention")


def test_memory_utility_requires_paired_ci_and_all_seed_directions():
    frame = pd.DataFrame(
        {
            "example_id": ["a", "a", "b", "b"],
            "delta": [0.03, 0.04, 0.02, 0.03],
        }
    )
    interval = paired_cluster_interval(
        frame,
        cluster="example_id",
        value="delta",
        n_resamples=500,
        confidence=0.95,
        seed=3,
    )
    reasons = memory_utility_reasons(
        cosine=interval,
        trajectory_reduction=0.25,
        teacher_fidelity_delta=-0.01,
        seed_deltas=[0.03, 0.02, 0.01],
        expected_seeds=3,
        config={
            "memory_effect_min_cosine": 0.02,
            "trajectory_reduction_fraction": 0.20,
            "semantic_noninferiority": 0.02,
        },
    )
    assert reasons == ()
    assert memory_utility_reasons(
        cosine=PairedInterval(0.03, 0.01, 0.05, 2),
        trajectory_reduction=0.25,
        teacher_fidelity_delta=0.0,
        seed_deltas=[0.03, -0.01, 0.02],
        expected_seeds=3,
        config={
            "memory_effect_min_cosine": 0.02,
            "trajectory_reduction_fraction": 0.20,
            "semantic_noninferiority": 0.02,
        },
    ) == ("seed_direction_inconsistent",)
