import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from jclosure.clamp_v3 import V3ClampValidation
from jclosure.clamp_v3_1 import (
    build_v31_schedule,
    validate_intervention_eligibility,
    validate_restoration_eligibility,
)
from jclosure.compact_memory_v3_1 import (
    autonomous_rollout,
    build_parameter_matched_controller,
    parameter_count,
    scheduled_feedback_probability,
)
from jclosure.config import load_config
from jclosure.datasets_v3_1 import (
    deterministic_memory_split,
    disjoint_arithmetic_domains,
    generate_iterated_modular_arithmetic,
    generate_sequential_state_machines,
)
from jclosure.experiments.calibrate_v3_1 import (
    _shard_invariant_config_digest,
    _write_partitioned_records,
)
from jclosure.provenance import sha256_file
from jclosure.runtime_v3_1 import answer_token_id
from jclosure.statistics_v3_1 import (
    common_valid_base_trials,
    paired_mediation_bootstrap,
)


def test_answer_token_prefers_canonical_bare_surface():
    class Tokenizer:
        def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return {"yes": [10], " yes": [11]}[surface]

    assert answer_token_id(Tokenizer(), "yes") == 10


def test_v31_config_freezes_primary_dense_null_tolerance():
    config = load_config("configs/closure_v3_1.yaml")
    assert config["geometry"]["formal_null_tolerance"] == pytest.approx(1e-4)


def test_v31_shard_digest_ignores_only_runtime_device():
    left = {"config": {"model": {"device": 0, "id": "model"}, "seed": 7}}
    right = {"config": {"model": {"device": 1, "id": "model"}, "seed": 7}}
    assert _shard_invariant_config_digest(left) == _shard_invariant_config_digest(
        right
    )
    right["config"]["seed"] = 8
    assert _shard_invariant_config_digest(left) != _shard_invariant_config_digest(
        right
    )


def test_partitioned_calibration_records_are_complete_and_hashed(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {"l1": 23, "position_scope": "final", "value": index}
            for index in range(7)
        ]
        + [
            {"l1": 24, "position_scope": "all_non_padding", "value": index}
            for index in range(3)
        ]
    )
    manifest_path = tmp_path / "records.json"
    manifest = _write_partitioned_records(
        frame,
        root=tmp_path,
        output_root=tmp_path / "records",
        manifest_path=manifest_path,
        rows_per_file=2,
    )
    assert manifest["rows"] == len(frame)
    assert sum(part["rows"] for part in manifest["parts"]) == len(frame)
    for part in manifest["parts"]:
        path = tmp_path / part["path"]
        assert path.is_file()
        assert sha256_file(path) == part["sha256"]


def test_autonomous_rollout_feedback_uses_only_predictions():
    class Increment(nn.Module):
        def forward(self, state):
            return state + 1, torch.cat((state, -state), dim=-1)

    states, _ = autonomous_rollout(
        Increment(), torch.zeros(1, 1), steps=3, family="markov"
    )
    assert states.flatten().tolist() == [1.0, 2.0, 3.0]


def test_gru_memory_persists_across_rollout_steps():
    class MemoryCounter(nn.Module):
        memory_dim = 1

        def forward(self, state, memory):
            del state
            updated = memory + 1
            return updated, torch.cat((updated, -updated), dim=-1), updated

    states, _ = autonomous_rollout(
        MemoryCounter(),
        torch.zeros(1, 1),
        steps=3,
        family="gru",
        memory_dim=1,
    )
    assert states.flatten().tolist() == [1.0, 2.0, 3.0]


def test_parameter_matched_controller_and_feedback_schedule():
    target = 100_000
    model = build_parameter_matched_controller(
        "gru",
        state_dim=16,
        action_count=4,
        target=target,
        tolerance=0.05,
        memory_dim=8,
    )
    assert abs(parameter_count(model) - target) / target <= 0.05
    assert (
        scheduled_feedback_probability(0, 50, warmup_fraction=0.2, maximum_feedback=0.8)
        == 0
    )
    assert scheduled_feedback_probability(
        49, 50, warmup_fraction=0.2, maximum_feedback=0.8
    ) == pytest.approx(0.8)


def test_common_valid_and_paired_mediation_bootstrap():
    rows = []
    for index in range(20):
        for mode, value in (
            ("single", 0.10),
            ("persistent_final", 0.05),
            ("persistent_all", 0.02),
        ):
            rows.append(
                {
                    "base_trial_id": str(index),
                    "prompt_id": str(index),
                    "condition": "state_preserving",
                    "mode": mode,
                    "valid": not (index == 0 and mode == "persistent_all"),
                    "js": value,
                }
            )
    frame = pd.DataFrame(rows)
    common = common_valid_base_trials(
        frame,
        required_pairs={
            ("state_preserving", "single"),
            ("state_preserving", "persistent_final"),
            ("state_preserving", "persistent_all"),
        },
    )
    assert common["base_trial_id"].nunique() == 19
    result = paired_mediation_bootstrap(
        common,
        cluster_col="prompt_id",
        value_col="js",
        n_resamples=200,
        confidence=0.95,
        seed=1,
        null_threshold=0.001,
    )
    assert result["ratio_interpretation_gate"]
    assert result["M_final"]["estimate"] == pytest.approx(0.5)
    assert result["M_all"]["estimate"] == pytest.approx(0.8)


def _validation(displacement: float, *, failures=()):
    formal = displacement >= 0.20 and not failures
    return V3ClampValidation(
        state_definition="V3-Dense",
        valid=formal,
        formal_valid=formal,
        small_perturbation_valid=0.05 <= displacement < 0.20 and not failures,
        failure_reasons=tuple(failures),
        dense_cosine=0.999,
        dense_profile_l2=0.001,
        top10_overlap=1.0,
        rms_drift=0.001,
        displacement_fraction=displacement,
        natural=True,
    )


def test_intervention_and_restoration_have_distinct_displacement_contracts():
    validation = _validation(
        0.01,
        failures=("displacement_below_sensitivity",),
    )
    intervention = validate_intervention_eligibility(
        validation,
        finite=True,
        activation_explosion=False,
        construction_status="PROJECTED",
    )
    restoration = validate_restoration_eligibility(
        validation,
        correction=torch.tensor([0.01, 0.0]),
        natural_scale=1.0,
        finite=True,
        activation_explosion=False,
        construction_status="PROJECTED",
    )
    assert not intervention.passed
    assert restoration.passed
    assert restoration.correction_natural_fraction == pytest.approx(0.01)


def test_initial_intervention_still_requires_formal_displacement():
    result = validate_intervention_eligibility(
        _validation(0.19, failures=("displacement_below_formal",)),
        finite=True,
        activation_explosion=False,
        construction_status="CONSTRUCTED",
    )
    assert not result.passed
    assert "displacement_below_formal" in result.reasons


def test_v31_single_final_and_all_schedules_are_distinct():
    kwargs = {
        "initial_layer": 23,
        "restoration_layers": [24, 25, 26],
        "initial_positions": [0, 1, 2],
        "final_position": 2,
    }
    single = build_v31_schedule(mode="single", **kwargs)
    final = build_v31_schedule(mode="persistent_final", **kwargs)
    all_positions = build_v31_schedule(mode="persistent_all", **kwargs)
    assert single.modified_layer_positions == ((23, 0), (23, 1), (23, 2))
    assert (24, 0) not in final.modified_layer_positions
    assert (24, 2) in final.modified_layer_positions
    assert (24, 0) in all_positions.modified_layer_positions
    assert (
        len(
            {
                single.modified_layer_positions,
                final.modified_layer_positions,
                all_positions.modified_layer_positions,
            }
        )
        == 3
    )


def test_final_only_initial_scope_makes_persistent_schedules_equivalent():
    kwargs = {
        "initial_layer": 23,
        "restoration_layers": [24, 25],
        "initial_positions": [7],
        "final_position": 7,
    }
    final = build_v31_schedule(mode="persistent_final", **kwargs)
    all_positions = build_v31_schedule(mode="persistent_all", **kwargs)
    assert final.modified_layer_positions == all_positions.modified_layer_positions


def test_v31_arithmetic_domains_are_deterministic_and_disjoint():
    declarations = {
        "fit": {"seed": 11, "candidates": 20},
        "calibration": {"seed": 12, "candidates": 20},
    }
    first = disjoint_arithmetic_domains(declarations)
    second = disjoint_arithmetic_domains(declarations)
    assert first == second
    left = {item.example_id for item in first["fit"]}
    right = {item.example_id for item in first["calibration"]}
    assert left.isdisjoint(right)


def test_memory_programs_and_splits_do_not_leak():
    tasks = [
        *generate_iterated_modular_arithmetic(30, seed=4),
        *generate_sequential_state_machines(30, seed=5),
    ]
    splits = deterministic_memory_split(tasks)
    hashes = {
        split: {item.program_hash for item in values}
        for split, values in splits.items()
    }
    assert hashes["train"].isdisjoint(hashes["validation"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["validation"].isdisjoint(hashes["test"])
    assert {item.length for item in tasks} == {8, 16, 32}


def test_schema_v4_protocols_are_separate():
    causal = json.loads(open("schemas/trial-record-v3-1.schema.json").read())
    memory = json.loads(open("schemas/compact-memory-record-v3-1.schema.json").read())
    assert causal["properties"]["schema_version"]["const"] == 4
    assert memory["properties"]["schema_version"]["const"] == 4
    assert (
        causal["properties"]["protocol_version"]["const"]
        != memory["properties"]["protocol_version"]["const"]
    )
