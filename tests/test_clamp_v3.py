import torch

from jclosure.clamp_v3 import (
    V3ClampThresholds,
    build_clamp_schedule,
    construct_dense_candidate,
    construct_sparse_candidate,
    scheduled_sparse_clamp_transforms,
    validate_v3_clamp,
)
from jclosure.experiments.clamp_v3_calibration import (
    _balanced_calibration_records,
    _calibration_trial_ids,
)
from jclosure.geometry import DenseJMap
from jclosure.jstate import ConceptVocabulary, JStateEncoder


def encoder() -> JStateEncoder:
    vocabulary = ConceptVocabulary((10, 11, 12), ("alpha", "beta", "gamma"))
    raw = {
        0: torch.tensor(
            [
                [2.0, 0, 0, 0],
                [0, 3.0, 0, 0],
                [0, 0, 4.0, 0],
            ]
        )
    }
    return JStateEncoder(raw, vocabulary, raw_directions=raw, k=3)


def test_v3_dense_and_sparse_equality_are_independent():
    state_encoder = encoder()
    dense_map = DenseJMap.from_encoder(state_encoder)
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    candidate = torch.tensor([2.0, 1.0, 0.5, 0.8])
    thresholds = V3ClampThresholds(
        rms_drift=1.0,
        formal_displacement=0.1,
        sparse_weighted_jaccard=0.9,
    )
    dense = validate_v3_clamp(
        clean,
        candidate,
        layer=0,
        state_definition="V3-Dense",
        encoder=state_encoder,
        dense_map=dense_map,
        natural_scale=1.0,
        natural=True,
        thresholds=thresholds,
    )
    sparse = validate_v3_clamp(
        clean,
        candidate,
        layer=0,
        state_definition="V3-Sparse",
        encoder=state_encoder,
        dense_map=dense_map,
        natural_scale=1.0,
        natural=True,
        thresholds=thresholds,
    )
    assert dense.sparse_equality is None
    assert sparse.sparse_equality is not None
    assert dense.formal_valid and sparse.formal_valid


def test_sparse_candidate_restores_sparse_component():
    state_encoder = encoder()
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    perturbed = torch.tensor([8.0, 7.0, 6.0, 0.9])
    candidate = construct_sparse_candidate(
        clean, perturbed, layer=0, encoder=state_encoder
    )
    assert torch.allclose(candidate[:3], clean[:3], atol=1e-5)
    assert candidate[3] == perturbed[3]


def test_dense_candidate_reports_zero_dimensional_intersection():
    raw = torch.eye(4)
    mapping = DenseJMap({0: raw})
    clean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    candidate, status = construct_dense_candidate(
        clean,
        torch.tensor([0.2, -0.1, 0.3, 0.4]),
        layer=0,
        dense_map=mapping,
        natural_scale=1.0,
        displacement_fraction=0.2,
        relative_tolerance=1e-8,
        optimized=False,
    )
    assert status["status"] == "FAILED"
    assert torch.equal(candidate, clean)


def test_schedule_resolves_padding_and_applies_only_recorded_positions():
    state_encoder = encoder()
    schedule = build_clamp_schedule(
        mode="persistent_final",
        initial_layer=0,
        future_layers=[],
        position_scope="all_non_padding",
        sequence_length=4,
        attention_mask=torch.tensor([0, 1, 1, 1]),
        explicit_positions=None,
        reasoning_span=None,
        state_definition="V3-Sparse",
        dictionary_size=4096,
    )
    assert schedule.resolved_positions == (1, 2, 3)
    clean = torch.tensor(
        [
            [9.0, 9.0, 9.0, 9.0],
            [2.0, 1.0, 0.5, 0.2],
            [1.0, 2.0, 0.5, 0.3],
            [0.5, 1.0, 2.0, 0.4],
        ]
    )
    activation = clean.clone()
    activation[:, :3] += 5
    transforms = scheduled_sparse_clamp_transforms(
        schedule, {0: clean}, encoder=state_encoder
    )
    output = transforms[0](activation.unsqueeze(0), 0)
    assert torch.equal(output[0, 0], activation[0])
    assert torch.allclose(output[0, 1:, :3], clean[1:, :3], atol=1e-5)


def test_calibration_batch_is_deterministic_and_task_balanced():
    records = [
        {
            "prompt_id": f"{family}-{index}",
            "prompt_hash": f"{index:02d}-{family}",
            "task_family": family,
        }
        for family in ("gamma", "alpha", "beta")
        for index in range(3)
    ]
    selected = _balanced_calibration_records(records, 6)
    assert [row["task_family"] for row in selected] == [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "gamma",
    ]
    assert selected == _balanced_calibration_records(list(reversed(records)), 6)


def test_calibration_pairing_is_shared_across_dictionaries_and_methods():
    base_local, paired_local = _calibration_trial_ids("anchor", "donor", 24, "local")
    base_optimized, paired_optimized = _calibration_trial_ids(
        "anchor", "donor", 24, "optimized"
    )
    assert base_local == base_optimized
    assert paired_local != paired_optimized
    assert (base_local, paired_local) == _calibration_trial_ids(
        "anchor", "donor", 24, "local"
    )
