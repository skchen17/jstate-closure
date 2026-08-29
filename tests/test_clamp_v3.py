import json
from types import SimpleNamespace

import pandas as pd
import torch

from jclosure.clamp_v3 import (
    V3ClampThresholds,
    build_clamp_schedule,
    construct_dense_candidate,
    construct_sparse_candidate,
    project_dense_candidate,
    scheduled_sparse_clamp_transforms,
    validate_v3_clamp,
)
from jclosure.config import load_config
from jclosure.experiments.clamp_v3_calibration import (
    _assigned_layers,
    _balanced_calibration_records,
    _calibration_trial_ids,
    _merge_shards,
    summarize_calibration,
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


def test_dense_projection_preserves_observed_null_displacement_without_rescaling():
    state_encoder = encoder()
    mapping = DenseJMap.from_encoder(state_encoder)
    clean = torch.tensor([2.0, 1.0, 0.5, 0.2])
    perturbed = torch.tensor([2.0, 1.0, 0.5, 0.7])
    candidate, status = project_dense_candidate(
        clean,
        perturbed,
        layer=0,
        dense_map=mapping,
        relative_tolerance=1e-6,
        optimized=False,
    )
    assert status["status"] == "PROJECTED"
    assert torch.linalg.vector_norm(candidate - clean) <= torch.linalg.vector_norm(
        perturbed - clean
    ) + 1e-6
    clean_state = mapping.dense_state(clean, 0)
    candidate_state = mapping.dense_state(candidate, 0)
    assert torch.nn.functional.cosine_similarity(
        clean_state[None], candidate_state[None]
    ).item() >= 0.995


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


def test_calibration_layer_shards_are_disjoint_and_complete():
    layers = list(range(23, 30))
    left = _assigned_layers(layers, shard_index=0, shard_count=2)
    right = _assigned_layers(layers, shard_index=1, shard_count=2)
    assert not set(left) & set(right)
    assert sorted([*left, *right]) == layers


def test_calibration_merge_requires_one_completed_record_per_shard(tmp_path):
    raw = tmp_path / "results/v3/raw"
    processed = tmp_path / "results/v3/processed"
    processed.mkdir(parents=True)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    config = load_config("configs/geometry_v3.yaml")
    layers = [int(value) for value in config["geometry"]["candidate_layers"]]
    for shard_index in range(2):
        directory = raw / f"clamp-v3-calibration-shard-{shard_index}"
        directory.mkdir(parents=True)
        assigned = _assigned_layers(
            layers, shard_index=shard_index, shard_count=2
        )
        frame = pd.DataFrame(
            [
                {
                    "base_trial_id": f"base-{layer}",
                    "method": "dense_local_null",
                    "dictionary_size": 4096,
                    "layer": layer,
                    "formal_valid": False,
                    "state_valid_before_naturality": False,
                    "natural": False,
                    "construction_failure_reason": "dense_equality_constraint",
                    "small_perturbation_valid": False,
                    "finite": True,
                    "activation_explosion": False,
                }
                for layer in assigned
            ]
        )
        candidates = directory / "candidates.parquet"
        frame.to_parquet(candidates, index=False)
        manifest = {
            "run_id": directory.name,
            "status": "COMPLETED_SHARD",
            "shard_group_id": "group",
            "shard_index": shard_index,
            "created_at": f"2026-01-0{shard_index + 1}",
            "git_commit": "commit",
            "config_digest": "config",
            "activation_bank_manifest": "bank.jsonl",
            "candidate_records": str(candidates.relative_to(tmp_path)),
            "hook_sanity": {
                "zero_exact": True,
                "identity_exact": True,
                "determinism_exact": True,
                "cleanup_exact": True,
                "finite": True,
            },
            "v2_hash_guard": {"status": "PASSED"},
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    context = SimpleNamespace(
        root=tmp_path,
        raw_dir=raw,
        processed_dir=processed,
        run_id="merge-run",
        config=config,
    )
    output, summary = _merge_shards(
        context,
        shard_group_id="group",
        shard_count=2,
        bank_manifest=bank,
    )
    assert output.is_file()
    assert summary["attempted"] == len(layers)
    assert summary["source_shards"] == [
        "clamp-v3-calibration-shard-0",
        "clamp-v3-calibration-shard-1",
    ]


def test_calibration_naturality_fraction_uses_pre_naturality_valid_set():
    frame = pd.DataFrame(
        [
            {
                "layer": 23,
                "dictionary_size": 4096,
                "method": "dense_local_null",
                "state_valid_before_naturality": True,
                "formal_valid": index < 180,
                "natural": index < 180,
                "construction_failure_reason": None,
                "small_perturbation_valid": False,
                "finite": True,
                "activation_explosion": False,
            }
            for index in range(200)
        ]
    )
    sanity = {
        "zero_exact": True,
        "identity_exact": True,
        "determinism_exact": True,
        "cleanup_exact": True,
        "finite": True,
    }
    summary = summarize_calibration(
        frame, config=load_config("configs/geometry_v3.yaml"), hook_sanity=sanity
    )
    layer = summary["layers"][0]
    assert layer["strict_valid"] == 200
    assert layer["formal_natural_valid"] == 180
    assert layer["natural_fraction_among_valid"] == 0.9
    assert "naturality_valid_fraction" in layer["reasons"]
