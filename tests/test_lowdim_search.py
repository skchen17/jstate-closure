import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from jclosure.experiments.lowdim_search import (
    _constrained_predictive_basis,
    _extract,
    _latest_completed_geometry_dirs,
    run_search,
)


def test_lowdim_search_uses_disjoint_fit_and_audit_samples():
    generator = np.random.default_rng(4)
    rows = []
    transition = generator.normal(size=(8, 8))
    for index in range(30):
        state = generator.normal(size=8)
        next_state = state @ transition + generator.normal(scale=0.01, size=8)
        rows.append(
            {
                "split": "fit" if index < 20 else "audit",
                "state": state.astype(np.float32),
                "next_state": next_state.astype(np.float32),
                "remainder": generator.normal(size=5).astype(np.float32),
                "sparse_state": np.where(np.abs(state) > 0.5, state, 0).astype(
                    np.float32
                ),
            }
        )
    results, summary = run_search(pd.DataFrame(rows), [2, 4])
    assert summary["train_samples"] == 20
    assert summary["test_samples"] == 10
    assert set(results["candidate"]) == {
        "dense_profile_pca",
        "deterministic_concept_clusters",
        "predictive_linear_bottleneck",
        "constrained_learned_encoder",
        "sparse_active_atoms",
    }
    assert set(results.loc[results["candidate"] != "sparse_active_atoms", "dimension"]) <= {
        2,
        4,
    }
    assert int(results.loc[results["candidate"] == "sparse_active_atoms", "dimension"].iloc[0]) <= 16


def test_constrained_basis_exactly_contains_frozen_concept_axes():
    predictive = np.eye(6, dtype=np.float32)
    basis = _constrained_predictive_basis(
        predictive,
        dimension=4,
        feature_count=6,
        frozen_indices=[1, 4],
    )
    assert basis is not None
    projector = basis @ basis.T
    assert np.allclose(projector[:, 1], np.eye(6)[:, 1])
    assert np.allclose(projector[:, 4], np.eye(6)[:, 4])


def test_completed_geometry_selection_ignores_failed_and_old_shards(tmp_path):
    raw = tmp_path / "results/v3/raw"
    manifests = [
        ("geometry-v3-old", "COMPLETED", "2026-01-01", 0),
        ("geometry-v3-new", "COMPLETED", "2026-01-02", 0),
        ("geometry-v3-failed", "FAILED", "2026-01-03", 1),
        ("geometry-v3-shard1", "COMPLETED", "2026-01-02", 1),
    ]
    for name, status, created, shard in manifests:
        directory = raw / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "stage": "spectrum",
                    "created_at": created,
                    "shard_index": shard,
                }
            ),
            encoding="utf-8",
        )
    selected = _latest_completed_geometry_dirs(tmp_path, "spectrum")
    assert [path.name for path in selected] == [
        "geometry-v3-new",
        "geometry-v3-shard1",
    ]


def test_extract_moves_activations_to_requested_device(tmp_path):
    activation_path = tmp_path / "activation.pt"
    torch.save(
        {
            "activations": {
                0: torch.ones(2, 4),
                1: torch.full((2, 4), 2.0),
            }
        },
        activation_path,
    )

    class FakeDenseMap:
        def dense_state(self, value, layer):
            assert value.device == torch.device("cpu")
            return value

    class FakeEncoder:
        vocabulary = SimpleNamespace(token_ids=(10, 11, 12))

        def decompose(self, value, layer):
            assert value.device == torch.device("cpu")
            return SimpleNamespace(
                remainder=torch.zeros_like(value),
                atom_indices=torch.tensor([1]),
                coefficients=torch.tensor([0.5]),
            )

    frame = _extract(
        tmp_path,
        [
            {
                "activation_path": activation_path.name,
                "prompt_id": "prompt",
                "task_family": "arithmetic",
                "split": "fit",
            }
        ],
        [0, 1],
        FakeEncoder(),
        FakeDenseMap(),
        device="cpu",
    )
    assert len(frame) == 1
    assert frame.iloc[0]["sparse_state"].tolist() == [0.0, 0.5, 0.0]
