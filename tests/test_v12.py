from __future__ import annotations

import numpy as np

from jclosure.experiments.causal_v12 import _continuous_semantic, _trajectory_error
from jclosure.experiments.jvp_v12 import _spectrum
from jclosure.state_models_v12 import fit_causal_basis, fit_pca_basis, project_basis


def test_requested_rank_is_never_silently_clamped() -> None:
    values = np.arange(30, dtype=np.float32).reshape(6, 5)
    model = fit_pca_basis(values, np.arange(4))
    try:
        project_basis(model, values, model.rank + 1)
    except ValueError as exc:
        assert "exceeds fitted rank" in str(exc)
    else:
        raise AssertionError("rank overflow was silently clamped")


def test_causal_basis_drops_numerically_null_directions() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(20, 8)).astype(np.float32)
    target = np.ones((20, 3), dtype=np.float32)
    target[:, 0] = np.linspace(-1, 1, 20)
    model = fit_causal_basis(values, target, np.arange(15))
    assert 1 <= model.rank <= 3


def test_semantic_audit_separates_continuous_and_topk_metrics() -> None:
    teacher = np.asarray([10.0, 9.0, 0.1, 0.0], dtype=np.float32)
    decoded = np.asarray([9.0, 10.0, 0.1, 0.0], dtype=np.float32)
    metrics = _continuous_semantic(teacher, decoded, 2)
    assert metrics["semantic_vector_cosine"] > 0.98
    assert metrics["semantic_topk_overlap"] == 1.0


def test_jvp_spectrum_reports_energy_ranks() -> None:
    matrix = np.diag(np.asarray([4.0, 2.0, 1.0], dtype=np.float32))
    metrics, right = _spectrum(matrix, [0.90, 0.95, 0.99])
    assert metrics["rank_90"] == 2
    assert right.shape == (3, 3)
    assert 1 < metrics["stable_rank"] < 2


def test_trajectory_error_decomposes_parallel_and_orthogonal() -> None:
    result = _trajectory_error(
        {
            "teacher_j_effect_norm": 2.0,
            "decoded_j_effect_norm": 2.0,
            "direction_cosine": 0.0,
            "j_l2_error": 2**1.5,
        }
    )
    assert result["error_parallel_to_teacher"] == 2.0
    assert result["error_orthogonal_to_teacher"] == 2.0
