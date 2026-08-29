import torch

from jclosure.datasets import TaskExample
from jclosure.decomposition import gradient_pursuit
from jclosure.experiments.geometry_v3 import (
    _local_diagnostics,
    _unique_prompt_examples,
)
from jclosure.geometry import (
    DenseJMap,
    DenseNullProjector,
    SparseStateEquality,
    SpectrumSummary,
    maximum_feasible_displacement,
    pareto_nondominated,
)


def dense_map() -> DenseJMap:
    raw = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    return DenseJMap({2: raw})


def test_raw_centered_map_and_dense_state():
    mapping = dense_map()
    raw = mapping.raw_map(2)
    centered = mapping.centered_map(2)
    assert raw.shape == centered.shape == (4, 4)
    assert torch.allclose(centered, raw - raw.mean(0, keepdim=True))
    assert torch.allclose(centered.mean(0), torch.zeros(4), atol=1e-7)
    state = mapping.dense_state(torch.tensor([1.0, 2.0, 3.0, 4.0]), 2)
    assert torch.allclose(torch.linalg.vector_norm(state), torch.tensor(1.0))


def test_analytic_jvp_vjp_and_gram_match_autograd():
    mapping = dense_map()
    h = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    v = torch.tensor([-0.2, 0.3, 0.7, 0.1])
    u = torch.tensor([0.4, -0.1, 0.2, -0.5])
    _, automatic_jvp = torch.autograd.functional.jvp(
        lambda value: mapping.dense_state(value, 2), h, v
    )
    analytic_jvp = mapping.dense_state_jvp(h.detach(), v, 2)
    assert torch.allclose(analytic_jvp, automatic_jvp, atol=1e-6, rtol=1e-5)
    _, automatic_vjp = torch.autograd.functional.vjp(
        lambda value: mapping.dense_state(value, 2), h, v=u
    )
    analytic_vjp = mapping.dense_state_vjp(h.detach(), u, 2)
    assert torch.allclose(analytic_vjp, automatic_vjp, atol=1e-6, rtol=1e-5)
    jacobian = mapping.local_jacobian(h.detach(), 2)
    gram = mapping.local_jacobian_gram(h.detach(), 2)
    assert torch.allclose(gram, jacobian.T @ jacobian, atol=1e-6, rtol=1e-5)


def test_local_diagnostics_uses_deterministic_device_safe_random_vectors():
    mapping = dense_map()
    diagnostics = _local_diagnostics(
        mapping,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        2,
        checks=2,
        seed=7,
        tolerances=[1e-4],
        full_spectrum=True,
    )
    assert diagnostics["jvp_passed"]
    assert len(diagnostics["jvp_relative_errors"]) == 2


def test_bounded_local_diagnostics_skip_full_gram():
    mapping = dense_map()
    centered = mapping.centered_map(2)
    map_summary = SpectrumSummary.from_singular_values(
        torch.linalg.svdvals(centered),
        rows=centered.shape[0],
        cols=centered.shape[1],
        relative_tolerances=[1e-4],
    )
    diagnostics = _local_diagnostics(
        mapping,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        2,
        checks=1,
        seed=7,
        tolerances=[1e-4],
        full_spectrum=False,
        map_summary=map_summary,
    )
    assert diagnostics["spectrum"] is None
    assert diagnostics["rank_status"] == "NUMERICALLY_BOUNDED"
    assert diagnostics["structural_null_dimension"] >= 1
    assert diagnostics["extremal_method"] == "power_top_only"
    lower, upper = diagnostics["tolerance_rank_bounds"]["relative_1e-04"]
    assert upper - lower <= 1
    assert diagnostics["jvp_passed"]


def test_geometry_prompt_selection_deduplicates_deterministically():
    examples = [
        TaskExample("a", "boolean", "t", "same prompt", "yes"),
        TaskExample("b", "boolean", "t", "same prompt", "yes"),
        TaskExample("c", "boolean", "t", "different prompt", "no"),
    ]
    selected = _unique_prompt_examples(examples, 2, family="boolean")
    assert [example.example_id for example in selected] == ["a", "c"]


def test_radial_null_and_tangent_intersection():
    mapping = dense_map()
    h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert mapping.radial_residual(h, 2) <= 1e-6
    projector = DenseNullProjector(mapping, 2)
    basis, _ = projector.low_singular_basis(h, relative_tolerance=1e-6)
    assert basis.shape[1] >= 1
    tangent = projector.tangent_intersection(basis, h)
    if tangent.shape[1]:
        assert torch.allclose(tangent.T @ h, torch.zeros(tangent.shape[1]), atol=1e-5)


def test_spectrum_rank_metrics_known_matrix():
    singular_values = torch.tensor([5.0, 1.0, 0.001, 0.0])
    summary = SpectrumSummary.from_singular_values(
        singular_values, rows=6, cols=4, dtype=torch.float32
    )
    assert summary.tolerance_ranks["relative_1e-02"] == 2
    assert summary.tolerance_ranks["relative_1e-05"] == 3
    assert 1.0 < summary.stable_rank < 2.0
    assert len(summary.cumulative_variance) == 4


def test_dense_null_projection_and_sphere_retraction():
    mapping = dense_map()
    h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    donor = torch.tensor([2.0, -1.0, 0.5, 1.0])
    projector = DenseNullProjector(mapping, 2)
    values, vectors = projector.local_singular_system(h)
    cached_basis = projector.low_singular_basis_from_system(
        values, vectors, relative_tolerance=1e-6
    )
    cached_basis = projector.tangent_intersection(cached_basis, h)
    delta, basis, _ = projector.donor_projection(
        h, donor, relative_tolerance=1e-6, sphere_tangent=True
    )
    assert torch.allclose(
        cached_basis @ cached_basis.T,
        basis @ basis.T,
        atol=1e-5,
        rtol=1e-5,
    )
    if basis.shape[1]:
        assert torch.allclose(basis.T @ (donor - delta), torch.zeros(basis.shape[1]), atol=1e-5)
    retracted = projector.retract_to_sphere(h, delta)
    assert torch.allclose(
        torch.linalg.vector_norm(h + retracted),
        torch.linalg.vector_norm(h),
        atol=1e-5,
    )


def test_sparse_equality_is_independent_of_dense_null():
    dictionary = torch.eye(4)
    clean = gradient_pursuit(torch.tensor([2.0, 1.0, 0.0, 0.5]), dictionary, k=4)
    same = gradient_pursuit(torch.tensor([2.0, 1.0, 0.0, 0.5]), dictionary, k=4)
    changed = gradient_pursuit(torch.tensor([1.0, 2.0, 0.0, 0.5]), dictionary, k=4)
    assert SparseStateEquality.compare(clean, same).passed
    assert not SparseStateEquality.compare(clean, changed).passed


def test_pareto_frontier_and_maximum_feasible_displacement():
    records = [
        {"displacement": 0.1, "error": 0.01},
        {"displacement": 0.2, "error": 0.01},
        {"displacement": 0.3, "error": 0.03},
    ]
    frontier = pareto_nondominated(
        records, maximize=("displacement",), minimize=("error",)
    )
    assert records[0] not in frontier
    feasibility = [
        {
            "displacement_fraction": 0.2,
            "dense_cosine": 0.996,
            "top10_overlap": 0.9,
            "rms_drift": 0.01,
            "natural": True,
        },
        {
            "displacement_fraction": 0.4,
            "dense_cosine": 0.99,
            "top10_overlap": 1.0,
            "rms_drift": 0.01,
            "natural": True,
        },
    ]
    assert maximum_feasible_displacement(feasibility) == 0.2
