import numpy as np
import pandas as pd

from jclosure.experiments.lowdim_search import run_search


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
            }
        )
    results, summary = run_search(pd.DataFrame(rows), [2, 4])
    assert summary["train_samples"] == 20
    assert summary["test_samples"] == 10
    assert set(results["candidate"]) == {
        "dense_profile_pca",
        "deterministic_concept_clusters",
        "predictive_linear_bottleneck",
    }
    assert set(results["dimension"]) <= {2, 4}
