import numpy as np
import pandas as pd

from jclosure.experiments.analyze_v32 import (
    bootstrap_median_lower,
    context_summary,
    localization_summary,
    rank_thresholds,
    restricted_rotation_fit,
    restricted_rotation_predict,
)


def test_bootstrap_median_lower_deterministic():
    values = np.arange(1.0, 21.0)
    x = bootstrap_median_lower(values, 200, 42)
    assert x == bootstrap_median_lower(values, 200, 42)
    assert 1.0 <= x <= np.median(values)


def test_rank_thresholds_one_direction():
    x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    r = rank_thresholds(x)
    assert r["r90"] == r["r95"] == r["r99"] == 1


def test_restricted_rotation_train_only_geometry():
    source = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, 1.0, 1.0]])
    target = source.copy()
    basis, q = restricted_rotation_fit(source, target, rank=2)
    pred = restricted_rotation_predict(source, basis, q)
    assert np.allclose(pred, source, atol=1e-8)


def test_context_requires_all_controls():
    rows = []
    for fam in ("a", "b"):
        for j in range(2):
            sid = f"{fam}-{j}"
            for condition, error in (("MATCHED_REC", 0.1), ("WRONG_TOKEN_REC", 0.2), ("SAME_FAMILY_WRONG_STATE_REC", 0.2), ("CROSS_FAMILY_WRONG_STATE_REC", 0.2), ("SHUFFLED_REC", 0.2)):
                rows.append({"state_id": sid, "family": fam, "condition": condition, "relative_l2_to_target_donor": error})
    cfg = {"context_gate": {"matched_better_fraction_min": 0.8, "families_required": 2}}
    result = context_summary(pd.DataFrame(rows), cfg)
    assert result["pass"]
    assert result["matched_better_than_all_four_fraction"] == 1.0


def test_localization_bidirectional_gate():
    frame = pd.DataFrame([{"family": "a", "benefit_positive": True, "removed_benefit_fraction": .8, "restored_benefit_fraction": .2, "remove_direction_cosine": .9, "restore_direction_cosine": .9, "writeback_exact": True}])
    cfg = {"localization_gate": {"benefit_removed_min": .5, "benefit_restored_min": .5, "donor_direction_cosine_min": .8, "family_success_fraction_min": .5, "families_required": 1}}
    result = localization_summary(frame, cfg)
    assert result["success_rows"] == 0
    assert not result["pass"]
