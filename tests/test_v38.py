"""CPU-only invariants for prospective V38 algebra and stage order."""
from pathlib import Path

import numpy as np

from jclosure.protocol_v38 import verify, verify_stage


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_pools_and_final_closed():
    base = verify(ROOT)
    assert base["prior_v37_formal_results_excluded_from_v38"] is True
    assert verify_stage(ROOT, "sample_pools")["models_not_queried"] is True
    assert verify_stage(ROOT, "final_closed")["opened"] is False
    assert verify_stage(ROOT, "composition_validation")["final_opening_eligible"] is False


def test_stage_order_and_prediction_preceded_full():
    for key in ("Q", "F"):
        for role in ("development", "validation"):
            order = [verify_stage(ROOT, f"trajectory_{stage}_{key}_{role}")
                     for stage in ("singles", "pairs", "predict", "full")]
            assert [x["created_utc"] for x in order] == sorted(x["created_utc"] for x in order)
            assert order[2]["R111_observed_before_prediction_freeze"] is False


def test_statewise_inclusion_exclusion_identity():
    rng = np.random.default_rng(38)
    y = rng.normal(size=(7, 6, 13))
    e100, e010, e001, e110, e101, e011, e111 = y
    additive = e100 + e010 + e001
    pair = (e110-e100-e010) + (e101-e100-e001) + (e011-e010-e001)
    second = e110+e101+e011-e100-e010-e001
    three = e111-e110-e101-e011+e100+e010+e001
    assert np.allclose(additive+pair, second)
    assert np.allclose(second+three, e111)
