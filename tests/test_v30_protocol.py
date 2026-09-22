"""V30 frozen split and provenance invariants."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v30 import verify, verify_stage


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v30/processed"


def test_frozen_v30_base_and_prewrite_stages():
    verify(ROOT)
    for stage in ("design", "execution_plan", "transport_source_amendment"):
        verify_stage(ROOT, stage)


def test_state_and_token_splits_are_disjoint():
    d = json.loads((OUT / "design_v30.json").read_text())
    ids = [{x["base_trial_id"] for x in d[role]} for role in ("calibration", "development", "validation", "independent_final")]
    assert [len(x) for x in ids] == [25, 120, 60, 50]
    assert len(set.union(*ids)) == sum(map(len, ids))
    historical = set()
    for version in (28, 29):
        old = json.loads((ROOT / f"results/v{version}/processed/design_v{version}.json").read_text())
        historical.update(x["base_trial_id"] for role in ("calibration", "development", "validation", "independent_final") for x in old[role])
    assert not set.union(*ids) & historical
    pairs = d["token_pair_library"]
    assert len(pairs) == 88
    assert {role: sum(p["role"] == role for p in pairs) for role in ("TOKEN_TRAIN", "TOKEN_VALIDATION", "TOKEN_FINAL")} == {"TOKEN_TRAIN": 72, "TOKEN_VALIDATION": 8, "TOKEN_FINAL": 8}
    assert len({p["pair_id"] for p in pairs}) == 88


def test_transport_amendment_preserves_targets_and_train_only_rule():
    d = json.loads((OUT / "design_v30.json").read_text())
    p = json.loads((OUT / "execution_plan_v30.json").read_text())
    a = json.loads((OUT / "transport_source_amendment_v30.json").read_text())
    by_id = {x["base_trial_id"]: x for role in ("development", "validation", "independent_final") for x in d[role]}
    assert set(a["transport_source_state"]) == set(p["transport_source_state"])
    assert len(set(a["source_by_family"].values())) == 5
    for key, source in a["transport_source_state"].items():
        target, pair = key.rsplit(":", 1)
        assert source != target
        assert source in p["fit_state_ids"]
        assert by_id[source]["family"] == by_id[target]["family"]
        assert pair in by_id[source]["eligible_pair_ids"]
    assert a["response_observed_before_amendment"] == 0
    assert p["independent_final_opened"] is False
