"""V31 prospective split, composition and provenance invariants."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v31 import verify, verify_stage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v31/processed"
ROLES = ("calibration", "development", "validation", "independent_final")


def test_base_and_pre_response_freezes():
    verify(ROOT)
    verify_stage(ROOT, "design")
    verify_stage(ROOT, "execution_plan")


def test_new_state_panels_exclude_frozen_history():
    design = json.loads((OUT / "design_v31.json").read_text())
    groups = [{x["base_trial_id"] for x in design[r]} for r in ROLES]
    assert [len(x) for x in groups] == [25, 120, 60, 50]
    assert len(set.union(*groups)) == sum(map(len, groups))
    old = set()
    for version in (28, 29, 30):
        prior = json.loads((ROOT / f"results/v{version}/processed/design_v{version}.json").read_text())
        old.update(x["base_trial_id"] for r in ROLES for x in prior[r])
    assert not set.union(*groups) & old


def test_token_and_composition_roles_are_disjoint():
    design = json.loads((OUT / "design_v31.json").read_text())
    pair_roles = {r: {x["candidate_token_id"] for x in design["token_pair_library"] if x["role"] == r} for r in ("TOKEN_TRAIN", "TOKEN_VALIDATION", "TOKEN_FINAL")}
    assert {k: len(v) for k, v in pair_roles.items()} == {"TOKEN_TRAIN": 160, "TOKEN_VALIDATION": 16, "TOKEN_FINAL": 16}
    assert len(set.union(*pair_roles.values())) == 192
    comps = design["surface_composition_splits"]
    assert {k: len(v) for k, v in comps.items()} == {"COMPOSITION_TRAIN": 14, "COMPOSITION_VALIDATION": 16, "COMPOSITION_FINAL": 16}
    for role, items in comps.items():
        expected = "TOKEN_TRAIN" if role == "COMPOSITION_TRAIN" else "TOKEN_VALIDATION" if role == "COMPOSITION_VALIDATION" else "TOKEN_FINAL"
        for x in items:
            assert x["surface_A"] + x["surface_B"] == x["surface_AB"]
            assert len({x["A"], x["B"], x["AB"]}) == 3
            assert x["A"] in pair_roles["TOKEN_TRAIN"]
            assert x["B"] in pair_roles["TOKEN_TRAIN"]
            assert x["AB"] in pair_roles[expected]


def test_final_remains_sealed_in_plan():
    plan = json.loads((OUT / "execution_plan_v31.json").read_text())
    assert plan["all_current_token_writes_unobserved_before_plan"] is True
    assert plan["all_future_responses_unobserved_before_plan"] is True
    assert plan["independent_final_opened"] is False
    assert len(plan["fit_state_ids"]) == 100
    assert len(plan["heldout_composition_eval"]["development"]) == 20
    assert len(plan["heldout_composition_eval"]["validation"]) == 60


def test_train_only_fits_and_exact_mechanism_panels():
    fit = json.loads((OUT / "composition_fit_v31.json").read_text())
    dictionary = json.loads((OUT / "primitive_dictionary_fit_v31.json").read_text())
    response = json.loads((OUT / "response_factor_fit_v31.json").read_text())
    assert fit["training_states"] == 100
    assert fit["heldout_token_used_in_fit"] is False
    assert fit["future_response_used_in_fit"] is False
    assert dictionary["heldout_write_used_in_fit"] is False
    assert response["training_states"] == 100
    assert response["heldout_token_or_state_used_in_fit"] is False
    for role in ("development", "validation"):
        summary = json.loads((OUT / f"mechanism_{role}_v31.json").read_text())
        assert summary["states"] == 10
        assert summary["token_pairs"] == 20
        assert summary["independent_final_opened"] is False


def test_final_gate_and_integrity_index_when_present():
    decision_path = OUT / "final_opening_v31.json"
    if not decision_path.exists():
        return
    decision = json.loads(decision_path.read_text())
    assert decision["independent_final_opened"] is False
    assert decision["TOKEN_FINAL_write_or_future_response_observed"] is False
    index_path = OUT / "v31_integrity_index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    assert index["independent_final_opened"] is False
    assert index["entry_count"] == len(index["entries"])
