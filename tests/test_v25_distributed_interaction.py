import json
from pathlib import Path

from jclosure.protocol_v25 import verify, verify_stage


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v25/processed"


def load(name):
    return json.loads((OUT / name).read_text())


def test_v25_protocol_and_append_only_stages():
    assert verify(ROOT)["freeze_digest"]
    for name in (
        "design",
        "component_basis",
        "rank_convergence",
        "rank_convergence_amendment",
        "interaction",
        "interaction_estimand_amendment",
        "analysis",
        "zero_rank_amendment",
        "adjudication",
        "final",
    ):
        assert verify_stage(ROOT, name)["freeze_digest"]


def test_v25_independent_final_is_balanced_and_sealed():
    d = load("design_v25.json")
    final = d["independent_final_states"]
    assert len(final) == 50
    assert {family: sum(x["family"] == family for x in final) for family in {x["family"] for x in final}} == {
        "boolean_logic": 10,
        "modular_arithmetic": 10,
        "short_graph_traversal": 10,
        "simple_state_transition": 10,
        "variable_binding": 10,
    }
    final_ids = {x["base_trial_id"] for x in final}
    observed_ids = {
        x["base_trial_id"]
        for key in ("rank_audit_states", "component_basis_states", "interaction_validation_states", "background_source_states")
        for x in d[key]
    }
    assert len(final_ids) == 50 and not final_ids & observed_ids
    assert not d["historical_final_opened"] and not d["v25_independent_final_opened"]


def test_v25_rank_audit_rejects_unqualified_rank_one_claim():
    r = load("v24_rank_audit_v25.json")
    assert set(r["v24_raw_broad_r95"]) == {1.0}
    assert min(r["corrected_centered_broad_r95"]) >= 13
    assert min(r["centered_normalized_broad_r95"]) >= 24
    assert not r["LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED"]
    assert r["V24_RANK1_STATUS"] == "MEASUREMENT_DEFINITION_SPECIFIC_UNCENTERED_EFFECT"


def test_v25_corrected_four_branch_estimand_and_cancellation():
    i = load("interaction_estimand_amendment_v25.json")
    assert i["estimand"] == "Y00=clean; Y10=clean+B; Y01=clean+C; Y11=clean+(B+C)"
    assert i["rows"] == 160 and not i["original_interaction_records_overwritten"]
    assert not i["STRONG_DISTRIBUTED_NONLINEAR_INTERACTION"]
    assert i["CAUSAL_CANCELLATION_SUPPORTED"]
    assert not i["writeback"]["numerical_gate_pass"]


def test_v25_zero_rank_correction_and_no_internal_convergence():
    z = load("zero_rank_amendment_v25.json")
    assert z["zero_response_layers"] == list(range(24))
    assert z["first_nonzero_layer"] == 24
    assert z["reference_r95"] == 56 and z["late_r95"] == 99
    assert z["r95_reduction_fraction"] < 0
    assert not z["MANY_TO_ONE_CAUSAL_CONVERGENCE"]
    assert not z["INTERNAL_CAUSAL_CONVERGENCE_SUPPORTED"]


def test_v25_formal_outcomes_authorization_and_sealing():
    a = load("v25_adjudication.json")
    assert a["formal_outcomes"] == [
        "V25-B_CAUSAL_CANCELLATION_SUPPORTED",
        "V25-E_READOUT_COMPRESSION_ONLY",
    ]
    assert not a["V25_A"] and a["V25_B"] and not a["V25_C"] and not a["V25_D"] and a["V25_E"]
    assert not a["V25_F"] and not a["V25_G"] and not a["V25_H"]
    assert a["H2_REMAINS"] and not a["H3_AUTHORIZED"] and not a["DYNAMIC_STATE_SEARCH_AUTHORIZED"]
    assert not a["historical_final_opened"] and not a["v25_independent_final_opened"]
