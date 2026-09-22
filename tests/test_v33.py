"""CPU-only V33 protocol, prospective, causal-audit and report checks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v33 import verify, verify_stage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/v33/processed"


def test_base_and_history_immutable():
    base=verify(ROOT)
    assert base["parent_commit"]=="09a00a67e5a286a05e47730c43586bce94b8095a"
    assert base["V1_V32_records_mutated"] is False
    assert base["historical_final_opened"] is False


def test_architecture_exact_fields():
    verify_stage(ROOT,"architecture")
    audit=json.loads((OUT/"architecture_audit_v33.json").read_text())
    assert audit["functional_structural_comparable"]
    assert len(audit["schema"])==96
    assert all(x["untouched_exact"] for x in audit["write_proofs"].values())
    assert set(audit["mapping"])=={"REC2","CONV2","KV"}


def test_design_prospective_and_disjoint():
    verify_stage(ROOT,"design")
    design=json.loads((OUT/"design_v33.json").read_text())
    roles={r:design[r] for r in ("calibration","development","validation","independent_final")}
    assert {r:len(v) for r,v in roles.items()}=={"calibration":25,"development":100,"validation":50,"independent_final":50}
    ids=[x["base_trial_id"] for rows in roles.values() for x in rows]
    assert len(ids)==len(set(ids))==225
    assert not design["causal_write_observed_before_freeze"]
    assert not design["future_causal_response_observed_before_freeze"]
    assert len(design["token_library"])==40
    assert all(len(x["future_probe_tokens"])==6 for rows in roles.values() for x in rows)


def test_factorial_kv_and_final_opening():
    opening=verify_stage(ROOT,"final_opening")
    assert opening["opened"]
    for role,n in (("development",100),("validation",50),("independent_final",50)):
        verify_stage(ROOT,f"factorial_{role}")
        rows=pd.read_parquet(OUT/f"factorial_{role}_v33.parquet")
        assert len(rows)==n
        assert rows.recipient_native_KV.all() and rows.writeback_exact.all()
        assert rows.family.nunique()==5


def test_controls_and_phase_b_limit():
    verify_stage(ROOT,"context_plan")
    verify_stage(ROOT,"controls")
    verify_stage(ROOT,"phase_b_instrumentation")
    phase=json.loads((OUT/"phase_b_instrumentation_v33.json").read_text())
    assert phase["model1"]["exact_delta_count"]==24
    assert phase["model2"]["exact_update_count"]==24
    assert not phase["bidirectional_mediation_executed"]


def test_adjudication_no_micro_overclaim():
    verify_stage(ROOT,"adjudication")
    a=json.loads((OUT/"v33_adjudication.json").read_text())
    assert all(a["formal_outcomes"][f"V33-{x}_{name}"] for x,name in (("A","CROSS_MODEL_REC_CORRECTION_REPLICATED"),("B","CONV_DOMINANT_HANDOFF_REPLICATED"),("C","CONDITIONAL_NOT_ADDITIVE_OR_GAIN_ONLY"),("D","MATCHED_CONTEXT_CORRECTION_REPLICATED")))
    assert not a["formal_outcomes"]["V33-H_SHARED_MICRO_MEDIATOR_IDENTIFIED"]
    assert not a["formal_outcomes"]["V33-I_SHARED_EFFECT_WITH_DIFFERENT_MICRO_MEDIATORS"]
    assert a["final_confirmation_pass"] and not a["model1_historical_final_reopened"]
