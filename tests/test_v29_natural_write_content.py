import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v29 import verify, verify_stage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v29/processed"


def load(name):
    return json.loads((OUT / name).read_text())


def test_v29_freezes_and_history():
    base = verify(ROOT)
    assert base["parent_commit"] and not base["historical_final_opened"]
    for name in (
        "design", "execution_plan", "forks_calibration", "interface_amendment",
        "forks_development", "development_analysis", "forks_validation",
        "validation_analysis", "final_opening", "secondary_development",
        "secondary_validation", "diagnostics", "realization_plan", "realization",
    ):
        assert verify_stage(ROOT, name)["freeze_digest"]
    if (ROOT / "artifacts/natural_write_content_v29_final.freeze.json").exists():
        assert verify_stage(ROOT, "final")["freeze_digest"]


def test_v29_disjoint_balanced_prewrite_design():
    d = load("design_v29.json")
    sets = []
    for role, n in (("calibration", 25), ("development", 75), ("validation", 50), ("independent_final", 50)):
        rows = d[role]
        assert len(rows) == n
        assert all(sum(x["family"] == family for x in rows) == n // 5 for family in d["families"])
        assert all(x["token_A"] != x["token_B"] and x["incoming_state_hashes"] for x in rows)
        sets.append({x["base_trial_id"] for x in rows})
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4))
    assert not d["future_response_observed_before_freeze"]
    assert d["prior_v28_ids_excluded"]


def test_v29_interface_amendment_and_full_native_ceiling():
    am = load("interface_amendment_v29.json")
    assert am["diagnostic_named_full_was_not_full_native_cache"]
    assert am["thresholds_unchanged"] and not am["formal_development_saved_before_amendment"]
    for role, n in (("development", 75), ("validation", 50)):
        data = load(f"forks_{role}_v29.json")
        assert data["states"] == n and data["all_writeback_pass"]
        assert len(data["native_recurrent_layers"]) == 24 and len(data["native_attention_layers"]) == 8
        assert load(f"{role}_analysis_v29.json")["full_native_cache_identity_ceiling"]
        transfer = pd.read_parquet(OUT / f"transfer_{role}_v29.parquet")
        full = transfer[transfer.condition == "REC+Conv+KV"]
        assert len(full) == 2 * n and (full.relative_l2_to_donor <= 1e-6).all()


def test_v29_reciprocal_content_not_KV_only():
    for role in ("development", "validation"):
        a = load(f"{role}_analysis_v29.json")
        assert a["branch_gate"] and a["strict_writeback_all"]
        assert a["profiles"]["REC+Conv"]["reciprocal_pass"]
        assert a["profiles"]["Conv"]["reciprocal_pass"]
        assert not a["profiles"]["KV"]["reciprocal_pass"]
        assert not a["profiles"]["REC"]["reciprocal_pass"]


def test_v29_secondary_controls_and_realization_records():
    for role in ("development", "validation"):
        s = load(f"secondary_{role}_v29.json")
        assert s["all_full_signatures_exact"] and s["signature_rows"] == 100
        assert s["factorial_rows"] == 10 and s["horizon_rows"] == 30
        c = pd.read_parquet(OUT / f"natural_vs_random_{role}_v29.parquet")
        natural = c[c.condition == "natural_REC+Conv"].relative_l2_to_donor.median()
        random = c[c.condition == "random_same_norm"].relative_l2_to_donor.median()
        assert natural < random
    r = load("write_effect_realization_v29.json")
    assert r["train_states"] == 90 and r["validation_states"] == 50
    assert r["train_span_rank"] >= max(r["tested_k"])
    assert set(r["not_estimable_k"]) == {128, 256}
    f = pd.read_parquet(OUT / "write_effect_realization_v29.parquet")
    assert np.isfinite(f.relative_l2_to_donor).all()


def test_v29_final_adjudication_and_authorization():
    opening = load("final_opening_v29.json")
    assert opening["final_opened"] and opening["fixed_finalist"] == "REC+Conv"
    adj = load("v29_adjudication.json")
    assert adj["independent_final"]["states"] == 50
    assert adj["independent_final"]["reciprocal_pass"]
    assert adj["formal_outcomes"]["V29_B_SAME_BACKGROUND_WRITE_TRANSFER_CONFIRMED"]
    assert adj["H2_REMAINS"] and not adj["H3_AUTHORIZED"]
