"""CPU-only invariants for the prospective V35 depth study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.depth_analysis_v35 import _late, _pair, _progressive, stage_gate
from jclosure.experiments.development_plan_v35 import FINAL_PRIORITY, candidates
from jclosure.experiments.final_opening_v35 import conditions
from jclosure.protocol_v35 import verify, verify_stage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v35/processed"


def test_panel_and_design_are_disjoint_and_paired():
    cfg = verify(ROOT)["config"]
    verify_stage(ROOT, "panel")
    verify_stage(ROOT, "design")
    panel = json.loads((OUT / "panel_v35.json").read_text())
    roles = tuple(cfg["roles_per_family"])
    ids = [x["base_trial_id"] for role in roles for x in panel[role]]
    assert len(ids) == len(set(ids)) == 180
    assert {role: len(panel[role]) for role in roles} == {
        "calibration": 20, "development": 80, "validation": 40, "independent_final": 40}
    for key in ("Q", "F"):
        verify_stage(ROOT, f"design_{key}")
        verify_stage(ROOT, f"interface_{key}")
        design = json.loads((OUT / f"design_{key}_v35.json").read_text())
        assert [x["base_trial_id"] for role in roles for x in design[role]] == ids
        assert all(len(x["future_probe_tokens"]) == 6 for role in roles for x in design[role])
        groups = design["relative_depth_layers"]
        assert [len(groups[f"Q{i}"]) for i in range(1, 5)] == [6] * 4
        assert len(set(sum((groups[f"Q{i}"] for i in range(1, 5)), []))) == 24
        assert len(design["condition_layers"]) == 15


def test_stage_gate_requires_both_directions_and_families():
    cfg = verify(ROOT)["config"]
    frame = pd.DataFrame({"condition": ["Q4"] * 5, "status": ["VALID"] * 5,
                          "family": cfg["families"], "removed_fraction": [0.6] * 5,
                          "restored_fraction": [0.6] * 5,
                          "reverse_correction_cosine": [0.9] * 5})
    assert stage_gate(frame, "Q4", cfg["quartile_gate"])["pass"]
    frame.loc[0, "restored_fraction"] = 0.1
    assert stage_gate(frame, "Q4", cfg["quartile_gate"])["pass"]
    frame.loc[1, "restored_fraction"] = 0.1
    assert not stage_gate(frame, "Q4", cfg["quartile_gate"])["pass"]


def test_qualitative_gates_are_bidirectional():
    cfg = verify(ROOT)["config"]
    frame = pd.DataFrame([{"condition": condition, "family": family, "status": "VALID",
                           "removed_fraction": value, "restored_fraction": value,
                           "reverse_correction_cosine": 1.0}
                          for family in cfg["families"] for condition, value in
                          (("Q1", 0.1), ("Q2", 0.1), ("Q3", 0.45), ("Q4", 0.55),
                           ("Q1_Q2", 0.2), ("Q3_Q4", 0.9),
                           ("Q1_Q2_Q3", 0.6), ("FULL", 1.0))])
    assert _late(frame, cfg)[0]
    frame.loc[frame.condition == "Q3_Q4", "restored_fraction"] = 0.2
    assert not _late(frame, cfg)[0]
    frame.loc[frame.condition == "Q1_Q2", "restored_fraction"] = 0.11
    assert not _progressive(frame, cfg)[0]
    assert not _pair(frame, cfg, "Q3_Q4", "complementary")[0]


def test_finalist_priority_is_frozen_and_all_conditions_are_predeclared():
    cfg = verify(ROOT)["config"]
    assert FINAL_PRIORITY[-1] == "FULL_DEPTH_ONLY"
    for name in FINAL_PRIORITY:
        assert set(conditions(name, cfg)) <= set(cfg["subset_conditions"])
    shared = {"late": False, "progressive": False, "serial_pairs": [],
              "complementary_pairs": ["Q3_Q4"], "redundant_pairs": [],
              "strong_quartiles": [], "strong_pairs": ["Q3_Q4"]}
    assert candidates(shared) == ["COMPLEMENTARY_Q3_Q4", "LOCALIZED_Q3_Q4", "FULL_DEPTH_ONLY"]


def test_development_records_exact_full_and_depth_audits():
    verify_stage(ROOT, "high_level_development")
    gate = verify_stage(ROOT, "full_gate_development")
    assert gate["formal_depth_authorized"]
    for key in ("Q", "F"):
        verify_stage(ROOT, f"full_read_{key}_development")
        verify_stage(ROOT, f"depth_{key}_development")
        full = pd.read_parquet(OUT / f"full_read_{key}_development_v35.parquet")
        depth = pd.read_parquet(OUT / f"depth_subset_{key}_development_v35.parquet")
        audit = pd.read_parquet(OUT / f"depth_audit_{key}_development_v35.parquet")
        assert len(full) == 80 and len(depth) == 80 * 15
        assert depth.state_id.nunique() == 80
        assert full.exact_writeback.all() and depth.exact_writeback.all()
        assert audit.exact_writeback.all()
        assert (audit.requested_hash == audit.realized_hash).all()
        assert np.allclose(full["removed_fraction"].dropna().median(), 1.0, atol=1e-5)
        assert np.allclose(full["restored_fraction"].dropna().median(), 1.0, atol=1e-5)


def test_validation_and_final_are_sealed_before_adjudication():
    dev = json.loads((OUT / "depth_analysis_development_v35.json").read_text())
    val = json.loads((OUT / "depth_analysis_validation_v35.json").read_text())
    plan = json.loads((OUT / "development_plan_v35.json").read_text())
    opening = json.loads((OUT / "final_opening_v35.json").read_text())
    verify_stage(ROOT, "depth_analysis_validation")
    verify_stage(ROOT, "development_plan")
    verify_stage(ROOT, "final_opening")
    assert not plan["validation_responses_seen"]
    assert not opening["final_responses_seen"]
    assert opening["selected_mechanism"] in plan["shared_development_candidates"]
    assert opening["selected_mechanism"] in candidates(val["shared_qualitative_both_roles"])
    assert not dev["historical_V34_exploratory_quartiles_used_as_formal"]
    for key in ("Q", "F"):
        verify_stage(ROOT, f"depth_{key}_validation")
        depth = pd.read_parquet(OUT / f"depth_subset_{key}_validation_v35.parquet")
        audit = pd.read_parquet(OUT / f"depth_audit_{key}_validation_v35.parquet")
        assert len(depth) == 40 * 15
        assert audit.exact_writeback.all()
        assert (audit.requested_hash == audit.realized_hash).all()


def test_independent_final_and_reports_are_integral():
    verify_stage(ROOT, "final_analysis")
    verify_stage(ROOT, "adjudication")
    verify_stage(ROOT, "integrity")
    opening = json.loads((OUT / "final_opening_v35.json").read_text())
    final = json.loads((OUT / "final_analysis_v35.json").read_text())
    index = json.loads((OUT / "v35_integrity_index.json").read_text())
    assert opening["independent_final_states_per_model"] == 40
    assert final["selected_mechanism"] == opening["selected_mechanism"]
    assert index["report_count"] >= 23
    assert "reports/V35_ALL_REPORTS.md" in index["files"]
    assert "reports/V35_COMPLETE_REPORT.md" in index["files"]
    for key in ("Q", "F"):
        verify_stage(ROOT, f"full_read_{key}_independent_final")
        frame = pd.read_parquet(OUT / f"full_read_{key}_independent_final_v35.parquet")
        assert len(frame) == 40 and frame.exact_writeback.all()
