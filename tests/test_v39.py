"""CPU-only V39 provenance, algebra, gate order, and sealed-outcome invariants."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v39 import verify, verify_stage
from jclosure.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_pools_and_unopened_roles():
    cfg = verify(ROOT)["config"]
    assert verify_stage(ROOT, "sample_pools")["models_not_queried"] is True
    programs, prompts = [], []
    for role, expected in (("calibration", 20), ("development", 80),
                           ("validation", 40), ("independent_final", 40)):
        pool = json.loads((ROOT / f"data/v39/{role}_pool_v39.json").read_text())
        assert len(pool["items"]) == expected
        assert pool["model_response_observed_before_seal"] is False
        for item in pool["items"]:
            assert item["horizon"] == cfg["formal_generator_horizon"][item["family"]]
            programs.append(item["program_hash"])
            prompts.append(item["prompt_sha256"])
    assert len(set(programs)) == len(set(prompts)) == 180
    adjudication = json.loads((ROOT / "results/v39/processed/v39_adjudication.json").read_text())
    assert adjudication["validation_opened"] is False
    assert adjudication["independent_final_opened"] is False
    assert adjudication["primary_finalist"] is None


def test_actual_prediction_frozen_before_full_and_corrected_metric_identity():
    for key in ("F", "Q"):
        seals = [verify_stage(ROOT, f"trajectory_{stage}_{key}_development")
                 for stage in ("singles", "pairs", "predict", "full")]
        assert [x["created_utc"] for x in seals] == sorted(x["created_utc"] for x in seals)
        assert seals[2]["R111_observed_before_prediction_freeze"] is False
        rows = pd.read_parquet(ROOT / f"results/v39/processed/trajectory_analysis_{key}_development_v39.parquet")
        p = rows.threeway_projection.to_numpy()
        fraction = rows.threeway_fraction.to_numpy()
        cosine = rows.threeway_cosine.to_numpy()
        assert np.allclose(p, fraction*cosine, atol=1e-8, rtol=1e-8)
        assert np.all(np.abs(p) <= fraction+1e-8)
        assert not rows.effect_below_floor.any()


def test_native_audits_and_gate_limited_reporting():
    for key in ("F", "Q"):
        assert verify_stage(ROOT, f"calibration_{key}")["all_bitwise"] is True
        assert verify_stage(ROOT, f"trace_interface_audit_{key}")["all_bitwise"] is True
    assert verify_stage(ROOT, "primitive_interface_audit_F")["all_bitwise"] is True
    finalist = verify_stage(ROOT, "primitive_finalist_F")
    assert finalist["primary_finalist"] is None
    assert finalist["development_removal_outcomes_seen"] is False
    report = verify_stage(ROOT, "report")
    assert report["report_count"] == 26
    assert report["validation_and_final_unopened"] is True
    assert (ROOT / "reports/V39_ALL_REPORTS.md").exists()
    index = json.loads((ROOT / "results/v39/processed/v39_integrity_index.json").read_text())
    assert index["indexed_files"]["reports/V39_COMPLETE_REPORT.md"] == sha256_file(
        ROOT / "reports/V39_COMPLETE_REPORT.md")


def test_statewise_inclusion_exclusion_with_nonzero_baseline():
    rng = np.random.default_rng(39)
    x = rng.normal(size=(8, 7, 11))
    x0, x1, x2, x3, x12, x13, x23, x123 = x
    second = x12+x13+x23-x1-x2-x3+x0
    interaction = x123-x12-x13-x23+x1+x2+x3-x0
    assert np.allclose(second+interaction, x123)
