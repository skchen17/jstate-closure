"""CPU integrity checks for independent V37 pools and numerical metrics."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd

from jclosure.experiments.local_observation_v37 import _ordering
from jclosure.protocol_v37 import verify, verify_stage


ROOT = Path(__file__).resolve().parents[1]


def test_v37_pools_sealed_before_model_response():
    base = verify(ROOT)
    assert base["prior_v36_validation_excluded_from_v37"]
    counts = {"calibration": 20, "development": 80,
              "validation": 40, "independent_final": 40}
    programs = set()
    prompts = set()
    for role, count in counts.items():
        seal = verify_stage(ROOT, f"pool_{role}")
        assert not seal["model_response_observed_before_seal"]
        rows = json.loads((ROOT / f"data/v37/{role}_pool_v37.json").read_text())["items"]
        assert len(rows) == count
        assert {row["family"] for row in rows} == set(base["config"]["families"])
        assert all(sum(row["family"] == family for row in rows) == count // 5
                   for family in base["config"]["families"])
        for row in rows:
            assert row["program_hash"] not in programs
            assert row["prompt_sha256"] not in prompts
            programs.add(row["program_hash"])
            prompts.add(row["prompt_sha256"])
    assert len(programs) == 180 and len(prompts) == 180


def test_v37_calibration_independent_from_v36():
    for key in ("Q", "F"):
        seal = verify_stage(ROOT, f"calibration_{key}")
        assert seal["all_within_tolerance"] and seal["all_bitwise"]
        record = json.loads((ROOT / f"results/v37/processed/calibration_{key}_v37.json").read_text())
        assert record["states"] == 20 and record["equality_rows"] == 3240
        frame = pd.read_parquet(ROOT / f"results/v37/processed/calibration_equality_{key}_v37.parquet")
        assert frame.max_abs_error.max() == 0.0


def test_pairwise_ordering_excludes_ties():
    assert _ordering([1, 2, 3], [1, 2, 3]) == 1.0
    assert _ordering([1, 2, 3], [3, 2, 1]) == 0.0
    assert _ordering([1, 1, 1], [1, 2, 3]) is None


def test_v37_finalist_independent_final_and_reports():
    for stage in ("sample_pools", "local_analysis_development",
                  "local_analysis_validation", "final_opening", "final_analysis",
                  "adjudication", "report"):
        assert verify_stage(ROOT, stage)["freeze_digest"]
    opening = json.loads((ROOT / "results/v37/processed/final_opening_v37.json").read_text())
    adjudication = json.loads((ROOT / "results/v37/processed/v37_adjudication.json").read_text())
    assert opening["selected_class"] == "READ_OPERATOR_MATCHING"
    assert opening["independent_final_outcomes_seen_at_opening"] is False
    assert adjudication["strongest_supported_level"] == "LEVEL_2_LOCAL_COMPUTATION"
    assert adjudication["cross_layer_mechanism_confirmed"] is False
    assert adjudication["v36_formal_results_counted"] is False
    for key in ("Q", "F"):
        for stage in (f"final_prediction_{key}", f"final_observation_{key}"):
            assert verify_stage(ROOT, stage)["freeze_digest"]
        observation = json.loads((ROOT / f"results/v37/processed/final_observation_{key}_v37.json").read_text())
        assert observation["states"] == 40
        assert observation["local_read_law_pass"] is True
        assert observation["natural_downstream_task_effect_tested"] is False
    reports = sorted((ROOT / "reports").glob("V37_*.md"))
    assert len(reports) == 30
    assert (ROOT / "reports/V37_ALL_REPORTS.md").is_file()
    changed = set(subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "reports/V36_*"],
                                          cwd=ROOT, text=True).splitlines())
    assert not changed
