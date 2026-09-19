"""V14 freeze, split, and historical-integrity regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v14 import digest, verify_base
from jclosure.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_v14_base_and_stage_freezes_are_self_consistent() -> None:
    base = verify_base(ROOT)
    assert base["baseline_commit"] == "37df399e4bba8afaca6d99721fe8ceee653e78ee"
    for path in ROOT.glob("artifacts/finite_causal_control_v14*.freeze.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["freeze_digest"] == digest(value), path
        if "parent_freeze_digest" in value:
            assert value["parent_freeze_digest"] == base["freeze_digest"]
        for name, expected in value["source_hashes"].items():
            assert sha256_file(ROOT / name) == expected, name


def test_v13_tracked_history_guard() -> None:
    from scripts.create_v13_immutable_for_v14 import verify

    result = verify(ROOT)
    assert result["file_count"] >= 1800
    assert result["excluded_mutable_paths"] == ["reports/FINAL_REPORT.md"]


def test_v14_diagnostic_and_development_are_disjoint() -> None:
    split = json.loads(
        (ROOT / "artifacts/finite_causal_control_v14_splits.freeze.json").read_text()
    )
    diagnostic = set(split["diagnostic_train"]["base_trial_ids"])
    development = set(split["development_validation"]["base_trial_ids"])
    assert len(diagnostic) == 5
    assert len(development) == 10
    assert diagnostic.isdisjoint(development)
    assert split["independent_final"]["status"] == "NOT_YET_CREATED_OR_SELECTED"


def test_v14_numeric_gate_is_not_retroactively_applied_to_v13() -> None:
    summary = json.loads(
        (ROOT / "results/v14/processed/numerical_snr_summary_v14.json").read_text()
    )
    assert summary["min_causal_effect_norm"] > 0
    assert summary["first_all_target_jvp_reliable_epsilon"] is None
    assert summary["v13_frozen_outcomes_changed"] is False
    labels = pd.read_parquet(
        ROOT / "results/v14/processed/v13_alpha_snr_reanalysis_v14.parquet"
    )
    assert set(labels.direction_snr_label) == {
        "SNR_QUALIFIED",
        "BELOW_DIRECTION_SNR_THRESHOLD",
    }


def test_transport_is_explicitly_aligned() -> None:
    summary = json.loads(
        (ROOT / "results/v14/processed/transport_analysis_v14.json").read_text()
    )
    assert summary["rank"] == 16
    pairs = pd.read_parquet(ROOT / "results/v14/processed/transport_pairs_v14.parquet")
    assert len(pairs) == 1900
    paired = pairs[pairs.relation == "successive_position"]
    procrustes = paired[paired.method == "procrustes"].raw_direction_cosine.mean()
    unaligned = paired[paired.method == "unaligned"].raw_direction_cosine.mean()
    assert procrustes > unaligned


def test_v14_integrity_manifest() -> None:
    from scripts.build_v14_integrity import verify

    result = verify(ROOT)
    assert result["file_count"] >= 50
    assert result["excluded_mutable_paths"] == ["reports/V14_COMPLETE_REPORT.md"]
