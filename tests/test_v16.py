"""V16 provenance, split, bank and report integrity guards."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from jclosure.experiments.action_bank_v16 import _verify_splits
from jclosure.protocol_v16 import PARENT, digest, verify
from jclosure.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]


def test_v16_protocol_and_splits() -> None:
    base=verify(ROOT);split=_verify_splits(ROOT)
    assert base["parent_commit"]==PARENT
    assert split["freeze_digest"]==digest(split)
    assert len(split["train"])==100 and len(split["validation"])==50
    assert not ({x["base_trial_id"] for x in split["train"]}&{x["base_trial_id"] for x in split["validation"]})
    assert split["independent_final"].startswith("NOT_CREATED")


def test_v16_bank_and_records() -> None:
    summary=json.loads((ROOT/"results/v16/processed/v16_analysis.json").read_text())
    bank=summary["bank"]
    assert bank["states_by_role"]=={"train":100,"validation":50}
    assert 0.0<=bank["reliable_fraction"]<=1.0
    for name,expected in (("decomposition","decomposition_sha256"),("models","models_sha256"),("reachability","reachability_sha256")):
        assert sha256_file(ROOT/summary["records"][name])==summary["records"][expected]
    frame=pd.read_parquet(ROOT/bank["records"])
    assert len(frame)==sum(bank["actions_by_role"].values())
    assert (frame.reliability_status=="RELIABLE").sum()==sum(bank["reliable_by_role"].values())


def test_v16_historical_bytes_unchanged_and_manifest() -> None:
    tracked=set(subprocess.check_output(["git","ls-tree","-r","--name-only",PARENT],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(["git","diff","--name-only",PARENT,"--","."],cwd=ROOT,text=True).splitlines())
    assert (tracked&changed)-{"reports/FINAL_REPORT.md"}==set()
    value=json.loads((ROOT/"results/v16/processed/v16_integrity.json").read_text())
    assert value["historical_modified_except_cumulative_final"]==[]
    for row in value["files"]:
        assert sha256_file(ROOT/row["path"])==row["sha256"]


def test_v16_reports_and_gates() -> None:
    decision=json.loads((ROOT/"artifacts/nonlinear_finite_causal_action_v16_finalist_decision.freeze.json").read_text())
    assert decision["freeze_digest"]==digest(decision)
    assert decision["independent_final"]=="NOT_CREATED_OR_OPENED"
    assert decision["eligible_finalist"] is None
    assert "<!-- V16_START -->" in (ROOT/"reports/FINAL_REPORT.md").read_text()
    assert (ROOT/"reports/V16_COMPLETE_REPORT.md").exists()
    assert len(list((ROOT/"reports").glob("*_V16.md")))>=11
