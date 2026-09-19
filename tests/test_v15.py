"""Synthetic checks for V15 actuator measurements and provenance."""

from __future__ import annotations

import math
import json
from pathlib import Path

import torch

from jclosure.experiments.actuation_v15 import local_bf16_ulp, transfer_metrics
from jclosure.protocol_v15 import digest
from jclosure.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_bf16_ulp_spacing() -> None:
    values = torch.tensor([1.0, 2.0, 0.5], dtype=torch.bfloat16)
    actual = local_bf16_ulp(values)
    expected = torch.tensor([2**-7, 2**-6, 2**-8])
    assert torch.equal(actual, expected)


def test_deadzone_and_surviving_writeback() -> None:
    old = torch.ones(32, dtype=torch.bfloat16)
    direction = torch.ones(32, dtype=torch.float32)
    tiny = (old.float() + 0.001 * direction).to(torch.bfloat16)
    strong = (old.float() + 0.02 * direction).to(torch.bfloat16)
    a = transfer_metrics(old, direction, tiny, 0.001)
    b = transfer_metrics(old, direction, strong, 0.02)
    assert a["zero_fraction"] == 1.0
    assert a["surviving_fraction"] == 0.0
    assert b["surviving_fraction"] == 1.0
    assert math.isclose(float(b["cosine"]), 1.0)


def test_protocol_digest_ignores_its_own_field() -> None:
    payload = {"a": 1, "freeze_digest": "anything"}
    assert digest(payload) == digest({"a": 1})


def test_historical_guard_with_declared_mutable_cumulative_report() -> None:
    old = json.loads((ROOT / "results/v14/processed/v14_integrity.json").read_text())
    changed = [
        name for name, expected in old["hashes"].items()
        if name != "reports/FINAL_REPORT.md" and sha256_file(ROOT / name) != expected
    ]
    assert changed == []
    current = json.loads((ROOT / "results/v15/processed/v15_integrity.json").read_text())
    assert current["historical_modified_except_cumulative_final"] == []
