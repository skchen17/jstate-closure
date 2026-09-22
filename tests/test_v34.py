"""CPU-only integrity checks for the frozen V34 functional-mediation study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from jclosure.experiments.functional_hooks_v34 import FunctionalIntervention
from jclosure.experiments.mediate_v34 import cosine
from jclosure.experiments.stage_analysis_v34 import _stats
from jclosure.protocol_v34 import verify, verify_stage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v34/processed"


def test_cosine_respects_direction_and_zero():
    assert cosine(np.array([1., 0.]), np.array([1., 0.])) == 1.0
    assert cosine(np.array([1., 0.]), np.array([-1., 0.])) == -1.0
    assert cosine(np.array([0., 0.]), np.array([1., 0.])) is None


def test_stage_hook_removed_even_after_exception():
    class Mixer(torch.nn.Module):
        def recurrent_gated_delta_rule(self):
            return None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_type = "linear_attention"
            self.linear_attn = Mixer()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Model()
    module = model.model.layers[0].linear_attn
    original = module.recurrent_gated_delta_rule
    with pytest.raises(ValueError):
        with FunctionalIntervention(model, "Q"):
            assert len(module._forward_hooks) == 1
            raise ValueError("intentional")
    assert not module._forward_hooks
    assert module.recurrent_gated_delta_rule.__func__ is original.__func__
    assert module.recurrent_gated_delta_rule.__self__ is module


def test_stage_gate_uses_correction_benefit_and_families():
    cfg = verify(ROOT)["config"]["stage_gate"]
    rows = []
    for family in verify(ROOT)["config"]["families"]:
        rows += [{"family": family, "status": "VALID", "removed_fraction": 0.6,
                  "restored_fraction": 0.7, "reverse_correction_cosine": 0.9,
                  "exact_writeback": True, "recipient_native_KV": True,
                  "six_frozen_probes": True}]
    frame = pd.DataFrame(rows)
    assert _stats(frame, cfg)["pass"]
    frame.loc[0, "restored_fraction"] = 0.0
    assert _stats(frame, cfg)["pass"]  # four of five families still pass
    frame.loc[1, "restored_fraction"] = 0.0
    assert not _stats(frame, cfg)["pass"]


def test_panel_disjoint_and_prospective():
    verify_stage(ROOT, "panel")
    verify_stage(ROOT, "design")
    panel = json.loads((OUT / "panel_v34.json").read_text())
    roles = ("calibration", "development", "validation", "independent_final")
    ids = [x["base_trial_id"] for role in roles for x in panel[role]]
    assert len(ids) == len(set(ids)) == 180
    assert {r: len(panel[r]) for r in roles} == {
        "calibration": 20, "development": 80, "validation": 40, "independent_final": 40}
    for model in ("Q", "F"):
        design = json.loads((OUT / f"design_{model}_v34.json").read_text())
        assert [x["base_trial_id"] for role in roles for x in design[role]] == ids
        assert all(len(x["future_probe_tokens"]) == 6
                   for role in roles for x in design[role])


def test_interface_and_formal_audit():
    for model in ("Q", "F"):
        verify_stage(ROOT, f"interface_{model}")
        for role, n in (("development", 80), ("validation", 40)):
            verify_stage(ROOT, f"mediation_{model}_{role}")
            rows = pd.read_parquet(OUT / f"mediation_{model}_{role}_v34.parquet")
            audit = pd.read_parquet(OUT / f"mediation_audit_{model}_{role}_v34.parquet")
            assert len(rows) == n * 4
            assert len(audit) == n * 6 * 4 * 2
            assert rows.exact_writeback.all() and audit.writeback_exact.all()
            assert (audit.requested_stage_hash == audit.realized_stage_hash).all()


def test_stage5_alias_not_promoted_and_final_gate():
    for role in ("development", "validation"):
        verify_stage(ROOT, f"stage_analysis_{role}")
        analysis = json.loads((OUT / f"stage_analysis_{role}_v34.json").read_text())
        assert analysis["stage5_alias_not_independently_identifiable"]
        for model in ("Q", "F"):
            assert analysis["models"][model]["RESIDUAL_INTEGRATION"]["pass"] is False
    final = json.loads((OUT / "v34_adjudication.json").read_text())
    if not final["shared_stage_or_pipeline_passes_both_roles"]:
        assert not final["independent_final_opened"]
        assert not final["application_experiments_authorized"]
