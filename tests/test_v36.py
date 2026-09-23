"""V36 research-integrity and native algebra regression checks."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from jclosure.experiments.operator_v36 import parts


ROOT = Path(__file__).resolve().parents[1]


def test_v36_panel_disjoint_and_sealed():
    panel = json.loads((ROOT / "results/v36/processed/panel_v36.json").read_text())
    roles = ("calibration", "development", "validation", "independent_final")
    assert [len(panel[r]) for r in roles] == [20, 80, 40, 40]
    ids = [x["base_trial_id"] for role in roles for x in panel[role]]
    assert len(ids) == len(set(ids)) == 180
    used = set()
    for version in (34, 35):
        previous = json.loads((ROOT / f"results/v{version}/processed/panel_v{version}.json").read_text())
        used.update(x["base_trial_id"] for role in roles for x in previous[role])
    assert not set(ids) & used
    opening = json.loads((ROOT / "results/v36/processed/final_opening_v36.json").read_text())
    assert opening["opened"] is False
    assert opening["independent_final_responses_seen"] is False


def test_qwen_old_state_dependent_true_update():
    f = {"q": torch.tensor([[[1., 2.]]]), "k": torch.tensor([[[1., 0.]]]),
         "v": torch.tensor([[[2., 3.]]]), "beta": torch.tensor([[.5]]),
         "decay": torch.tensor([[.8]]), "postconv": torch.zeros(1, 1, 2, dtype=torch.float32)}
    sa = torch.zeros(1, 1, 2, 2)
    sb = torch.ones_like(sa)
    a, b = parts("Q", f, sa), parts("Q", f, sb)
    assert not torch.equal(a["update"], b["update"])
    assert torch.allclose(a["new_state"], a["decayed"] + a["update"])
    assert torch.allclose(b["new_state"], b["decayed"] + b["update"])


def test_falcon_current_update_independent_of_old_state():
    f = {"dA": torch.ones(1, 1, 1, 2) * .7,
         "dBx": torch.ones(1, 1, 1, 2) * .2,
         "C": torch.tensor([[[.3, .4]]]),
         "hidden": torch.tensor([[[.5]]]),
         "D": torch.tensor([[.1]])}
    sa = torch.zeros(1, 1, 1, 2)
    sb = torch.ones_like(sa)
    a, b = parts("F", f, sa), parts("F", f, sb)
    assert torch.equal(a["update"], b["update"])
    assert torch.allclose(a["new_state"], a["decayed"] + a["update"])
    assert torch.allclose(b["new_state"], b["decayed"] + b["update"])
    assert torch.allclose(b["raw"] - a["raw"], torch.tensor([[[.49]]]))


def test_v35_triple_amendment_is_separate_from_v36_final():
    audit = json.loads((ROOT / "results/v36/processed/v35_audit_v36.json").read_text())
    assert audit["audit_outcome"] == "V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED"
    assert all(audit["triple_strong_gate"][role][model]["pass"]
               for role in ("development", "validation") for model in ("Q", "F"))
    assert audit["counterfactual_final_not_observed"] is True
