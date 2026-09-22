"""Freeze V29 secondary-panel membership and fixed-final rule before development responses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/plan_v29.py"
OUT = Path("results/v29/processed")


def prepare(root):
    verify_stage(root, "design")
    d = json.loads((root / OUT / "design_v29.json").read_text())
    cfg = verify(root)["config"]
    controls = {}
    factorial = {}
    for role in ("development", "validation"):
        controls[role] = []
        factorial[role] = []
        for fam in d["families"]:
            group = [x for x in d[role] if x["family"] == fam]
            controls[role].extend(x["base_trial_id"] for x in group[:2])
            factorial[role].extend([[group[0]["base_trial_id"], group[1]["base_trial_id"]], [group[2]["base_trial_id"], group[3]["base_trial_id"]]])
    payload = {
        "control_state_ids": controls,
        "multi_probe_state_ids": controls,
        "horizon_state_ids": controls,
        "factorial_pairs": factorial,
        "control_conditions": ["natural_REC+Conv", "random_same_norm_REC+Conv", "shuffled_natural_REC+Conv", "sign_flipped_REC+Conv", "wrong_state_REC+Conv"],
        "future_signature_probes": "per-state top4 frozen prefix-distribution candidates, not selected from write response",
        "horizons": [1, 2, 4],
        "realization_k": cfg["realization_k"],
        "primary_partial_channels": cfg["channels"],
        "fixed_finalist": "REC+Conv",
        "final_rule": cfg["final_rule"],
        "final_panel_sealed": True,
        "development_responses_observed_before_plan": False,
        "validation_responses_observed_before_plan": False,
        "historical_final_opened": False,
    }
    path = root / OUT / "execution_plan_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "execution_plan", [SOURCE, str(path.relative_to(root)), "artifacts/natural_write_content_v29_design.freeze.json"], {"plan_hash": hashlib.sha256(path.read_bytes()).hexdigest(), **payload})
    return {"freeze_digest": fr["freeze_digest"], "controls_per_role": {r: len(x) for r, x in controls.items()}}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
