"""Freeze response-blind V29 write-effect basis training/evaluation split."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.forks_formal_v29 import OUT, design
from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/realization_plan_v29.py"


def run(root):
    verify_stage(root, "execution_plan")
    d = design(root)
    cfg = verify(root)["config"]
    train = list(d["calibration"])
    held = []
    for family in d["families"]:
        rows = [x for x in d["development"] if x["family"] == family]
        train += rows[:13]
        held += rows[13:]
    assert len(train) == 90 and len(held) == 10
    payload = {
        "train_ids": [x["base_trial_id"] for x in train],
        "development_holdout_ids": [x["base_trial_id"] for x in held],
        "validation_ids": [x["base_trial_id"] for x in d["validation"]],
        "train_size": 90,
        "development_holdout_size": 10,
        "validation_size": 50,
        "source": "natural same-incoming REC+Conv B-minus-A outgoing contrast over all 24 recurrent layers",
        "basis": "train-only centered PCA/SVD of native write contrasts",
        "approximation": "mean plus rank-k projection; BF16 native recipient cache writeback",
        "k_grid": cfg["realization_k"],
        "rank_above_training_span_not_estimable": [128, 256],
        "donor_target": "full natural B-versus-A next response under fixed shared z",
        "directions": ["A_from_B", "B_from_A"],
        "gates": {"cosine_min": cfg["realization_cosine_min"], "magnitude_min": cfg["realization_magnitude_min"], "magnitude_max": cfg["realization_magnitude_max"], "relative_l2_max": cfg["realization_relative_l2_max"], "families_required": cfg["families_required"]},
        "split_response_selected": False,
        "historical_final_opened": False,
    }
    path = root / OUT / "realization_plan_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "realization_plan", [SOURCE, str(path.relative_to(root)), "artifacts/natural_write_content_v29_execution_plan.freeze.json"], payload)
    return {"freeze_digest": fr["freeze_digest"], "train": 90, "development_holdout": 10, "validation": 50}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
