"""Freeze the V35 secondary-control and descriptive-flow subset before validation responses."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/control_plan_v35.py"


def run(root: Path):
    verify_stage(root, "design")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_Q_v35.json").read_text())
    selected = {}
    for family in cfg["families"]:
        ids = sorted(x["base_trial_id"] for x in design["validation"] if x["family"] == family)
        if len(ids) != 8:
            raise RuntimeError(f"V35 control family supply drift: {family}")
        selected[family] = ids[:2]
    record = {"role": "validation", "two_states_per_family": selected,
              "context_conv_rec_states": [sid for ids in selected.values() for sid in ids],
              "KV_states": [ids[0] for ids in selected.values()],
              "flow_states": [ids[1] for ids in selected.values()],
              "flow_probe_index": 0, "control_probe_count": 6,
              "quartiles": ["Q1", "Q2", "Q3", "Q4"],
              "KV_depth_condition_rule": "full depth unless a smaller same-organization finalist qualifies in both models",
              "validation_responses_used_for_selection": False}
    path = root / OUT / "control_plan_v35.json"
    write_json_atomic(path, record)
    seal = stage_freeze(root, "control_plan", [SOURCE, str(path.relative_to(root)),
                                               "artifacts/hierarchical_read_v35_design.freeze.json"],
                        {"selected_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], "states": 10, "KV_states": 5, "flow_states": 5}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
