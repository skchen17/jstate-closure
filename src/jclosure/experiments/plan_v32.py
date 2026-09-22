"""Freeze V32 factorial, trace, context and final-opening rules before outcomes."""
from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments.design_v32 import hd
from jclosure.protocol_v32 import stage_freeze, verify, verify_stage
from jclosure.provenance import write_json_atomic

OUT = Path("results/v32/processed")
SOURCE = "src/jclosure/experiments/plan_v32.py"


def prepare(root: Path):
    verify_stage(root, "design")
    cfg = verify(root)["config"]
    design = json.loads((root / OUT / "design_v32.json").read_text())
    selected = {}
    for role in ("development", "validation"):
        selected[role] = {}
        for family in cfg["families"]:
            group = [x for x in design[role] if x["family"] == family]
            selected[role][family] = {
                "factorial_state_ids": [x["base_trial_id"] for x in group],
                "trace_state_ids": [x["base_trial_id"] for x in group[:2]],
                "context_state_ids": [x["base_trial_id"] for x in group[:2]],
                "cross_token_state_ids": [x["base_trial_id"] for x in group[:5 if role == "development" else 2]],
            }
    plan = {
        "roles": selected,
        "factorial_conditions": cfg["factorial"],
        "primary_pair": "anchor token 25 versus frozen primary natural candidate token in each incoming state",
        "secondary_pair": "frozen second natural candidate of a different surface category when eligible",
        "primary_recipient": "A outgoing cache",
        "primary_donor": "B outgoing cache",
        "persistent_fields": {"REC": "recurrent_states", "Conv": "conv_states", "KV": "recipient-native keys and values"},
        "response_signature": cfg["primary_signature"],
        "replication_gate": cfg["replication_gate"],
        "alignment_gate": cfg["alignment_gate"],
        "context_gate": cfg["context_gate"],
        "localization_gate": cfg["localization_gate"],
        "bootstrap": cfg["bootstrap"],
        "layer_groups": cfg["layer_groups"],
        "candidate_rule": cfg["trace_candidate_rule"],
        "intervention": cfg["site_intervention"],
        "gain_rule": cfg["gain_rule"],
        "rotation_rule": cfg["rotation_rule"],
        "final_rule": cfg["final_rule"],
        "finalist_priority": ["recurrent_operator_output", "recurrent_gate", "qkv_or_update", "residual_integration", "distributed_group"],
        "independent_final_opened": False,
        "future_responses_observed_before_plan": 0,
    }
    path = root / OUT / "execution_plan_v32.json"
    write_json_atomic(path, plan)
    stage = stage_freeze(root, "execution_plan", [SOURCE, str(path.relative_to(root)), "artifacts/rec_conv_mechanism_v32_design.freeze.json"], {"plan_hash": hd(plan), "sample_hash": hd(selected), "layer_groups_hash": hd(cfg["layer_groups"]), "future_responses_observed_before_plan": 0})
    return {"freeze_digest": stage["freeze_digest"], "factorial_states": {role: sum(len(x["factorial_state_ids"]) for x in selected[role].values()) for role in selected}, "trace_states": {role: sum(len(x["trace_state_ids"]) for x in selected[role].values()) for role in selected}}


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2))
