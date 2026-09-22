"""Prospective source-pairing amendment using only prewrite eligibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jclosure.protocol_v30 import verify_stage, stage_freeze
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/transport_amend_v30.py"
OUT = Path("results/v30/processed")


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(root: Path):
    verify_stage(root, "execution_plan")
    design = json.loads((root / OUT / "design_v30.json").read_text())
    plan = json.loads((root / OUT / "execution_plan_v30.json").read_text())
    by_id = {x["base_trial_id"]: x for role in ("development", "validation", "independent_final") for x in design[role]}
    fit = [by_id[sid] for sid in plan["fit_state_ids"]]
    original = plan["transport_source_state"]
    source_by_family = {}
    revised = {}
    for family in design["families"]:
        keys = [key for key in original if by_id[key.rsplit(":", 1)[0]]["family"] == family]
        needed = {key.rsplit(":", 1)[1] for key in keys}
        candidates = [x for x in fit if x["family"] == family and needed <= set(x["eligible_pair_ids"])]
        if not candidates:
            raise RuntimeError(f"no prewrite-complete transport source: {family}")
        candidates.sort(key=lambda x: digest(["V30_CANONICAL_SOURCE", family, x["base_trial_id"]]))
        chosen = candidates[0]["base_trial_id"]
        source_by_family[family] = chosen
        for key in keys:
            sid, pair_id = key.rsplit(":", 1)
            if sid == chosen:
                raise RuntimeError("transport source cannot equal target")
            revised[key] = chosen
    if set(revised) != set(original):
        raise RuntimeError("transport test set changed")
    record = {
        "reason": "Resource-bounded prospective amendment: one prewrite-eligible canonical source per family replaces 85 independently selected sources; target states, token contrasts, TRAIN-only transport fitting, metrics, gates and final rule remain unchanged.",
        "scope": "transport_source_state map only",
        "selection_input": "frozen design prewrite eligibility and frozen execution-plan target pairs",
        "response_observed_before_amendment": 0,
        "source_by_family": source_by_family,
        "transport_source_state": revised,
        "original_map_sha256": digest(original),
        "revised_map_sha256": digest(revised),
        "target_pair_keys_unchanged": True,
        "independent_final_opened": False,
    }
    path = root / OUT / "transport_source_amendment_v30.json"
    write_json_atomic(path, record)
    freeze = stage_freeze(root, "transport_source_amendment", [SOURCE, str(path.relative_to(root)), "artifacts/transferable_natural_writes_v30_execution_plan.freeze.json"], {"record_sha256": digest(record), "source_count": len(source_by_family), "response_observed_before_amendment": 0})
    return {"freeze_digest": freeze["freeze_digest"], "source_count": len(source_by_family), "target_pairs": len(revised)}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
