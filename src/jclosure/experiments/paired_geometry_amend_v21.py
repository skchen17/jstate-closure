"""Append-only handling of the one calibrated but unreliable V21 probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.experiments import paired_geometry_v21 as paired
from jclosure.protocol_v21 import stage_freeze, verify_stage

SOURCE = "src/jclosure/experiments/paired_geometry_amend_v21.py"


def prepare(root: Path) -> dict:
    design = verify_stage(root, "paired_geometry_design")
    absent = [index for index in design["probe_direction_indices"]
              if design["probe_alpha_by_direction"][str(index)] is None]
    if absent != [254]:
        raise RuntimeError(f"V21 unreliable probe set drift: {absent}")
    return stage_freeze(root, "paired_geometry_unreliable_probe_amendment_1",
                        [SOURCE, "artifacts/action_coordinate_geometry_v21_paired_geometry_design.freeze.json",
                         "results/v21/processed/probe_calibration_v21.json"],
                        {"reason": "One frozen probe (254) failed train-only reliability at all alpha candidates; retain its exact JVP at fallback alpha 0.25, measure finite response for audit, but exclude its finite column from reliability-qualified rank/alignment.",
                         "unreliable_direction_index": 254, "fallback_alpha": 0.25,
                         "reliable_probe_count": 63,
                         "V21_paired_JVP_or_finite_responses_observed_before_amendment": 0,
                         "final_six_action_responses_opened": False})


def run(root: Path, role: str, limit: int | None) -> dict:
    amend = verify_stage(root, "paired_geometry_unreliable_probe_amendment_1")
    original = paired.verify_stage
    def with_fallback(path: Path, stage: str):
        result = original(path, stage)
        if stage == "paired_geometry_design":
            result = dict(result)
            result["probe_alpha_by_direction"] = dict(result["probe_alpha_by_direction"])
            result["probe_alpha_by_direction"][str(amend["unreliable_direction_index"])] = amend["fallback_alpha"]
        return result
    paired.verify_stage = with_fallback
    try:
        return paired.run(root, role, limit)
    finally:
        paired.verify_stage = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--role", choices=("development", "validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "run" and not args.role:
        parser.error("--role required for run")
    result = prepare(Path.cwd()) if args.stage == "prepare" else run(Path.cwd(), args.role, args.limit)
    print(json.dumps(result, indent=2))
