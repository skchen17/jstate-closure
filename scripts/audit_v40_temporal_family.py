"""Descriptive task-family delay profiles from sealed V40 state-level outcomes.

This appendix is exploratory: no family-specific threshold or decay law is fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from scripts.analyze_v40_a4 import CONDITIONS, bootstrap_mean, load_rows, primary_index

SOURCE = "scripts/audit_v40_temporal_family.py"
CHANNELS = ("REC_only", "Conv_only", "KV_only", "REC_Conv", "REC_Conv_KV")


def channel_summary(index: dict, qualified: list[dict], delay: int,
                    channel: str, seed: int, iterations: int) -> dict:
    selected = [index[(item["state_id"], delay, channel)] for item in qualified]
    base = [index[(item["state_id"], delay, "BASE")] for item in qualified]
    margins = [row["donor_minus_recipient_logit"] - ref["donor_minus_recipient_logit"]
               for row, ref in zip(selected, base, strict=True)]
    return {"donor_recovered_qualified": sum(row["donor_recovered"] for row in selected),
            "qualified_denominator": len(qualified),
            "median_margin_shift_vs_base": float(np.median(margins)) if margins else None,
            "mean_margin_shift_vs_base": bootstrap_mean(margins, seed, iterations)}


def analyze(root: Path, role: str) -> dict:
    cfg = verify(root)["config"]
    verify_stage(root, f"analysis_temporal_Q_{role}")
    first_rows, first_summary = load_rows(root, "Q", role, "functional")
    later_rows, later_summary = load_rows(root, "Q", role, "temporal")
    index = {**primary_index(first_rows), **primary_index(later_rows)}
    states = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    result = {"model": "Q", "role": role, "exploratory_descriptive": True,
              "independent_unit": "state", "families": {},
              "functional_raw_sha256": first_summary["raw_sha256"],
              "temporal_raw_sha256": later_summary["raw_sha256"]}
    seed = int(cfg["bootstrap_seed"]) + sum(map(ord, role + "family"))
    iterations = int(cfg["bootstrap_iterations"])
    for family in sorted({item["family"] for item in states}):
        members = [item for item in states if item["family"] == family]
        profiles = {}
        for delay in cfg["temporal_delays"]:
            complete = [item for item in members if all(
                (item["state_id"], delay, condition) in index for condition in CONDITIONS)]
            qualified = [item for item in complete if
                         index[(item["state_id"], delay, "BASE")]["recipient_correct"]
                         and index[(item["state_id"], delay, "DONOR_NATIVE")]["donor_recovered"]]
            channels = {}
            for channel in CHANNELS:
                channels[channel] = channel_summary(
                    index, qualified, delay, channel,
                    seed + delay * 41 + len(channel) + len(family), iterations)
            profiles[str(delay)] = {"sealed_states": len(members),
                                    "complete_states": len(complete),
                                    "native_qualified_states": len(qualified),
                                    "channels": channels}
        result["families"][family] = {"sealed_states": len(members), "profiles": profiles}
    return result


def run(root: Path, role: str) -> dict:
    result = analyze(root, role)
    path = root / f"results/v40/processed/temporal_family_Q_{role}_v40.json"
    if path.exists():
        raise RuntimeError("V40 family temporal appendix already exists")
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"temporal_family_Q_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         f"results/v40/processed/analysis_temporal_Q_{role}_v40.json"],
                        {"role": role, "exploratory_descriptive": True,
                         "appendix_sha256": sha256_file(path)})
    return {"path": str(path.relative_to(root)), "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
