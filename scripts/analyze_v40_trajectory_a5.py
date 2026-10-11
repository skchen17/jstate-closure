"""Task-answer and hidden-checkpoint analysis for sealed V40 REC trajectories."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jclosure.protocol_v40_a5 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic
from scripts.analyze_v40_a4 import bootstrap_mean

SOURCE = "scripts/analyze_v40_trajectory_a5.py"
ORDER = ("BASE", "DONOR_NATIVE", "R000", "R100", "R010", "R001",
         "R110", "R101", "R011", "R111")


def interaction(row: dict[str, dict]) -> float:
    y = {key: value["donor_minus_recipient_logit"] for key, value in row.items()}
    return (y["R111"] - y["R110"] - y["R101"] - y["R011"]
            + y["R100"] + y["R010"] + y["R001"] - y["R000"])


def analyze(root: Path, role: str) -> dict:
    cfg = verify(root)["config"]
    verify_stage(root, f"trajectory_Q_{role}")
    summary = json.loads((root / f"results/v40/processed/trajectory_Q_{role}_v40.json").read_text())
    raw_path = root / summary["raw_path"]
    if sha256_file(raw_path) != summary["raw_sha256"]:
        raise RuntimeError("V40 a5 trajectory raw checksum drift")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    states = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    by_state = defaultdict(list)
    indexed = {}
    for row in rows:
        by_state[row["state_id"]].append(row)
        if row["status"] != "VALID":
            continue
        label = row["state_id"], row["delay"], row["condition"]
        if label in indexed:
            raise RuntimeError(f"V40 a5 duplicate trajectory label: {label}")
        indexed[label] = row
    if set(by_state) != {task["state_id"] for task in states}:
        raise RuntimeError("V40 a5 state denominator drift")
    seed = int(cfg["bootstrap_seed"]) + sum(map(ord, role))
    iterations = int(cfg["bootstrap_iterations"])
    role_result = {"model": "Q", "role": role, "sealed_states": len(states),
                   "raw_sha256": summary["raw_sha256"], "delays": {},
                   "future_output_copied": False}
    for delay in cfg["delays"]:
        complete = [task for task in states if all(
            (task["state_id"], delay, condition) in indexed for condition in ORDER)]
        qualified = [task for task in complete if
                     indexed[(task["state_id"], delay, "BASE")]["recipient_correct"]
                     and indexed[(task["state_id"], delay, "DONOR_NATIVE")]["donor_recovered"]]
        def condition_rows(condition, *, delay=delay, qualified=qualified):
            return [indexed[(task["state_id"], delay, condition)] for task in qualified]
        conditions = {}
        r000 = condition_rows("R000")
        for condition in ORDER:
            condition_values = condition_rows(condition)
            margins = [row["donor_minus_recipient_logit"] - base["donor_minus_recipient_logit"]
                       for row, base in zip(condition_values, r000, strict=True)]
            hidden = defaultdict(list)
            for row in condition_values:
                for checkpoint, layers in row["checkpoint_hidden_cosines"].items():
                    for layer, values in layers.items():
                        if values["to_native_donor"] is not None:
                            hidden[f"{checkpoint}:{layer}"].append(values["to_native_donor"])
            conditions[condition] = {
                "donor_recovered_all_complete": sum(indexed[(task["state_id"], delay, condition)]["donor_recovered"]
                                                   for task in complete),
                "all_state_denominator": len(states),
                "qualified_donor_recovered": sum(row["donor_recovered"] for row in condition_values),
                "qualified_denominator": len(qualified),
                "margin_shift_vs_common_conv": bootstrap_mean(margins, seed + delay * 37 + len(condition), iterations),
                "median_hidden_cosine_to_donor": {name: float(np.median(values))
                                                   for name, values in hidden.items()},
            }
        def interaction_values(members, *, delay=delay):
            return [interaction({condition: indexed[(task["state_id"], delay, condition)]
                                 for condition in ORDER[2:]}) for task in members]
        interactions = interaction_values(qualified)
        family_interactions = {family: {
            "qualified_states": sum(task["family"] == family for task in qualified),
            "median_three_way_margin": float(np.median(interaction_values(
                [task for task in qualified if task["family"] == family])))
                if any(task["family"] == family for task in qualified) else None}
            for family in sorted({task["family"] for task in states})}
        ci = bootstrap_mean(interactions, seed + delay * 53, iterations)
        gate = cfg["task_grounded_gate"]
        claimed = (len(qualified) >= gate["minimum_native_qualified_states"]
                   and ci["ci95"] is not None
                   and ci["ci95"][0] > gate["positive_bootstrap_lower_bound"]
                   and sum(value["median_three_way_margin"] is not None
                           and value["median_three_way_margin"] > 0
                           for value in family_interactions.values())
                   >= gate["positive_family_medians_required"])
        role_result["delays"][str(delay)] = {
            "complete_states": len(complete), "qualified_states": len(qualified),
            "invalid_states": [task["state_id"] for task in states if task not in complete],
            "conditions": conditions, "three_way_margin_interaction": ci,
            "family_interactions": family_interactions,
            "task_grounded_three_way_gate": claimed,
            "gate_rule": "at least 16 qualified states, positive 95% bootstrap lower bound, and positive median in at least three families"}
    return role_result


def run(root: Path, role: str) -> dict:
    result = analyze(root, role)
    path = root / f"results/v40/processed/trajectory_analysis_Q_{role}_v40.json"
    if path.exists():
        raise RuntimeError("V40 a5 trajectory analysis already exists")
    write_json_atomic(path, result)
    summary = root / f"results/v40/processed/trajectory_Q_{role}_v40.json"
    seal = stage_freeze(root, f"trajectory_analysis_Q_{role}",
                        [SOURCE, str(path.relative_to(root)), str(summary.relative_to(root))],
                        {"role": role, "analysis_sha256": sha256_file(path)})
    return {"role": role, "path": str(path.relative_to(root)),
            "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
