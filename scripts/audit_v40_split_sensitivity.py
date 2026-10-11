"""Append-only sensitivity to native split/full answer disagreements in V40."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "scripts/audit_v40_split_sensitivity.py"
CONDITIONS = ("BASE", "REC_only", "Conv_only", "KV_only",
              "REC_Conv", "REC_Conv_KV", "DONOR_NATIVE")


def disagreed_states(rows: list[dict]) -> tuple[set[str], list[dict]]:
    details = []
    for row in rows:
        audit = row.get("split_full_audit")
        if audit is not None and not audit["candidate_prediction_agrees"]:
            details.append({"state_id": row["state_id"], "family": row["family"],
                            "condition": row["condition"],
                            "cached_prediction": row["predicted_answer"],
                            "direct_prediction": audit["direct_predicted_answer"],
                            "max_full_vocab_abs": audit["full_vocab_max_abs"]})
    return {item["state_id"] for item in details}, details


def run(root: Path, key: str, role: str) -> dict:
    cfg = verify(root)["config"]
    if key not in ("Q", "F") or role not in ("development", "validation", "independent_final"):
        raise ValueError((key, role))
    verify_stage(root, f"analysis_functional_{key}_{role}")
    analysis_path = root / f"results/v40/processed/analysis_functional_{key}_{role}_v40.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary_path = root / f"results/v40/processed/formal_functional_{key}_{role}_v40.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_path = root / summary["raw_path"]
    if sha256_file(raw_path) != summary["raw_sha256"]:
        raise RuntimeError("V40 split audit raw checksum drift")
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    mismatched, details = disagreed_states(rows)
    primary = {(row["state_id"], row["condition"]): row for row in rows
               if row.get("kind") == "primary" and row.get("delay") == 0
               and row["status"] == "VALID"}
    strict = sorted(set(analysis["qualified_state_ids"]) - mismatched)
    if any((state_id, condition) not in primary for state_id in strict for condition in CONDITIONS):
        raise RuntimeError("V40 split audit missing primary condition")
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_id"]].append(row)
    family = {}
    for name in sorted(analysis["families"]):
        family[name] = sum(primary[(state_id, "BASE")]["family"] == name for state_id in strict)
    gate_cfg = cfg["qwen_primary_interpretability_gate"]
    gate = None
    if key == "Q":
        if role == "development":
            gate = (len(strict) >= gate_cfg["minimum_development_qualified_states"]
                    and sum(value >= 4 for value in family.values())
                    >= gate_cfg["minimum_development_families_with_four_qualified"])
        elif role == "validation":
            gate = len(strict) >= gate_cfg["minimum_validation_qualified_states"]
    controls = {}
    for kind, condition in (("same_prefix_mismatch", "ALT_REC_Conv"),
                            ("shuffled_REC_Conv", "SHUFFLED")):
        differences = []
        for state_id in strict:
            matches = [row for row in by_state[state_id] if row.get("kind") == kind
                       and row.get("condition") == condition and row["status"] == "VALID"]
            if len(matches) > 1:
                raise RuntimeError("V40 split audit duplicate control")
            if matches:
                differences.append(int(primary[(state_id, "REC_Conv")]["donor_recovered"])
                                   - int(matches[0]["donor_recovered"]))
        controls[kind] = {"n": len(differences),
                          "mean_recconv_minus_control":
                              sum(differences) / len(differences) if differences else None}
    result = {"model": key, "role": role,
              "policy": "exclude an entire task state if either native BASE or DONOR_NATIVE cached prediction differs from direct full prediction; retain primary all-state analysis unchanged",
              "original_qualified_states": analysis["qualified_states"],
              "strict_qualified_states": len(strict),
              "mismatched_state_ids": sorted(mismatched),
              "mismatch_details": details,
              "strict_qualified_by_family": family,
              "strict_qwen_interpretability_gate": gate,
              "strict_donor_recovery": {condition: {
                  "n": len(strict),
                  "recovered": sum(bool(primary[(state_id, condition)]["donor_recovered"])
                                   for state_id in strict)} for condition in CONDITIONS},
              "strict_matched_control_differences": controls,
              "raw_sha256": summary["raw_sha256"],
              "primary_analysis_sha256": sha256_file(analysis_path),
              "append_only_sensitivity_not_replacement": True}
    path = root / f"results/v40/processed/split_sensitivity_{key}_{role}_v40.json"
    if path.exists():
        raise RuntimeError("V40 split sensitivity already exists")
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"split_sensitivity_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         str(summary_path.relative_to(root)),
                         str(analysis_path.relative_to(root))],
                        {"model": key, "role": role,
                         "sensitivity_sha256": sha256_file(path)})
    return {"model": key, "role": role,
            "mismatch_states": len(mismatched),
            "strict_qualified": len(strict),
            "freeze_digest": seal["freeze_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role), indent=2))
