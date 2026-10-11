"""State-unit V40 analysis from immutable formal JSONL partitions."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from jclosure.protocol_v40_a4 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "scripts/analyze_v40_a4.py"
CONDITIONS = ("BASE", "REC_only", "Conv_only", "KV_only",
              "REC_Conv", "REC_Conv_KV", "DONOR_NATIVE")
ROLES = ("development", "validation", "independent_final")


def bootstrap_mean(values: list[float], seed: int, iterations: int) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    samples = [sum(values[rng.randrange(n)] for _ in range(n)) / n
               for _ in range(iterations)]
    return {"mean": float(np.mean(values)),
            "ci95": [float(np.quantile(samples, .025)),
                     float(np.quantile(samples, .975))], "n": n}


def load_rows(root: Path, key: str, role: str, phase: str) -> tuple[list[dict], dict]:
    verify_stage(root, f"{phase}_{key}_{role}")
    summary = json.loads((root / f"results/v40/processed/formal_{phase}_{key}_{role}_v40.json").read_text())
    path = root / summary["raw_path"]
    if sha256_file(path) != summary["raw_sha256"]:
        raise RuntimeError("V40 a4 raw checksum drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(row["model"] != key or row["role"] != role or row["phase"] != phase
           for row in rows):
        raise RuntimeError("V40 a4 role/phase contamination")
    return rows, summary


def primary_index(rows: list[dict]) -> dict:
    index = {}
    for row in rows:
        if row.get("kind") != "primary" or row["status"] != "VALID":
            continue
        label = (row["state_id"], row["delay"], row["condition"])
        if label in index:
            raise RuntimeError(f"V40 a4 duplicate primary: {label}")
        index[label] = row
    return index


def functional(root: Path, key: str, role: str, cfg: dict) -> dict:
    rows, summary = load_rows(root, key, role, "functional")
    states = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_id"]].append(row)
    if set(by_state) != {s["state_id"] for s in states}:
        raise RuntimeError("V40 a4 state denominator drift")
    index = primary_index(rows)
    complete = [s for s in states if all((s["state_id"], 0, c) in index for c in CONDITIONS)
                and not any(r["status"] == "INVALID" for r in by_state[s["state_id"]])]
    qualified = [s for s in complete if
                 index[(s["state_id"], 0, "BASE")]["recipient_correct"] and
                 index[(s["state_id"], 0, "DONOR_NATIVE")]["donor_recovered"]]
    qset = {s["state_id"] for s in qualified}
    complete_ids = {s["state_id"] for s in complete}
    seed = int(cfg["bootstrap_seed"]) + sum(map(ord, key + role))
    iterations = int(cfg["bootstrap_iterations"])
    metrics = {}
    for condition in CONDITIONS:
        all_success = sum(bool(index[(s["state_id"], 0, condition)]["donor_recovered"])
                          for s in complete)
        qualified_success = sum(bool(index[(s["state_id"], 0, condition)]["donor_recovered"])
                                for s in qualified)
        differences = [float(index[(s["state_id"], 0, condition)]["donor_recovered"])
                       - float(index[(s["state_id"], 0, "BASE")]["donor_recovered"])
                       for s in qualified]
        margin_shift = [index[(s["state_id"], 0, condition)]["donor_minus_recipient_logit"]
                        - index[(s["state_id"], 0, "BASE")]["donor_minus_recipient_logit"]
                        for s in qualified]
        metrics[condition] = {
            "donor_recovered_all_states": all_success,
            "all_state_denominator": len(states),
            "valid_state_denominator": len(complete),
            "donor_recovered_qualified": qualified_success,
            "qualified_denominator": len(qualified),
            "qualified_rate": qualified_success / len(qualified) if qualified else None,
            "recovery_delta_vs_base": bootstrap_mean(differences, seed + len(condition), iterations),
            "margin_shift_vs_base": bootstrap_mean(margin_shift, seed + len(condition) * 2, iterations),
        }
    recconv_vs_conv = [
        float(index[(s["state_id"], 0, "REC_Conv")]["donor_recovered"])
        - float(index[(s["state_id"], 0, "Conv_only")]["donor_recovered"])
        for s in qualified]
    margin_recconv_vs_conv = [
        index[(s["state_id"], 0, "REC_Conv")]["donor_minus_recipient_logit"]
        - index[(s["state_id"], 0, "Conv_only")]["donor_minus_recipient_logit"]
        for s in qualified]
    families = {}
    for family in sorted({s["family"] for s in states}):
        members = [s for s in qualified if s["family"] == family]
        families[family] = {"sealed_states": sum(s["family"] == family for s in states),
                            "complete_states": sum(s["family"] == family for s in complete),
                            "qualified_states": len(members),
                            "donor_recovered": {condition:
                                sum(bool(index[(s["state_id"], 0, condition)]["donor_recovered"])
                                    for s in members)
                                for condition in CONDITIONS}}
    controls = {}
    matched_control_contrasts = {}
    for kind, name in (("same_prefix_mismatch", "ALT_REC_Conv"),
                       ("shuffled_REC_Conv", "SHUFFLED")):
        differences = []
        for task in qualified:
            sid = task["state_id"]
            matches = [r for r in by_state[sid] if r.get("kind") == kind
                       and r.get("condition") == name and r["status"] == "VALID"]
            if len(matches) > 1:
                raise RuntimeError(f"V40 a4 duplicate control: {sid}:{kind}")
            if matches:
                differences.append(float(index[(sid, 0, "REC_Conv")]["donor_recovered"])
                                   - float(matches[0]["donor_recovered"]))
        matched_control_contrasts[kind] = bootstrap_mean(
            differences, seed + 991 + len(kind), iterations)
    for kind in ("surface_only", "irrelevant_history", "same_prefix_mismatch", "shuffled_REC_Conv"):
        subset = [r for r in rows if r.get("kind") == kind]
        valid = [r for r in subset if r["status"] == "VALID"]
        controls[kind] = {"valid_rows": len(valid),
                          "unavailable_rows": sum(r["status"] == "UNAVAILABLE" for r in subset),
                          "invalid_rows": sum(r["status"] == "INVALID" for r in subset),
                          "correct_task_rows": sum(r.get("task_correct", False) for r in valid),
                          "paired_donor_recovery_rows": sum(r.get("donor_recovered", False)
                                                            for r in valid),
                          "by_condition": {name: {
                              "n": sum(r.get("condition") == name for r in valid),
                              "correct": sum(r.get("condition") == name and
                                             r.get("task_correct", False) for r in valid),
                              "paired_donor": sum(r.get("condition") == name and
                                                  r.get("donor_recovered", False) for r in valid)}
                              for name in sorted({r.get("condition") for r in valid})}}
    audits = [index[(s["state_id"], 0, condition)]["split_full_audit"]
              for s in complete for condition in ("BASE", "DONOR_NATIVE")]
    gate_cfg = cfg["qwen_primary_interpretability_gate"]
    gate = (len(qualified) >= int(gate_cfg["minimum_development_qualified_states"])
            and sum(f["qualified_states"] >= 4 for f in families.values())
            >= int(gate_cfg["minimum_development_families_with_four_qualified"])) if role == "development" else (
            len(qualified) >= int(gate_cfg["minimum_validation_qualified_states"]))
    return {"model": key, "role": role, "phase": "functional",
            "sealed_states": len(states), "complete_states": len(complete),
            "invalid_states": [s["state_id"] for s in states if s["state_id"] not in complete_ids],
            "qualified_states": len(qualified), "qualified_state_ids": sorted(qset),
            "baseline_qualification_gate": gate if key == "Q" else None,
            "channels": metrics, "recconv_vs_conv_recovery":
                bootstrap_mean(recconv_vs_conv, seed + 901, iterations),
            "recconv_vs_conv_margin":
                bootstrap_mean(margin_recconv_vs_conv, seed + 902, iterations),
            "families": families, "controls": controls,
            "recconv_recovery_vs_matched_controls": matched_control_contrasts,
            "split_full": {"audits": len(audits),
                           "candidate_prediction_agrees": sum(a["candidate_prediction_agrees"] for a in audits),
                           "max_full_vocab_abs": max((a["full_vocab_max_abs"] for a in audits), default=None)},
            "raw_sha256": summary["raw_sha256"]}


def temporal(root: Path, key: str, role: str, cfg: dict) -> dict:
    first_rows, first_summary = load_rows(root, key, role, "functional")
    later_rows, later_summary = load_rows(root, key, role, "temporal")
    states = json.loads((root / f"data/v40/{role}_pool_v40.json").read_text())["items"]
    index = {**primary_index(first_rows), **primary_index(later_rows)}
    seed = int(cfg["bootstrap_seed"]) + sum(map(ord, key + role + "temporal"))
    iterations = int(cfg["bootstrap_iterations"])
    profiles = {}
    for delay in cfg["temporal_delays"]:
        complete = [s for s in states if all((s["state_id"], delay, c) in index for c in CONDITIONS)]
        qualified = [s for s in complete if
                     index[(s["state_id"], delay, "BASE")]["recipient_correct"] and
                     index[(s["state_id"], delay, "DONOR_NATIVE")]["donor_recovered"]]
        channel = {}
        for condition in CONDITIONS:
            effects = [index[(s["state_id"], delay, condition)]["donor_minus_recipient_logit"]
                       - index[(s["state_id"], delay, "BASE")]["donor_minus_recipient_logit"]
                       for s in qualified]
            ceiling = [index[(s["state_id"], delay, "DONOR_NATIVE")]["donor_minus_recipient_logit"]
                       - index[(s["state_id"], delay, "BASE")]["donor_minus_recipient_logit"]
                       for s in qualified]
            ratios = [effect / native for effect, native in zip(effects, ceiling, strict=True)
                      if native > 1e-6]
            channel[condition] = {
                "donor_recovered_all_states": sum(bool(index[(s["state_id"], delay, condition)]["donor_recovered"])
                                                  for s in complete),
                "all_state_denominator": len(states),
                "qualified_recovery": sum(bool(index[(s["state_id"], delay, condition)]["donor_recovered"])
                                          for s in qualified),
                "qualified_denominator": len(qualified),
                "margin_effect": bootstrap_mean(effects, seed + delay * 31 + len(condition), iterations),
                "median_margin_effect": float(np.median(effects)) if effects else None,
                "median_fraction_of_native_ceiling": float(np.median(ratios)) if ratios else None,
                "native_ceiling_positive_states": len(ratios),
            }
        profiles[str(delay)] = {"complete_states": len(complete),
                                "qualified_states": len(qualified), "channels": channel,
                                "qualified_by_family": {family: sum(s["family"] == family for s in qualified)
                                                        for family in sorted({s["family"] for s in states})}}
    common = [s for s in states if all(
        all((s["state_id"], d, c) in index for c in CONDITIONS)
        and index[(s["state_id"], d, "BASE")]["recipient_correct"]
        and index[(s["state_id"], d, "DONOR_NATIVE")]["donor_recovered"]
        for d in cfg["temporal_delays"])]
    crossings = {}
    for condition in CONDITIONS[1:-1]:
        values = [float(np.median([
            index[(s["state_id"], d, condition)]["donor_minus_recipient_logit"]
            - index[(s["state_id"], d, "BASE")]["donor_minus_recipient_logit"]
            for s in common])) if common else None for d in cfg["temporal_delays"]]
        base = values[0]
        cross = None
        if len(common) >= 16 and base is not None and abs(base) > 1e-6 and all(v is not None for v in values):
            for i, delay in enumerate(cfg["temporal_delays"][1:], 1):
                if all(abs(v) <= .5 * abs(base) for v in values[i:]):
                    cross = delay
                    break
        crossings[condition] = {"observed_grid_half_crossing": cross,
                                "common_qualified_state_count": len(common),
                                "common_state_median_margin_effects": values,
                                "criterion": "first delay at or below half of delay-zero absolute median effect, remaining below at all later tested delays"}
    return {"model": key, "role": role, "phase": "temporal",
            "sealed_states": len(states), "profiles": profiles,
            "common_qualified_states": len(common),
            "observed_grid_half_crossings": crossings,
            "functional_raw_sha256": first_summary["raw_sha256"],
            "temporal_raw_sha256": later_summary["raw_sha256"],
            "no_decay_law_fitted": True}


def run(root: Path, key: str, role: str, phase: str) -> dict:
    cfg = verify(root)["config"]
    if key not in ("Q", "F") or role not in ROLES or phase not in ("functional", "temporal"):
        raise ValueError((key, role, phase))
    result = functional(root, key, role, cfg) if phase == "functional" else temporal(root, key, role, cfg)
    path = root / f"results/v40/processed/analysis_{phase}_{key}_{role}_v40.json"
    if path.exists():
        raise RuntimeError("V40 a4 analysis already exists")
    write_json_atomic(path, result)
    raw_summary = root / f"results/v40/processed/formal_{phase}_{key}_{role}_v40.json"
    seal = stage_freeze(root, f"analysis_{phase}_{key}_{role}",
                        [SOURCE, str(path.relative_to(root)), str(raw_summary.relative_to(root))],
                        {"model": key, "role": role, "phase": phase,
                         "analysis_sha256": sha256_file(path)})
    return {"path": str(path.relative_to(root)), "freeze_digest": seal["freeze_digest"],
            "sealed_states": result["sealed_states"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("functional", "temporal"))
    parser.add_argument("model", choices=("Q", "F"))
    parser.add_argument("role", choices=ROLES)
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.model, args.role, args.phase), indent=2))
