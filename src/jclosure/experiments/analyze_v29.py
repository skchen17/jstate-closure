"""Predeclared reciprocal donor-direction gates for V29 formal panels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import write_json_atomic

SOURCE = "src/jclosure/experiments/analyze_v29.py"
OUT = Path("results/v29/processed")
PARTIAL = ("REC", "Conv", "KV", "REC+Conv", "REC+KV", "Conv+KV")


def summarize(f, audit, cfg):
    merged = f.merge(audit[["base_trial_id", "condition", "direction", "writeback_pass", "requested_exact", "untouched_exact"]], on=["base_trial_id", "condition", "direction"], validate="one_to_one")
    profiles = {}
    for condition, rows in merged.groupby("condition"):
        directions = {}
        for direction, x in rows.groupby("direction"):
            eligible = x.future_q >= cfg["branch_future_q_min"]
            success = eligible & x.writeback_pass & x.requested_exact & x.untouched_exact & (x.donor_cosine >= cfg["partial_cosine_min"]) & (x.magnitude_ratio >= cfg["partial_magnitude_min"])
            fam = {}
            for name, part in x.groupby("family"):
                accepted = success.loc[part.index]
                fam[name] = {"rows": len(part), "success_fraction": float(accepted.mean()), "median_cosine": float(part.donor_cosine.median()), "median_magnitude": float(part.magnitude_ratio.median()), "pass": bool(float(accepted.mean()) >= cfg["partial_success_fraction_min"])}
            gate = bool(
                eligible.mean() >= cfg["partial_success_fraction_min"]
                and x.writeback_pass.all()
                and x.requested_exact.all()
                and x.untouched_exact.all()
                and float(x.donor_cosine.median()) >= cfg["partial_cosine_min"]
                and float(x.magnitude_ratio.median()) >= cfg["partial_magnitude_min"]
                and float(success.mean()) >= cfg["partial_success_fraction_min"]
                and sum(v["pass"] for v in fam.values()) >= cfg["families_required"]
            )
            directions[direction] = {"rows": len(x), "eligible_fraction": float(eligible.mean()), "median_cosine": float(x.donor_cosine.median()), "median_magnitude": float(x.magnitude_ratio.median()), "median_relative_l2": float(x.relative_l2_to_donor.median()), "success_fraction": float(success.mean()), "families": fam, "pass": gate}
        profiles[condition] = {"directions": directions, "reciprocal_pass": all(v["pass"] for v in directions.values())}
    return profiles, merged


def run(root: Path, role: str):
    verify_stage(root, f"forks_{role}")
    if role == "validation":
        verify_stage(root, "development_analysis")
    cfg = verify(root)["config"]
    f = pd.read_parquet(root / OUT / f"transfer_{role}_v29.parquet")
    a = pd.read_parquet(root / OUT / f"audit_{role}_v29.parquet")
    forks = pd.read_parquet(root / OUT / f"forks_{role}_v29.parquet")
    profiles, merged = summarize(f, a, cfg)
    full = merged[merged.condition == "REC+Conv+KV"]
    full_exact = bool(full.writeback_pass.all() and (full.relative_l2_to_donor <= 1e-6).all() and (full.donor_cosine >= 0.999999).all())
    branch = bool((forks.future_q >= cfg["branch_future_q_min"]).mean() >= cfg["partial_success_fraction_min"] and forks.same_token_replay_exact.all() and forks.incoming_identical.all() and forks.current_token_only_differs.all())
    payload = {
        "role": role,
        "states": len(forks),
        "rows": len(f),
        "median_current_q": float(forks.current_q.median()),
        "median_future_q": float(forks.future_q.median()),
        "branch_future_eligible_fraction": float((forks.future_q >= cfg["branch_future_q_min"]).mean()),
        "branch_gate": branch,
        "full_native_cache_identity_ceiling": full_exact,
        "profiles": profiles,
        "partial_reciprocal_pass": {c: profiles[c]["reciprocal_pass"] for c in PARTIAL},
        "fixed_finalist": "REC+Conv",
        "fixed_finalist_pass": profiles["REC+Conv"]["reciprocal_pass"],
        "strict_writeback_all": bool(a.writeback_pass.all() and a.requested_exact.all() and a.untouched_exact.all()),
        "current_preserved_all": bool(a.recipient_current_unchanged.all()),
        "same_next_token_all": bool(a.next_token_identical.all()),
        "historical_final_opened": False,
        "independent_final_opened": False,
    }
    if not full_exact:
        raise RuntimeError("all-layer full-cache identity ceiling failed")
    path = root / OUT / f"{role}_analysis_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, f"{role}_analysis", [SOURCE, str(path.relative_to(root)), f"results/v29/processed/forks_{role}_v29.parquet", f"results/v29/processed/transfer_{role}_v29.parquet", f"results/v29/processed/audit_{role}_v29.parquet", f"artifacts/natural_write_content_v29_forks_{role}.freeze.json"], {"analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "branch_gate": branch, "full_identity_ceiling": full_exact, "partial_reciprocal_pass": payload["partial_reciprocal_pass"], "fixed_finalist_pass": payload["fixed_finalist_pass"]})
    return {"freeze_digest": fr["freeze_digest"], "branch_gate": branch, "full_identity_ceiling": full_exact, "partial_reciprocal_pass": payload["partial_reciprocal_pass"], "median_future_q": payload["median_future_q"]}


def final_opening(root: Path):
    dev = verify_stage(root, "development_analysis")
    val = verify_stage(root, "validation_analysis")
    cfg = verify(root)["config"]
    opened = bool(dev["branch_gate"] and val["branch_gate"] and dev["full_identity_ceiling"] and val["full_identity_ceiling"] and dev["fixed_finalist_pass"] and val["fixed_finalist_pass"])
    payload = {"final_opened": opened, "fixed_finalist": "REC+Conv", "rule": cfg["final_rule"], "reason": "fixed REC+Conv reciprocal donor-direction gate passed development and validation" if opened else "fixed REC+Conv failed a development/validation reciprocal gate; final remains sealed", "development_analysis_sha256": hashlib.sha256((root / OUT / "development_analysis_v29.json").read_bytes()).hexdigest(), "validation_analysis_sha256": hashlib.sha256((root / OUT / "validation_analysis_v29.json").read_bytes()).hexdigest(), "historical_final_opened": False}
    payload["final_opening_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = root / OUT / "final_opening_v29.json"
    write_json_atomic(path, payload)
    fr = stage_freeze(root, "final_opening", [SOURCE, str(path.relative_to(root)), "artifacts/natural_write_content_v29_development_analysis.freeze.json", "artifacts/natural_write_content_v29_validation_analysis.freeze.json"], payload)
    return {"freeze_digest": fr["freeze_digest"], **payload}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("development", "validation", "final_opening"))
    a = p.parse_args()
    print(json.dumps(final_opening(Path.cwd()) if a.role == "final_opening" else run(Path.cwd(), a.role), indent=2))
