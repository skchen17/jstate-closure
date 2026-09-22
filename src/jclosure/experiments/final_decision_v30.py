"""Freeze V30 finalist adjudication without accessing sealed independent final responses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v30 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/final_decision_v30.py"
OUT = Path("results/v30/processed")


def gate(frame, cfg, require_families=True):
    x = frame.dropna(subset=["donor_cosine", "magnitude_ratio", "relative_l2_to_donor"])
    if x.empty:
        return {"rows": len(frame), "pass": False, "reason": "no valid rows"}
    ok = (x.donor_cosine >= cfg["cosine_min"]) & x.magnitude_ratio.between(cfg["magnitude_min"], cfg["magnitude_max"]) & (x.relative_l2_to_donor <= cfg["relative_l2_max"])
    family = {fam: float(ok.loc[g.index].mean()) for fam, g in x.groupby("family")}
    result = {"rows": len(frame), "valid_rows": len(x), "median_cosine": float(x.donor_cosine.median()), "median_magnitude": float(x.magnitude_ratio.median()), "median_relative_l2": float(x.relative_l2_to_donor.median()), "success_fraction": float(ok.mean()), "family_success_fraction": family}
    result["pass"] = bool(len(x) == len(frame) and result["median_cosine"] >= cfg["cosine_min"] and cfg["magnitude_min"] <= result["median_magnitude"] <= cfg["magnitude_max"] and result["median_relative_l2"] <= cfg["relative_l2_max"] and result["success_fraction"] >= cfg["success_fraction_min"] and (not require_families or sum(v >= cfg["success_fraction_min"] for v in family.values()) >= cfg["families_required"]))
    return result


def run(root):
    for stage in ("causal_global_development", "causal_global_validation", "local_transport_development", "local_transport_validation"):
        verify_stage(root, stage)
    plan = json.loads((root / OUT / "execution_plan_v30.json").read_text())
    cfg = plan["gates"]
    frames = {role: pd.read_parquet(root / OUT / f"causal_global_{role}_v30.parquet") for role in ("development", "validation")}
    transports = {role: pd.read_parquet(root / OUT / f"local_transport_{role}_v30.parquet") for role in ("development", "validation")}
    candidate_tests = {}
    for finalist in plan["finalist_priority"]:
        if finalist.startswith("GLOBAL"):
            source = frames
            condition = finalist
            axes = {"development": ("STATE_OOD", "TOKEN_OOD", "JOINT_OOD"), "validation": ("STATE_OOD", "JOINT_OOD")}
        else:
            source = transports
            condition = finalist.replace("TRANSPORTED_LOCAL", "TRANSPORTED_LOCAL_RIDGE")
            axes = {"development": ("JOINT_OOD",), "validation": ("JOINT_OOD",)}
        checks = {}
        for role, role_axes in axes.items():
            for axis in role_axes:
                data = source[role]
                if "axis" in data:
                    data = data[data.axis == axis]
                checks[f"{role}:{axis}"] = gate(data[data.condition == condition], cfg)
        lofo = {}
        if finalist.startswith("GLOBAL"):
            for role in ("development", "validation"):
                data = frames[role]
                for family, group in data[(data.condition == condition.replace("GLOBAL", "LOFO")) & (data.axis == "JOINT_OOD")].groupby("family"):
                    lofo[f"{role}:{family}"] = gate(group, cfg, require_families=False)
        pass_axes = all(x["pass"] for x in checks.values())
        pass_lofo = True if not lofo else all(sum(lofo.get(f"{role}:{family}", {}).get("pass", False) for family in frames[role].family.unique()) >= cfg["families_required"] for role in ("development", "validation"))
        candidate_tests[finalist] = {"decoder_condition": condition, "axes": checks, "LOFO": lofo, "pass_axes": pass_axes, "pass_LOFO": pass_lofo, "qualified": bool(pass_axes and pass_lofo)}
    qualified = [f for f in plan["finalist_priority"] if candidate_tests[f]["qualified"]]
    selected = qualified[0] if qualified else None
    result = {"selected_finalist": selected, "final_opened": bool(selected), "independent_final_responses_observed": False, "independent_final_state_count_sealed": len(plan["independent_final_ids"]) if not selected else None, "candidate_tests": candidate_tests, "finalist_priority": plan["finalist_priority"], "transported_local_primary_map": "RIDGE (first frozen transport method); Procrustes remains diagnostic", "rule": plan["final_rule"], "reason": "No candidate passes all frozen development and validation causal OOD gates" if not selected else "One priority finalist qualifies before final response observation", "H2_REMAINS": True, "H3_AUTHORIZED": False, "CROSS_MODEL_REPLICATION_AUTHORIZED": bool(selected and selected.startswith(("GLOBAL", "TRANSPORTED_LOCAL"))), "source_hashes": {f"global_{r}": sha256_file(root / OUT / f"causal_global_{r}_v30.parquet") for r in frames} | {f"transport_{r}": sha256_file(root / OUT / f"local_transport_{r}_v30.parquet") for r in transports}}
    jp = root / OUT / "final_opening_v30.json"
    write_json_atomic(jp, result)
    fr = stage_freeze(root, "final_opening", [SOURCE, str(jp.relative_to(root)), *[f"artifacts/transferable_natural_writes_v30_{s}.freeze.json" for s in ("causal_global_development", "causal_global_validation", "local_transport_development", "local_transport_validation")]], {"final_opened": bool(selected), "selected_finalist": selected, "decision_sha256": sha256_file(jp), "reason": result["reason"]})
    return {"freeze_digest": fr["freeze_digest"], "final_opened": bool(selected), "selected_finalist": selected, "qualified": qualified}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
