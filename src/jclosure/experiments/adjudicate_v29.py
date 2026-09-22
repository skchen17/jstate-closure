"""Integrate frozen V29 causal gates without promoting diagnostics into formal claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments.analyze_v29 import summarize
from jclosure.experiments.forks_formal_v29 import OUT
from jclosure.protocol_v29 import verify, verify_stage, stage_freeze
from jclosure.provenance import sha256_file, write_json_atomic

SOURCE = "src/jclosure/experiments/adjudicate_v29.py"


def read(root, name):
    return json.loads((root / OUT / name).read_text())


def median(frame, column, **query):
    x = frame
    for key, value in query.items():
        x = x[x[key] == value]
    return float(x[column].median()) if len(x) else None


def run(root: Path):
    stages = ("design", "execution_plan", "forks_calibration", "interface_amendment", "forks_development", "development_analysis", "forks_validation", "validation_analysis", "final_opening", "secondary_development", "secondary_validation", "diagnostics", "realization_plan", "realization")
    for stage in stages:
        verify_stage(root, stage)
    opening = verify_stage(root, "final_opening")
    if opening["final_opened"]:
        verify_stage(root, "forks_independent_final")
    cfg = verify(root)["config"]
    dev, val = read(root, "development_analysis_v29.json"), read(root, "validation_analysis_v29.json")
    final = None
    if opening["final_opened"]:
        f = pd.read_parquet(root / OUT / "transfer_independent_final_v29.parquet")
        a = pd.read_parquet(root / OUT / "audit_independent_final_v29.parquet")
        forks = pd.read_parquet(root / OUT / "forks_independent_final_v29.parquet")
        profiles, merged = summarize(f, a, cfg)
        final = {"states": len(forks), "median_future_q": float(forks.future_q.median()), "future_eligible_fraction": float((forks.future_q >= cfg["branch_future_q_min"]).mean()), "REC_Conv": profiles["REC+Conv"], "reciprocal_pass": profiles["REC+Conv"]["reciprocal_pass"], "writeback_all": bool(a.writeback_pass.all()), "current_preserved_all": bool(a.recipient_current_unchanged.all()), "same_next_token_all": bool(a.next_token_identical.all())}
    signature = {role: pd.read_parquet(root / OUT / f"future_signature_{role}_v29.parquet") for role in ("development", "validation")}
    controls = {role: pd.read_parquet(root / OUT / f"natural_vs_random_{role}_v29.parquet") for role in ("development", "validation")}
    factorial = {role: pd.read_parquet(root / OUT / f"state_token_factorial_{role}_v29.parquet") for role in ("development", "validation")}
    horizons = {role: pd.read_parquet(root / OUT / f"write_horizon_{role}_v29.parquet") for role in ("development", "validation")}
    geometry = {role: pd.read_parquet(root / OUT / f"geometry_{role}_v29.parquet") for role in ("development", "validation")}
    diagnostic = read(root, "workspace_write_diagnostics_v29.json")
    realization = read(root, "write_effect_realization_v29.json")
    gate = bool(dev["branch_gate"] and val["branch_gate"] and dev["full_native_cache_identity_ceiling"] and val["full_native_cache_identity_ceiling"])
    partial = bool(gate and final and final["reciprocal_pass"] and final["writeback_all"])
    recurrent = bool(partial and dev["fixed_finalist_pass"] and val["fixed_finalist_pass"])
    compact = bool(realization["compact_write_effect_dimension_identified"])
    outcomes = {
        "V29_A_TOKEN_CONDITIONED_STATE_WRITE_CONFIRMED": bool(gate and final and final["future_eligible_fraction"] >= cfg["partial_success_fraction_min"]),
        "V29_B_SAME_BACKGROUND_WRITE_TRANSFER_CONFIRMED": partial,
        "V29_C_RECURRENT_WRITE_CARRIES_NONTRIVIAL_CONTENT": recurrent,
        "V29_D_KV_DOMINATED_TOKEN_CARRYOVER": bool(dev["profiles"]["KV"]["reciprocal_pass"] and val["profiles"]["KV"]["reciprocal_pass"] and not recurrent),
        "V29_E_DISTRIBUTED_WRITE_CONTENT": bool(not dev["profiles"]["Conv"]["reciprocal_pass"] and not val["profiles"]["Conv"]["reciprocal_pass"] and recurrent),
        "V29_F_STATE_DEPENDENT_WRITE_CODE": False,
        "V29_G_GLOBAL_WRITE_CODE": False,
        "V29_H_CURRENT_WORKSPACE_INCOMPLETE_FOR_WRITE": False,
        "V29_I_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED": compact,
        "V29_J_NO_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED": not compact,
        "V29_K_WRITE_CONTENT_REMAINS_UNIDENTIFIED": not partial,
    }
    state = {role: {"pairs": len(factorial[role]), "median_write_contrast_cosine": float(factorial[role].cross_state_write_contrast_cosine.median()), "median_cross_state_delta_to_native_cosine": float(factorial[role].cross_state_delta_cosine_to_native_same_state_effect.median()), "median_cross_state_delta_relative_l2": float(factorial[role].cross_state_delta_relative_l2.median())} for role in factorial}
    multi = {role: {c: {direction: {"median_cosine": median(signature[role], "donor_cosine", condition=c, direction=direction), "median_relative_l2": median(signature[role], "relative_l2_to_donor", condition=c, direction=direction)} for direction in ("A_from_B", "B_from_A")} for c in ("REC", "Conv", "KV", "REC+Conv", "REC+Conv+KV")} for role in signature}
    natural = {role: {c: {"median_cosine": median(controls[role], "donor_cosine", condition=c), "median_relative_l2": median(controls[role], "relative_l2_to_donor", condition=c), "median_realized_to_requested": median(controls[role], "realized_to_requested", condition=c)} for c in controls[role].condition.unique()} for role in controls}
    horizon = {role: {str(h): {"median_cosine": median(horizons[role], "donor_cosine", horizon=h), "median_magnitude": median(horizons[role], "magnitude_ratio", horizon=h), "median_relative_l2": median(horizons[role], "relative_l2_to_donor", horizon=h)} for h in (1, 2, 4)} for role in horizons}
    geom = {role: {ch: {"fields": len(x), "median_contrast_norm": float(x.contrast_norm.median()), "median_write_A_B_cosine": float(x.A_B_cosine.median()), "positive_contrast_fraction": float((x.contrast_norm > 0).mean())} for ch, x in geometry[role].groupby("channel")} for role in geometry}
    summary = {"formal_outcomes": outcomes, "development": {"states": dev["states"], "median_future_q": dev["median_future_q"], "profiles": dev["profiles"]}, "validation": {"states": val["states"], "median_future_q": val["median_future_q"], "profiles": val["profiles"]}, "independent_final": final, "full_cache_is_identity_ceiling_not_selective_carrier": True, "state_dependence": state, "state_code_adjudication": "inconclusive: no prospectively frozen cross-state success/degradation gate; off-manifold delta injection is diagnostic", "multi_probe": multi, "natural_vs_random": natural, "horizons": horizon, "write_geometry": geom, "workspace_write_prediction": diagnostic["models"], "write_geometry_from_current_J": diagnostic["write_geometry_from_current_J"], "workspace_incomplete_formal_reason": "prediction gain is diagnostic; no J-matched causal equivalence or valid direct J transplant", "realization": {"tested_k": realization["tested_k"], "not_estimable_k": realization["not_estimable_k"], "qualified_k": realization["qualified_k"], "train_span_rank": realization["train_span_rank"], "write_dimension": realization["write_dimension"]}, "H2_REMAINS": True, "H3_AUTHORIZED": False, "DYNAMIC_STATE_SEARCH_AUTHORIZED": False, "AUTONOMOUS_CONTROLLER_AUTHORIZED": False, "historical_final_opened": False, "V29_independent_final_opened": bool(opening["final_opened"]), "calibration_selected_layer_diagnostic_invalid_as_full_cache": True, "V29_formal_interface_all_32_layers": True}
    summary["ADJUDICATION_HASH"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = root / OUT / "v29_adjudication.json"
    write_json_atomic(path, summary)
    inputs = [SOURCE, str(path.relative_to(root)), "artifacts/natural_write_content_v29_realization.freeze.json", "artifacts/natural_write_content_v29_diagnostics.freeze.json", "artifacts/natural_write_content_v29_secondary_validation.freeze.json", "artifacts/natural_write_content_v29_final_opening.freeze.json"]
    if final:
        inputs.append("artifacts/natural_write_content_v29_forks_independent_final.freeze.json")
    fr = stage_freeze(root, "adjudication", inputs, {"ADJUDICATION_HASH": summary["ADJUDICATION_HASH"], "formal_outcomes": outcomes, "historical_final_opened": False, "H3_AUTHORIZED": False})
    return {"freeze_digest": fr["freeze_digest"], "formal_outcomes": outcomes, "final": final}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
