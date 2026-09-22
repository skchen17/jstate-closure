"""Frozen V34 per-role bidirectional mediation gates and cross-model profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.protocol_v34 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v34/processed")
SOURCE = "src/jclosure/experiments/stage_analysis_v34.py"
STAGES = ("TRUE_UPDATE", "TRANSFORMED_CONTROL", "POSTCONV_INPUT", "RECURRENT_READ", "RESIDUAL_INTEGRATION")


def _stats(frame: pd.DataFrame, gate: dict) -> dict:
    good = frame[frame.status == "VALID"]
    family = {}
    for name, part in frame.groupby("family"):
        valid = part[part.status == "VALID"]
        medians = {col: float(valid[col].median()) if len(valid) else None for col in
                   ("removed_fraction", "restored_fraction", "reverse_correction_cosine")}
        passed = (len(valid) > 0 and bool(part.exact_writeback.all()) and
                  medians["removed_fraction"] is not None and
                  medians["removed_fraction"] >= gate["family_median_removed_min"] and
                  medians["restored_fraction"] >= gate["family_median_restored_min"] and
                  medians["reverse_correction_cosine"] >= gate["family_median_reverse_cosine_min"])
        family[name] = {"rows": len(part), "valid_rows": len(valid), **medians, "pass": bool(passed)}
    medians = {col: float(good[col].median()) if len(good) else None for col in
               ("removed_fraction", "restored_fraction", "reverse_correction_cosine")}
    family_pass = sum(x["pass"] for x in family.values())
    passed = (len(good) > 0 and bool(frame.exact_writeback.all()) and
              bool(frame.recipient_native_KV.all()) and bool(frame.six_frozen_probes.all()) and
              medians["removed_fraction"] is not None and
              medians["removed_fraction"] >= gate["median_removed_min"] and
              medians["restored_fraction"] >= gate["median_restored_min"] and
              medians["reverse_correction_cosine"] >= gate["median_reverse_correction_cosine_min"] and
              family_pass >= gate["families_required"])
    return {"rows": len(frame), "valid_rows": len(good), **medians,
            "families_passing": family_pass, "family": family, "pass": bool(passed)}


def run(root: Path, role: str) -> dict:
    if role not in ("development", "validation"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "stage_analysis_development")
    high = verify_stage(root, f"high_level_{role}")
    high_json = json.loads((root / OUT / f"high_level_{role}_v34.json").read_text())
    if not high["formal_mediation_authorized"] or not high_json["both_pass"]:
        raise RuntimeError("V34 high-level gate failed")
    gate = verify(root)["config"]["stage_gate"]
    result = {"role": role, "high_level_reconfirmed": True, "models": {},
              "stage_order": list(STAGES), "stage5_alias_not_independently_identifiable": True,
              "same_semantic_states": True, "thresholds_retuned": False}
    inputs = [SOURCE, f"artifacts/functional_mediation_v34_high_level_{role}.freeze.json"]
    for key in ("Q", "F"):
        verify_stage(root, f"mediation_{key}_{role}")
        inputs.append(f"artifacts/functional_mediation_v34_mediation_{key}_{role}.freeze.json")
        path = root / OUT / f"mediation_{key}_{role}_v34.parquet"
        frame = pd.read_parquet(path)
        expected_states = 80 if role == "development" else 40
        if frame.state_id.nunique() != expected_states or len(frame) != expected_states * 4:
            raise RuntimeError(f"V34 incomplete mediation {key}:{role}")
        model = {}
        for stage in STAGES:
            if stage == "RESIDUAL_INTEGRATION":
                model[stage] = {"status": "ALIASED_NOT_SEPARATELY_IDENTIFIABLE", "pass": False,
                                "source_stage": "RECURRENT_READ"}
            else:
                model[stage] = {"status": "TESTED", **_stats(frame[frame.stage == stage], gate)}
        result["models"][key] = model
    result["shared_stage_passes_this_role"] = [s for s in STAGES if
        result["models"]["Q"][s]["pass"] and result["models"]["F"][s]["pass"]]
    result["model_stage_passes_this_role"] = {key: [s for s in STAGES if
        result["models"][key][s]["pass"]] for key in ("Q", "F")}
    result["pipeline_testing_authorized"] = not bool(result["shared_stage_passes_this_role"])
    if role == "validation":
        dev = json.loads((root / OUT / "stage_analysis_development_v34.json").read_text())
        result["shared_stage_passes_development_and_validation"] = [s for s in STAGES if
            s in dev["shared_stage_passes_this_role"] and s in result["shared_stage_passes_this_role"]]
        result["model_stage_passes_both_roles"] = {key: [s for s in STAGES if
            s in dev["model_stage_passes_this_role"][key] and s in result["model_stage_passes_this_role"][key]]
            for key in ("Q", "F")}
        inputs.append("artifacts/functional_mediation_v34_stage_analysis_development.freeze.json")
    path = root / OUT / f"stage_analysis_{role}_v34.json"
    write_json_atomic(path, result)
    inputs.append(str(path.relative_to(root)))
    seal = stage_freeze(root, f"stage_analysis_{role}", inputs,
                        {"role": role, "result_sha256": sha256_file(path),
                         "shared_stage_passes_this_role": result["shared_stage_passes_this_role"]})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
