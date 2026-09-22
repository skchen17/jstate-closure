"""Gate hierarchical V35 work on fresh high-level and full-read replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/full_gate_v35.py"


def model_gate(frame, threshold):
    valid = frame[frame.status == "VALID"]
    family = {}
    for name, part in frame.groupby("family"):
        good = part[part.status == "VALID"]
        values = {"removed": float(good.removed_fraction.median()) if len(good) else None,
                  "restored": float(good.restored_fraction.median()) if len(good) else None,
                  "cosine": float(good.reverse_correction_cosine.median()) if len(good) else None}
        passed = (values["removed"] is not None and
                  values["removed"] >= threshold["median_removed_min"] and
                  values["restored"] >= threshold["median_restored_min"] and
                  values["cosine"] >= threshold["median_cosine_min"])
        family[name] = {"rows": len(part), "valid_rows": len(good), **values, "pass": bool(passed)}
    values = {"removed": float(valid.removed_fraction.median()) if len(valid) else None,
              "restored": float(valid.restored_fraction.median()) if len(valid) else None,
              "cosine": float(valid.reverse_correction_cosine.median()) if len(valid) else None}
    family_pass = sum(x["pass"] for x in family.values())
    passed = (values["removed"] is not None and bool(frame.exact_writeback.all()) and
              bool(frame.recipient_native_KV.all()) and bool(frame.six_frozen_probes.all()) and
              values["removed"] >= threshold["median_removed_min"] and
              values["restored"] >= threshold["median_restored_min"] and
              values["cosine"] >= threshold["median_cosine_min"] and
              family_pass >= threshold["families_required"])
    return {"rows": len(frame), "valid_rows": len(valid), **values,
            "families_passing": family_pass, "family": family, "pass": bool(passed)}


def run(root: Path, role: str):
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    high = verify_stage(root, f"high_level_{role}")
    threshold = verify(root)["config"]["full_read_gate"]
    result = {"role": role, "high_level_pass": high["formal_depth_authorized"],
              "models": {}, "thresholds_retuned": False}
    for key in ("Q", "F"):
        verify_stage(root, f"full_read_{key}_{role}")
        frame = pd.read_parquet(root / OUT / f"full_read_{key}_{role}_v35.parquet")
        expected = {"development": 80, "validation": 40, "independent_final": 40}[role]
        if len(frame) != expected:
            raise RuntimeError(f"V35 incomplete full depth {key}:{role}")
        result["models"][key] = model_gate(frame, threshold)
    result["both_pass"] = bool(result["high_level_pass"] and all(x["pass"] for x in result["models"].values()))
    result["formal_depth_authorized"] = bool(result["both_pass"] if role == "development" else
                                              result["both_pass"] and
                                              json.loads((root / OUT / "full_gate_development_v35.json").read_text())["both_pass"])
    path = root / OUT / f"full_gate_{role}_v35.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"full_gate_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         f"artifacts/hierarchical_read_v35_high_level_{role}.freeze.json",
                         *(f"artifacts/hierarchical_read_v35_full_read_{key}_{role}.freeze.json" for key in ("Q", "F"))],
                        {"role": role, "both_pass": result["both_pass"],
                         "formal_depth_authorized": result["formal_depth_authorized"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
