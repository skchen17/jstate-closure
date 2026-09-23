"""Bidirectional full read-cut and secondary Q2/Q3/Q4 replication gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/read_gate_v37.py"


def _gate(frame: pd.DataFrame, prefix: str, threshold: dict) -> dict:
    def stats(part):
        eligible = part[part.eligible_ratio]
        if eligible.empty:
            return {"states": len(part), "eligible": 0, "removed": None,
                    "restored": None, "cosine": None, "pass": False}
        result = {"states": len(part), "eligible": len(eligible),
                  "removed": float(eligible[f"{prefix}_removed_fraction"].median()),
                  "restored": float(eligible[f"{prefix}_restored_fraction"].median()),
                  "cosine": float(eligible[f"{prefix}_cosine"].median())}
        result["pass"] = bool(result["removed"] >= threshold["median_removed_min"] and
                              result["restored"] >= threshold["median_restored_min"] and
                              result["cosine"] >= threshold["median_cosine_min"])
        return result
    family = {name: stats(part) for name, part in frame.groupby("family")}
    overall = stats(frame)
    overall["family"] = family
    overall["families_passing"] = sum(value["pass"] for value in family.values())
    overall["pass"] = bool(overall["pass"] and overall["families_passing"] >= threshold["families_required"]
                           and frame.exact_interface_writeback.all())
    return overall


def run(root: Path, role: str) -> dict:
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    high = verify_stage(root, f"high_level_{role}")
    cfg = verify(root)["config"]
    result = {"role": role, "high_level_both_pass": high["both_pass"],
              "models": {}, "thresholds_retuned": False}
    for key in ("Q", "F"):
        verify_stage(root, f"read_baselines_{key}_{role}")
        frame = pd.read_parquet(root / OUT / f"read_baselines_{key}_{role}_v37.parquet")
        expected = {"development": 80, "validation": 40, "independent_final": 40}[role]
        if len(frame) != expected:
            raise RuntimeError(f"Incomplete full read run {key}:{role}")
        result["models"][key] = {
            "full_read": _gate(frame.rename(columns={"full_reverse_cosine": "full_cosine"}),
                               "full", cfg["full_read_gate"]),
            "q2_q3_q4_secondary": _gate(frame, "triple", cfg["q2_q3_q4_secondary_gate"]),
        }
    result["both_pass"] = bool(high["both_pass"] and
                               all(value["full_read"]["pass"] for value in result["models"].values()))
    result["secondary_triple_both_pass"] = bool(all(value["q2_q3_q4_secondary"]["pass"]
                                                 for value in result["models"].values()))
    path = root / OUT / f"read_gate_{role}_v37.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"read_gate_{role}", [SOURCE, str(path.relative_to(root)),
                        *[f"artifacts/computational_origin_v37_read_baselines_{key}_{role}.freeze.json"
                          for key in ("Q", "F")]],
                        {"role": role, "both_pass": result["both_pass"],
                         "secondary_triple_both_pass": result["secondary_triple_both_pass"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("development", "validation", "independent_final"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.role), indent=2))
