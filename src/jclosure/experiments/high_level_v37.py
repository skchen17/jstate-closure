"""State-level high-level gate for a prospectively disjoint revision panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.analyze_v32 import bootstrap_median_lower
from jclosure.protocol_v37 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


OUT = Path("results/v37/processed")
SOURCE = "src/jclosure/experiments/high_level_v37.py"


def summarize(frame: pd.DataFrame, cfg: dict) -> dict:
    gate = cfg["high_level_gate"]
    boot = cfg["bootstrap"]
    families = {}
    for family, part in frame.groupby("family"):
        families[family] = {"states": len(part),
                            "positive_fraction": float(part.improvement_positive.mean()),
                            "median_reduction": float(part.relative_conv_error_reduction.median())}
    positive = float(frame.improvement_positive.mean())
    reduction = float(frame.relative_conv_error_reduction.median())
    lower = bootstrap_median_lower(frame.relative_conv_error_reduction,
                                   boot["iterations"], boot["seed"], boot["lower_tail"])
    passing = sum(row["positive_fraction"] >= gate["positive_fraction_min"]
                  and row["median_reduction"] >= gate["median_donor_error_reduction_min"]
                  for row in families.values())
    passed = positive >= gate["positive_fraction_min"] and reduction >= gate["median_donor_error_reduction_min"]
    passed = passed and passing >= gate["families_required"] and lower > 0
    return {"states": len(frame), "positive_fraction": positive, "median_reduction": reduction,
            "bootstrap_lower_reduction": lower, "families_passing": passing,
            "families": families, "pass": bool(passed)}


def run(root: Path, role: str) -> dict:
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "high_level_development")
    if role == "independent_final":
        opening = verify_stage(root, "final_opening")
        if not opening["opened"]:
            raise RuntimeError("Independent final remains sealed")
    cfg = verify(root)["config"]
    result = {"role": role, "models": {}, "thresholds_retuned": False,
              "same_semantic_state_ids": True}
    for key in ("Q", "F"):
        verify_stage(root, f"factorial_{key}_{role}")
        frame = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v37.parquet")
        result["models"][key] = summarize(frame, cfg)
    result["both_pass"] = all(value["pass"] for value in result["models"].values())
    result["formal_mechanism_authorized"] = bool(result["both_pass"] if role == "development" else
                                                  result["both_pass"] and
                                                  json.loads((root / OUT / "high_level_development_v37.json").read_text())["both_pass"])
    path = root / OUT / f"high_level_{role}_v37.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"high_level_{role}", [SOURCE, str(path.relative_to(root)),
                        *[f"artifacts/computational_origin_v37_factorial_{key}_{role}.freeze.json"
                          for key in ("Q", "F")]],
                        {"role": role, "both_pass": result["both_pass"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("development", "validation", "independent_final"))
    a = p.parse_args()
    print(json.dumps(run(Path.cwd(), a.role), indent=2))
