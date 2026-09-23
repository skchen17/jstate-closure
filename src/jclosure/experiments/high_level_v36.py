"""Frozen fresh-panel V36 REC conditional-effect gate before operator analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.high_level_v34 import summarize
from jclosure.protocol_v36 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v36/processed")
SOURCE = "src/jclosure/experiments/high_level_v36.py"


def run(root: Path, role: str):
    if role not in ("development", "validation", "independent_final"):
        raise ValueError(role)
    if role == "validation":
        verify_stage(root, "high_level_development")
    if role == "independent_final":
        verify_stage(root, "final_opening")
    cfg = verify(root)["config"]
    result = {"role": role, "models": {}, "thresholds_retuned": False,
              "same_semantic_state_ids": True}
    for key in ("Q", "F"):
        verify_stage(root, f"factorial_{key}_{role}")
        frame = pd.read_parquet(root / OUT / f"factorial_{key}_{role}_v36.parquet")
        result["models"][key] = summarize(frame, cfg)
    result["both_pass"] = all(x["pass"] for x in result["models"].values())
    result["formal_operator_authorized"] = bool(result["both_pass"] if role == "development" else
                                             result["both_pass"] and
                                             json.loads((root / OUT / "high_level_development_v36.json").read_text())["both_pass"])
    path = root / OUT / f"high_level_{role}_v36.json"
    write_json_atomic(path, result)
    seal = stage_freeze(root, f"high_level_{role}",
                        [SOURCE, str(path.relative_to(root)),
                         *(f"artifacts/computational_origin_v36_factorial_{key}_{role}.freeze.json" for key in ("Q", "F"))],
                        {"role": role, "summary_sha256": sha256_file(path),
                         "both_pass": result["both_pass"],
                         "formal_operator_authorized": result["formal_operator_authorized"]})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("development", "validation", "independent_final"))
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), args.role), indent=2))
