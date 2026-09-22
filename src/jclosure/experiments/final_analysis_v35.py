"""Evaluate the pre-opened V35 mechanism on paired independent-final states."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.depth_analysis_v35 import _late, _pair, _progressive, _serial, stage_gate
from jclosure.protocol_v35 import stage_freeze, verify, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v35/processed")
SOURCE = "src/jclosure/experiments/final_analysis_v35.py"


def model_test(frame, serial, cfg, name):
    families = sorted(frame.family.unique())
    if name == "LATE_Q3_Q4":
        passed, values = _late(frame, cfg)
        family = {family: _late(frame, cfg, family)[0] for family in families}
        required = cfg["late_consolidation_gate"]["families_required"]
    elif name == "PROGRESSIVE_CUMULATIVE":
        passed, values = _progressive(frame, cfg)
        family = {family: _progressive(frame, cfg, family)[0] for family in families}
        required = cfg["progressive_gate"]["families_required"]
    elif name.startswith("SERIAL_"):
        early, late = name.removeprefix("SERIAL_").split("_TO_")
        passed, values = _serial(frame, serial, cfg, early, late)
        family = {family: _serial(frame, serial, cfg, early, late, family)[0] for family in families}
        required = cfg["serial_gate"]["families_required"]
    elif name.startswith(("COMPLEMENTARY_", "REDUNDANT_")):
        kind = "complementary" if name.startswith("COMPLEMENTARY_") else "redundancy"
        pair = name.split("_", 1)[1]
        passed, values = _pair(frame, cfg, pair, kind)
        family = {family: _pair(frame, cfg, pair, kind, family)[0] for family in families}
        required = cfg[f"{kind}_gate"]["families_required"]
    elif name.startswith("LOCALIZED_"):
        gate = stage_gate(frame, name.removeprefix("LOCALIZED_"), cfg["quartile_gate"])
        return {"pass": gate["pass"], "values": gate, "family_pass": {
            family: record["pass"] for family, record in gate["family"].items()}}
    else:
        raise RuntimeError(f"V35 unknown final mechanism: {name}")
    return {"pass": bool(passed and sum(family.values()) >= required),
            "values": values, "family_pass": family}


def run(root: Path):
    verify_stage(root, "final_opening")
    full = verify_stage(root, "full_gate_independent_final")
    opening = json.loads((root / OUT / "final_opening_v35.json").read_text())
    name = opening["selected_mechanism"]
    cfg = verify(root)["config"]
    result = {"selected_mechanism": name, "independent_final_opened": True,
              "models": {}, "thresholds_retuned": False,
              "historical_independent_finals_reopened": False}
    inputs = [SOURCE, "artifacts/hierarchical_read_v35_final_opening.freeze.json",
              "artifacts/hierarchical_read_v35_full_gate_independent_final.freeze.json"]
    for key in ("Q", "F"):
        if name == "FULL_DEPTH_ONLY":
            gate = json.loads((root / OUT / "full_gate_independent_final_v35.json").read_text())["models"][key]
            result["models"][key] = {"pass": gate["pass"], "values": gate,
                                      "family_pass": {family: value["pass"] for family, value in gate["family"].items()}}
        else:
            verify_stage(root, f"finalist_{key}")
            inputs.append(f"artifacts/hierarchical_read_v35_finalist_{key}.freeze.json")
            frame = pd.read_parquet(root / OUT / f"finalist_{key}_v35.parquet")
            if frame.state_id.nunique() != 40 or not frame.exact_writeback.all():
                raise RuntimeError(f"V35 finalist incomplete {key}")
            serial = (pd.read_parquet(root / OUT / f"finalist_serial_{key}_v35.parquet")
                      if name.startswith("SERIAL_") else pd.DataFrame())
            result["models"][key] = model_test(frame, serial, cfg, name)
    result["both_pass"] = bool(full["formal_depth_authorized"] and
                                all(value["pass"] for value in result["models"].values()))
    path = root / OUT / "final_analysis_v35.json"
    write_json_atomic(path, result)
    inputs.append(str(path.relative_to(root)))
    seal = stage_freeze(root, "final_analysis", inputs,
                        {"selected_mechanism": name, "both_pass": result["both_pass"],
                         "summary_sha256": sha256_file(path)})
    return {"freeze_digest": seal["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
