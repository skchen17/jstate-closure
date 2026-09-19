"""Gate independent V15 confirmation without opening any final bank prematurely."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v15 import stage_freeze, verify


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    linearity = json.loads((root / "results/v15/processed/finite_response_linearity_v15.json").read_text())
    repeat = json.loads((root / "results/v15/processed/finite_response_repeat_v15.json").read_text())
    control = json.loads((root / "results/v15/processed/finite_causal_control_v15.json").read_text())
    fidelity = pd.DataFrame(control["fidelity"])
    h1 = fidelity[(fidelity.method == "actuator_calibrated_h1") & (fidelity.horizon == 1)]
    section = base["config"]["development"]
    h1_pass = bool(len(h1) == 1 and h1.iloc[0].direction >= section["h1_direction_min"] and section["h1_magnitude_min"] <= h1.iloc[0].magnitude <= section["h1_magnitude_max"] and h1.iloc[0].output >= section["h1_output_min"] and h1.iloc[0].semantic_continuous >= section["h1_semantic_continuous_min"])
    numerical = bool(linearity["FINITE_RESPONSE_LINEARITY_GATE"] and repeat["max_repeat_relative_l2"] <= 0.01)
    eligible = bool(numerical and h1_pass)
    if eligible:
        raise RuntimeError("Numerical and development gates passed; new independent bank design requires a separate frozen capture before selection. Do not reuse V13/V14 final data.")
    result = stage_freeze(root, "finalist_decision", [
        "scripts/freeze_v15_finalist_decision.py",
        "results/v15/processed/finite_response_linearity_v15.json",
        "results/v15/processed/finite_response_repeat_v15.json",
        "results/v15/processed/finite_causal_control_v15.json",
        "artifacts/quantization_aware_actuation_v15_splits.freeze.json",
    ], {
        "role": "development_gate_decision", "numerical_gate_pass": numerical,
        "finite_response_linearity_pass": bool(linearity["FINITE_RESPONSE_LINEARITY_GATE"]),
        "finite_response_repeatability_pass": bool(repeat["max_repeat_relative_l2"] <= 0.01),
        "development_h1_gate_pass": h1_pass,
        "finalist_status": "NO_ELIGIBLE_FINALIST",
        "independent_confirmatory_bank": "NOT_CREATED_OR_OPENED",
        "historical_semantic_gate_unchanged": True,
    })
    print(json.dumps({"freeze_digest": result["freeze_digest"], "finalist_status": result["finalist_status"], "numerical_gate_pass": numerical, "development_h1_gate_pass": h1_pass}, sort_keys=True))


if __name__ == "__main__":
    main()
