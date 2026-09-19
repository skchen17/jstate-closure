"""Terminal and within-case monotonicity analysis of V15 development control."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")


def main() -> None:
    root = Path.cwd()
    verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_control_terminal_analysis.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "control_terminal_analysis", ["scripts/analyze_control_v15.py", "results/v15/processed/control_development_v15.parquet"], {"role": "development_validation", "analysis": "terminal ratio includes stopped cases; monotonicity evaluated within each case across all recorded steps from initial ratio 1", "no_new_control_run": True})
    else:
        stage = json.loads(stage_path.read_text())
    frame = pd.read_parquet(root / OUT / "control_development_v15.parquet")
    rows = []
    for (base_id, method), group in frame.groupby(["base_trial_id", "method"]):
        group = group.sort_values("step")
        raw = [1.0, *group.raw_residual_ratio.astype(float).tolist()]
        weighted = [1.0, *group.weighted_residual_ratio.astype(float).tolist()]
        rows.append({"role": "development_validation", "freeze_digest": stage["freeze_digest"], "base_trial_id": base_id, "method": method, "steps_recorded": len(group), "accepted_steps": int(group.accepted.sum()), "terminal_raw_ratio": raw[-1], "terminal_weighted_ratio": weighted[-1], "raw_monotone_nonincreasing": all(b <= a + 1e-10 for a, b in zip(raw, raw[1:], strict=False)), "weighted_monotone_nonincreasing": all(b <= a + 1e-10 for a, b in zip(weighted, weighted[1:], strict=False)), "terminal_raw_improved": raw[-1] < 1.0 - 1e-8})
    path = root / OUT / "control_terminal_v15.parquet"
    result = pd.DataFrame(rows)
    result.to_parquet(path, index=False, compression="zstd")
    pooled = result.groupby("method").agg(case_count=("base_trial_id", "size"), terminal_raw_ratio=("terminal_raw_ratio", "median"), terminal_weighted_ratio=("terminal_weighted_ratio", "median"), raw_monotone_fraction=("raw_monotone_nonincreasing", "mean"), weighted_monotone_fraction=("weighted_monotone_nonincreasing", "mean"), terminal_improvement_fraction=("terminal_raw_improved", "mean"), median_accepted_steps=("accepted_steps", "median")).reset_index().to_dict("records")
    summary = {"role": "development_validation", "freeze_digest": stage["freeze_digest"], "records": str(path), "records_sha256": sha256_file(path), "pooled": pooled}
    write_json_atomic(root / OUT / "control_terminal_v15.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
