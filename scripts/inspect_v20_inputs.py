from pathlib import Path
import json
import pandas as pd

root = Path.cwd()
split = json.loads((root / "artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json").read_text())
bank = pd.read_parquet(root / "results/v16/processed/finite_action_bank_v16.parquet")
print("columns", bank.columns.tolist())
print("roles", bank.role.value_counts().to_dict())
print("designs", bank.design.value_counts().to_dict())
subset = bank[(bank.role == "train") & (bank.design == "single")]
for i, direction in enumerate(split["direction_indices"]):
    group = subset[subset.coordinate_index == i]
    print(i, direction, split["direction_families"][i], len(group),
          group.reliability_status.value_counts().to_dict(),
          round(float(group.calibration_alpha.median()), 3),
          round(float(group.j_effect_norm.median()), 4))
