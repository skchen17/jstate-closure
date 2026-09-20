from pathlib import Path
import json

import numpy as np
import pandas as pd

root = Path("/data/CSK/J-space-project/v19-counterfactual-work")
repo = Path("/data/CSK/J-space-project/jstate-closure")
split = json.loads((repo / "artifacts/counterfactual_workspace_v19_splits.freeze.json").read_text())
train_ids = [x["base_trial_id"] for x in split["train"]]
seen = {x.name.removeprefix("factorial_").removesuffix(".parquet") for x in (root / "train").glob("factorial_*.parquet")}
prefix = next((i for i, key in enumerate(train_ids) if key not in seen), len(train_ids))
suffix = next((i for i, key in enumerate(reversed(train_ids)) if key not in seen), len(train_ids))
print("train contiguous prefix/suffix/gap", prefix, suffix, len(train_ids)-prefix-suffix)
for role in ("train", "validation"):
    paths = sorted((root / role).glob("factorial_*.parquet"))
    print(role, "states", len(paths))
    if not paths:
        continue
    parts = [pd.read_parquet(path) for path in paths[: min(len(paths), 30)]]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame[(frame.horizon == 1) & frame.q_reliable]
    matched = frame[frame.action_status == "MATCHED_REALIZED_ACTION"]
    print("sample states", matched.base_trial_id.nunique(), "rows", len(frame),
          "match rate", len(matched) / max(len(frame), 1))
    if len(matched):
        y = {key: np.stack(matched[key]).astype(float) for key in
             ("y00_stack", "y01_stack", "y10_stack", "y11_stack")}
        n = y["y10_stack"] - y["y00_stack"]
        r0 = y["y01_stack"] - y["y00_stack"]
        rq = y["y11_stack"] - y["y10_stack"]
        m = rq - r0
        ratio = np.linalg.norm(m, axis=1) / np.maximum(np.maximum(np.linalg.norm(r0, axis=1), np.linalg.norm(rq, axis=1)), 1e-8)
        natural = np.linalg.norm(n, axis=1) / np.maximum(np.linalg.norm(r0, axis=1), 1e-8)
        print("sample M/R median", np.median(ratio), "N/R median", np.median(natural))
