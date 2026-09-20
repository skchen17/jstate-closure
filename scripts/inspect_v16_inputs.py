"""Read-only shape inspection for V16 target alignment."""

from pathlib import Path

import numpy as np

path = Path("artifacts/causal/v13/causal_features_v13.npz")
with np.load(path, allow_pickle=False) as payload:
    for name in payload.files:
        if name.startswith("endpoint__") or name in ("base_trial_id", "family", "split", "layerwise"):
            print(name, payload[name].shape, payload[name].dtype)
