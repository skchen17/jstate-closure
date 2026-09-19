"""Independent re-execution repeatability of finite writeback responses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jclosure.experiments import operator_v15 as operator
from jclosure.experiments.numerics_v14 import _panels
from jclosure.protocol_v15 import stage_freeze, verify
from jclosure.provenance import sha256_file, write_json_atomic

OUT = Path("results/v15/processed")


def main() -> None:
    root = Path.cwd()
    base = verify(root)
    stage_path = root / "artifacts/quantization_aware_actuation_v15_repeat.freeze.json"
    if not stage_path.exists():
        stage = stage_freeze(root, "repeat", ["src/jclosure/experiments/repeat_v15.py", "results/v15/processed/autograd_vs_actuator_v15.parquet"], {"role": "diagnostic_train", "directions": [0, 206], "epsilon": 1.0, "repeats": 2})
    else:
        stage = json.loads(stage_path.read_text())
    _, bundle, dense_map, metadata, values, measured, recurrent, attention = operator._setup(root)
    panel = _panels(root, int(base["config"]["diagnostic"]["anchors_per_family"]))
    rows = []
    for declaration in panel:
        cache, kwargs, baseline, _, _ = operator._anchor(root, declaration, bundle, dense_map, metadata, values, measured)
        for index in (0, 206):
            direction = {name: values["directions"][name][index] for name in ("recurrent", "conv", "kv")}
            responses = []
            for _ in range(2):
                _, _, response, plus_cache, minus_cache = operator._effects(cache, kwargs, baseline, direction, recurrent, attention, 1.0)
                responses.append(response)
                del plus_cache, minus_cache
            for target in operator.TARGETS:
                a, b = responses[0][target], responses[1][target]
                rows.append({"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "base_trial_id": str(declaration["base_trial_id"]), "direction_index": index, "target": target, "response_norm": float(np.linalg.norm(a)), "repeat_absolute_l2": float(np.linalg.norm(a - b)), "repeat_relative_l2": float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), 1e-20))})
    path = root / OUT / "finite_response_repeat_v15.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    summary = {"role": "diagnostic_train", "freeze_digest": stage["freeze_digest"], "records": str(path), "records_sha256": sha256_file(path), "max_repeat_relative_l2": float(pd.DataFrame(rows).repeat_relative_l2.max()), "max_repeat_absolute_l2": float(pd.DataFrame(rows).repeat_absolute_l2.max())}
    write_json_atomic(root / OUT / "finite_response_repeat_v15.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
