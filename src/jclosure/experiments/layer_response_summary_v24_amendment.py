"""Build the V24 response index with the audited natural-transition exclusions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jclosure.experiments.layer_response_v24 import OUT, SCRATCH
from jclosure.protocol_v24 import stage_freeze, verify_stage
from jclosure.provenance import sha256_file, write_json_atomic


SOURCE = "src/jclosure/experiments/layer_response_summary_v24_amendment.py"


def run(root: Path) -> dict:
    verify_stage(root, "natural_transition_amendment")
    design = json.loads((root / OUT / "bottleneck_design_v24.json").read_text())
    natural_audit = json.loads((root / OUT / "natural_transition_amendment_v24.json").read_text())
    rows = []
    for role in ("development", "validation"):
        for item in design[f"{role}_states"]:
            path = root / SCRATCH / f"finite_{role}" / f"layer_{item['base_trial_id']}.npz"
            if not path.exists():
                raise RuntimeError(f"missing {path}")
            rows.append({"kind": "finite", "role": role, **item, "path": str(path), "sha256": sha256_file(path)})
    for item in design["jvp_states"]:
        path = root / SCRATCH / "jvp" / f"layer_jvp_{item['base_trial_id']}.npz"
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        rows.append({"kind": "JVP", "role": "development", **item, "path": str(path), "sha256": sha256_file(path)})
    for item in natural_audit["included"]:
        path = Path(item["path"])
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        rows.append({"kind": "natural", "role": "development", **item, "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    index_path = root / OUT / "layer_response_index_v24.parquet"
    frame.to_parquet(index_path, index=False, compression="zstd")
    result = {
        "finite_development_bases": 50,
        "finite_validation_bases": 25,
        "finite_operator_states": 150,
        "jvp_base_states": 10,
        "jvp_operator_states": 20,
        "natural_transition_design_count": natural_audit["design_count"],
        "natural_transition_eligible_count": natural_audit["eligible_transition_count"],
        "natural_transition_excluded_count": natural_audit["excluded_count"],
        "index_rows": len(frame),
        "index_sha256": sha256_file(index_path),
        "historical_final_opened": False,
        "v24_independent_final_opened": False,
    }
    summary_path = root / OUT / "layer_response_summary_v24.json"
    write_json_atomic(summary_path, result)
    frozen = stage_freeze(
        root,
        "layer_response_summary_amendment",
        [SOURCE, str(index_path.relative_to(root)), str(summary_path.relative_to(root)),
         "results/v24/processed/natural_transition_amendment_v24.json"],
        result,
    )
    return {"freeze_digest": frozen["freeze_digest"], **result}


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
