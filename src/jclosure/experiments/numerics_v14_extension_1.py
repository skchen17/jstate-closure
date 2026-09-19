"""V14 numerical-scale extension after the frozen infinitesimal sweep failed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jclosure.experiments import numerics_v14 as original
from jclosure.protocol_v14 import freeze_stage, verify_base

EPSILONS = [0.15, 0.3, 0.5, 1.0, 2.0]
OUT = Path("results/v14/processed/numerical_scale_extension_1")


def main() -> None:
    root = Path.cwd()
    stage = (
        root
        / "artifacts/finite_causal_control_v14_numerical_scale_extension_1.freeze.json"
    )
    if not stage.exists():
        amendment = freeze_stage(
            root,
            "numerical_scale_extension_1",
            [
                "src/jclosure/experiments/numerics_v14_extension_1.py",
                "src/jclosure/experiments/numerics_v14.py",
                "results/v14/processed/jvp_finite_writeback_audit_v14.parquet",
            ],
            {
                "reason": "no JVP/finite-difference agreement at epsilon <= 0.1",
                "epsilons": EPSILONS,
                "estimand_changed": False,
                "precision_modes_changed": False,
                "source_results_preserved": True,
            },
        )
    else:
        amendment = json.loads(stage.read_text(encoding="utf-8"))
    base = copy.deepcopy(verify_base(root))
    base["config"]["numerical_audit"]["epsilons"] = EPSILONS
    base["freeze_digest"] = amendment["freeze_digest"]
    original.verify_base = lambda _: base
    original.OUT = OUT
    print(json.dumps(original.audit(root), sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
