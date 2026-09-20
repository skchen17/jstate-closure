"""Append-only GPU1 placement for the frozen small V19 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments import diagnostics_v19
from jclosure.experiments.gpu1_v19 import load_context_gpu1
from jclosure.protocol_v19 import stage_freeze, verify_stage

SOURCE = "src/jclosure/experiments/diagnostic_gpu1_v19.py"


def prepare(root: Path):
    runtime = verify_stage(root, "runtime_gpu1_amendment_4")
    diagnostics = verify_stage(root, "diagnostics")
    if not runtime["equivalence_passed"]:
        raise RuntimeError("V19 GPU1 equivalence not established")
    return stage_freeze(root, "diagnostic_gpu1_amendment_5",
                        [SOURCE, "artifacts/counterfactual_workspace_v19_runtime_gpu1_amendment_4.freeze.json",
                         "artifacts/counterfactual_workspace_v19_diagnostics.freeze.json"],
                        {"why": "Reuse zero-error GPU0/GPU1 four-endpoint numerical equivalence for frozen small sign/scale and order diagnostics while GPU0 completes the larger train bank.",
                         "when": "after GPU1 validation bank completion and before V19 sign/scale or order diagnostic responses",
                         "responses_already_observed": {"factorial_train_state_records": len(list((bank.SCRATCH / "train").glob("factorial_*.parquet"))),
                                                        "factorial_validation_state_records": len(list((bank.SCRATCH / "validation").glob("factorial_*.parquet"))),
                                                        "sign_scale_rows": 0, "order_rows": 0},
                         "prior_runtime_digest": runtime["freeze_digest"],
                         "diagnostic_freeze_digest": diagnostics["freeze_digest"],
                         "weights_changed": False, "runtime_device_map": {"": 1},
                         "equivalence_evidence": runtime["comparison"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "sign_scale", "order"))
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == "prepare":
        result = prepare(root)
    else:
        amendment = verify_stage(root, "diagnostic_gpu1_amendment_5")
        if amendment["prior_runtime_digest"] != verify_stage(root, "runtime_gpu1_amendment_4")["freeze_digest"]:
            raise RuntimeError("V19 diagnostic GPU1 amendment chain mismatch")
        bank.load_context_v18 = load_context_gpu1
        result = {"sign_scale": diagnostics_v19.sign_scale, "order": diagnostics_v19.order}[args.stage](root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
