"""Append-only binding of the V19 wrapper that resolves q and teacher amendments."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.protocol_v19 import stage_freeze, verify_stage

SOURCE = "src/jclosure/experiments/execution_binding_v19.py"


def run(root: Path):
    q = verify_stage(root, "q_amendment_2")
    teacher = verify_stage(root, "teacher_amendment_3")
    return stage_freeze(root, "execution_binding_amendment_6",
                        [SOURCE, "src/jclosure/experiments/run_amended_v19.py",
                         "artifacts/counterfactual_workspace_v19_q_amendment_2.freeze.json",
                         "artifacts/counterfactual_workspace_v19_teacher_amendment_3.freeze.json"],
                        {"why": "The V19 execution wrapper resolves the append-only q correction and frozen teacher manifest without modifying the initially frozen bank source; bind its unchanged executed bytes explicitly for long-term verification.",
                         "when": "after validation and diagnostic response completion while train factorial continues; wrapper bytes unchanged since before the first successful factorial response",
                         "responses_already_observed": {"factorial_train_state_records": len(list((bank.SCRATCH / "train").glob("factorial_*.parquet"))),
                                                        "factorial_validation_state_records": len(list((bank.SCRATCH / "validation").glob("factorial_*.parquet"))),
                                                        "sign_scale_completed": (root / bank.OUT / "q_sign_scale_v19.parquet").exists(),
                                                        "order_completed": (root / bank.OUT / "writeback_order_audit_v19.parquet").exists()},
                         "q_amendment_digest": q["freeze_digest"],
                         "teacher_amendment_digest": teacher["freeze_digest"],
                         "estimand_changed": False, "response_records_modified": False})


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
