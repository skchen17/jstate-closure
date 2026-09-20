"""Run the already-frozen V19 train bank on GPU1 in reverse state order.

This changes scheduling only.  The state split, interventions, teacher tokens,
four-way estimand, and per-state output schema are identical to the GPU0 run.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.experiments import gpu1_v19, run_amended_v19
from jclosure.protocol_v19 import verify_stage


def main() -> None:
    if sys.argv[1:2] != ["train"]:
        raise RuntimeError("GPU1 reverse runtime is frozen for train only")
    amendment = verify_stage(Path.cwd(), "gpu1_reverse_train_amendment_7")
    runtime = verify_stage(Path.cwd(), "runtime_gpu1_amendment_4")
    if amendment["gpu1_runtime_parent_digest"] != runtime["freeze_digest"] or not runtime["equivalence_passed"]:
        raise RuntimeError("GPU1 runtime chain or equivalence mismatch")
    bank.load_context_v18 = gpu1_v19.load_context_gpu1
    original_run = bank.run

    def reverse_train(root: Path, role: str, limit: int | None = None) -> dict:
        if role != "train" or limit is not None:
            raise RuntimeError("GPU1 reverse wrapper only accepts the full train bank")
        original_verify = bank.verify_stage

        def reversed_split(path: Path, stage: str) -> dict:
            value = original_verify(path, stage)
            if stage == "splits":
                value = copy.deepcopy(value)
                value["train"] = list(reversed(value["train"]))
            return value

        bank.verify_stage = reversed_split
        try:
            return original_run(root, role)
        finally:
            bank.verify_stage = original_verify

    bank.run = reverse_train
    run_amended_v19.main()


if __name__ == "__main__":
    main()
