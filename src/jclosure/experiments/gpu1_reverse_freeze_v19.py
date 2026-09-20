"""Append-only V19 scheduling amendment for an idle equivalent GPU1."""

from __future__ import annotations

import json
from pathlib import Path

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.protocol_v19 import stage_freeze, verify_stage


def main() -> None:
    root = Path.cwd()
    runtime = verify_stage(root, "runtime_gpu1_amendment_4")
    binding = verify_stage(root, "execution_binding_amendment_6")
    split = verify_stage(root, "splits")
    train_observed = len(list((bank.SCRATCH / "train").glob("factorial_*.parquet")))
    val_observed = len(list((bank.SCRATCH / "validation").glob("factorial_*.parquet")))
    value = stage_freeze(
        root,
        "gpu1_reverse_train_amendment_7",
        ["src/jclosure/experiments/gpu1_reverse_train_v19.py",
         "src/jclosure/experiments/gpu1_reverse_freeze_v19.py",
         "src/jclosure/experiments/gpu1_v19.py",
         "src/jclosure/experiments/run_amended_v19.py",
         "artifacts/counterfactual_workspace_v19_runtime_gpu1_amendment_4.freeze.json",
         "artifacts/counterfactual_workspace_v19_execution_binding_amendment_6.freeze.json",
         "artifacts/counterfactual_workspace_v19_splits.freeze.json"],
        {"why": "Run the remaining immutable train-state factorials on idle GPU1 in reverse order to reduce wall time; no protocol, split, q, action, teacher or estimand change.",
         "responses_already_observed": {"train_states": train_observed,
                                        "validation_states": val_observed,
                                        "independent_final_states": 0},
         "gpu1_runtime_parent_digest": runtime["freeze_digest"],
         "binding_parent_digest": binding["freeze_digest"],
         "split_parent_digest": split["freeze_digest"],
         "schedule": "GPU0 ascending, GPU1 descending; stop GPU1 before any unfinished-state overlap; existing per-state parquet skipped",
         "state_order_changes_estimand": False,
         "hash_alg": "sha256"},
    )
    print(json.dumps({"freeze_digest": value["freeze_digest"],
                      "responses_already_observed": value["responses_already_observed"]}, indent=2))


if __name__ == "__main__":
    main()
