"""Run frozen V19 factorial using the append-only q-bank correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jclosure.experiments import counterfactual_bank_v19 as bank
from jclosure.protocol_v19 import verify_stage


def _verify(root: Path, name: str):
    if name == "interventions":
        amendment = verify_stage(root, "q_amendment_2")
        prior = verify_stage(root, "q_amendment_1")
        initial = verify_stage(root, "interventions")
        if (prior["prior_intervention_freeze_digest"] != initial["freeze_digest"]
                or amendment["prior_amendment_digest"] != prior["freeze_digest"]):
            raise RuntimeError("V19 q amendment parent mismatch")
        return amendment
    return verify_stage(root, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "validation", "aggregate_train", "aggregate_validation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    bank.verify_stage = _verify
    root = Path.cwd()
    teacher = verify_stage(root, "teacher_amendment_3")
    if teacher["prior_q_amendment_digest"] != verify_stage(root, "q_amendment_2")["freeze_digest"]:
        raise RuntimeError("V19 teacher amendment chain mismatch")
    teacher_frame = pd.read_parquet(root / teacher["teacher_path"])
    teacher_map = {str(row.base_trial_id): [int(x) for x in row.teacher_tokens]
                   for row in teacher_frame.itertuples()}
    original_metadata = bank._state_metadata

    def metadata_with_teacher(repo: Path, item: dict) -> dict:
        value = original_metadata(repo, item)
        value["teacher_tokens_h8_or_h1"] = teacher_map[str(item["base_trial_id"])]
        return value

    bank._state_metadata = metadata_with_teacher
    result = bank.aggregate(root, args.stage.split("_", 1)[1]) if args.stage.startswith("aggregate_") else bank.run(root, args.stage, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
