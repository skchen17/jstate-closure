"""Run v10 decoded-causal validation with the frozen v8 task mapping."""

from __future__ import annotations

import json
from typing import Any

import jclosure.experiments.decoded_causal_v10 as base
from jclosure.datasets_v8 import load_tasks
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.persistent_state_v8 import FORMAL_DATA
from jclosure.protocol_v10 import verify_candidate_freeze
from jclosure.protocol_v10_causal_amendment import (
    PROTOCOL_V10_CAUSAL_AMENDMENT,
    build_causal_amendment_freeze,
    verify_causal_amendment_freeze,
)
from jclosure.provenance import write_json_atomic


def v8_task_mapping(context: Any) -> dict[str, Any]:
    return {
        task.example_id: task
        for _, task in load_tasks(context.root / FORMAL_DATA)
    }


def main() -> None:
    parser = standard_parser(
        "decoded compact-state causal validation v10 metadata amendment",
        "configs/causal_sufficiency_v10.yaml",
    )
    parser.add_argument("--stage", required=True, choices=("freeze", "causal"))
    args = parser.parse_args()
    context = initialize_context("decoded-causal-v10-amendment", args)
    try:
        if args.stage == "freeze":
            context.finish(
                "COMPLETED_V10_CAUSAL_AMENDMENT_FREEZE",
                freeze=build_causal_amendment_freeze(context.root, context.config),
            )
            return
        amendment = verify_causal_amendment_freeze(context.root, context.config)
        candidate = verify_candidate_freeze(context.root, context.config)
        tasks = v8_task_mapping(context)
        pair_rows = base._pair_metadata(context.root)
        selected = candidate["causal_confirmatory_base_trial_ids"]
        missing = [
            base_id
            for base_id in selected
            if str(pair_rows[base_id]["prompt_id"]) not in tasks
        ]
        if missing:
            raise RuntimeError(f"selected v8 prompt mapping is incomplete: {missing}")
        if args.dry_run:
            context.finish(
                "DRY_RUN",
                causal_amendment_freeze_digest=amendment["freeze_digest"],
                mapped_selected_pairs=len(selected),
            )
            return
        base._tasks = v8_task_mapping
        base._causal(context, candidate)
        summary = json.loads((context.root / base.CAUSAL_SUMMARY).read_text())
        summary["causal_metadata_amendment"] = {
            "protocol_version": PROTOCOL_V10_CAUSAL_AMENDMENT,
            "freeze_digest": amendment["freeze_digest"],
            "mapping_definition": amendment["mapping_definition"],
            "mapped_selected_pairs": len(selected),
        }
        write_json_atomic(context.root / base.CAUSAL_SUMMARY, summary)
        context.finish("COMPLETED_V10_CAUSAL_AMENDED", summary=summary)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
