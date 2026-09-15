"""Difficulty calibration and formal data generation for protocol v8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.datasets_v8 import FAMILIES, load_tasks, write_calibration, write_formal
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic

CALIBRATION_DATA = Path("data/v8/difficulty_calibration_r6.json")
FORMAL_DATA = Path("data/v8/persistent_causal_formal.json")
CALIBRATION_SUMMARY = Path("results/v8/processed/difficulty_calibration_v8_r6.json")


@torch.no_grad()
def teacher_actions(bundle: Any, task: Any) -> tuple[list[str], str | None]:
    from jclosure.runtime_v3_1 import encode_direct_prompt

    input_ids = encode_direct_prompt(bundle, task.prompt)
    maximum = 2 * int(task.horizon) + 8
    generated = bundle.hf_model.generate(
        input_ids=input_ids,
        max_new_tokens=maximum,
        do_sample=False,
        use_cache=True,
        pad_token_id=int(bundle.tokenizer.eos_token_id),
    )[0, input_ids.shape[1] :]
    allowed = (
        {str(value) for value in range(32)}
        if all(str(value).isdigit() for value in task.semantic_actions)
        else set("ABCDEFGHIJKL")
    )
    actions: list[str] = []
    error = None
    specials = set(getattr(bundle.tokenizer, "all_special_ids", ()) or ())
    for token in generated.tolist():
        if int(token) in specials:
            error = "special_token_before_complete"
            break
        surface = bundle.tokenizer.decode(
            [int(token)], skip_special_tokens=True
        ).strip()
        if not surface:
            continue
        if surface not in allowed:
            error = "nonsemantic_generated_token"
            break
        actions.append(surface)
        if len(actions) == task.horizon:
            break
    if len(actions) != task.horizon and error is None:
        error = "incomplete_trajectory"
    return actions, error


def _generate(context: Any) -> None:
    path = context.root / CALIBRATION_DATA
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = write_calibration(context.root, context.config, path)
    context.finish(
        "COMPLETED_CALIBRATION_GENERATION",
        data=str(path.relative_to(context.root)),
        data_sha256=sha256_file(path),
        item_count=len(payload["items"]),
    )


def _calibrate(context: Any, bundle: Any, limit: int | None) -> None:
    items = load_tasks(context.root / CALIBRATION_DATA)
    if limit is not None:
        items = items[: int(limit)]
    rows: list[dict[str, Any]] = []
    progress = context.raw_dir / context.run_id / "calibration_progress.json"
    for index, (_, task) in enumerate(items):
        actions, error = teacher_actions(bundle, task)
        row = {
            "schema_version": 11,
            "protocol_version": "structured_persistent_state_protocol_v8",
            "record_type": "difficulty_calibration_trial",
            "run_id": context.run_id,
            "prompt_id": task.example_id,
            "family": task.family,
            "variant": task.variant,
            "horizon": task.horizon,
            "attempted": True,
            "parseable": error is None and len(actions) == task.horizon,
            "full_trajectory_correct": tuple(actions) == task.semantic_actions,
            "final_answer_correct": bool(actions and actions[-1] == task.final_answer),
            "generated_actions": actions,
            "expected_actions": list(task.semantic_actions),
            "error": error,
        }
        rows.append(row)
        if index % 25 == 0 or index + 1 == len(items):
            write_json_atomic(
                progress,
                {"status": "RUNNING", "completed": index + 1, "total": len(items)},
            )
    raw = context.raw_dir / context.run_id / "difficulty_calibration_v8.jsonl"
    append_jsonl(raw, rows)
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["family", "horizon"], sort=True)
        .agg(
            attempted=("attempted", "sum"),
            parseable=("parseable", "sum"),
            full_trajectory_correct=("full_trajectory_correct", "sum"),
            final_answer_correct=("final_answer_correct", "sum"),
        )
        .reset_index()
    )
    grouped["parseable_rate"] = grouped["parseable"] / grouped["attempted"]
    grouped["full_trajectory_accuracy"] = (
        grouped["full_trajectory_correct"] / grouped["attempted"]
    )
    grouped["final_answer_accuracy"] = (
        grouped["final_answer_correct"] / grouped["attempted"]
    )
    section = context.config["persistent_state_v8"]["calibration"]
    preferred = float(section["preferred_accuracy"])
    minimum = float(section["minimum_accuracy"])
    selected: dict[str, int] = {}
    for family in FAMILIES:
        family_rows = grouped[grouped["family"] == family]
        eligible = family_rows[family_rows["full_trajectory_accuracy"] >= preferred]
        if eligible.empty:
            eligible = family_rows[family_rows["full_trajectory_accuracy"] >= minimum]
        if not eligible.empty:
            selected[family] = int(eligible["horizon"].max())
    summary = {
        "schema_version": 11,
        "protocol_version": "structured_persistent_state_protocol_v8",
        "run_id": context.run_id,
        "data": str(CALIBRATION_DATA),
        "data_sha256": sha256_file(context.root / CALIBRATION_DATA),
        "records": str(raw.relative_to(context.root)),
        "rows": grouped.to_dict("records"),
        "selected_horizons": selected,
        "all_families_authorized": set(selected) == set(FAMILIES),
        "minimum_accuracy": minimum,
        "preferred_accuracy": preferred,
    }
    write_json_atomic(context.root / CALIBRATION_SUMMARY, summary)
    write_json_atomic(
        progress,
        {"status": "COMPLETED", "completed": len(items), "total": len(items)},
    )
    context.finish("COMPLETED_DIFFICULTY_CALIBRATION", summary=summary)


def _formal(context: Any) -> None:
    summary = json.loads((context.root / CALIBRATION_SUMMARY).read_text())
    if not summary["all_families_authorized"]:
        context.finish(
            "GATED_TEACHER_ACCURACY",
            selected_horizons=summary["selected_horizons"],
        )
        return
    path = context.root / FORMAL_DATA
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_horizons = {
        str(key): int(value) for key, value in summary["selected_horizons"].items()
    }
    selected_horizons.update(
        {
            str(key): int(value)
            for key, value in context.config["persistent_state_v8"]["formal"]
            .get("horizon_overrides", {})
            .items()
        }
    )
    payload = write_formal(
        context.root,
        context.config,
        selected_horizons,
        context.root / CALIBRATION_DATA,
        path,
    )
    counts = pd.DataFrame(payload["items"]).groupby(["split", "family"]).size()
    context.finish(
        "COMPLETED_FORMAL_GENERATION",
        data=str(path.relative_to(context.root)),
        data_sha256=sha256_file(path),
        item_count=len(payload["items"]),
        counts=[
            {"split": split, "family": family, "count": int(value)}
            for (split, family), value in counts.items()
        ],
    )


def main() -> None:
    parser = standard_parser(
        "protocol-v8 task calibration and generation",
        "configs/persistent_state_v8.yaml",
    )
    parser.add_argument(
        "--stage", choices=("generate", "calibrate", "formal"), required=True
    )
    args = parser.parse_args()
    context = initialize_context("calibrate-v8", args)
    try:
        if args.stage == "generate":
            _generate(context)
        elif args.stage == "formal":
            _formal(context)
        elif args.dry_run:
            context.finish("DRY_RUN")
        else:
            from jclosure.model import load_model_bundle

            _calibrate(context, load_model_bundle(context.config), args.limit)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
