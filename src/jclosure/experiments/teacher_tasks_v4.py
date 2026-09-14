"""Calibrate teacher competence, freeze formal tasks, and audit formal splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jclosure.datasets_v4 import (
    PROTOCOL_V4,
    ProgramTraceTask,
    advance_environment_state,
    environment_step_prompt,
    generate_program_tasks,
    initial_environment_state,
    verify_disjoint_domains,
)
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.model import load_model_bundle
from jclosure.protocol_v4 import build_program_freeze, verify_program_freeze
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.records_v4 import TeacherCompetenceRecord
from jclosure.runtime_v3_1 import encode_direct_prompt


def _semantic_universe(task: ProgramTraceTask) -> set[str]:
    if all(value.isdigit() for value in task.semantic_actions):
        return set("0123456789")
    return set("ABCDEF")


def parse_semantic_generation(
    tokenizer: Any,
    token_ids: list[int],
    task: ProgramTraceTask,
) -> tuple[tuple[str, ...], bool, str | None]:
    """Parse exactly one-character task actions while allowing whitespace tokens."""

    actions: list[str] = []
    allowed = _semantic_universe(task)
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for token_id in token_ids:
        if int(token_id) in special:
            if len(actions) < task.horizon:
                return tuple(actions), False, "special_token_before_complete"
            break
        surface = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        pieces = surface.split()
        for piece in pieces:
            if len(piece) != 1 or piece not in allowed:
                return tuple(actions), False, "nonsemantic_generated_token"
            actions.append(piece)
            if len(actions) == task.horizon:
                return tuple(actions), True, None
    return tuple(actions), False, "trajectory_length_mismatch"


def _padded_inputs(
    bundle: Any, prompts: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = bundle.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad nor EOS token")
    encoded = [encode_direct_prompt(bundle, prompt)[0] for prompt in prompts]
    maximum_prompt = max(int(value.numel()) for value in encoded)
    device = next(bundle.hf_model.parameters()).device
    input_ids = torch.full(
        (len(prompts), maximum_prompt), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, value in enumerate(encoded):
        length = int(value.numel())
        input_ids[row, -length:] = value.to(device)
        attention_mask[row, -length:] = 1
    return input_ids, attention_mask


def _first_semantic_action(
    tokenizer: Any, token_ids: list[int], task: ProgramTraceTask
) -> tuple[str | None, str | None]:
    allowed = _semantic_universe(task)
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for token_id in token_ids:
        if int(token_id) in special:
            return None, "special_token_before_action"
        surface = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        pieces = surface.split()
        if not pieces:
            continue
        if len(pieces) != 1 or len(pieces[0]) != 1 or pieces[0] not in allowed:
            return None, "nonsemantic_generated_token"
        return pieces[0], None
    return None, "missing_semantic_action"


@torch.no_grad()
def generate_teacher_stepwise_records(
    bundle: Any,
    tasks: list[ProgramTraceTask],
    *,
    domain: str,
    run_id: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Autonomously roll the teacher through one-token environment macrosteps."""

    tokenizer = bundle.tokenizer
    records: list[dict[str, Any]] = []
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset : offset + batch_size]
        states = [initial_environment_state(task) for task in batch]
        actions: list[list[str]] = [[] for _ in batch]
        token_traces: list[list[int]] = [[] for _ in batch]
        prompts_by_task: list[list[str]] = [[] for _ in batch]
        errors: list[str | None] = [None for _ in batch]
        for step_index in range(max(task.horizon for task in batch)):
            active = [
                index
                for index, task in enumerate(batch)
                if step_index < task.horizon and errors[index] is None
            ]
            if not active:
                break
            prompts = [
                environment_step_prompt(batch[index], step_index, states[index])
                for index in active
            ]
            for index, prompt in zip(active, prompts, strict=True):
                prompts_by_task[index].append(prompt)
            input_ids, attention_mask = _padded_inputs(bundle, prompts)
            generated = bundle.hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                pad_token_id=int(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, input_ids.shape[1] :].detach().cpu().tolist()
            for index, token_ids in zip(active, continuation, strict=True):
                action, error = _first_semantic_action(
                    tokenizer, token_ids, batch[index]
                )
                token_traces[index].extend(int(value) for value in token_ids)
                if error is not None or action is None:
                    errors[index] = error
                    continue
                actions[index].append(action)
                states[index] = advance_environment_state(
                    batch[index], action, states[index]
                )
        for index, task in enumerate(batch):
            generated_actions = tuple(actions[index])
            parseable = len(generated_actions) == task.horizon and errors[index] is None
            full_correct = bool(
                parseable and generated_actions == task.semantic_actions
            )
            final_correct = bool(
                generated_actions and generated_actions[-1] == task.final_answer
            )
            record = TeacherCompetenceRecord(
                run_id=run_id,
                domain=domain,
                example_id=task.example_id,
                family=task.family,
                variant=task.variant,
                horizon=task.horizon,
                parseable=parseable,
                full_trajectory_correct=full_correct,
                final_answer_correct=final_correct,
                expected_actions=task.semantic_actions,
                generated_actions=generated_actions,
                error=errors[index],
            ).to_dict()
            record.update(
                {
                    "template_id": task.template_id,
                    "prompt": task.prompt,
                    "program_hash": task.program_hash,
                    "generator_seed": task.generator_seed,
                    "generator_index": task.generator_index,
                    "generated_token_ids": token_traces[index],
                    "step_prompts": prompts_by_task[index],
                    "evaluation_mode": "environment_stepwise",
                }
            )
            records.append(record)
    return records


@torch.no_grad()
def generate_teacher_records(
    bundle: Any,
    tasks: list[ProgramTraceTask],
    *,
    domain: str,
    run_id: str,
    batch_size: int,
    maximum_extra_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tokenizer = bundle.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad nor EOS token")
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset : offset + batch_size]
        encoded = [encode_direct_prompt(bundle, task.prompt)[0] for task in batch]
        maximum_prompt = max(int(value.numel()) for value in encoded)
        device = next(bundle.hf_model.parameters()).device
        input_ids = torch.full(
            (len(batch), maximum_prompt), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, value in enumerate(encoded):
            length = int(value.numel())
            input_ids[row, -length:] = value.to(device)
            attention_mask[row, -length:] = 1
        maximum_horizon = max(task.horizon for task in batch)
        generated = bundle.hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2 * maximum_horizon + int(maximum_extra_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=int(pad_id),
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = generated[:, maximum_prompt:].detach().cpu().tolist()
        for task, token_ids in zip(batch, continuation, strict=True):
            actions, parseable, error = parse_semantic_generation(
                tokenizer, [int(value) for value in token_ids], task
            )
            expected = task.semantic_actions
            full_correct = bool(parseable and actions == expected)
            final_correct = bool(actions and actions[-1] == task.final_answer)
            record = TeacherCompetenceRecord(
                run_id=run_id,
                domain=domain,
                example_id=task.example_id,
                family=task.family,
                variant=task.variant,
                horizon=task.horizon,
                parseable=parseable,
                full_trajectory_correct=full_correct,
                final_answer_correct=final_correct,
                expected_actions=expected,
                generated_actions=actions,
                error=error,
            ).to_dict()
            record.update(
                {
                    "template_id": task.template_id,
                    "prompt": task.prompt,
                    "program_hash": task.program_hash,
                    "generator_seed": task.generator_seed,
                    "generator_index": task.generator_index,
                    "generated_token_ids": [int(value) for value in token_ids],
                }
            )
            records.append(record)
    return records


def _rates(frame: pd.DataFrame, group_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "attempted": int(len(group)),
                "parseable": int(group["parseable"].sum()),
                "full_trajectory_correct": int(group["full_trajectory_correct"].sum()),
                "final_answer_correct": int(group["final_answer_correct"].sum()),
                "parseable_rate": float(group["parseable"].mean()),
                "full_trajectory_accuracy": float(
                    group["full_trajectory_correct"].mean()
                ),
                "final_answer_accuracy": float(group["final_answer_correct"].mean()),
            }
        )
        rows.append(row)
    return rows


def _calibration_tasks(config: dict[str, Any]) -> list[ProgramTraceTask]:
    section = config["teacher_tasks_v4"]
    seed = int(section["calibration_seed"])
    per_cell = int(section["calibration_per_cell"])
    horizons = tuple(int(value) for value in section["horizons"])
    tasks: list[ProgramTraceTask] = []
    cell_index = 0
    for family, variants in section["variants"].items():
        for variant in variants:
            for horizon in horizons:
                tasks.extend(
                    generate_program_tasks(
                        str(family),
                        str(variant),
                        per_cell,
                        seed=seed + cell_index * 1009,
                        horizons=(horizon,),
                    )
                )
                cell_index += 1
    return tasks


def _calibration_summary(context: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    cells = _rates(frame, ["family", "variant", "horizon"])
    threshold = float(
        context.config["teacher_tasks_v4"]["minimum_cell_full_trajectory_accuracy"]
    )
    eligible = [row for row in cells if row["full_trajectory_accuracy"] >= threshold]
    selected: dict[str, dict[str, str]] = defaultdict(dict)
    by_family_horizon: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_family_horizon[(str(row["family"]), int(row["horizon"]))].append(row)
    for (family, horizon), candidates in by_family_horizon.items():
        winner = sorted(
            candidates,
            key=lambda row: (
                -float(row["full_trajectory_accuracy"]),
                -float(row["parseable_rate"]),
                -float(row["final_answer_accuracy"]),
                str(row["variant"]),
            ),
        )[0]
        selected[family][str(horizon)] = str(winner["variant"])
    complete_families = sorted(
        family
        for family, horizons in selected.items()
        if set(horizons)
        == {str(int(value)) for value in context.config["teacher_tasks_v4"]["horizons"]}
    )
    return {
        "schema_version": 6,
        "protocol_version": PROTOCOL_V4,
        "run_id": context.run_id,
        "domain": "calibration",
        "minimum_cell_full_trajectory_accuracy": threshold,
        "preferred_cell_full_trajectory_accuracy": float(
            context.config["teacher_tasks_v4"][
                "preferred_cell_full_trajectory_accuracy"
            ]
        ),
        "rates_by_family_variant_horizon": cells,
        "selected_variant_by_family_horizon": dict(selected),
        "complete_horizon_families": complete_families,
        "overall": _rates(frame.assign(scope="all"), ["scope"])[0],
    }


def _write_task_file(path: Path, domain: str, tasks: list[ProgramTraceTask]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 6,
            "protocol_version": PROTOCOL_V4,
            "domain": domain,
            "items": [task.to_dict() for task in tasks],
        },
    )


def _freeze_formal_data(context: Any) -> dict[str, Any]:
    calibration_path = context.processed_dir / "teacher_calibration_stepwise_v4.json"
    if not calibration_path.is_file():
        raise RuntimeError("teacher calibration summary is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    families = list(calibration["complete_horizon_families"])
    if not families:
        raise RuntimeError("no family passed teacher calibration at every horizon")
    section = context.config["teacher_tasks_v4"]
    horizons = [int(value) for value in section["horizons"]]
    blocked: set[str] = set()
    domains: dict[str, list[ProgramTraceTask]] = {}
    for domain, requested_value in section["formal_counts_per_family"].items():
        requested = int(requested_value)
        seed = int(section["formal_seeds"][domain])
        tasks: list[ProgramTraceTask] = []
        for family_index, family in enumerate(families):
            base, remainder = divmod(requested, len(horizons))
            for horizon_index, horizon in enumerate(horizons):
                count = base + int(horizon_index < remainder)
                variant = calibration["selected_variant_by_family_horizon"][family][
                    str(horizon)
                ]
                generated = generate_program_tasks(
                    family,
                    variant,
                    count,
                    seed=seed + family_index * 100_003 + horizon_index * 1009,
                    horizons=(horizon,),
                    blocked_program_hashes=blocked,
                )
                blocked.update(task.program_hash for task in generated)
                tasks.extend(generated)
        domains[str(domain)] = tasks
    verify_disjoint_domains(domains)
    data_root = context.root / "data/v4"
    data_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for domain, tasks in domains.items():
        path = data_root / f"{domain}.json"
        _write_task_file(path, domain, tasks)
        paths[domain] = path
    freeze = build_program_freeze(
        context.root,
        context.config,
        calibration_summary=calibration_path,
        data_paths=paths,
        selection={
            "families": families,
            "variant_by_family_horizon": calibration[
                "selected_variant_by_family_horizon"
            ],
            "formal_counts_per_family": section["formal_counts_per_family"],
        },
    )
    freeze["data_counts"] = {key: len(value) for key, value in domains.items()}
    return freeze


def _load_frozen_tasks(
    context: Any, freeze: dict[str, Any], domain: str
) -> list[ProgramTraceTask]:
    payload = json.loads(
        (context.root / freeze["formal_data"][domain]).read_text(encoding="utf-8")
    )
    tasks = []
    for item in payload["items"]:
        value = dict(item)
        value["semantic_actions"] = tuple(value["semantic_actions"])
        tasks.append(ProgramTraceTask(**value))
    return tasks


def main() -> None:
    parser = standard_parser(
        "Calibrate and freeze protocol-v4 teacher tasks",
        "configs/predictive_v4.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("calibrate", "calibrate_stepwise", "freeze", "formal"),
        required=True,
    )
    parser.add_argument(
        "--domain",
        choices=("train", "validation", "causal_test", "rollout_test", "all"),
        default="all",
    )
    args = parser.parse_args()
    context = initialize_context("teacher-tasks-v4", args)
    try:
        if args.dry_run:
            context.finish("DRY_RUN", stage=args.stage)
            return
        if args.stage == "freeze":
            freeze = _freeze_formal_data(context)
            context.finish(
                "COMPLETED_FREEZE",
                freeze="artifacts/program_tasks_v4.freeze.json",
                freeze_digest=freeze["freeze_digest"],
            )
            return
        bundle = load_model_bundle(context.config)
        batch_size = int(context.config["teacher_tasks_v4"]["generation_batch_size"])
        extra = int(context.config["teacher_tasks_v4"]["maximum_extra_tokens"])
        if args.stage in {"calibrate", "calibrate_stepwise"}:
            tasks = _calibration_tasks(context.config)
            if args.limit is not None:
                tasks = tasks[: int(args.limit)]
            if args.stage == "calibrate_stepwise":
                records = generate_teacher_stepwise_records(
                    bundle,
                    tasks,
                    domain="calibration_stepwise",
                    run_id=context.run_id,
                    batch_size=batch_size,
                )
                label = "teacher_calibration_stepwise"
            else:
                records = generate_teacher_records(
                    bundle,
                    tasks,
                    domain="calibration_sequence",
                    run_id=context.run_id,
                    batch_size=batch_size,
                    maximum_extra_tokens=extra,
                )
                label = "teacher_calibration"
            raw = context.raw_dir / context.run_id / f"{label}.jsonl"
            append_jsonl(raw, records)
            summary = _calibration_summary(context, records)
            summary["records"] = str(raw.relative_to(context.root))
            summary["evaluation_mode"] = (
                "environment_stepwise"
                if args.stage == "calibrate_stepwise"
                else "single_sequence"
            )
            summary_path = context.processed_dir / f"{label}_v4.json"
            write_json_atomic(summary_path, summary)
            context.finish(
                "COMPLETED_CALIBRATION",
                summary=str(summary_path.relative_to(context.root)),
                records_sha256=sha256_file(raw),
            )
            return
        freeze = verify_program_freeze(context.root, context.config)
        selected_domains = (
            list(freeze["formal_data"]) if args.domain == "all" else [args.domain]
        )
        all_records: list[dict[str, Any]] = []
        for domain in selected_domains:
            tasks = _load_frozen_tasks(context, freeze, domain)
            if args.limit is not None:
                tasks = tasks[: int(args.limit)]
            all_records.extend(
                generate_teacher_stepwise_records(
                    bundle,
                    tasks,
                    domain=domain,
                    run_id=context.run_id,
                    batch_size=batch_size,
                )
            )
        raw = context.raw_dir / context.run_id / "teacher_formal.jsonl"
        append_jsonl(raw, all_records)
        frame = pd.DataFrame(all_records)
        rates = _rates(frame, ["domain", "family", "horizon"])
        overall = _rates(frame.assign(scope="all"), ["scope"])[0]
        summary = {
            "schema_version": 6,
            "protocol_version": PROTOCOL_V4,
            "run_id": context.run_id,
            "freeze_digest": freeze["freeze_digest"],
            "records": str(raw.relative_to(context.root)),
            "rates_by_domain_family_horizon": rates,
            "overall": overall,
            "teacher_competence_target_met": bool(
                overall["full_trajectory_accuracy"] >= 0.70
            ),
        }
        output = context.processed_dir / "teacher_formal_v4.json"
        write_json_atomic(output, summary)
        frame.to_parquet(
            context.processed_dir / "teacher_formal_v4.parquet",
            index=False,
            compression="zstd",
        )
        context.finish(
            "COMPLETED_FORMAL", summary=str(output.relative_to(context.root))
        )
    except KeyboardInterrupt:
        context.finish("FAILED", error="KeyboardInterrupt")
        raise
    except Exception as exc:
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
