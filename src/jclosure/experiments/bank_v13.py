"""Generate, screen, and capture the independent V13 causal-state bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jclosure.clamp_v3 import V3ClampThresholds
from jclosure.datasets_v4 import ProgramTraceTask
from jclosure.datasets_v8 import FAMILIES, _generate_one, load_tasks
from jclosure.experiments.causal_single_v4 import _load_encoder
from jclosure.experiments.common import initialize_context, standard_parser
from jclosure.experiments.geometry_v3 import NaturalityModel
from jclosure.experiments.persistent_channels_v7 import (
    _prefill,
    _teacher_forced_trajectory,
    _teacher_tokens,
)
from jclosure.experiments.persistent_state_v8 import (
    _load_bank,
    _semantic_ids,
)
from jclosure.model import load_model_bundle
from jclosure.persistent_state_v8 import extract_architecture_state, save_state_shard
from jclosure.protocol_v13 import (
    PROTOCOL_V13,
    SCHEMA_VERSION_V13,
    SELECTION_PATH,
    TASKS_PATH,
    TEACHER_SUMMARY,
    build_base_freeze,
    build_guard,
    build_stage_freeze,
    verify_base_freeze,
)
from jclosure.provenance import append_jsonl, sha256_file, write_json_atomic
from jclosure.runtime_v3_1 import encode_direct_prompt
from jclosure.single_arm_v4 import (
    construct_from_shared_basis,
    shared_low_singular_basis,
)

CAPTURE_FREEZE_PATH = Path("artifacts/causal_bank_v13_capture.freeze.json")
BANK_SUMMARY = Path("results/v13/processed/causal_bank_expansion_v13.json")


def _parser() -> argparse.ArgumentParser:
    parser = standard_parser(
        "V13 independent causal bank", "configs/causal_geometry_v13.yaml"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "generate",
            "teacher",
            "select",
            "guard",
            "freeze",
            "capture",
            "freeze-capture",
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "final_test"))
    parser.add_argument("--family", choices=FAMILIES)
    return parser


def _all_historical_hashes(root: Path) -> set[str]:
    values: set[str] = set()
    for path in sorted((root / "data").rglob("*.json")):
        if "v13" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in payload.get("items", []):
            if "program_hash" in row:
                values.add(str(row["program_hash"]))
    return values


def _generate(context: Any) -> dict[str, Any]:
    section = context.config["causal_geometry_v13"]
    v8 = json.loads(
        (context.root / "data/v8/persistent_causal_formal.json").read_text()
    )
    horizons = {str(k): int(v) for k, v in v8["selected_horizons"].items()}
    blocked = _all_historical_hashes(context.root)
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "final_test"):
        for family_index, family in enumerate(FAMILIES):
            seed = int(section["split_seeds"][split]) + family_index * 100_003
            generator = np.random.default_rng(seed)
            tasks = []
            count = int(section["candidate_per_family"][split])
            attempt = 0
            while len(tasks) < count:
                if attempt >= count * 500:
                    raise RuntimeError(f"V13 generator exhausted for {split}/{family}")
                if family == "modular_arithmetic":
                    horizon = horizons[family]
                    # Keep every reported state to one decimal digit.  The
                    # teacher parser is token based, so multi-token numerals
                    # would measure tokenization rather than recurrence skill.
                    modulus = int(generator.integers(5, 11))
                    base_step = int(generator.choice(np.asarray((-2, -1, 1, 2))))
                    lift = int(generator.integers(-8, 9))
                    step = base_step + lift * modulus
                    start = int(generator.integers(0, modulus))
                    actions = [str(start)]
                    current = start
                    for _ in range(horizon - 1):
                        current = (current + step) % modulus
                        actions.append(str(current))
                    ast = {
                        "kind": "v13_lifted_modular_recurrence",
                        "start": start,
                        "step": step,
                        "base_step": base_step,
                        "lift": lift,
                        "modulus": modulus,
                        "reported_states": horizon,
                    }
                    variant = "v13_lifted_modular"
                    prompt = (
                        f"Modular recurrence: x0 = {start}. At each step add "
                        f"{base_step} modulo {modulus}. The same update can be written "
                        f"as adding {step}, because {step} is congruent to {base_step} "
                        f"modulo {modulus}. Starting with x0, output x0 "
                        f"through x{horizon - 1} as exactly {horizon} "
                        "space-separated integers and no explanation:"
                    )
                    base = _generate_one(family, generator, horizon, seed, attempt)
                    base = replace(
                        base,
                        variant=variant,
                        template_id=f"v13:{family}:{variant}:h{horizon}",
                        prompt=prompt,
                        semantic_actions=tuple(actions),
                        final_answer=actions[-1],
                        ast=ast,
                    )
                else:
                    base = _generate_one(
                        family,
                        generator,
                        horizons[family],
                        seed,
                        attempt,
                    )
                    ast = base.ast
                program_hash = hashlib.sha256(
                    json.dumps(
                        {"family": family, "variant": base.variant, "ast": ast},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                attempt += 1
                if program_hash in blocked:
                    continue
                blocked.add(program_hash)
                tasks.append(
                    replace(
                        base,
                        example_id=f"v8-{family}-{program_hash[:20]}",
                        program_hash=program_hash,
                        ast=ast,
                    )
                )
            for task in tasks:
                renamed = replace(
                    task,
                    example_id=task.example_id.replace("v8-", f"v13-{split}-", 1),
                )
                rows.append({**renamed.to_dict(), "split": split})
    hashes = [row["program_hash"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("V13 candidate program overlap")
    payload = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "domain": "v13_independent_causal_bank_candidates",
        "historical_program_overlap": False,
        "selected_horizons": horizons,
        "items": sorted(rows, key=lambda row: (row["split"], row["example_id"])),
    }
    write_json_atomic(context.root / TASKS_PATH, payload)
    return {
        "count": len(rows),
        "path": str(TASKS_PATH),
        "sha256": sha256_file(context.root / TASKS_PATH),
    }


def _task_rows(path: Path) -> list[tuple[str, ProgramTraceTask]]:
    return load_tasks(path)


def _parse_teacher_tokens(
    bundle: Any, task: ProgramTraceTask, tokens: list[int]
) -> tuple[list[str], str | None]:
    allowed = (
        {str(value) for value in range(1000)}
        if all(str(value).isdigit() for value in task.semantic_actions)
        else set("ABCDEFGHIJKL")
    )
    actions: list[str] = []
    error = None
    specials = set(getattr(bundle.tokenizer, "all_special_ids", ()) or ())
    for token in tokens:
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


@torch.no_grad()
def _teacher_batch(
    bundle: Any, tasks: list[ProgramTraceTask]
) -> list[tuple[list[str], str | None]]:
    encoded = [encode_direct_prompt(bundle, task.prompt)[0] for task in tasks]
    maximum_length = max(int(value.numel()) for value in encoded)
    pad = int(bundle.tokenizer.eos_token_id)
    device = next(bundle.hf_model.parameters()).device
    input_ids = torch.full(
        (len(tasks), maximum_length), pad, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(input_ids)
    for index, value in enumerate(encoded):
        length = int(value.numel())
        input_ids[index, -length:] = value.to(device)
        attention[index, -length:] = 1
    maximum = max(2 * int(task.horizon) + 8 for task in tasks)
    generated = bundle.hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_new_tokens=maximum,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad,
    )[:, maximum_length:]
    return [
        _parse_teacher_tokens(bundle, task, generated[index].tolist())
        for index, task in enumerate(tasks)
    ]


def _teacher(context: Any, family: str | None = None) -> dict[str, Any]:
    bundle = load_model_bundle(context.config)
    rows: list[dict[str, Any]] = []
    repair_source_records_sha256 = None
    tasks = _task_rows(context.root / TASKS_PATH)
    if family is not None:
        current_tasks = {task.example_id: task for _, task in tasks}
        previous = json.loads((context.root / TEACHER_SUMMARY).read_text())
        repair_source_records_sha256 = previous["records_sha256"]
        previous_rows = [
            json.loads(line)
            for line in (context.root / previous["records"]).read_text().splitlines()
        ]
        rows.extend(
            row
            for row in previous_rows
            if row["family"] != family
            and row["prompt_id"] in current_tasks
            and row["program_hash"] == current_tasks[row["prompt_id"]].program_hash
        )
        tasks = [(split, task) for split, task in tasks if task.family == family]
    progress = context.raw_dir / context.run_id / "teacher_progress.json"
    records = context.raw_dir / context.run_id / "teacher_screen_v13.jsonl"
    batch_size = int(context.config["causal_geometry_v13"]["teacher_generation_batch"])
    for start in range(0, len(tasks), batch_size):
        current = tasks[start : start + batch_size]
        results = _teacher_batch(bundle, [task for _, task in current])
        for (split, task), (actions, error) in zip(current, results, strict=True):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION_V13,
                    "protocol_version": PROTOCOL_V13,
                    "run_id": context.run_id,
                    "split": split,
                    "prompt_id": task.example_id,
                    "family": task.family,
                    "program_hash": task.program_hash,
                    "expected_actions": list(task.semantic_actions),
                    "generated_actions": actions,
                    "parseable": error is None and len(actions) == task.horizon,
                    "teacher_correct": tuple(actions) == task.semantic_actions,
                    "error": error,
                }
            )
        completed = min(start + len(current), len(tasks))
        if start % 100 == 0 or completed == len(tasks):
            write_json_atomic(
                progress,
                {"status": "RUNNING", "completed": completed, "total": len(tasks)},
            )
    append_jsonl(records, rows)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["teacher_correct"]:
            counts[row["split"]][row["family"]] += 1
    summary = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "candidate_count": len(rows),
        "repaired_family": family,
        "carried_teacher_rows": len(rows) - len(tasks),
        "repair_source_records_sha256": repair_source_records_sha256,
        "records": str(records.relative_to(context.root)),
        "records_sha256": sha256_file(records),
        "teacher_correct_counts": {
            split: dict(value) for split, value in counts.items()
        },
        "candidate_manifest_sha256": sha256_file(context.root / TASKS_PATH),
    }
    write_json_atomic(context.root / TEACHER_SUMMARY, summary)
    write_json_atomic(
        progress, {"status": "COMPLETED", "completed": len(tasks), "total": len(tasks)}
    )
    return summary


def _select(context: Any) -> dict[str, Any]:
    section = context.config["causal_geometry_v13"]
    tasks = {
        task.example_id: (split, task)
        for split, task in _task_rows(context.root / TASKS_PATH)
    }
    summary = json.loads((context.root / TEACHER_SUMMARY).read_text())
    rows = [
        json.loads(line)
        for line in (context.root / summary["records"]).read_text().splitlines()
    ]
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "final_test"):
        counts[split] = {}
        for family in FAMILIES:
            eligible = [
                row
                for row in rows
                if row["split"] == split
                and row["family"] == family
                and row["teacher_correct"]
            ]
            eligible.sort(
                key=lambda row: hashlib.sha256(
                    f"{section['task_seed']}:{row['prompt_id']}".encode()
                ).hexdigest()
            )
            target = int(section["target_per_family"][split])
            reserve = min(len(eligible), int(math.ceil(target * 1.35)))
            if reserve < target:
                raise RuntimeError(
                    f"teacher-correct pool shortfall {split}/{family}: {reserve} < {target}"
                )
            for row in eligible[:reserve]:
                _, task = tasks[str(row["prompt_id"])]
                selected.append({**task.to_dict(), "split": split})
            counts[split][family] = reserve
    payload = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "domain": "teacher-correct V13 capture pool with frozen reserves",
        "source_teacher_records_sha256": summary["records_sha256"],
        "counts": counts,
        "items": sorted(
            selected, key=lambda row: (row["split"], row["family"], row["example_id"])
        ),
    }
    write_json_atomic(context.root / SELECTION_PATH, payload)
    return {
        "count": len(selected),
        "counts": counts,
        "path": str(SELECTION_PATH),
        "sha256": sha256_file(context.root / SELECTION_PATH),
    }


def _thresholds(config: dict[str, Any]) -> V3ClampThresholds:
    section = config["persistent_state_v8"]["intervention"]
    return V3ClampThresholds(
        dense_cosine=float(section["dense_cosine_threshold"]),
        dense_top10_overlap=float(section["top10_overlap_threshold"]),
        rms_drift=float(section["rms_drift_threshold"]),
        formal_displacement=float(section["formal_displacement_fraction"]),
        sensitivity_displacement=0.05,
    )


def _selected_tasks(root: Path, split: str) -> list[ProgramTraceTask]:
    return [task for role, task in load_tasks(root / SELECTION_PATH) if role == split]


@torch.no_grad()
def _capture(context: Any, split: str) -> dict[str, Any]:
    freeze = verify_base_freeze(context.root, context.config)
    section = context.config["causal_geometry_v13"]
    v8 = context.config["persistent_state_v8"]
    intervention = v8["intervention"]
    layer = int(intervention["layer"])
    measured = [int(value) for value in intervention["measured_layers"]]
    main_layer = max(measured)
    recurrent = [
        int(value) for value in v8["persistent_components"]["recurrent_layers"]
    ]
    attention = [
        int(value) for value in v8["persistent_components"]["attention_layers"]
    ]
    horizons = [
        int(value)
        for value in (
            section["train_capture_horizons"]
            if split == "train"
            else section["horizons"]
        )
    ]
    target = int(section["target_per_family"][split])
    tasks = _selected_tasks(context.root, split)
    bundle = load_model_bundle(context.config)
    bank_rows, bank_states = _load_bank(context.root)
    naturality = NaturalityModel(128, 10, 0.99).fit(bank_states)
    _, encoder, dense_map = _load_encoder(context, bundle)
    device = next(bundle.hf_model.parameters()).device
    shared = shared_low_singular_basis(
        dense_map, layer, relative_tolerance=1e-4, device=device
    )
    threshold = _thresholds(context.config)
    counts: dict[str, int] = defaultdict(int)
    excluded: dict[str, int] = defaultdict(int)
    pair_rows: list[dict[str, Any]] = []
    endpoint: dict[str, list[Any]] = defaultdict(list)
    shard_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    artifact_dir = context.root / "artifacts/causal/v13" / context.run_id / split
    artifact_dir.mkdir(parents=True, exist_ok=True)
    progress = context.raw_dir / context.run_id / f"capture_{split}_progress.json"

    def flush() -> None:
        if not shard_rows:
            return
        path = artifact_dir / f"state_shard_{len(shards):04d}.pt"
        save_state_shard(path, shard_rows)
        shards.append(
            {
                "path": str(path.relative_to(context.root)),
                "sha256": sha256_file(path),
                "count": len(shard_rows),
                "base_trial_ids": [row["base_trial_id"] for row in shard_rows],
            }
        )
        shard_rows.clear()

    ordered = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"capture:{task.example_id}".encode()
        ).hexdigest(),
    )
    for attempt, task in enumerate(ordered):
        if all(counts[family] >= target for family in FAMILIES):
            break
        if counts[task.family] >= target:
            continue
        matches = [
            (row, bank_states[index])
            for index, row in enumerate(bank_rows)
            if row["family"] == task.family and int(row["horizon"]) == task.horizon
        ]
        if not matches:
            excluded["no_matched_donor"] += 1
            continue
        donor_row, donor_array = matches[
            int(hashlib.sha256(task.example_id.encode()).hexdigest(), 16) % len(matches)
        ]
        clean = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=layer,
            candidate=None,
        )
        donor = torch.from_numpy(donor_array).to(device)
        natural_scale = float(
            torch.linalg.vector_norm(donor - clean["hidden"][layer]).item()
        )
        candidate, quality, valid = construct_from_shared_basis(
            clean["hidden"][layer],
            donor,
            shared=shared,
            dense_map=dense_map,
            encoder=encoder,
            natural_scale=natural_scale,
            displacement_fraction=float(intervention["initial_strength"]),
            thresholds=threshold,
            naturality=naturality,
        )
        singular_values = quality.get("construction", {}).pop("singular_values", [])
        if singular_values:
            quality["construction"]["singular_value_count"] = len(singular_values)
            quality["construction"]["singular_value_sha256"] = hashlib.sha256(
                np.asarray(singular_values, dtype=np.float32).tobytes()
            ).hexdigest()
        if not valid:
            excluded["intervention_ineligible"] += 1
            continue
        perturbed = _prefill(
            bundle,
            task.prompt,
            measured_layers=measured,
            dense_map=dense_map,
            intervention_layer=layer,
            candidate=candidate,
        )
        tokens = _teacher_tokens(
            bundle,
            clean,
            count=max(horizons),
            measured_layers=measured,
            dense_map=dense_map,
        )
        clean_traj = _teacher_forced_trajectory(
            bundle,
            clean["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        pert_traj = _teacher_forced_trajectory(
            bundle,
            perturbed["cache"],
            tokens,
            prompt_length=clean["prompt_length"],
            measured_layers=measured,
            dense_map=dense_map,
        )
        base_id = (
            f"v13-{split}-{hashlib.sha256(task.example_id.encode()).hexdigest()[:20]}"
        )
        clean_state = extract_architecture_state(
            clean["cache"], recurrent_layers=recurrent, attention_layers=attention
        )
        perturbed_state = extract_architecture_state(
            perturbed["cache"], recurrent_layers=recurrent, attention_layers=attention
        )
        shard_rows.append(
            {
                "base_trial_id": base_id,
                "prompt_id": task.example_id,
                "family": task.family,
                "split": split,
                "prompt_length": clean["prompt_length"],
                "teacher_tokens": tokens,
                "clean": clean_state,
                "perturbed": perturbed_state,
            }
        )
        semantic_ids = _semantic_ids(
            bundle.tokenizer, task.semantic_actions[min(1, task.horizon - 1)]
        )
        endpoint["base_trial_id"].append(base_id)
        endpoint["prompt_id"].append(task.example_id)
        endpoint["family"].append(task.family)
        endpoint["current_j_clean"].append(
            clean["j"][layer].cpu().numpy().astype(np.float16)
        )
        endpoint["current_j_perturbed"].append(
            perturbed["j"][layer].cpu().numpy().astype(np.float16)
        )
        for horizon in horizons:
            index = horizon - 1
            clean_j = clean_traj["j"][main_layer][index].astype(np.float16)
            pert_j = pert_traj["j"][main_layer][index].astype(np.float16)
            endpoint[f"clean_j_h{horizon}"].append(clean_j)
            endpoint[f"perturbed_j_h{horizon}"].append(pert_j)
            clean_logits = clean_traj["logits"][index].astype(np.float32)
            pert_logits = pert_traj["logits"][index].astype(np.float32)
            selected_ids = np.asarray(
                list(
                    dict.fromkeys(
                        [
                            *semantic_ids,
                            *np.argsort(clean_logits)[-32:][::-1].tolist(),
                            *np.argsort(pert_logits)[-32:][::-1].tolist(),
                        ]
                    )
                )[:64],
                dtype=int,
            )
            padded_ids = np.full(64, -1, dtype=np.int32)
            clean_values = np.zeros(64, dtype=np.float16)
            pert_values = np.zeros(64, dtype=np.float16)
            padded_ids[: len(selected_ids)] = selected_ids
            clean_values[: len(selected_ids)] = clean_logits[selected_ids]
            pert_values[: len(selected_ids)] = pert_logits[selected_ids]
            endpoint[f"logit_ids_h{horizon}"].append(padded_ids)
            endpoint[f"clean_logits_h{horizon}"].append(clean_values)
            endpoint[f"perturbed_logits_h{horizon}"].append(pert_values)
            endpoint[f"semantic_delta_h{horizon}"].append(
                np.float32(
                    np.mean(pert_logits[semantic_ids])
                    - np.mean(clean_logits[semantic_ids])
                    if semantic_ids
                    else 0.0
                )
            )
        pair_rows.append(
            {
                "schema_version": SCHEMA_VERSION_V13,
                "protocol_version": PROTOCOL_V13,
                "run_id": context.run_id,
                "base_trial_id": base_id,
                "prompt_id": task.example_id,
                "family": task.family,
                "split": split,
                "donor_id": donor_row["prompt_id"],
                "valid": True,
                "teacher_correct": True,
                "natural_scale": natural_scale,
                "intervention_quality": quality,
                "teacher_tokens": tokens,
            }
        )
        counts[task.family] += 1
        if len(shard_rows) >= int(section["state_shard_size"]):
            flush()
        if attempt % 10 == 0:
            write_json_atomic(
                progress,
                {
                    "status": "RUNNING",
                    "attempted": attempt + 1,
                    "valid_counts": dict(counts),
                    "excluded": dict(excluded),
                },
            )
    flush()
    complete = all(counts[family] >= target for family in FAMILIES)
    if not complete:
        raise RuntimeError(
            f"V13 {split} capture shortfall: {dict(counts)} target={target}"
        )
    pair_path = context.raw_dir / context.run_id / f"causal_pairs_{split}_v13.jsonl"
    append_jsonl(pair_path, pair_rows)
    endpoint_path = artifact_dir / f"endpoints_{split}_v13.npz"
    np.savez_compressed(
        endpoint_path, **{key: np.asarray(value) for key, value in endpoint.items()}
    )
    manifest = {
        "schema_version": SCHEMA_VERSION_V13,
        "protocol_version": PROTOCOL_V13,
        "run_id": context.run_id,
        "source_freeze_digest": freeze["freeze_digest"],
        "split": split,
        "complete": True,
        "target_per_family": target,
        "valid_count": len(pair_rows),
        "valid_counts": dict(counts),
        "excluded_reasons": dict(excluded),
        "pair_records": str(pair_path.relative_to(context.root)),
        "pair_records_sha256": sha256_file(pair_path),
        "endpoint_artifact": str(endpoint_path.relative_to(context.root)),
        "endpoint_artifact_sha256": sha256_file(endpoint_path),
        "state_shards": shards,
        "horizons": horizons,
        "architecture": {
            "recurrent_layers": recurrent,
            "attention_layers": attention,
            "state_dtype": "bfloat16",
            "kv_token_scope": "all_prefill_positions",
        },
    }
    path = context.root / f"results/v13/processed/causal_capture_{split}_v13.json"
    write_json_atomic(path, manifest)
    write_json_atomic(
        progress,
        {
            "status": "COMPLETED",
            "valid_counts": dict(counts),
            "excluded": dict(excluded),
        },
    )
    return manifest


def _freeze_capture(context: Any) -> dict[str, Any]:
    inputs = [
        Path(f"results/v13/processed/causal_capture_{split}_v13.json")
        for split in ("train", "validation", "final_test")
    ]
    manifests = [json.loads((context.root / path).read_text()) for path in inputs]
    for manifest in manifests:
        inputs.append(Path(manifest["pair_records"]))
        inputs.append(Path(manifest["endpoint_artifact"]))
        inputs.extend(Path(row["path"]) for row in manifest["state_shards"])
    payload = build_stage_freeze(
        context.root,
        context.config,
        path=CAPTURE_FREEZE_PATH,
        purpose="freeze V13 teacher-correct raw persistent-state captures and independent splits",
        inputs=inputs,
        payload={
            "bank_sizes": {item["split"]: item["valid_count"] for item in manifests},
            "family_counts": {
                item["split"]: item["valid_counts"] for item in manifests
            },
            "split_hashes": {
                item["split"]: hashlib.sha256(
                    "\n".join(
                        sorted(
                            row
                            for shard in item["state_shards"]
                            for row in shard["base_trial_ids"]
                        )
                    ).encode()
                ).hexdigest()
                for item in manifests
            },
            "confirmatory_used_for_selection": False,
        },
    )
    write_json_atomic(context.root / BANK_SUMMARY, payload)
    return payload


def main() -> None:
    args = _parser().parse_args()
    context = initialize_context("bank-v13", args)
    try:
        if args.stage == "generate":
            result = _generate(context)
        elif args.stage == "teacher":
            result = _teacher(context, args.family)
        elif args.stage == "select":
            result = _select(context)
        elif args.stage == "guard":
            result = build_guard(
                context.root, context.config["causal_geometry_v13"]["baseline_commit"]
            )
        elif args.stage == "freeze":
            result = build_base_freeze(context.root, context.config)
        elif args.stage == "capture":
            if not args.split:
                raise ValueError("--split is required for capture")
            result = _capture(context, args.split)
        else:
            result = _freeze_capture(context)
        context.finish("COMPLETED_V13_BANK_STAGE", summary=result)
    except Exception as exc:
        context.fail_progress(f"{type(exc).__name__}: {exc}")
        context.finish("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
