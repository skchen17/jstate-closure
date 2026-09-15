"""Deterministic high-accuracy task domains for persistent-state protocol v8."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jclosure.datasets_v4 import ProgramTraceTask
from jclosure.provenance import write_json_atomic

PROTOCOL_V8 = "structured_persistent_state_protocol_v8"
FAMILIES = (
    "boolean_logic",
    "simple_state_transition",
    "modular_arithmetic",
    "short_graph_traversal",
    "variable_binding",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _task(
    family: str,
    variant: str,
    prompt: str,
    actions: list[str],
    horizon: int,
    seed: int,
    index: int,
    ast: dict[str, Any],
) -> ProgramTraceTask:
    program_hash = _digest({"family": family, "variant": variant, "ast": ast})
    return ProgramTraceTask(
        example_id=f"v8-{family}-{program_hash[:20]}",
        family=family,
        variant=variant,
        template_id=f"v8:{family}:{variant}:h{horizon}",
        prompt=prompt,
        semantic_actions=tuple(actions),
        final_answer=actions[-1],
        horizon=horizon,
        generator_seed=seed,
        generator_index=index,
        program_hash=program_hash,
        ast=ast,
    )


def _generate_one(
    family: str, generator: np.random.Generator, horizon: int, seed: int, index: int
) -> ProgramTraceTask:
    if family == "boolean_logic":
        operators = generator.choice(np.asarray(("AND", "OR", "XOR")), horizon)
        left = generator.integers(0, 2, horizon).tolist()
        right = generator.integers(0, 2, horizon).tolist()
        alphabet = np.asarray(tuple("ABCDEFGHIJKL"))
        labels = [
            generator.choice(alphabet, 2, replace=False).tolist()
            for _ in range(horizon)
        ]
        actions = []
        expressions = []
        for operator, a, b, names in zip(
            operators.tolist(), left, right, labels, strict=True
        ):
            value = a & b if operator == "AND" else a | b if operator == "OR" else a ^ b
            actions.append(str(value))
            expressions.append(
                f"{names[0]}={a}, {names[1]}={b}, compute {names[0]} {operator} {names[1]}"
            )
        ast = {
            "kind": "mixed_boolean_pairs",
            "operators": operators.tolist(),
            "left": left,
            "right": right,
            "labels": labels,
            "horizon": horizon,
        }
        prompt = (
            f"Boolean expressions: {'; '.join(expressions)}. Evaluate each "
            f"independently. Output exactly {horizon} result bits in order, "
            "separated by single spaces and no explanation:"
        )
        return _task(
            family, "mixed_boolean_pairs", prompt, actions, horizon, seed, index, ast
        )
    if family == "modular_arithmetic":
        modulus = int(generator.integers(5, 16))
        step = int(generator.choice(np.asarray((-2, -1, 1, 2))))
        start = int(generator.integers(0, modulus))
        current = start
        actions = [str(start)]
        for _ in range(horizon - 1):
            current = (current + step) % modulus
            actions.append(str(current))
        operation = f"add {step}" if step > 0 else f"subtract {abs(step)}"
        ast = {
            "kind": "wide_modular_recurrence",
            "start": start,
            "step": step,
            "modulus": modulus,
            "reported_states": horizon,
        }
        prompt = (
            f"Modular recurrence: x0 = {start}. For each step, {operation} "
            f"modulo {modulus}. Starting with x0, give x0 through x{horizon - 1} "
            f"as exactly {horizon} "
            "space-separated integers. Answer only with the integers:"
        )
        return _task(
            family, "short_modular", prompt, actions, horizon, seed, index, ast
        )
    if family == "simple_state_transition":
        alphabet = np.asarray(tuple("ABCDEFGHIJKL"))
        size = int(generator.integers(3, 7))
        states = generator.choice(alphabet, size, replace=False).tolist()
        start = int(generator.integers(0, size))
        actions = [str(states[(start + step + 1) % size]) for step in range(horizon)]
        ast = {
            "kind": "clocked_finite_state_machine",
            "states": states,
            "start": states[start],
            "updates": horizon,
        }
        transitions = ", ".join(
            f"{states[position]} -> {states[(position + 1) % size]}"
            for position in range(size)
        )
        prompt = (
            f"State-transition edges: {transitions}. Start at {states[start]} and "
            f"follow one transition per tick. Output the state after each of exactly "
            f"{horizon} ticks, space-separated, with no explanation:"
        )
        return _task(
            family, "clocked_cycle_machine", prompt, actions, horizon, seed, index, ast
        )
    if family == "short_graph_traversal":
        alphabet = np.asarray(tuple("ABCDEFGHIJKL"))
        size = int(generator.integers(3, 9))
        nodes = generator.choice(alphabet, size, replace=False).tolist()
        start = int(generator.integers(0, size))
        actions = [str(nodes[(start + offset + 1) % size]) for offset in range(horizon)]
        edges = ", ".join(
            f"{nodes[position]}->{nodes[(position + 1) % size]}"
            for position in range(size)
        )
        ast = {
            "kind": "wide_directed_cycle",
            "nodes": nodes,
            "start": nodes[start],
            "horizon": horizon,
        }
        prompt = (
            f"Directed edges: {edges}. Start at {nodes[start]} and follow one "
            f"edge per step. Output the node after each of exactly {horizon} "
            "steps, space-separated, with no explanation:"
        )
        return _task(
            family, "wide_directed_cycle", prompt, actions, horizon, seed, index, ast
        )
    if family == "variable_binding":
        variables = tuple("WXYZ")
        values = generator.choice(10, 4, replace=False).tolist()
        queries = generator.integers(0, 4, horizon).tolist()
        actions = [str(values[position]) for position in queries]
        bindings = dict(zip(variables, values, strict=True))
        ast = {
            "kind": "wide_variable_lookup",
            "bindings": bindings,
            "queries": [variables[position] for position in queries],
            "horizon": horizon,
        }
        prompt = (
            "Bindings: "
            + ", ".join(f"{key}={value}" for key, value in bindings.items())
            + ". Queries: "
            + " ".join(ast["queries"])
            + f". Output exactly {horizon} bound digits in order, space-separated, "
            "with no explanation:"
        )
        return _task(
            family, "wide_variable_lookup", prompt, actions, horizon, seed, index, ast
        )
    raise ValueError(f"unknown v8 family {family}")


def generate_tasks(
    family: str,
    count: int,
    *,
    seed: int,
    horizon: int,
    blocked_hashes: set[str] | None = None,
) -> list[ProgramTraceTask]:
    generator = np.random.default_rng(seed)
    blocked = set(blocked_hashes or ())
    tasks: list[ProgramTraceTask] = []
    attempt = 0
    while len(tasks) < count and attempt < max(10_000, count * 200):
        task = _generate_one(family, generator, horizon, seed, attempt)
        attempt += 1
        if task.program_hash in blocked:
            continue
        blocked.add(task.program_hash)
        tasks.append(task)
    if len(tasks) != count:
        raise RuntimeError(f"generated {len(tasks)}/{count} {family} tasks")
    return tasks


def historical_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for directory in (root / "data/v4", root / "data/v5"):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text())
            hashes.update(
                str(item["program_hash"]) for item in payload.get("items", [])
            )
    return hashes


def write_calibration(root: Path, config: dict[str, Any], path: Path) -> dict[str, Any]:
    section = config["persistent_state_v8"]["calibration"]
    blocked = historical_hashes(root)
    items: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for horizon_index, horizon in enumerate(section["horizons"]):
            tasks = generate_tasks(
                family,
                int(section["count_per_family_horizon"]),
                seed=int(section["seed"])
                + family_index * 100_003
                + horizon_index * 10_007,
                horizon=int(horizon),
                blocked_hashes=blocked,
            )
            blocked.update(task.program_hash for task in tasks)
            items.extend({**task.to_dict(), "split": "calibration"} for task in tasks)
    payload = {
        "schema_version": 11,
        "protocol_version": PROTOCOL_V8,
        "domain": "v8_difficulty_calibration",
        "items": sorted(items, key=lambda value: value["example_id"]),
    }
    write_json_atomic(path, payload)
    return payload


def write_formal(
    root: Path,
    config: dict[str, Any],
    selected_horizons: dict[str, int],
    calibration_path: Path,
    path: Path,
) -> dict[str, Any]:
    section = config["persistent_state_v8"]["formal"]
    calibration = json.loads(calibration_path.read_text())
    blocked = historical_hashes(root) | {
        str(item["program_hash"]) for item in calibration["items"]
    }
    items: list[dict[str, Any]] = []
    for split in ("train", "validation", "final_test"):
        count = int(section["candidates_per_family"][split])
        for family_index, family in enumerate(FAMILIES):
            tasks = generate_tasks(
                family,
                count,
                seed=int(section["split_seeds"][split]) + family_index * 100_003,
                horizon=int(selected_horizons[family]),
                blocked_hashes=blocked,
            )
            blocked.update(task.program_hash for task in tasks)
            for task in tasks:
                renamed = replace(
                    task, example_id=task.example_id.replace("v8-", f"v8-{split}-", 1)
                )
                items.append({**renamed.to_dict(), "split": split})
    hashes = [str(item["program_hash"]) for item in items]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("v8 formal program overlap")
    payload = {
        "schema_version": 11,
        "protocol_version": PROTOCOL_V8,
        "domain": "v8_persistent_causal_formal",
        "selected_horizons": selected_horizons,
        "items": sorted(items, key=lambda value: (value["split"], value["example_id"])),
    }
    write_json_atomic(path, payload)
    return payload


def load_tasks(path: Path) -> list[tuple[str, ProgramTraceTask]]:
    payload = json.loads(path.read_text())
    output = []
    for item in payload["items"]:
        value = dict(item)
        split = str(value.pop("split"))
        value["semantic_actions"] = tuple(value["semantic_actions"])
        output.append((split, ProgramTraceTask(**value)))
    return output
