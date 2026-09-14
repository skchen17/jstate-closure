"""Protocol-v4 program traces and domain-separated data generation.

The v4 tasks are deliberately simple enough to calibrate teacher competence before
they are admitted to the formal splits.  Difficulty selection is performed only on
the calibration domain; the four formal domains use independent seeds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PROTOCOL_V4 = "jstate_predictive_protocol_v4"


@dataclass(frozen=True)
class ProgramTraceTask:
    example_id: str
    family: str
    variant: str
    template_id: str
    prompt: str
    semantic_actions: tuple[str, ...]
    final_answer: str
    horizon: int
    generator_seed: int
    generator_index: int
    program_hash: str
    ast: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(payload: object) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _task(
    *,
    family: str,
    variant: str,
    prompt: str,
    actions: Iterable[str],
    horizon: int,
    seed: int,
    index: int,
    ast: dict[str, Any],
) -> ProgramTraceTask:
    semantic_actions = tuple(str(value) for value in actions)
    digest = _digest((family, variant, ast))
    return ProgramTraceTask(
        example_id=f"v4-{family}-{digest[:20]}",
        family=family,
        variant=variant,
        template_id=f"{family}:{variant}:h{horizon}",
        prompt=prompt,
        semantic_actions=semantic_actions,
        final_answer=semantic_actions[-1],
        horizon=int(horizon),
        generator_seed=int(seed),
        generator_index=int(index),
        program_hash=digest,
        ast=ast,
    )


def _arithmetic(
    generator: np.random.Generator,
    *,
    variant: str,
    horizon: int,
    seed: int,
    index: int,
) -> ProgramTraceTask:
    step_by_variant = {"increment_1": 1, "increment_2": 2, "decrement_1": -1}
    step = step_by_variant[variant]
    modulus = int(generator.integers(max(5, abs(step) + 2), 11))
    start = int(generator.integers(0, modulus))
    current = start
    actions: list[str] = []
    for _ in range(horizon):
        current = (current + step) % modulus
        actions.append(str(current))
    operation = f"add {step}" if step >= 0 else "subtract 1"
    ast = {
        "kind": "modular_recurrence",
        "start": start,
        "step": step,
        "modulus": modulus,
        "horizon": horizon,
    }
    prompt = (
        f"Start at digit {start}. Repeatedly {operation} modulo {modulus} for "
        f"exactly {horizon} steps. The first output must be the state AFTER the "
        f"first step. Output exactly the {horizon} resulting digits separated by "
        "single spaces and no explanation:"
    )
    return _task(
        family="modular_arithmetic",
        variant=variant,
        prompt=prompt,
        actions=actions,
        horizon=horizon,
        seed=seed,
        index=index,
        ast=ast,
    )


def _boolean(
    generator: np.random.Generator,
    *,
    variant: str,
    horizon: int,
    seed: int,
    index: int,
) -> ProgramTraceTask:
    if variant == "binary_xor_pairs":
        left = generator.integers(0, 2, size=horizon).tolist()
        right = generator.integers(0, 2, size=horizon).tolist()
        actions = [str(a ^ b) for a, b in zip(left, right, strict=True)]
        pairs = " ".join(f"{a}{b}" for a, b in zip(left, right, strict=True))
        ast = {
            "kind": "pointwise_xor_pairs",
            "left": left,
            "right": right,
            "horizon": horizon,
        }
        prompt = (
            f"Bit pairs: {pairs}. XOR the two bits in each pair independently. "
            f"Output exactly the {horizon} result bits in order, separated by "
            "single spaces and no explanation:"
        )
        return _task(
            family="boolean_logic",
            variant=variant,
            prompt=prompt,
            actions=actions,
            horizon=horizon,
            seed=seed,
            index=index,
            ast=ast,
        )
    bits = generator.integers(0, 2, size=horizon).tolist()
    if variant == "not_each":
        actions = [str(1 - value) for value in bits]
        rule = "logical NOT to each input bit independently"
        kind = "pointwise_not"
    elif variant == "xor_one":
        actions = [str(value ^ 1) for value in bits]
        rule = "XOR 1 with each input bit independently"
        kind = "pointwise_xor_one"
    else:
        raise ValueError(f"unknown Boolean variant: {variant}")
    ast = {"kind": kind, "inputs": bits, "horizon": horizon}
    prompt = (
        f"Input bits: {' '.join(map(str, bits))}. Apply {rule}. Output exactly "
        f"the {horizon} result bits in the same order, separated by single spaces "
        "and no explanation:"
    )
    return _task(
        family="boolean_logic",
        variant=variant,
        prompt=prompt,
        actions=actions,
        horizon=horizon,
        seed=seed,
        index=index,
        ast=ast,
    )


def _graph(
    generator: np.random.Generator,
    *,
    variant: str,
    horizon: int,
    seed: int,
    index: int,
) -> ProgramTraceTask:
    if variant != "directed_cycle":
        raise ValueError(f"unknown graph variant: {variant}")
    size = int(generator.integers(3, 7))
    nodes = tuple(
        np.asarray(tuple("ABCDEF"), dtype="U1")[
            generator.permutation(6)[:size]
        ].tolist()
    )
    start_index = int(generator.integers(0, size))
    current = start_index
    actions: list[str] = []
    for _ in range(horizon):
        current = (current + 1) % size
        actions.append(nodes[current])
    ast = {
        "kind": "directed_cycle_walk",
        "nodes": nodes,
        "start": nodes[start_index],
        "horizon": horizon,
    }
    edges = ", ".join(
        f"{nodes[position]}->{nodes[(position + 1) % size]}" for position in range(size)
    )
    prompt = (
        f"Directed graph edges: {edges}. Start at {nodes[start_index]} and follow "
        f"one outgoing edge per step for exactly {horizon} steps. The first output "
        f"must be the node AFTER the first step. Output exactly the {horizon} "
        "visited nodes separated by single spaces and no explanation:"
    )
    return _task(
        family="short_graph_traversal",
        variant=variant,
        prompt=prompt,
        actions=actions,
        horizon=horizon,
        seed=seed,
        index=index,
        ast=ast,
    )


def _binding(
    generator: np.random.Generator,
    *,
    variant: str,
    horizon: int,
    seed: int,
    index: int,
) -> ProgramTraceTask:
    if variant != "four_way_lookup":
        raise ValueError(f"unknown binding variant: {variant}")
    variables = tuple("WXYZ")
    values = generator.choice(10, size=4, replace=False).tolist()
    queries = generator.integers(0, 4, size=horizon).tolist()
    actions = [str(values[value]) for value in queries]
    ast = {
        "kind": "variable_lookup",
        "bindings": dict(zip(variables, values, strict=True)),
        "queries": [variables[value] for value in queries],
        "horizon": horizon,
    }
    declarations = ", ".join(
        f"{variable}={value}" for variable, value in zip(variables, values, strict=True)
    )
    query_text = " ".join(variables[value] for value in queries)
    prompt = (
        f"Bindings: {declarations}. Queries: {query_text}. Replace each query by "
        f"its bound digit. Output exactly the {horizon} digits in query order, "
        "separated by single spaces and no explanation:"
    )
    return _task(
        family="variable_binding",
        variant=variant,
        prompt=prompt,
        actions=actions,
        horizon=horizon,
        seed=seed,
        index=index,
        ast=ast,
    )


def _state_machine(
    generator: np.random.Generator,
    *,
    variant: str,
    horizon: int,
    seed: int,
    index: int,
) -> ProgramTraceTask:
    if variant not in {"toggle_every_step", "toggle_or_keep"}:
        raise ValueError(f"unknown state-machine variant: {variant}")
    state_names = tuple(
        np.asarray(tuple("ABCDEF"), dtype="U1")[generator.permutation(6)[:2]].tolist()
    )
    start = int(generator.integers(0, 2))
    commands = (
        [1] * horizon
        if variant == "toggle_every_step"
        else generator.integers(0, 2, size=horizon).tolist()
    )
    current = start
    actions: list[str] = []
    for command in commands:
        if command == 1:
            current = 1 - current
        actions.append(state_names[current])
    ast = {
        "kind": "two_state_machine",
        "states": state_names,
        "start": state_names[start],
        "commands": commands,
        "horizon": horizon,
    }
    prompt = (
        f"A machine has states {state_names[0]} and {state_names[1]} and starts "
        f"at {state_names[start]}. Commands are "
        f"{' '.join(map(str, commands))}. Command 1 flips the current state; "
        f"command 0 keeps it. Process exactly {horizon} commands. The first output "
        f"must be the state AFTER command one. Output exactly the {horizon} states "
        "separated by single spaces and no explanation:"
    )
    return _task(
        family="simple_state_transition",
        variant=variant,
        prompt=prompt,
        actions=actions,
        horizon=horizon,
        seed=seed,
        index=index,
        ast=ast,
    )


GENERATORS = {
    "modular_arithmetic": _arithmetic,
    "boolean_logic": _boolean,
    "short_graph_traversal": _graph,
    "variable_binding": _binding,
    "simple_state_transition": _state_machine,
}


def initial_environment_state(task: ProgramTraceTask) -> str | None:
    """Return the state carried between teacher macrosteps, when applicable."""

    kind = str(task.ast["kind"])
    if kind == "modular_recurrence":
        return str(task.ast["start"])
    if kind == "directed_cycle_walk":
        return str(task.ast["start"])
    if kind == "two_state_machine":
        return str(task.ast["start"])
    return None


def environment_step_prompt(
    task: ProgramTraceTask, step_index: int, current_state: str | None
) -> str:
    """Build a one-action prompt for an environment-mediated teacher rollout."""

    if not 0 <= step_index < task.horizon:
        raise IndexError("step index is outside the task horizon")
    kind = str(task.ast["kind"])
    if kind == "modular_recurrence":
        if current_state is None:
            raise ValueError("modular recurrence requires a current state")
        step = int(task.ast["step"])
        modulus = int(task.ast["modulus"])
        operation = f"add {step}" if step >= 0 else "subtract 1"
        return (
            f"Current digit: {current_state}. Apply one step: {operation} modulo "
            f"{modulus}. Output the next digit only:"
        )
    if kind in {"pointwise_not", "pointwise_xor_one"}:
        value = int(task.ast["inputs"][step_index])
        rule = "logical NOT" if kind == "pointwise_not" else "XOR 1"
        return f"Input bit: {value}. Apply {rule}. Output the result bit only:"
    if kind == "pointwise_xor_pairs":
        left = int(task.ast["left"][step_index])
        right = int(task.ast["right"][step_index])
        return (
            f"First bit: {left}. Second bit: {right}. XOR them. Output the result "
            "bit only:"
        )
    if kind == "directed_cycle_walk":
        if current_state is None:
            raise ValueError("graph walk requires a current state")
        nodes = tuple(str(value) for value in task.ast["nodes"])
        edges = ", ".join(
            f"{nodes[index]}->{nodes[(index + 1) % len(nodes)]}"
            for index in range(len(nodes))
        )
        return (
            f"Directed edges: {edges}. Current node: {current_state}. Follow its "
            "one outgoing edge. Output the next node only:"
        )
    if kind == "variable_lookup":
        bindings = task.ast["bindings"]
        query = str(task.ast["queries"][step_index])
        declarations = ", ".join(
            f"{key}={value}" for key, value in sorted(bindings.items())
        )
        return f"Bindings: {declarations}. Query: {query}. Output its bound digit only:"
    if kind == "two_state_machine":
        if current_state is None:
            raise ValueError("state machine requires a current state")
        left, right = (str(value) for value in task.ast["states"])
        command = int(task.ast["commands"][step_index])
        return (
            f"Machine states: {left} and {right}. Current state: {current_state}. "
            f"Command: {command}. Command 1 switches state and command 0 keeps it. "
            "Output the next state only:"
        )
    raise ValueError(f"unsupported environment task kind: {kind}")


def advance_environment_state(
    task: ProgramTraceTask, predicted_action: str, current_state: str | None
) -> str | None:
    """Feed a teacher prediction into the next environment macrostep."""

    del current_state
    if str(task.ast["kind"]) in {
        "modular_recurrence",
        "directed_cycle_walk",
        "two_state_machine",
    }:
        return str(predicted_action)
    return None


def terminal_environment_prompt(task: ProgramTraceTask) -> str:
    """Return the final-decision macrostate after all declared program steps."""

    return (
        f"The {task.family} program has completed {task.horizon} steps. Its final "
        f"state is {task.final_answer}. Output that final state only:"
    )


def generate_program_tasks(
    family: str,
    variant: str,
    n: int,
    *,
    seed: int,
    horizons: tuple[int, ...] = (4, 8, 16, 32),
    blocked_program_hashes: set[str] | None = None,
) -> list[ProgramTraceTask]:
    """Generate deterministic unique tasks for one family/variant."""

    if family not in GENERATORS:
        raise ValueError(f"unknown task family: {family}")
    generator = np.random.default_rng(seed)
    blocked = set(blocked_program_hashes or ())
    output: list[ProgramTraceTask] = []
    attempts = 0
    maximum_attempts = max(1000, n * 100)
    while len(output) < n and attempts < maximum_attempts:
        # Duplicate rejection must not change the declared horizon balance.
        horizon = int(horizons[len(output) % len(horizons)])
        task = GENERATORS[family](
            generator,
            variant=variant,
            horizon=horizon,
            seed=seed,
            index=attempts,
        )
        attempts += 1
        if task.program_hash in blocked:
            continue
        blocked.add(task.program_hash)
        output.append(task)
    if len(output) != n:
        raise RuntimeError(
            f"generated {len(output)}/{n} unique {family}:{variant} tasks"
        )
    return output


def verify_disjoint_domains(domains: dict[str, list[ProgramTraceTask]]) -> None:
    seen_examples: dict[str, str] = {}
    seen_programs: dict[str, str] = {}
    for domain, tasks in domains.items():
        for task in tasks:
            if task.example_id in seen_examples:
                raise ValueError(
                    f"example overlap: {task.example_id} in {seen_examples[task.example_id]} and {domain}"
                )
            if task.program_hash in seen_programs:
                raise ValueError(
                    f"program overlap: {task.program_hash} in {seen_programs[task.program_hash]} and {domain}"
                )
            seen_examples[task.example_id] = domain
            seen_programs[task.program_hash] = domain
