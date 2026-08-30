"""Deterministic, domain-separated datasets for protocol v3.1."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from jclosure.datasets import TaskExample, normalize_prompt, stable_id


def expression_key(example: TaskExample) -> str:
    variables = example.variables
    values = tuple(variables.get(key) for key in ("a", "b", "c", "d"))
    return hashlib.sha256(repr(values).encode()).hexdigest()


def disjoint_arithmetic_domains(
    declarations: dict[str, dict[str, int]],
    *,
    blocked_prompt_hashes: set[str] | None = None,
    blocked_expression_hashes: set[str] | None = None,
) -> dict[str, list[TaskExample]]:
    """Generate exact-size domains while excluding every earlier selected item."""

    prompt_hashes = set(blocked_prompt_hashes or ())
    expression_hashes = set(blocked_expression_hashes or ())
    output: dict[str, list[TaskExample]] = {}
    for domain, declaration in declarations.items():
        required = int(declaration["candidates"])
        seed = int(declaration["seed"])
        pool = generate_modular_arithmetic(required * 4, seed=seed)
        selected: list[TaskExample] = []
        for example in pool:
            prompt_hash = hashlib.sha256(
                normalize_prompt(example.prompt).encode()
            ).hexdigest()
            expression_hash = expression_key(example)
            if prompt_hash in prompt_hashes or expression_hash in expression_hashes:
                continue
            prompt_hashes.add(prompt_hash)
            expression_hashes.add(expression_hash)
            selected.append(example)
            if len(selected) == required:
                break
        if len(selected) != required:
            raise RuntimeError(
                f"v3.1 domain {domain} produced {len(selected)}/{required} unique items"
            )
        output[domain] = selected
    return output


def generate_modular_arithmetic(n: int, *, seed: int) -> list[TaskExample]:
    """Generate scalable parity arithmetic with single-token balanced labels.

    The name is retained for the pre-freeze API, but the declared AST is an
    integer parity predicate.  This avoids selecting only one class from a
    teacher that cannot reliably solve the earlier multi-operation candidate
    domain with explicit thinking disabled.
    """

    generator = np.random.default_rng(seed)
    examples: list[TaskExample] = []
    seen: set[int] = set()
    while len(examples) < n:
        value = int(generator.integers(10_000, 1_000_000_000))
        if value in seen:
            continue
        seen.add(value)
        answer = "yes" if value % 2 == 0 else "no"
        template_index = len(examples) % 4
        prompts = (
            f"Is the integer {value} even? Respond with exactly yes or no:",
            f"Parity question: {value} is even. Is this true? Answer exactly yes or no:",
            f"Does {value} have even parity? Output only yes or no:",
            f"Arithmetic classification: is {value} divisible by 2? Reply exactly yes or no:",
        )
        examples.append(
            TaskExample(
                example_id=stable_id("v3.1-parity-arithmetic", value),
                family="arithmetic",
                template_id=f"parity(integer):v{template_index}",
                prompt=prompts[template_index],
                answer=answer,
                intermediates=(str(value % 10), "even" if value % 2 == 0 else "odd"),
                variables={
                    "a": value,
                    "b": 0,
                    "c": 1,
                    "d": 0,
                    "predicate": "divisible_by_2",
                },
            )
        )
    return examples


@dataclass(frozen=True)
class SequentialTask:
    example_id: str
    family: str
    template_id: str
    prompt: str
    semantic_actions: tuple[str, ...]
    final_answer: str
    length: int
    generator_seed: int
    generator_index: int
    program_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_iterated_modular_arithmetic(
    n: int, *, seed: int, lengths: tuple[int, ...] = (8, 16, 32)
) -> list[SequentialTask]:
    generator = np.random.default_rng(seed)
    output: list[SequentialTask] = []
    for index in range(n):
        length = int(lengths[index % len(lengths)])
        start = int(generator.integers(0, 10))
        operations: list[tuple[str, int]] = []
        values: list[str] = []
        current = start
        for _ in range(length):
            operation = "+" if int(generator.integers(0, 2)) == 0 else "*"
            operand = int(generator.integers(1, 10))
            operations.append((operation, operand))
            current = (
                (current + operand) % 10
                if operation == "+"
                else (current * operand) % 10
            )
            values.append(str(current))
        program = ", ".join(f"{op}{value}" for op, value in operations)
        digest = hashlib.sha256(
            repr((start, operations, seed, index)).encode()
        ).hexdigest()
        prompt = (
            f"Start at digit {start}. Apply these operations modulo 10: {program}. "
            f"Output exactly {length} resulting digits separated by spaces, and no other text:"
        )
        output.append(
            SequentialTask(
                example_id=f"memory-arithmetic-{digest[:20]}",
                family="iterated_modular_arithmetic",
                template_id=f"mod10-chain-l{length}",
                prompt=prompt,
                semantic_actions=tuple(values),
                final_answer=values[-1],
                length=length,
                generator_seed=seed,
                generator_index=index,
                program_hash=digest,
            )
        )
    return output


def generate_sequential_state_machines(
    n: int, *, seed: int, lengths: tuple[int, ...] = (8, 16, 32)
) -> list[SequentialTask]:
    generator = np.random.default_rng(seed)
    names = tuple("ABCDEF")
    output: list[SequentialTask] = []
    for index in range(n):
        length = int(lengths[index % len(lengths)])
        transition0 = generator.permutation(len(names)).tolist()
        transition1 = generator.permutation(len(names)).tolist()
        start = int(generator.integers(0, len(names)))
        inputs = generator.integers(0, 2, size=length).tolist()
        current = start
        states: list[str] = []
        for symbol in inputs:
            current = (transition0 if symbol == 0 else transition1)[current]
            states.append(names[current])
        table = "; ".join(
            f"{names[state]}:0->{names[transition0[state]]},1->{names[transition1[state]]}"
            for state in range(len(names))
        )
        digest = hashlib.sha256(
            repr((transition0, transition1, start, inputs, seed, index)).encode()
        ).hexdigest()
        prompt = (
            f"State machine transitions are {table}. Start at {names[start]}. "
            f"Read input {''.join(map(str, inputs))}. Output exactly {length} visited "
            "states separated by spaces, and no other text:"
        )
        output.append(
            SequentialTask(
                example_id=f"memory-machine-{digest[:20]}",
                family="synthetic_state_machine",
                template_id=f"six-state-l{length}",
                prompt=prompt,
                semantic_actions=tuple(states),
                final_answer=states[-1],
                length=length,
                generator_seed=seed,
                generator_index=index,
                program_hash=digest,
            )
        )
    return output


def deterministic_memory_split(
    tasks: list[SequentialTask],
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[SequentialTask]]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("memory split fractions must sum to one")
    grouped: dict[tuple[str, int], list[SequentialTask]] = {}
    for task in tasks:
        grouped.setdefault((task.family, task.length), []).append(task)
    output: dict[str, list[SequentialTask]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: item.program_hash)
        first = int(round(len(ordered) * fractions[0]))
        second = int(round(len(ordered) * (fractions[0] + fractions[1])))
        output["train"].extend(ordered[:first])
        output["validation"].extend(ordered[first:second])
        output["test"].extend(ordered[second:])
    return output
