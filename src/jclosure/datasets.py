"""Pinned upstream loaders and deterministic procedural reasoning datasets."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskExample:
    example_id: str
    family: str
    template_id: str
    prompt: str
    answer: str
    intermediates: tuple[str, ...] = ()
    variables: dict[str, Any] = field(default_factory=dict)
    facts: tuple[tuple[str, str, str], ...] = ()


def stable_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def load_upstream_json(data_root: str | Path, relative: str) -> dict[str, Any]:
    root = Path(data_root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("upstream data path escapes data root")
    return json.loads(path.read_text(encoding="utf-8"))


def upstream_multihop(data_root: str | Path) -> list[TaskExample]:
    payload = load_upstream_json(data_root, "evaluations/lens-eval-multihop.json")
    return [
        TaskExample(
            example_id=str(item["name"]),
            family="factual_two_hop",
            template_id=str(item["name"]).split("-")[0],
            prompt=str(item["prompt"]),
            answer=str(item["target"]),
            intermediates=tuple(str(value) for value in item["intermediates"]),
        )
        for item in payload["items"]
    ]


def upstream_order_ops(data_root: str | Path) -> list[TaskExample]:
    payload = load_upstream_json(data_root, "evaluations/lens-eval-order-ops.json")
    return [
        TaskExample(
            example_id=str(item["name"]),
            family="order_of_operations",
            template_id=str(item["name"]).split("-")[0],
            prompt=str(item["prompt"]),
            answer=str(item["target"]),
            intermediates=tuple(str(value) for value in item["intermediates"]),
        )
        for item in payload["items"]
    ]


def generate_arithmetic(
    n: int,
    *,
    seed: int = 2_026_082_8,
    max_value: int = 20,
) -> list[TaskExample]:
    generator = np.random.default_rng(seed)
    examples: list[TaskExample] = []
    seen: set[tuple[int, int, int, int]] = set()
    while len(examples) < n:
        a, b = (int(value) for value in generator.integers(0, max_value + 1, size=2))
        c = int(generator.integers(1, 10))
        d = int(generator.integers(0, max_value + 1))
        values = (a, b, c, d)
        if values in seen:
            continue
        seen.add(values)
        intermediate = a + b
        answer = intermediate * c + d
        template_index = len(examples) % 4
        prompts = (
            f"({a} + {b}) * {c} + {d} =",
            f"Compute exactly: add {a} and {b}, multiply by {c}, then add {d}. Answer:",
            f"Start with {a} + {b}; take that result times {c}; increase it by {d}. Result:",
            f"Evaluate the integer expression (({a}+{b})*{c})+{d}. The value is",
        )
        prompt = prompts[template_index]
        examples.append(
            TaskExample(
                example_id=stable_id("arithmetic", *values),
                family="arithmetic",
                template_id=f"(a+b)*c+d:v{template_index}",
                prompt=prompt,
                answer=str(answer),
                intermediates=(str(intermediate), str(intermediate * c)),
                variables={"a": a, "b": b, "c": c, "d": d, "sum": intermediate},
            )
        )
    return examples


def generate_equal_intermediate_pairs(
    n_pairs: int,
    *,
    seed: int = 2_026_082_8,
) -> list[tuple[TaskExample, TaskExample]]:
    generator = np.random.default_rng(seed)
    pairs: list[tuple[TaskExample, TaskExample]] = []
    while len(pairs) < n_pairs:
        total = int(generator.integers(3, 31))
        a1 = int(generator.integers(0, total + 1))
        a2 = int(generator.integers(0, total + 1))
        if a1 == a2:
            continue
        b1, b2 = total - a1, total - a2
        c1, c2 = (int(value) for value in generator.choice(np.arange(1, 10), size=2, replace=False))
        d1, d2 = (int(value) for value in generator.integers(0, 20, size=2))
        left = TaskExample(
            example_id=stable_id("collision", len(pairs), "left", seed),
            family="arithmetic",
            template_id=f"equal-sum-pair-c{c1}-d{d1 % 2}",
            prompt=f"({a1} + {b1}) * {c1} + {d1} =",
            answer=str(total * c1 + d1),
            intermediates=(str(total), str(total * c1)),
            variables={"a": a1, "b": b1, "c": c1, "d": d1, "sum": total},
        )
        right = TaskExample(
            example_id=stable_id("collision", len(pairs), "right", seed),
            family="arithmetic",
            template_id=f"equal-sum-pair-c{c2}-d{d2 % 2}",
            prompt=f"({a2} + {b2}) * {c2} + {d2} =",
            answer=str(total * c2 + d2),
            intermediates=(str(total), str(total * c2)),
            variables={"a": a2, "b": b2, "c": c2, "d": d2, "sum": total},
        )
        pairs.append((left, right))
    return pairs


def generate_boolean(n: int, *, seed: int = 2_026_082_8) -> list[TaskExample]:
    generator = np.random.default_rng(seed)
    operators = ("and", "or", "xor")
    examples: list[TaskExample] = []
    for index in range(n):
        x, y, z = (bool(value) for value in generator.integers(0, 2, size=3))
        first, second = generator.choice(operators, size=2, replace=True)

        def apply(op: str, a: bool, b: bool) -> bool:
            return (a and b) if op == "and" else (a or b) if op == "or" else (a != b)

        intermediate = apply(str(first), x, y)
        answer = apply(str(second), intermediate, z)
        prompt = f"Let x={x}, y={y}, z={z}. Compute (x {first} y) {second} z. Answer True or False:"
        examples.append(
            TaskExample(
                example_id=stable_id("boolean", index, seed),
                family="boolean_logic",
                template_id=f"{first}-{second}",
                prompt=prompt,
                answer=str(answer),
                intermediates=(str(intermediate),),
                variables={"x": x, "y": y, "z": z, "first": str(first), "second": str(second)},
            )
        )
    return examples


def generate_graph_traversal(
    n: int, *, seed: int = 2_026_082_8, nodes: int = 7
) -> list[TaskExample]:
    generator = np.random.default_rng(seed)
    examples: list[TaskExample] = []
    for index in range(n):
        adjacency: dict[int, set[int]] = {node: set() for node in range(nodes)}
        for node in range(nodes - 1):
            adjacency[node].add(node + 1)
            adjacency[node + 1].add(node)
        for _ in range(nodes):
            left, right = (int(value) for value in generator.integers(0, nodes, size=2))
            if left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
        source, target = (int(value) for value in generator.choice(nodes, size=2, replace=False))
        queue = deque([(source, [source])])
        visited = {source}
        path: list[int] | None = None
        while queue:
            current, prefix = queue.popleft()
            if current == target:
                path = prefix
                break
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*prefix, neighbor]))
        assert path is not None
        edges = sorted((a, b) for a, values in adjacency.items() for b in values if a < b)
        edge_text = ", ".join(f"{a}-{b}" for a, b in edges)
        answer = str(path[1])
        examples.append(
            TaskExample(
                example_id=stable_id("graph", index, seed),
                family="graph_traversal",
                template_id="shortest-next-node",
                prompt=f"Undirected edges: {edge_text}. On a shortest path from {source} to {target}, the next node after {source} is",
                answer=answer,
                intermediates=tuple(str(value) for value in path[1:-1]),
                variables={"source": source, "target": target, "path": path},
            )
        )
    return examples


def generate_variable_binding(n: int, *, seed: int = 2_026_082_8) -> list[TaskExample]:
    generator = np.random.default_rng(seed)
    names = np.array(["amber", "birch", "cedar", "dawn", "ember", "frost", "grove", "hazel"])
    values = np.array(["circle", "square", "triangle", "star", "moon", "sun", "river", "stone"])
    examples: list[TaskExample] = []
    for index in range(n):
        chosen_names = generator.choice(names, size=4, replace=False)
        chosen_values = generator.choice(values, size=4, replace=False)
        query = int(generator.integers(0, 4))
        bindings = "; ".join(
            f"{name} means {value}"
            for name, value in zip(chosen_names, chosen_values, strict=True)
        )
        examples.append(
            TaskExample(
                example_id=stable_id("binding", index, seed),
                family="variable_binding",
                template_id="four-bindings",
                prompt=f"{bindings}. What does {chosen_names[query]} mean? Answer in one word:",
                answer=str(chosen_values[query]),
                intermediates=(str(chosen_names[query]),),
                variables={"query_index": query},
            )
        )
    return examples


def generate_symbolic_planning(n: int, *, seed: int = 2_026_082_8) -> list[TaskExample]:
    """Generate knowledge-light state updates with a held-out action template."""

    generator = np.random.default_rng(seed)
    objects = np.array(["amber", "birch", "cedar", "dawn", "ember", "frost"])
    locations = np.array(["red", "blue", "green", "white", "black", "gold"])
    examples: list[TaskExample] = []
    for index in range(n):
        chosen_objects = generator.choice(objects, size=3, replace=False)
        chosen_locations = generator.choice(locations, size=3, replace=False)
        first, second, query = (int(value) for value in generator.integers(0, 3, size=3))
        while second == first:
            second = int(generator.integers(0, 3))
        initial = ", ".join(
            f"{object_} at {location}"
            for object_, location in zip(chosen_objects, chosen_locations, strict=True)
        )
        state = dict(zip(chosen_objects.tolist(), chosen_locations.tolist(), strict=True))
        state[str(chosen_objects[first])], state[str(chosen_objects[second])] = (
            state[str(chosen_objects[second])],
            state[str(chosen_objects[first])],
        )
        answer = state[str(chosen_objects[query])]
        template = index % 3
        prompts = (
            f"Initial state: {initial}. Swap the locations of {chosen_objects[first]} and {chosen_objects[second]}. Where is {chosen_objects[query]}?",
            f"Objects are placed as follows: {initial}. Exchange {chosen_objects[first]} with {chosen_objects[second]}. Final location of {chosen_objects[query]}:",
            f"Given {initial}, perform SWAP({chosen_objects[first]}, {chosen_objects[second]}). Query {chosen_objects[query]} ->",
        )
        examples.append(
            TaskExample(
                example_id=stable_id("planning", index, seed),
                family="symbolic_planning",
                template_id=f"swap-location:v{template}",
                prompt=prompts[template],
                answer=answer,
                intermediates=(state[str(chosen_objects[first])],),
                variables={
                    "first": str(chosen_objects[first]),
                    "second": str(chosen_objects[second]),
                    "query": str(chosen_objects[query]),
                },
            )
        )
    return examples


def generate_state_machines(n: int, *, seed: int = 2_026_082_8) -> list[TaskExample]:
    """Generate random finite-state-machine rollouts with explicit transition tables."""

    generator = np.random.default_rng(seed)
    names = np.array(["A", "B", "C", "D"])
    examples: list[TaskExample] = []
    for index in range(n):
        zero = generator.permutation(4)
        one = generator.permutation(4)
        start = int(generator.integers(0, 4))
        inputs = generator.integers(0, 2, size=4)
        current = start
        intermediates: list[str] = []
        for symbol in inputs:
            current = int(zero[current] if symbol == 0 else one[current])
            intermediates.append(str(names[current]))
        table = "; ".join(
            f"{names[state]}:0->{names[zero[state]]},1->{names[one[state]]}"
            for state in range(4)
        )
        template = index % 3
        prompts = (
            f"State transitions are {table}. Start at {names[start]} and read {''.join(map(str, inputs))}. Final state:",
            f"Machine table: {table}. From {names[start]}, apply inputs {' '.join(map(str, inputs))}. End at",
            f"FSM [{table}] initial={names[start]} input={''.join(map(str, inputs))} output-state=",
        )
        examples.append(
            TaskExample(
                example_id=stable_id("state-machine", index, seed),
                family="state_machine",
                template_id=f"four-state:v{template}",
                prompt=prompts[template],
                answer=str(names[current]),
                intermediates=tuple(intermediates[:-1]),
                variables={"start": str(names[start]), "input": "".join(map(str, inputs))},
            )
        )
    return examples


def split_by_template(
    examples: list[TaskExample],
    *,
    seed: int = 2_026_082_8,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[TaskExample]]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    groups: dict[str, list[TaskExample]] = {}
    for example in examples:
        groups.setdefault(example.template_id, []).append(example)
    keys = sorted(groups)
    generator = np.random.default_rng(seed)
    generator.shuffle(keys)
    first = int(round(fractions[0] * len(keys)))
    second = int(round((fractions[0] + fractions[1]) * len(keys)))
    assignments = {
        "train": keys[:first],
        "validation": keys[first:second],
        "test": keys[second:],
    }
    return {
        split: [example for key in selected for example in groups[key]]
        for split, selected in assignments.items()
    }
