"""Response-blind, externally labelled paired tasks for V40."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

FAMILIES = (
    "variable_binding",
    "state_transition",
    "long_context_dependency",
    "boolean_logic",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def make_task(family: str, seed: int, index: int) -> dict[str, Any]:
    """Produce one natural, single-symbol counterfactual with an external answer."""
    if family not in FAMILIES:
        raise ValueError(family)
    rng = random.Random(digest(["v40", family, seed, index]))
    if family == "variable_binding":
        other = rng.randrange(10)
        a, b = rng.sample([x for x in range(10) if x != other], 2)
        prefix = f"Bindings: Bob={other}; Alice=\n"
        suffix = "\nQuestion: What is Alice's bound value? Answer with one digit:"
        answers = [str(a), str(b)]
        ast = {"other": other, "recipient_value": a, "donor_value": b}
    elif family == "state_transition":
        a, b = rng.sample(tuple("ABCD"), 2)
        steps = rng.choice((1, 2, 3))
        cycle = "".join(rng.sample(tuple("ABCD"), 4))
        prefix = (f"State cycle: {cycle[0]}->{cycle[1]}, {cycle[1]}->{cycle[2]}, "
                  f"{cycle[2]}->{cycle[3]}, {cycle[3]}->{cycle[0]}. Current state=\n")
        suffix = f"\nAfter {steps} transition(s), which state? Answer with one letter:"
        answers = [cycle[(cycle.index(x) + steps) % 4] for x in (a, b)]
        ast = {"recipient_state": a, "donor_state": b, "steps": steps,
               "cycle": cycle}
    elif family == "long_context_dependency":
        a, b = rng.sample(range(10), 2)
        archive = rng.randrange(100_000, 999_999)
        filler = " ".join(f"Archive line {i}: neutral context." for i in range(1, 13))
        prefix = f"Archive #{archive}. The relevant code is\n"
        suffix = f"\n{filler} Question: What was the relevant code? Answer with one digit:"
        answers = [str(a), str(b)]
        ast = {"archive": archive, "recipient_code": a, "donor_code": b,
               "neutral_lines": 12}
    else:
        a, b = (0, 1) if rng.randrange(2) == 0 else (1, 0)
        fixed = rng.randrange(2)
        scenario = rng.randrange(1_000_000)
        prefix = f"Boolean scenario #{scenario}. Rule: XOR. A={fixed}; B=\n"
        suffix = "\nQuestion: What is A XOR B? Answer with one bit:"
        answers = [str(fixed ^ x) for x in (a, b)]
        ast = {"scenario": scenario, "fixed_A": fixed,
               "recipient_B": a, "donor_B": b}
    fork_values = [str(ast[k]) for k in (
        ("recipient_value", "donor_value") if family == "variable_binding" else
        ("recipient_state", "donor_state") if family == "state_transition" else
        ("recipient_code", "donor_code") if family == "long_context_dependency" else
        ("recipient_B", "donor_B"))]
    assert len(fork_values) == 2 and fork_values[0] != fork_values[1]
    assert answers[0] != answers[1]
    program = {"family": family, "ast": ast, "prefix": prefix, "suffix": suffix}
    program_hash = digest(program)
    return {
        "state_id": f"v40-{family}-{program_hash[:20]}",
        "family": family, "program_hash": program_hash,
        "generator_seed": seed, "generator_index": index,
        "prefix": prefix, "suffix": suffix,
        "recipient_fork": fork_values[0], "donor_fork": fork_values[1],
        "recipient_prompt": prefix + fork_values[0] + suffix,
        "donor_prompt": prefix + fork_values[1] + suffix,
        "recipient_answer": answers[0], "donor_answer": answers[1],
        "ast": ast, "labels_generated_before_model_execution": True,
    }


def generate(family: str, count: int, seed: int, blocked: set[str]) -> list[dict[str, Any]]:
    items = []
    attempt = 0
    while len(items) < count and attempt < max(10_000, count * 200):
        task = make_task(family, seed, attempt)
        attempt += 1
        if task["program_hash"] in blocked:
            continue
        blocked.add(task["program_hash"])
        items.append(task)
    if len(items) != count:
        raise RuntimeError(f"V40 pool shortage {family}: {len(items)}/{count}")
    return items
