"""Prepare and hash disjoint v3.1 causal and compact-memory inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jclosure.config import load_config
from jclosure.datasets import normalize_prompt, task_examples_to_json
from jclosure.datasets_v3_1 import (
    deterministic_memory_split,
    disjoint_arithmetic_domains,
    expression_key,
    generate_iterated_modular_arithmetic,
    generate_sequential_state_machines,
)
from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic


def _old_hashes(root: Path) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    expressions: set[str] = set()
    for path in sorted((root / "results/v3/raw").glob("*/activation_bank_manifest.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt_hash = row.get("prompt_hash")
            if prompt_hash:
                prompts.add(str(prompt_hash))
            variables = row.get("variables", {})
            if all(key in variables for key in ("a", "b", "c", "d")):
                surrogate = type("Example", (), {"variables": variables})()
                expressions.add(expression_key(surrogate))  # type: ignore[arg-type]
    return prompts, expressions


def prepare(root: Path) -> dict[str, object]:
    closure_config = load_config(root / "configs/closure_v3_1.yaml")
    memory_config = load_config(root / "configs/compact_memory_v3_1.yaml")
    old_prompts, old_expressions = _old_hashes(root)
    declarations = {
        key: dict(value) for key, value in closure_config["v3_1_data"].items()
    }
    domains = disjoint_arithmetic_domains(
        declarations,
        blocked_prompt_hashes=old_prompts,
        blocked_expression_hashes=old_expressions,
    )
    data_root = root / "data/v3_1"
    data_root.mkdir(parents=True, exist_ok=True)
    domain_files: dict[str, dict[str, object]] = {}
    seen_prompts: set[str] = set(old_prompts)
    seen_expressions: set[str] = set(old_expressions)
    for domain, examples in domains.items():
        path = data_root / f"arithmetic_{domain}.json"
        write_json_atomic(path, task_examples_to_json(examples))
        current_prompts = {
            hashlib.sha256(normalize_prompt(item.prompt).encode()).hexdigest()
            for item in examples
        }
        current_expressions = {expression_key(item) for item in examples}
        overlap = bool(current_prompts & seen_prompts or current_expressions & seen_expressions)
        if overlap:
            raise RuntimeError(f"v3.1 {domain} overlaps an earlier domain")
        seen_prompts.update(current_prompts)
        seen_expressions.update(current_expressions)
        domain_files[domain] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "count": len(examples),
            "seed": int(declarations[domain]["seed"]),
        }

    memory = memory_config["compact_memory"]
    maximum = int(memory["maximum_candidates_per_family"])
    lengths = tuple(int(value) for value in memory["balanced_lengths"])
    memory_tasks = [
        *generate_iterated_modular_arithmetic(maximum, seed=20260905, lengths=lengths),
        *generate_sequential_state_machines(maximum, seed=20260906, lengths=lengths),
    ]
    splits = deterministic_memory_split(
        memory_tasks,
        fractions=tuple(float(value) for value in memory["split_fractions"]),
    )
    memory_files: dict[str, dict[str, object]] = {}
    program_hashes: set[str] = set()
    for split, tasks in splits.items():
        hashes = {task.program_hash for task in tasks}
        if hashes & program_hashes:
            raise RuntimeError("compact-memory program leaked across splits")
        program_hashes.update(hashes)
        path = data_root / f"compact_memory_{split}.json"
        write_json_atomic(
            path,
            {
                "schema_version": 4,
                "protocol_version": "compact_memory_exploratory_v3_1",
                "split": split,
                "items": [task.to_dict() for task in tasks],
            },
        )
        memory_files[split] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "count": len(tasks),
        }
    manifest = {
        "schema_version": 4,
        "protocol_version": "corrective_exploratory_protocol_v3_1",
        "baseline_commit": "6919d445ba82f8604df109d394517f51bdf2dc0a",
        "old_prompt_hashes_considered": len(old_prompts),
        "old_expression_hashes_considered": len(old_expressions),
        "causal_domains": domain_files,
        "memory_splits": memory_files,
        "cross_domain_overlap": False,
    }
    target = root / "artifacts/v3_1_data_manifest.json"
    write_json_atomic(target, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(repository_root()))
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.root).resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
