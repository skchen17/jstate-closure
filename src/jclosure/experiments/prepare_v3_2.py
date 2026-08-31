"""Prepare deterministic, non-overlapping causal domains for protocol v3.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jclosure.config import load_config
from jclosure.datasets import (
    normalize_prompt,
    task_examples_from_json,
    task_examples_to_json,
)
from jclosure.datasets_v3_1 import disjoint_arithmetic_domains, expression_key
from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic


def _blocked_inputs(root: Path) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    expressions: set[str] = set()
    for manifest_path in (
        root / "artifacts/v3_1_data_manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for value in manifest.get("causal_domains", {}).values():
            for example in task_examples_from_json(root / value["path"]):
                prompts.add(
                    hashlib.sha256(normalize_prompt(example.prompt).encode()).hexdigest()
                )
                expressions.add(expression_key(example))
    for bank in sorted((root / "results/v3/raw").glob("*/activation_bank_manifest.jsonl")):
        for line in bank.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("prompt_hash"):
                prompts.add(str(row["prompt_hash"]))
            variables = row.get("variables", {})
            if all(key in variables for key in ("a", "b", "c", "d")):
                surrogate = type("Example", (), {"variables": variables})()
                expressions.add(expression_key(surrogate))  # type: ignore[arg-type]
    return prompts, expressions


def prepare(root: Path) -> dict[str, object]:
    config = load_config(root / "configs/closure_v3_2.yaml")
    declarations = {key: dict(value) for key, value in config["v3_2_data"].items()}
    blocked_prompts, blocked_expressions = _blocked_inputs(root)
    domains = disjoint_arithmetic_domains(
        declarations,
        blocked_prompt_hashes=blocked_prompts,
        blocked_expression_hashes=blocked_expressions,
    )
    data_root = root / "data/v3_2"
    data_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    seen_prompts = set(blocked_prompts)
    seen_expressions = set(blocked_expressions)
    for domain, examples in domains.items():
        current_prompts = {
            hashlib.sha256(normalize_prompt(item.prompt).encode()).hexdigest()
            for item in examples
        }
        current_expressions = {expression_key(item) for item in examples}
        if current_prompts & seen_prompts or current_expressions & seen_expressions:
            raise RuntimeError(f"v3.2 {domain} overlaps an earlier domain")
        seen_prompts.update(current_prompts)
        seen_expressions.update(current_expressions)
        path = data_root / f"arithmetic_{domain}.json"
        write_json_atomic(path, task_examples_to_json(examples))
        files[domain] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "count": len(examples),
            "seed": int(declarations[domain]["seed"]),
        }
    manifest: dict[str, object] = {
        "schema_version": 5,
        "protocol_version": "corrective_causal_protocol_v3_2",
        "baseline_commit": "a0d164e1167fae21a7d3e7ec41dae9fd7e819019",
        "blocked_prompt_count": len(blocked_prompts),
        "blocked_expression_count": len(blocked_expressions),
        "causal_domains": files,
        "cross_domain_overlap": False,
    }
    write_json_atomic(root / "artifacts/v3_2_data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(repository_root()))
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.root).resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
