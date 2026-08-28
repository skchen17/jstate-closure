"""Prepare and hash the output-independent Phase 0 v2 holdout datasets."""

from __future__ import annotations

import argparse
import json

from jclosure.datasets import (
    fresh_probe_swap_multihop,
    generate_phase0_order_ops_holdout,
    normalize_prompt,
    task_examples_to_json,
    upstream_multihop,
    upstream_order_ops,
)
from jclosure.experiments.common import repository_root
from jclosure.provenance import sha256_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", default="data/phase0_v2")
    args = parser.parse_args()
    root = repository_root()
    upstream = root / "data/upstream/anthropic"
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    multihop = fresh_probe_swap_multihop(upstream)
    calibration_order = upstream_order_ops(upstream)
    order = generate_phase0_order_ops_holdout(
        args.count,
        seed=args.seed,
        calibration_prompts={normalize_prompt(item.prompt) for item in calibration_order},
    )
    multihop_path = output / "fresh_multihop.json"
    order_path = output / "fresh_order_ops.json"
    write_json_atomic(multihop_path, task_examples_to_json(multihop))
    write_json_atomic(order_path, task_examples_to_json(order))
    calibration = [*upstream_multihop(upstream), *calibration_order]
    calibration_prompts = {normalize_prompt(item.prompt) for item in calibration}
    fresh_prompts = {normalize_prompt(item.prompt) for item in [*multihop, *order]}
    overlap = sorted(calibration_prompts & fresh_prompts)
    if overlap:
        raise RuntimeError(f"fresh holdout overlaps calibration: {overlap[:3]}")
    manifest = {
        "schema_version": 2,
        "protocol_version": "phase0_protocol_v2",
        "seed": args.seed,
        "counts": {"factual_two_hop": len(multihop), "order_of_operations": len(order)},
        "calibration_prompt_count": len(calibration_prompts),
        "fresh_unique_prompt_count": len(fresh_prompts),
        "calibration_overlap_count": 0,
        "files": {
            str(multihop_path.relative_to(root)): sha256_file(multihop_path),
            str(order_path.relative_to(root)): sha256_file(order_path),
        },
    }
    write_json_atomic(output / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
