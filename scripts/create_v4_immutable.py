#!/usr/bin/env python3
"""Create the one-time byte guard for frozen v4 inputs and results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic

BASELINE = "63f4869bbe2542551b0e4b34a44e0f9ae622d133"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    names = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", BASELINE],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    selected = []
    for name in names:
        if name == "artifacts/program_tasks_v4.freeze.json":
            selected.append(name)
        elif name == "configs/predictive_v4.yaml":
            selected.append(name)
        elif name.startswith("data/v4/"):
            selected.append(name)
        elif name.startswith("results/v4/"):
            selected.append(name)
        elif name == "reports/PREDICTIVE_JSTATE_V4.md":
            selected.append(name)
    files = {name: sha256_file(root / name) for name in sorted(set(selected))}
    write_json_atomic(
        root / "artifacts/v4_immutable.sha256.json",
        {"schema_version": 1, "baseline_commit": BASELINE, "files": files},
    )
    print(json.dumps({"baseline_commit": BASELINE, "files": len(files)}))


if __name__ == "__main__":
    main()
