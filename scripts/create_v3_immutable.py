#!/usr/bin/env python3
"""Create the one-time baseline v3 immutable hash manifest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jclosure.provenance import sha256_file, write_json_atomic

BASELINE = "6919d445ba82f8604df109d394517f51bdf2dc0a"


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
        if name == "artifacts/phase3_protocol_v3.freeze.json":
            selected.append(name)
        elif name.startswith("configs/") and "v3" in name:
            selected.append(name)
        elif name.startswith("results/v3/"):
            selected.append(name)
        elif name.startswith("reports/") and name != "reports/FINAL_REPORT.md":
            selected.append(name)
    files = {name: sha256_file(root / name) for name in sorted(set(selected))}
    write_json_atomic(
        root / "artifacts/v3_immutable.sha256.json",
        {"schema_version": 1, "baseline_commit": BASELINE, "files": files},
    )
    print(json.dumps({"baseline_commit": BASELINE, "files": len(files)}))


if __name__ == "__main__":
    main()
