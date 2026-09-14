#!/usr/bin/env python3
"""Verify that protocol-v4 frozen files remain byte-identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.provenance import sha256_file


def verify(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        if observed != expected:
            failures.append(
                {"path": relative, "expected": expected, "observed": observed}
            )
    if failures:
        raise RuntimeError(f"v4 immutable guard failed: {failures}")
    return {
        "status": "PASSED",
        "baseline_commit": manifest["baseline_commit"],
        "verified_files": len(manifest["files"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="artifacts/v4_immutable.sha256.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(verify(root, root / args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
