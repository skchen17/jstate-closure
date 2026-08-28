#!/usr/bin/env python3
"""Create or verify the immutable Phase 0 v1/v2 SHA-256 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jclosure.baseline_guard import verify_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--manifest", default="artifacts/phase0_v2_immutable.sha256.json"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / args.manifest
    if args.write:
        write_manifest(root, manifest)
    print(json.dumps(verify_manifest(root, manifest), sort_keys=True))


if __name__ == "__main__":
    main()
