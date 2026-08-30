"""Create protocol-v3.1 closure or compact-memory freeze manifests."""

from __future__ import annotations

import argparse
import json

from jclosure.experiments.common import repository_root
from jclosure.protocol_v3_1 import create_closure_freeze, create_memory_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("closure", "memory"), required=True)
    parser.add_argument(
        "--calibration",
        default="results/v3_1/processed/closure_v3_1_calibration.json",
    )
    args = parser.parse_args()
    root = repository_root()
    payload = (
        create_closure_freeze(root, calibration_path=args.calibration)
        if args.kind == "closure"
        else create_memory_freeze(root)
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
