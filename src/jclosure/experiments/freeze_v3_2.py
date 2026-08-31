"""Create additive v3.2 freeze manifests."""

from __future__ import annotations

import argparse
import json

from jclosure.experiments.common import repository_root
from jclosure.protocol_v3_2 import (
    create_calibration_freeze,
    create_closure_freeze,
    create_memory_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("calibration", "closure", "memory"), required=True)
    parser.add_argument("--calibration", default="results/v3_2/processed/closure_v3_2_calibration.json")
    args = parser.parse_args()
    root = repository_root()
    if args.kind == "calibration":
        payload = create_calibration_freeze(root)
    elif args.kind == "closure":
        payload = create_closure_freeze(root, args.calibration)
    else:
        payload = create_memory_freeze(root)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
