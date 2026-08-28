"""Freeze the calibrated exploratory-v3 behavioral protocol."""

from __future__ import annotations

import argparse
import json

from jclosure.experiments.common import repository_root
from jclosure.protocol_v3 import create_v3_freeze, verify_v3_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry_v3.yaml")
    parser.add_argument(
        "--calibration", default="results/v3/processed/clamp_v3_calibration.json"
    )
    args = parser.parse_args()
    root = repository_root()
    payload = create_v3_freeze(
        root,
        config_path=root / args.config,
        calibration_path=root / args.calibration,
    )
    verified = verify_v3_freeze(root, require_behavioral_authorization=False)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "eligible_protocols": sorted(payload["eligible_protocols"]),
                "verified": verified["protocol_version"],
                "path": str(root / "artifacts/phase3_protocol_v3.freeze.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
