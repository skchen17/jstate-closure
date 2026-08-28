"""Create the immutable Phase 0 v2 confirmatory protocol manifest."""

from __future__ import annotations

import argparse
import json

from jclosure.config import load_config
from jclosure.experiments.common import repository_root
from jclosure.protocol import build_protocol_freeze, write_protocol_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0_v2_confirmatory.yaml")
    parser.add_argument("--calibration", default="results/processed/phase0_v2_calibration.json")
    parser.add_argument("--output", default="artifacts/phase0_protocol_v2.freeze.json")
    args = parser.parse_args()
    root = repository_root()
    config_path = root / args.config
    config = load_config(config_path)
    calibration = json.loads((root / args.calibration).read_text(encoding="utf-8"))
    band = [int(layer) for layer in calibration["workspace_band"]]
    positive_layer = int(calibration["positive_control_layer"])
    configured_band = [int(layer) for layer in config["phase0_v2"]["workspace_band"]]
    configured_positive = config["phase0_v2"]["positive_control_layer"]
    if configured_band != band or int(configured_positive) != positive_layer:
        raise RuntimeError(
            "confirmatory config must exactly contain the calibration-selected band and positive layer"
        )
    tracked = [
        "src/jclosure/phase0.py",
        "src/jclosure/datasets.py",
        "src/jclosure/experiments/validate_lens_v2.py",
        "data/phase0_v2/MANIFEST.json",
        "data/phase0_v2/fresh_multihop.json",
        "data/phase0_v2/fresh_order_ops.json",
    ]
    manifest = build_protocol_freeze(
        root=root,
        config_path=config_path,
        workspace_band=band,
        positive_control_layer=positive_layer,
        tracked_files=tracked,
        thresholds={
            "hit10": float(config["validation"]["hit10_threshold"]),
            "rank_advantage_ci_lower": 0.0,
            "positive_control_vs_null": "ci_lower_gt_q99_null_and_1e-4_floor",
        },
        exclusion_policy={
            "official_main_positions": "all_valid",
            "copied_targets": "included_and_flagged",
            "position16": "sensitivity_only",
            "tokenization": "single_token_candidates_required",
        },
        notes=(
            "Exact Anthropic synonym implementation was not present in the pinned public repository.",
            "Any later protocol change must use a new exploratory protocol version.",
        ),
    )
    write_protocol_freeze(root / args.output, manifest)
    print(json.dumps(manifest.to_dict(), indent=2))


if __name__ == "__main__":
    main()
