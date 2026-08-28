#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/phase0_v2_calibration.yaml}"
python -m jclosure.experiments.validate_lens_v2 --config "$CONFIG" ${ARGS:-}
