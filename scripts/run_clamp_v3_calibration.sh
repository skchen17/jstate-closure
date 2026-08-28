#!/usr/bin/env bash
set -euo pipefail
timeout 12h python -m jclosure.experiments.clamp_v3_calibration \
  --config "${CONFIG:-configs/geometry_v3.yaml}" "$@"
