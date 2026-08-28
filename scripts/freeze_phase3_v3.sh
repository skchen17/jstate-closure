#!/usr/bin/env bash
set -euo pipefail
python -m jclosure.experiments.freeze_phase3_v3 \
  --config "${CONFIG:-configs/geometry_v3.yaml}" "$@"
