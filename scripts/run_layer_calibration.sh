#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/phase0_v2_confirmatory.yaml}"
python -m jclosure.experiments.calibrate_layers --config "$CONFIG" ${ARGS:-}
