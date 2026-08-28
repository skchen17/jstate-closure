#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/pilot.yaml}"
python -m jclosure.experiments.closure --config "$CONFIG" ${ARGS:-}
