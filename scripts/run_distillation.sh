#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/confirm.yaml}"
python -m jclosure.experiments.distill_controller --config "$CONFIG" ${ARGS:-}
