#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/confirm.yaml}"
python -m jclosure.experiments.memory_order --config "$CONFIG" ${ARGS:-}
