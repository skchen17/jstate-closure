#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/confirm.yaml}"
python -m jclosure.experiments.closure --config "$CONFIG" ${ARGS:-}
