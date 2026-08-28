#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/confirm.yaml}"
python -m jclosure.experiments.natural_collisions --config "$CONFIG" ${ARGS:-}
